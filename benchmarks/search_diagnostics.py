"""Local DEV diagnostics and resource controls for reproducible notebook comparisons."""

from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import json
import marshal
import os
import re
import shutil
import sys
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np

from honestml import AutoML, CVConfig, HPOConfig, SearchConfig
from honestml.adapters import Reader, outer_holdout_carve
from honestml.application.search import bounded_probe_folds, probe_models
from honestml.application.slice import (
    EstimatorFactory,
    _CandidateFailed,
    _run_candidate,
    design_matrix,
)
from honestml.composition.build import Components, build_default_components
from honestml.composition.run_report import configure_run_cost
from honestml.core import Candidate, Dataset, FeatureSubsetSelector, Fold, RunContext, Task
from honestml.core.exceptions import AutoMLError
from honestml.core.ports.estimator import (
    Estimator,
    SupportsIterationBudget,
    SupportsIterationPlan,
    SupportsNativeModel,
    SupportsThreadLimit,
)
from honestml.core.schema import categorical_positions, native_routing
from honestml.core.task import resolve_positive

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "ops/loop/logs/training-improvements-20260908"
SOURCES = {
    "otto": ("05-otto-product-classification.ipynb", "otto", ("train.csv",)),
    "store": (
        "06-store-sales.ipynb",
        "store-sales",
        ("train.csv", "stores.csv", "oil.csv", "holidays_events.csv"),
    ),
    "adult": ("03-adult-income.ipynb", "adult", ("adult.csv",)),
    "credit": ("04-credit-card-fraud.ipynb", "creditcardfraud", ("creditcard.csv",)),
}
THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


@dataclass(frozen=True)
class ProbeResource:
    name: str
    rows: int = 4096
    iterations: int = 64
    folds: int = 1
    es_rows: int = 1
    merge_es: bool = False


def resource_grid(temporal: bool) -> tuple[ProbeResource, ...]:
    base = ProbeResource("reference")
    grid = (
        base,
        replace(base, name="rows", rows=16384),
        replace(base, name="iterations", iterations=256),
        replace(base, name="folds", folds=2),
        replace(base, name="without_es", merge_es=True),
    )
    return grid + ((replace(base, name="es_rows", es_rows=512),) if temporal else ())


def paired_order(seed_index: int, names: Sequence[str]) -> tuple[str, ...]:
    return tuple(names if seed_index % 2 == 0 else reversed(names))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


SOURCE_FILES = (
    "benchmarks/search_diagnostics.py",
    "src/honestml/application/search.py",
    "src/honestml/application/slice.py",
    "src/honestml/core/config.py",
    "src/honestml/composition/facade.py",
    "src/honestml/adapters/boosting.py",
    "src/honestml/adapters/feature_rankers.py",
    "src/honestml/adapters/significance.py",
)
IMPORT_SNAPSHOT = {name: (ROOT / name).read_bytes() for name in SOURCE_FILES}
IMPORT_HASHES = {
    name: hashlib.sha256(contents).hexdigest() for name, contents in IMPORT_SNAPSHOT.items()
}
ENTRYPOINT_HASHES = {
    function.__name__: hashlib.sha256(marshal.dumps(function.__code__)).hexdigest()
    for function in (bounded_probe_folds, probe_models, _run_candidate)
}


def preserve_sources(output: Path) -> None:
    for name, contents in IMPORT_SNAPSHOT.items():
        path = output / "sources" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents)
    write_json(
        output / "source-provenance.json",
        {
            "import_boundary_sha256": IMPORT_HASHES,
            "imported_entrypoint_bytecode_sha256": ENTRYPOINT_HASHES,
        },
    )


def verify_sources() -> None:
    changed = [name for name, digest in IMPORT_HASHES.items() if sha256(ROOT / name) != digest]
    if changed:
        raise RuntimeError(f"benchmark sources changed after import: {changed}")


def prerequisites(scenario: str, root: Path = ROOT) -> dict[str, Any]:
    notebook, directory, files = SOURCES[scenario]
    source = root / "notebooks" / notebook
    data = root / "notebooks/data" / directory
    return {
        "notebook": str(source),
        "source_sha256": sha256(source),
        "data": [
            {
                "path": str(data / name),
                "exists": (data / name).is_file(),
                "bytes": (data / name).stat().st_size if (data / name).is_file() else None,
            }
            for name in files
        ],
    }


def prepare_inputs(scenario: str, root: Path = ROOT) -> tuple[Any, np.ndarray, Any, dict[str, Any]]:
    """Execute only the trusted local notebook's data-preparation cell, without download or fit."""
    import pandas as pd

    manifest = prerequisites(scenario, root)
    missing = [entry["path"] for entry in manifest["data"] if not entry["exists"]]
    if missing:
        raise FileNotFoundError(str(missing))
    source = Path(manifest["notebook"])
    notebook = json.loads(source.read_text(encoding="utf-8"))
    cell = notebook["cells"][3]
    if cell["cell_type"] != "code":
        raise ValueError("expected notebook data-preparation code at cell 3")
    text = "".join(cell["source"])
    constants = {}
    for statement in ast.parse("".join(notebook["cells"][1]["source"])).body:
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            name = statement.targets[0]
            if isinstance(name, ast.Name) and name.id in ("SEED", "DATE_FROM"):
                constants[name.id] = ast.literal_eval(statement.value)
    namespace: dict[str, Any] = {
        "pd": pd,
        "np": np,
        "DATA": Path(manifest["data"][0]["path"]).parent,
        **constants,
    }
    exec(compile(text, f"{source}:cell3", "exec"), namespace)
    manifest["preparation_cell"] = 3
    manifest["preparation_sha256"] = hashlib.sha256(text.encode()).hexdigest()
    manifest["data"] = [
        dict(entry, sha256=sha256(Path(entry["path"]))) for entry in manifest["data"]
    ]
    return namespace["X"], np.asarray(namespace["y"]), namespace.get("tval"), manifest


def scenario_config(scenario: str, es_rows: int = 1) -> tuple[Task, CVConfig]:
    if scenario == "store":
        return Task(kind="regression", metric="rmse"), CVConfig(
            scheme="timeseries_period",
            period="week",
            n_splits=4,
            n_test=4,
            purge=1,
            embargo=1,
            n_es=es_rows,
            outer_holdout=0.2,
        )
    if scenario == "otto":
        return Task(kind="multiclass", metric="log_loss"), CVConfig(outer_holdout=0.2)
    return Task(kind="binary", metric="pr_auc" if scenario == "credit" else "roc_auc"), CVConfig(
        outer_holdout=0.2
    )


def instrument_fit(
    estimator: Estimator, audit: list[dict[str, Any]], family: str, threads: int
) -> None:
    """Observe one factory-created estimator fit, restoring its original instance before serialization."""
    original = estimator.fit
    missing = object()
    local = vars(estimator).get("fit", missing)

    def fit(
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> Estimator:
        try:
            result = original(X, y, X_val=X_val, y_val=y_val, sample_weight=sample_weight)
            audit.append(
                {
                    "family": family,
                    "rows": X.shape[0],
                    "columns": X.shape[1],
                    "es_rows": 0 if X_val is None else X_val.shape[0],
                    "iterations": estimator.fitted_iterations
                    if isinstance(estimator, SupportsIterationBudget)
                    else None,
                    **effective_native(estimator, threads),
                }
            )
            return result
        finally:
            if local is missing:
                delattr(estimator, "fit")
            else:
                setattr(estimator, "fit", local)

    setattr(estimator, "fit", fit)


def limit_factory(
    factory: EstimatorFactory,
    threads: int,
    iterations: int | None = None,
    *,
    audit: list[dict[str, Any]] | None = None,
    family: str = "unknown",
) -> EstimatorFactory:
    def make() -> Estimator:
        estimator = factory()
        if isinstance(estimator, SupportsThreadLimit):
            estimator.set_threads(threads)
        if iterations is not None and isinstance(estimator, SupportsIterationBudget):
            estimator.set_refit_iterations(iterations)
        if audit is not None:
            instrument_fit(estimator, audit, family, threads)
        return estimator

    return make


def limit_components(
    components: Components, threads: int, audit: list[dict[str, Any]] | None = None
) -> Components:
    for strategy in [
        components.feature_ranker,
        *(s for _, s in components.feature_strategies or ()),
    ]:
        if strategy is not None:
            if isinstance(strategy, SupportsThreadLimit):
                strategy.set_threads(threads)
            elif not isinstance(strategy, FeatureSubsetSelector):
                raise TypeError(f"ranker lacks thread-limit port: {type(strategy).__name__}")
    original = components.make_factory

    def make_factory(name: str, parameters: Mapping[str, Any]) -> EstimatorFactory:
        assert original is not None
        return limit_factory(original(name, parameters), threads, audit=audit, family=name)

    return components._replace(
        estimators={
            name: limit_factory(factory, threads, audit=audit, family=name)
            for name, factory in components.estimators.items()
        },
        make_factory=make_factory if original is not None else None,
    )


@contextmanager
def comparable_resources(threads: int = 4) -> Iterator[list[dict[str, Any]]]:
    """Apply common native/BLAS/FS worker limits and cap resolved HPO, including presets.

    Benchmark-only process-global patches require sequential execution. Kernel callers set
    THREAD_ENV before Python imports; threadpool_limits also constrains already loaded BLAS.
    """
    from threadpoolctl import threadpool_limits

    import honestml.adapters as adapters
    import honestml.adapters.feature_rankers as rankers
    import honestml.composition.facade as facade

    if threads < 1:
        raise ValueError("threads must be positive")
    original_build = facade.build_default_components
    original_hpo = AutoML._resolve_hpo
    original_proxy = adapters.make_ranker_fit_predict
    original_ranker = rankers._fit_ranker_model
    audit: list[dict[str, Any]] = []

    def build(*args: Any, **kwargs: Any) -> Components:
        return limit_components(original_build(*args, **kwargs), threads, audit)

    def resolve_hpo(model: AutoML, hpo: HPOConfig | None) -> HPOConfig | None:
        resolved = original_hpo(model, hpo)
        return (
            None
            if resolved is None
            else resolved.model_copy(update={"n_trials": min(5, resolved.n_trials)})
        )

    def proxy(*args: Any, **kwargs: Any) -> Any:
        kwargs["threads"] = threads
        return original_proxy(*args, **kwargs)

    def ranker(*args: Any, **kwargs: Any) -> Any:
        kwargs["threads"] = threads
        bound = inspect.signature(original_ranker).bind(*args, **kwargs)
        model = original_ranker(*args, **kwargs)
        workers = model.get_params()["n_jobs"]
        if workers != threads:
            raise ValueError(f"ranker worker mismatch: {workers}; expected {threads}")
        x = bound.arguments["x"]
        audit.append(
            {
                "family": "proxy",
                "rows": x.shape[0],
                "columns": x.shape[1],
                "es_rows": 0,
                "workers": {"n_jobs": workers},
                "iterations": len(model.estimators_),
                "tree_budget": model.get_params()["n_estimators"],
            }
        )
        return model

    with ExitStack() as stack:
        stack.enter_context(patch.dict(os.environ, {key: str(threads) for key in THREAD_ENV}))
        stack.enter_context(threadpool_limits(limits=threads))
        stack.enter_context(patch.object(facade, "build_default_components", build))
        stack.enter_context(patch.object(AutoML, "_resolve_hpo", resolve_hpo))
        stack.enter_context(patch.object(adapters, "make_ranker_fit_predict", proxy))
        stack.enter_context(patch.object(rankers, "_fit_ranker_model", ranker))
        yield audit


def notebook_setup(mode: str, seed: int, output: Path, threads: int = 4) -> str:
    """Kernel injection for unchanged notebook cells; use a fresh output directory and kernel.

    The caller rewrites only SEED, DATA/RESULTS locations and local leaderboard access, sets
    KAGGLE_SUBMIT=0, and saves the notebook source hash. Full notebook evaluation is separate
    from DEV-only probes and requires explicit execution authorization.
    """
    if mode not in ("baseline", "fast"):
        raise ValueError("mode must be baseline or fast")
    return f"""
import sys as _dsys
_dsys.path.insert(0, {str(ROOT)!r})
from benchmarks.search_diagnostics import comparable_resources as _dresources, verify_sources as _dverify, IMPORT_HASHES as _dhashes
import honestml as _dh
import functools as _df
import json as _dj
import time as _dt
from pathlib import Path as _dPath
_dout = _dPath({str(output)!r})
_dcontext = _dresources({threads})
_daudit = _dcontext.__enter__()
_dfit = _dh.AutoML.fit
_dcount = 0
def _diagnostic_fit(self, *args, **kwargs):
    global _dcount
    if self.cache is not None:
        raise ValueError("notebook comparisons require cache=None")
    if self.random_state != {seed}:
        raise ValueError("notebook SEED does not match declared seed")
    _dcount += 1
    path = _dout / ("fit-%02d.json" % _dcount)
    if path.exists():
        raise FileExistsError(path)
    record = {{"status": "running", "mode": {mode!r}, "seed": {seed},
               "native_threads": {threads}, "cache": "disabled", "hpo_cap_per_family": 5}}
    path.write_text(_dj.dumps(record), encoding="utf-8")
    audit_start = len(_daudit)
    started = _dt.perf_counter()
    try:
        result = _dfit(self, *args, **kwargs)
        _dverify()
    except BaseException as error:
        record.update(status="failed", elapsed_s=_dt.perf_counter()-started,
                      error_type=type(error).__name__, error=str(error))
        path.write_text(_dj.dumps(record), encoding="utf-8")
        raise
    record.update(status="completed", elapsed_s=_dt.perf_counter()-started, report=self.run_report_,
                  source_sha256=_dhashes,
                  effective_fits=_daudit[audit_start:])
    path.write_text(_dj.dumps(record, allow_nan=False), encoding="utf-8")
    return result
_dh.AutoML.fit = _diagnostic_fit
if {mode!r} == "fast":
    _dh.AutoML = _df.partial(_dh.AutoML, search=_dh.SearchConfig(threads={threads}))
"""


def run_notebook(
    source: Path, mode: str, seed: int, output: Path, threads: int = 4
) -> dict[str, Any]:
    """Execute an isolated notebook copy with common resource limits and disabled submissions."""
    import nbformat
    from jupyter_client import KernelManager
    from nbclient import NotebookClient
    from nbclient.exceptions import CellExecutionError, CellTimeoutError, DeadKernelError

    output.mkdir(parents=True, exist_ok=False)
    preserve_sources(output)
    notebook = nbformat.read(source, as_version=4)
    source_hash = sha256(source)
    seed_count = 0
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        text, replaced = re.subn(r"(?m)^SEED = .*?$", f"SEED = {seed}", cell.source)
        seed_count += replaced
        text = text.replace(
            'DATA = Path("data/', f'DATA = Path("{(ROOT / "notebooks/data").as_posix()}/'
        )
        text = re.sub(
            r"(?m)^RESULTS = Path\(.*$",
            lambda _: f"RESULTS = Path({str(output / 'results')!r})",
            text,
        )
        if '"leaderboard", COMP, "--download"' in text:
            directory = "titanic" if source.name.startswith("01-") else "house-prices"
            archives = list((ROOT / "notebooks/results" / directory / "leaderboard").glob("*.zip"))
            if len(archives) != 1:
                raise FileNotFoundError("one local leaderboard snapshot is required")
            destination = output / "results/leaderboard"
            destination.mkdir(parents=True)
            shutil.copy2(archives[0], destination / archives[0].name)
            before = 'subprocess.run(\n    [KAGGLE, "competitions", "leaderboard", COMP, "--download", "-p", str(lb_dir)], check=True\n)'
            if before not in text:
                raise ValueError("unrecognized leaderboard cell")
            text = text.replace(before, "None  # local leaderboard snapshot")
        cell.source, cell.outputs, cell.execution_count = text, [], None
    if seed_count != 1:
        raise ValueError("expected exactly one notebook SEED assignment")
    notebook.cells.insert(0, nbformat.v4.new_code_cell(notebook_setup(mode, seed, output, threads)))
    record: dict[str, Any] = {
        "status": "running",
        "notebook": str(source),
        "source_sha256": source_hash,
        "mode": mode,
        "seed": seed,
        "native_threads": threads,
        "hpo_cap_per_family": 5,
        "cache": "disabled",
        "submissions": False,
        "sampling_overrides": None,
    }
    write_json(output / "status.json", record)
    manager = KernelManager(kernel_name="python3")
    assert manager.kernel_spec is not None
    manager.kernel_spec.argv = [
        sys.executable,
        "-m",
        "ipykernel_launcher",
        "-f",
        "{connection_file}",
    ]
    client = NotebookClient(notebook, km=manager, timeout=None, startup_timeout=120)
    started = time.perf_counter()
    try:
        with patch.dict(
            os.environ, {**{key: str(threads) for key in THREAD_ENV}, "KAGGLE_SUBMIT": "0"}
        ):
            client.execute(cwd=str(output), cleanup_kc=True)
        record["status"] = "completed"
    except (CellExecutionError, CellTimeoutError, DeadKernelError, RuntimeError, OSError) as error:
        record.update(status="failed", error_type=type(error).__name__, error=str(error))
    finally:
        record["elapsed_s"] = time.perf_counter() - started
        nbformat.write(notebook, output / source.name)
        write_json(output / "status.json", record)
        if sha256(source) != source_hash:
            raise RuntimeError("source notebook changed during execution")
    return record


def partition_report(indices: np.ndarray, dataset: Dataset, task: Task) -> dict[str, Any]:
    target = dataset.target()
    assert target is not None
    classes, counts = (
        np.unique(target[indices], return_counts=True) if task.is_classification else ([], [])
    )
    timestamps = dataset.time()
    groups = dataset.groups()
    result: dict[str, Any] = {
        "rows": len(indices),
        "indices_sha256": hashlib.sha256(indices.astype("<i8").tobytes()).hexdigest(),
        "class_counts": {str(c): int(n) for c, n in zip(classes, counts)},
        "groups": len(np.unique(groups[indices])) if groups is not None else None,
    }
    if timestamps is not None:
        import pandas as pd

        selected = pd.to_datetime(timestamps[indices])
        result.update(
            time_min=str(selected.min()),
            time_max=str(selected.max()),
            weeks=len(selected.to_period("W").unique()),
        )
    return result


def probe_partitions(folds: Sequence[Fold], dataset: Dataset, task: Task) -> list[dict[str, Any]]:
    return [
        {
            name: partition_report(indices, dataset, task)
            for name, indices in (
                ("fit", fold.fit_idx),
                ("es", fold.es_idx),
                ("test", fold.test_idx),
            )
        }
        for fold in folds
    ]


def diagnostic_folds(
    original: Sequence[Fold],
    y: np.ndarray,
    task: Task,
    resource: ProbeResource,
    *,
    seed: int,
    threads: int,
) -> tuple[Fold, ...]:
    """Vary fit rows while holding ES/test fixed; additional folds retain per-fold resources."""
    positions = np.linspace(0, len(original) - 1, min(resource.folds, len(original)), dtype=int)
    result = []
    for position in positions:
        fold = original[int(position)]
        baseline = SearchConfig(max_rows=4096, max_folds=1, threads=threads)
        base = bounded_probe_folds([fold], y=y, task=task, config=baseline, seed=seed)
        expanded = bounded_probe_folds(
            [fold],
            y=y,
            task=task,
            config=baseline.model_copy(update={"max_rows": resource.rows}),
            seed=seed,
        )
        if not base or not expanded:
            continue
        fit, es, test = expanded[0].fit_idx, base[0].es_idx, base[0].test_idx
        if resource.merge_es:
            fit, es = np.concatenate((fit, es)), np.empty(0, dtype=int)
        result.append(Fold(fit, es, test))
    return tuple(result)


def effective_native(estimator: Estimator, threads: int) -> dict[str, Any]:
    """Read native model parameters after fitting and reject a mismatched worker setting."""
    if not isinstance(estimator, SupportsThreadLimit):
        return {"native_thread_port": False, "blas_limit": threads}
    if not isinstance(estimator, SupportsNativeModel):
        raise TypeError("native resource verification requires SupportsNativeModel")
    native = estimator.native_model()
    parameters = native.get_params()
    workers = {key: parameters[key] for key in ("thread_count", "n_jobs") if key in parameters}
    if not workers or any(value != threads for value in workers.values()):
        raise ValueError(f"native worker mismatch: {workers}; expected {threads}")
    return {
        "native_thread_port": True,
        "workers": workers,
        "tree_budget": estimator.iteration_budget
        if isinstance(estimator, SupportsIterationBudget)
        else None,
    }


def evaluate_resource(
    dataset: Dataset,
    task: Task,
    cv: CVConfig,
    resource: ProbeResource,
    *,
    seed: int,
    threads: int,
    model_order: Sequence[str],
    folds_override: Sequence[Fold] | None = None,
    candidate_sink: list[Candidate] | None = None,
) -> dict[str, Any]:
    y = dataset.target()
    assert y is not None
    components = build_default_components(
        task,
        random_state=seed,
        metric=task.metric,
        cv=cv,
        has_time=dataset.time() is not None,
        has_group=dataset.groups() is not None,
        has_missing=bool(np.isnan(dataset.to_numpy()).any()),
        classes=np.unique(y),
    )
    original = list(components.splitter.split(dataset))
    folds = (
        tuple(folds_override)
        if folds_override is not None
        else diagnostic_folds(original, y, task, resource, seed=seed, threads=threads)
    )
    if not folds:
        return {"status": "infeasible", "resource": asdict(resource)}
    context = RunContext()
    configure_run_cost(context)
    context.start_run()
    x = design_matrix(dataset)
    names = list(dataset.schema.numeric) + list(dataset.schema.categorical)
    routing = native_routing(dataset.schema, task.native_cat_max_unique)
    cats = categorical_positions(
        names, [name for name, route in routing.items() if route == "native"]
    )
    classes = np.unique(y)
    scores: list[dict[str, Any]] = []
    native_settings: list[dict[str, Any]] = []
    for name in model_order:
        if name not in components.estimators:
            raise ValueError(f"requested model unavailable: {name}")
        created: list[Estimator] = []
        limited = limit_factory(components.estimators[name], threads, resource.iterations)

        def factory() -> Estimator:
            estimator = limited()
            created.append(estimator)
            return estimator

        candidate = _run_candidate(
            name,
            factory,
            x_full=x,
            y=y,
            feature_names=names,
            categorical_indices=cats,
            kind=task.kind,
            positive=resolve_positive(task, classes) if task.is_classification else None,
            global_classes=classes,
            metric=components.metric,
            folds=list(folds),
            sample_weight=dataset.sample_weight(),
            n_features=x.shape[1],
            need_proba=True,
            capture_oof=True,
            capture_proba=True,
            block_index=None,
            weighting="pooled",
            logger=context.logger,
            ctx=context,
            stage="scouting",
        )
        if candidate_sink is not None:
            candidate_sink.append(candidate)
        native_settings.extend(
            {"model": name, "fold": i, **effective_native(estimator, threads)}
            for i, estimator in enumerate(created)
        )
        scores.append({"model": name, "score": candidate.score, "train_time": candidate.train_time})
    context.finish_run()
    return {
        "status": "completed",
        "resource": asdict(resource),
        "resolved_cv": components.cv.model_dump(mode="json"),
        "partitions": probe_partitions(folds, dataset, task),
        "model_order": list(model_order),
        "features": names,
        "native_routing": routing,
        "effective_native": native_settings,
        "scores": scores,
        "cost": context.cost_report(),
    }


def evaluate_policy(
    dataset: Dataset, task: Task, cv: CVConfig, *, seed: int, threads: int
) -> dict[str, Any]:
    y = dataset.target()
    assert y is not None
    components = build_default_components(
        task,
        random_state=seed,
        metric=task.metric,
        cv=cv,
        has_time=dataset.time() is not None,
        has_group=dataset.groups() is not None,
        has_missing=bool(np.isnan(dataset.to_numpy()).any()),
        classes=np.unique(y),
    )
    folds = list(components.splitter.split(dataset))
    evaluations: list[dict[str, Any]] = []
    config = SearchConfig(threads=threads)

    def evaluate(name: str, selected: Sequence[Fold], iterations: int) -> Candidate:
        candidates: list[Candidate] = []
        resource = ProbeResource("policy", iterations=iterations, folds=len(selected))
        evaluations.append(
            evaluate_resource(
                dataset,
                task,
                cv,
                resource,
                seed=seed,
                threads=threads,
                model_order=[name],
                folds_override=selected,
                candidate_sink=candidates,
            )
        )
        return candidates[0]

    full_iterations = {}
    for name, factory in components.estimators.items():
        estimator = factory()
        full_iterations[name] = (
            estimator.iteration_limit(early_stopping=any(f.es_idx.size for f in folds))
            if isinstance(estimator, SupportsIterationPlan)
            else None
        )
    outcome = probe_models(
        tuple(components.estimators),
        folds,
        y=y,
        groups=dataset.groups(),
        task=task,
        metric=components.metric,
        config=config,
        seed=seed,
        evaluate=evaluate,
        significance_test=components.significance,
        sample_weight=dataset.sample_weight(),
        times=dataset.time(),
        full_iterations=full_iterations,
    )
    return {
        "status": "completed",
        "seed": seed,
        "policy": "current",
        "configuration": config.model_dump(mode="json"),
        "winner": outcome.winner,
        "reason": outcome.reason,
        "issues": list(outcome.issues),
        "rank_changed": outcome.rank_changed,
        "cost_estimates": outcome.cost_estimates,
        "failures": list(outcome.failures),
        "confirmation_failures": list(outcome.confirmation_failures),
        "skipped": list(outcome.skipped),
        "diagnostics": outcome.diagnostics,
        "evaluations": evaluations,
    }


def run_probes(
    scenario: str,
    output: Path,
    seeds: Sequence[int],
    resources: Sequence[ProbeResource],
    threads: int = 4,
    policy: str = "none",
) -> None:
    output.mkdir(parents=True, exist_ok=False)
    preserve_sources(output)
    x, y, timestamps, provenance = prepare_inputs(scenario)
    task, cv = scenario_config(scenario)
    full = Reader(task).read(x, y, time=timestamps)
    scheme = "timeseries_period" if scenario == "store" else "stratified"
    dev_indices, reserved = outer_holdout_carve(
        full,
        scheme=scheme,
        fraction=0.2,
        stratify=task.is_classification,
        random_state=42,
        purge=cv.purge,
        period=cv.period,
    )
    dev = full.take(dev_indices)
    del full
    manifest = {
        "scenario": scenario,
        "provenance": provenance,
        "boundary": "DEV only; reserved holdout is never scored",
        "outer_split_seed": 42,
        "reserved_rows": len(reserved),
        "dev_rows": len(dev_indices),
        "dev_indices_sha256": hashlib.sha256(dev_indices.astype("<i8").tobytes()).hexdigest(),
        "seeds": list(seeds),
        "native_threads": threads,
        "blas_threads": threads,
        "cache": "disabled; fresh estimator per fit; OS file cache uncontrolled",
        "hpo": "disabled for model probes",
        "resource_grid": [asdict(resource) for resource in resources],
        "policy": policy,
        "budget": {"time_budget_s": None, "memory_limit_mb": None},
        "sampling": "current bounded_probe_folds per original fold; evenly spaced folds; shared ES/test",
        "ablation_boundary": "rows changes fit only; folds preserves per-fold budget; no FS/HPO/calibration/ensemble",
        "import_boundary_source_sha256": IMPORT_HASHES,
    }
    write_json(output / "manifest.json", manifest)
    del x, y, timestamps, reserved
    with comparable_resources(threads):
        for repeat, seed in enumerate(seeds):
            order = paired_order(repeat, [r.name for r in resources])
            model_order = paired_order(repeat, ("baseline", "catboost", "lightgbm", "linear"))
            for position, name in enumerate(order):
                resource = next(r for r in resources if r.name == name)
                _, variant_cv = scenario_config(scenario, resource.es_rows)
                path = output / f"seed-{seed}-{name}.json"
                record: dict[str, Any] = {"seed": seed, "position": position, "status": "running"}
                write_json(path, record)
                try:
                    record.update(
                        evaluate_resource(
                            dev,
                            task,
                            variant_cv,
                            resource,
                            seed=seed,
                            threads=threads,
                            model_order=model_order,
                        )
                    )
                except (
                    AutoMLError,
                    _CandidateFailed,
                    ValueError,
                    RuntimeError,
                    MemoryError,
                    OSError,
                ) as error:
                    record.update(
                        status="failed", error_type=type(error).__name__, error=str(error)
                    )
                    raise
                finally:
                    write_json(path, record)

            if policy == "current":
                path = output / f"seed-{seed}-policy.json"
                write_json(path, {"status": "running", "seed": seed})
                write_json(path, evaluate_policy(dev, task, cv, seed=seed, threads=threads))
                if scenario == "store" and any(r.name == "es_rows" for r in resources):
                    _, es_cv = scenario_config(
                        scenario, next(r.es_rows for r in resources if r.name == "es_rows")
                    )
                    path = output / f"seed-{seed}-policy-es_rows.json"
                    write_json(path, {"status": "running", "seed": seed})
                    write_json(path, evaluate_policy(dev, task, es_cv, seed=seed, threads=threads))
    verify_sources()


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", choices=(*SOURCES, "notebook"))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument(
        "--variants", nargs="+", default=["reference", "rows", "iterations", "folds"]
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--policy", choices=("none", "current"), default="none")
    args = parser.parse_args()
    if args.seeds is None:
        args.seeds = [42] if args.scenario == "notebook" else [42, 43, 44]
    if args.threads < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error("threads must be positive and seeds unique")
    if args.scenario == "notebook":
        if args.source is None:
            parser.error("notebook requires --source")
        notebook_offset = int(args.source.name[:2]) - 1 if args.source.name[:2].isdigit() else 0
        if not args.execute:
            sys.stdout.write(
                json.dumps(
                    {
                        "source": str(args.source),
                        "seeds": args.seeds,
                        "order": [
                            paired_order(i + notebook_offset, ("baseline", "fast"))
                            for i in range(len(args.seeds))
                        ],
                        "native_threads": args.threads,
                        "execution": "not started",
                    }
                )
                + "\n"
            )
            return
        if args.output is None:
            parser.error("--execute requires a new --output directory")
        args.output.mkdir(parents=True, exist_ok=False)
        records = []
        for repeat, seed in enumerate(args.seeds):
            for mode in paired_order(repeat + notebook_offset, ("baseline", "fast")):
                records.append(
                    run_notebook(
                        args.source.resolve(),
                        mode,
                        seed,
                        args.output / f"seed-{seed}" / mode,
                        args.threads,
                    )
                )
                write_json(args.output / "batch.json", records)
        return
    grid = resource_grid(args.scenario == "store")
    unknown = set(args.variants) - {r.name for r in grid}
    if unknown:
        parser.error(f"unknown variants: {sorted(unknown)}")
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.variants)) != len(args.variants):
        parser.error("seeds and variants must be unique")
    resources = tuple(r for r in grid if r.name in args.variants)
    if args.execute:
        if args.output is None:
            parser.error("--execute requires a new --output directory")
        run_probes(args.scenario, args.output, args.seeds, resources, args.threads, args.policy)
    else:
        sys.stdout.write(
            json.dumps(
                {
                    "prerequisites": prerequisites(args.scenario),
                    "seeds": args.seeds,
                    "resources": [asdict(r) for r in resources],
                    "native_threads": args.threads,
                    "execution": "not started",
                },
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
