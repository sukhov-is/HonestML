"""Diagnostic experiments keep evaluation rows and resources comparable."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from honestml import AutoML, HPOConfig
from honestml.adapters import Reader
from honestml.core import Fold, Task

pytestmark = pytest.mark.unit


@pytest.fixture
def runner() -> Any:
    path = Path(__file__).resolve().parents[2] / "benchmarks/search_diagnostics.py"
    spec = importlib.util.spec_from_file_location("search_diagnostics", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_rows_ablation_preserves_es_and_test(runner: Any) -> None:
    original = [Fold(np.arange(20000), np.arange(20000, 22000), np.arange(22000, 26000))]
    y = np.arange(26000) % 3
    base = runner.ProbeResource("reference")
    a = runner.diagnostic_folds(original, y, Task(kind="multiclass"), base, seed=42, threads=4)[0]
    b = runner.diagnostic_folds(
        original, y, Task(kind="multiclass"), replace(base, rows=16384), seed=42, threads=4
    )[0]
    assert len(b.fit_idx) > len(a.fit_idx)
    np.testing.assert_array_equal(a.es_idx, b.es_idx)
    np.testing.assert_array_equal(a.test_idx, b.test_idx)
    assert not np.intersect1d(b.fit_idx, b.test_idx).size
    c = runner.diagnostic_folds(
        original, y, Task(kind="multiclass"), replace(base, iterations=256), seed=42, threads=4
    )[0]
    np.testing.assert_array_equal(a.fit_idx, c.fit_idx)


def test_more_folds_keeps_first_fold_resource(runner: Any) -> None:
    original = [
        Fold(np.arange(1000), np.arange(1000, 1200), np.arange(1200, 1500)),
        Fold(np.arange(1000, 2000), np.arange(2000, 2200), np.arange(2200, 2500)),
    ]
    y = np.arange(2500) % 2
    a = runner.diagnostic_folds(
        original, y, Task(kind="binary"), runner.ProbeResource("a"), seed=42, threads=4
    )
    b = runner.diagnostic_folds(
        original, y, Task(kind="binary"), runner.ProbeResource("b", folds=2), seed=42, threads=4
    )
    assert len(a) == 1 and len(b) == 2
    np.testing.assert_array_equal(a[0].fit_idx, b[0].fit_idx)
    np.testing.assert_array_equal(a[0].test_idx, b[0].test_idx)


@pytest.mark.slow
def test_resource_context_caps_preset_and_restores_after_exception(runner: Any) -> None:
    pytest.importorskip("catboost")
    pytest.importorskip("lightgbm")
    import honestml.composition.facade as facade

    original = AutoML._resolve_hpo
    original_build = facade.build_default_components
    environment = dict(os.environ)
    with pytest.raises(RuntimeError, match="sentinel"), runner.comparable_resources(4):
        assert AutoML()._resolve_hpo(None) is None
        model = AutoML(preset="best")
        effective, _ = model._resolve_preset()
        assert model._resolve_hpo(effective["hpo"]).n_trials == 5
        config = model._resolve_hpo(HPOConfig(n_trials=3, timeout_s=7))
        assert config.n_trials == 3 and config.timeout_s == 7
        components = facade.build_default_components(
            Task(kind="binary"),
            random_state=42,
            models=("catboost", "lightgbm"),
            classes=np.array([0, 1]),
            hpo=HPOConfig(),
        )
        assert components.estimators["catboost"]()._params["thread_count"] == 4
        assert components.estimators["lightgbm"]()._params["n_jobs"] == 4
        assert components.make_factory("lightgbm", {"n_estimators": 50})()._params["n_jobs"] == 4
        assert all(os.environ[key] == "4" for key in runner.THREAD_ENV)
        raise RuntimeError("sentinel")
    assert AutoML._resolve_hpo is original
    assert facade.build_default_components is original_build
    assert dict(os.environ) == environment


def test_prepare_executes_only_data_cell(runner: Any, tmp_path: Path) -> None:
    folder = tmp_path / "notebooks/data/otto"
    folder.mkdir(parents=True)
    (folder / "train.csv").write_text("id,a,target\n1,2,x\n2,3,y\n", encoding="utf-8")
    source = tmp_path / "notebooks/05-otto-product-classification.ipynb"
    cells = [
        "raise RuntimeError('markdown')",
        "SEED = 42",
        "raise RuntimeError('download')",
        "df=pd.read_csv(DATA/'train.csv'); y=df.pop('target'); X=df.drop(columns='id')",
        "raise RuntimeError('fit')",
    ]
    source.write_text(
        json.dumps({"cells": [{"cell_type": "code", "source": [s]} for s in cells]}),
        encoding="utf-8",
    )
    before = source.read_bytes()
    x, y, t, provenance = runner.prepare_inputs("otto", tmp_path)
    assert x.shape == (2, 1) and y.tolist() == ["x", "y"] and t is None
    assert source.read_bytes() == before
    assert provenance["data"][0]["sha256"] == runner.sha256(folder / "train.csv")


def test_partitions_report_rare_events_and_single_row_es(runner: Any) -> None:
    task = Task(kind="binary")
    dataset = Reader(task).read(
        pd.DataFrame({"x": np.arange(10)}),
        np.array([0] * 9 + [1]),
        time=np.arange(10).astype("datetime64[D]"),
    )
    rows = runner.probe_partitions(
        [Fold(np.arange(6), np.array([6]), np.array([7, 8, 9]))], dataset, task
    )
    assert rows[0]["es"]["rows"] == 1
    assert rows[0]["test"]["class_counts"] == {"0": 2, "1": 1}
    assert rows[0]["test"]["weeks"] >= 1


def test_dev_only_run_alternates_order_and_rejects_stale_output(
    runner: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    x = pd.DataFrame({"x": np.linspace(0, 1, 200)})
    y = np.arange(200) % 2
    monkeypatch.setattr(runner, "prepare_inputs", lambda scenario: (x, y, None, {"source": "test"}))
    observed = []

    def evaluate(dataset: Any, task: Any, cv: Any, resource: Any, **kwargs: Any) -> dict[str, Any]:
        observed.append((dataset.n_rows, resource.name, kwargs["model_order"]))
        return {"status": "completed", "holdout_score": None}

    monkeypatch.setattr(runner, "evaluate_resource", evaluate)
    output = tmp_path / "fresh"
    resources = [
        runner.ProbeResource("reference"),
        runner.ProbeResource("iterations", iterations=256),
    ]
    runner.run_probes("otto", output, [42, 43], resources)
    assert all(row[0] == 160 for row in observed)
    assert [row[1] for row in observed] == ["reference", "iterations", "iterations", "reference"]
    assert observed[0][2] == tuple(reversed(observed[2][2]))
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["reserved_rows"] == 40 and manifest["outer_split_seed"] == 42
    with pytest.raises(FileExistsError):
        runner.run_probes("otto", output, [42], resources)


def test_probe_dev_preserves_full_reader_schema_and_vocabulary(
    runner: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    x = pd.DataFrame(
        {
            "border": np.concatenate([np.tile(np.arange(20), 8), np.full(40, 20)]),
            "category": ["a", "b"] * 80 + ["z"] * 40,
        }
    )
    y = np.arange(200) % 3
    task = Task(kind="multiclass")
    full = Reader(task).read(x, y)
    dev_indices = np.arange(160)
    reread = Reader(task).read(x.iloc[dev_indices], y[dev_indices])
    assert "border" in full.schema.numeric and "border" in reread.schema.categorical
    assert full.schema.categories["category"] != reread.schema.categories["category"]
    monkeypatch.setattr(runner, "prepare_inputs", lambda scenario: (x, y, None, {}))
    monkeypatch.setattr(
        runner, "outer_holdout_carve", lambda *args, **kwargs: (dev_indices, np.arange(160, 200))
    )
    seen: list[str] = []

    def check_dataset(dataset: Any) -> None:
        assert dataset.schema == full.schema
        np.testing.assert_array_equal(dataset.target(), y[dev_indices])
        np.testing.assert_array_equal(
            runner.design_matrix(dataset), runner.design_matrix(full.take(dev_indices))
        )

    def evaluate_resource(dataset: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
        check_dataset(dataset)
        seen.append("resource")
        return {"status": "completed"}

    def evaluate_policy(dataset: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
        check_dataset(dataset)
        seen.append("policy")
        return {"status": "completed"}

    monkeypatch.setattr(runner, "evaluate_resource", evaluate_resource)
    monkeypatch.setattr(runner, "evaluate_policy", evaluate_policy)
    runner.run_probes(
        "otto", tmp_path / "schema", [42], [runner.ProbeResource("reference")], policy="current"
    )
    assert seen == ["resource", "policy"]


def test_notebook_injection_checks_seed_cache_and_hpo(runner: Any, tmp_path: Path) -> None:
    text = runner.notebook_setup("fast", 43, tmp_path)
    compile(text, "injection", "exec")
    assert "self.cache is not None" in text
    assert "self.random_state != 43" in text
    assert "SearchConfig(threads=4)" in text
    assert '"hpo_cap_per_family": 5' in text
    with pytest.raises(ValueError):
        runner.notebook_setup("typo", 42, tmp_path)


def test_evaluation_records_actual_native_resource_and_fold_rows(
    runner: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    task = Task(kind="regression", metric="rmse")
    dataset = Reader(task).read(
        pd.DataFrame({"x": np.sin(np.arange(100))}), np.arange(100, dtype=float)
    )
    components = runner.build_default_components(task, random_state=42, models=("baseline",))

    class Splitter:
        def split(self, dataset: Any) -> Any:
            yield Fold(np.arange(60), np.arange(60, 80), np.arange(80, 100))

    class Native:
        supports_early_stopping = True
        native_format = "fake"
        feature_names: list[str] = []
        fitted_iterations = 8
        iteration_budget = 64

        def set_threads(self, threads: int) -> None:
            self.threads = threads

        def set_refit_iterations(self, count: int) -> None:
            self.iteration_budget = count

        def fit(
            self, x: Any, y: Any, X_val: Any = None, y_val: Any = None, sample_weight: Any = None
        ) -> Any:
            assert len(x) == 60 and len(X_val) == 20
            return self

        def predict(self, x: Any) -> np.ndarray:
            return np.zeros(len(x))

        def native_model(self) -> Any:
            return self

        def get_params(self) -> dict[str, int]:
            return {"n_jobs": self.threads, "n_estimators": self.iteration_budget}

    components = components._replace(estimators={"fake": Native}, splitter=Splitter())
    monkeypatch.setattr(runner, "build_default_components", lambda *a, **k: components)
    result = runner.evaluate_resource(
        dataset,
        task,
        runner.CVConfig(),
        runner.ProbeResource("reference"),
        seed=42,
        threads=4,
        model_order=["fake"],
    )
    assert result["effective_native"][0]["workers"] == {"n_jobs": 4}
    fit = result["cost"]["work"][0]
    assert fit["rows"] == 60 and fit["iterations"] == 8 and fit["tree_budget"] == 64
    assert result["partitions"][0]["es"]["rows"] == 20
    estimator = Native()
    estimator.set_threads(2)
    with pytest.raises(ValueError, match="worker mismatch"):
        runner.effective_native(estimator, 4)


def test_notebook_copy_preserves_windows_paths_without_kernel(
    runner: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import ast
    from types import ModuleType, SimpleNamespace

    def code_cell(source: str) -> SimpleNamespace:
        return SimpleNamespace(cell_type="code", source=source, outputs=[], execution_count=None)

    def read_notebook(path: Path, *, as_version: int) -> SimpleNamespace:
        assert as_version == 4
        return json.loads(
            path.read_text(encoding="utf-8"), object_hook=lambda d: SimpleNamespace(**d)
        )

    def write_notebook(notebook: Any, path: Path) -> None:
        path.write_text(json.dumps(notebook, default=vars), encoding="utf-8")

    nbformat = ModuleType("nbformat")
    nbformat.read = read_notebook
    nbformat.write = write_notebook
    nbformat.v4 = SimpleNamespace(new_code_cell=code_cell)
    nbclient = ModuleType("nbclient")
    exceptions = ModuleType("nbclient.exceptions")
    exceptions.CellExecutionError = ValueError
    exceptions.CellTimeoutError = TimeoutError
    exceptions.DeadKernelError = ConnectionError
    jupyter_client = ModuleType("jupyter_client")
    manager = SimpleNamespace(kernel_spec=SimpleNamespace(argv=[]))

    def kernel_manager(*, kernel_name: str) -> SimpleNamespace:
        assert kernel_name == "python3"
        return manager

    jupyter_client.KernelManager = kernel_manager
    for module in (nbformat, nbclient, exceptions, jupyter_client):
        monkeypatch.setitem(sys.modules, module.__name__, module)

    source = tmp_path / "sample.ipynb"
    notebook = SimpleNamespace(
        cells=[
            code_cell(
                'from pathlib import Path\nSEED = 42\nDATA = Path("data/otto")\nRESULTS = Path("results/otto")'
            )
        ]
    )
    write_notebook(notebook, source)
    original = source.read_bytes()
    output = tmp_path / "result space"

    class Client:
        def __init__(self, notebook: Any, **kwargs: Any) -> None:
            self.notebook = notebook
            assert kwargs["km"] is manager
            assert manager.kernel_spec.argv[:3] == [sys.executable, "-m", "ipykernel_launcher"]

        def execute(self, **kwargs: Any) -> None:
            assert os.environ["KAGGLE_SUBMIT"] == "0"
            tree = ast.parse(self.notebook.cells[1].source)
            assignment = next(
                n
                for n in tree.body
                if isinstance(n, ast.Assign)
                and isinstance(n.targets[0], ast.Name)
                and n.targets[0].id == "RESULTS"
            )
            assert ast.literal_eval(assignment.value.args[0]) == str(output / "results")

    nbclient.NotebookClient = Client
    result = runner.run_notebook(source, "baseline", 43, output)
    assert result["status"] == "completed"
    assert source.read_bytes() == original


def test_comparable_resources_accepts_real_sequential_strategy(runner: Any) -> None:
    import honestml.composition.facade as facade
    from honestml import FeatureSelectionConfig
    from honestml.core import FeatureSubsetSelector

    with runner.comparable_resources(4) as audit:
        components = facade.build_default_components(
            Task(kind="binary"),
            random_state=42,
            classes=np.array([0, 1]),
            feature_selection=FeatureSelectionConfig(compare=("importance", "sequential")),
        )
        assert any(
            isinstance(strategy, FeatureSubsetSelector)
            for _, strategy in components.feature_strategies
        )
        assert audit == []


def test_instance_fit_audit_preserves_capabilities_and_restores_method(runner: Any) -> None:
    from honestml.core import SupportsEarlyStopping, SupportsNativeCategorical

    class Native:
        supports_early_stopping = True
        supports_native_categorical = True
        categorical_indices: list[int] = []
        feature_names: list[str] = []
        native_format = "fake"

        def set_threads(self, count: int) -> None:
            self.workers = count

        def native_model(self) -> Any:
            return self

        def get_params(self) -> dict[str, int]:
            return {"n_jobs": self.workers}

        def fit(
            self, X: Any, y: Any, X_val: Any = None, y_val: Any = None, sample_weight: Any = None
        ) -> Any:
            assert self.categorical_indices == [0] and self.feature_names == ["x"]
            return self

    native = Native()
    audit = []
    factory = runner.limit_factory(lambda: native, 4, audit=audit, family="fake")
    estimator = factory()
    assert estimator is native and isinstance(estimator, SupportsEarlyStopping)
    assert isinstance(estimator, SupportsNativeCategorical)
    estimator.categorical_indices = [0]
    estimator.feature_names = ["x"]
    estimator.fit(np.ones((8, 1)), np.arange(8), X_val=np.ones((2, 1)), y_val=np.arange(2))
    assert "fit" not in vars(native)
    assert audit == [
        {
            "family": "fake",
            "rows": 8,
            "columns": 1,
            "es_rows": 2,
            "iterations": None,
            "native_thread_port": True,
            "workers": {"n_jobs": 4},
            "tree_budget": None,
        }
    ]


def test_fs_proxy_records_effective_workers(runner: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    import honestml.adapters.feature_rankers as rankers

    def fake(
        task: Any,
        x: Any,
        y: Any,
        random_state: int,
        sample_weight: Any,
        *,
        threads: Any = None,
        **kwargs: Any,
    ) -> Any:
        return SimpleNamespace(
            get_params=lambda: {"n_jobs": threads, "n_estimators": 3}, estimators_=[1, 2, 3]
        )

    monkeypatch.setattr(rankers, "_fit_ranker_model", fake)
    with runner.comparable_resources(4) as audit:
        rankers._fit_ranker_model(Task(kind="binary"), np.ones((9, 2)), np.arange(9) % 2, 42, None)
        assert audit[0]["workers"] == {"n_jobs": 4}
        assert audit[0]["iterations"] == 3 and audit[0]["rows"] == 9


def test_source_drift_is_rejected(
    runner: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "module.py"
    source.write_text("value = 1", encoding="utf-8")
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "IMPORT_HASHES", {"module.py": runner.sha256(source)})
    runner.verify_sources()
    source.write_text("value = 2", encoding="utf-8")
    with pytest.raises(RuntimeError, match="changed after import"):
        runner.verify_sources()
