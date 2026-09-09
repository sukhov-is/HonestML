"""Completion estimates follow the requested CV/refit plan without extra fits."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from honestml import AutoML, CVConfig, SearchConfig
from honestml.application.search import probe_models
from honestml.core import Fold, Task
from honestml.core.selection_policy import Candidate

from .test_search_confirmation import Metric, folds

pytestmark = pytest.mark.unit


def test_cost_uses_fit_only_probe_rows_and_model_specific_completion_plan() -> None:
    calls = []

    def evaluate(name: str, parts: Sequence[Fold], iterations: int) -> Candidate:
        calls.append(name)
        return Candidate(
            name,
            0.9,
            train_time=2.0 if iterations == 64 else 18.0,
            refit_iterations=min(1000, iterations),
        )

    result = probe_models(
        ["es", "plain"],
        folds(),
        y=np.arange(1400.0),
        groups=None,
        task=Task(kind="regression"),
        metric=Metric(),
        config=SearchConfig(max_rows=100, confirmation_rows=600),
        seed=1,
        evaluate=evaluate,
        full_iterations={"es": 1000, "plain": None},
        full_training_rows={"es": 1600, "plain": 2000},
        completion_refit_rows=(1400, 1800),
    )
    assert len(calls) == 4
    assert result.initial_cost_estimates["es"] == pytest.approx(2 * 4800 / 60 * 1000 / 64)
    assert result.cost_estimates["es"] == pytest.approx(18 * 4800 / 360 * 1000 / 256)
    assert result.cost_estimates["plain"] == pytest.approx(18 * 5200 / 360)
    diagnostic = result.cost_model["es"]
    assert diagnostic["predicted_confirmation_s"] == pytest.approx(48.0)
    assert diagnostic["confirmation_error_s"] == pytest.approx(-30.0)
    assert diagnostic["confirmation_relative_error"] == pytest.approx(-30 / 48)


@pytest.mark.parametrize(
    ("run_mode", "finalize", "holdout", "expected"),
    [
        ("full", True, 0.25, [90, 120]),
        ("full", False, 0.25, [90]),
        ("selection", True, 0.25, []),
        ("full", True, 0.0, [120]),
    ],
)
def test_facade_reports_actual_refit_plan(
    run_mode: str,
    finalize: bool,
    holdout: float,
    expected: list[int],
) -> None:
    rng = np.random.default_rng(4)
    x = pl.DataFrame(rng.normal(size=(120, 3)), schema=["a", "b", "c"])
    model = AutoML(
        task="regression",
        models=("baseline",),
        cv=CVConfig(n_splits=2, outer_holdout=holdout),
        search=SearchConfig(),
        run_mode=run_mode,
        finalize=finalize,
        significance="off",
    ).fit(x, x["a"].to_numpy())
    report = model.run_report_
    assert report["search"]["completion_refit_rows"] == expected
    assert [work["rows"] for work in report["cost"]["work"] if work["stage"] == "refit"] == expected
    assert {"fs", "hpo"} <= set(report["search"]["cost_estimate_excludes"])
    assert report["search"]["fs_execution"]["requested"] is False
    assert report["search"]["fs_execution"]["approximate"] is False


def test_search_completion_plan_invalidates_completed_cache(tmp_path: Path) -> None:
    x = pl.DataFrame({"a": np.arange(120.0), "b": np.arange(120.0) % 3})
    fingerprints = []
    for run_mode, finalize in [("full", False), ("full", True), ("selection", True)]:
        model = AutoML(
            task="regression",
            models=("baseline",),
            cv=CVConfig(n_splits=2, outer_holdout=0.25),
            search=SearchConfig(),
            cache=tmp_path,
            run_mode=run_mode,
            finalize=finalize,
            significance="off",
        ).fit(x, x["a"].to_numpy())
        report = model.run_report_
        fingerprints.append(report["run_fingerprint"])
        assert not report["search"].get("stage_reused", False)
    assert len(set(fingerprints)) == 3


@pytest.mark.slow
def test_explicit_native_tree_ceiling_is_preserved_in_both_probe_levels() -> None:
    pytest.importorskip("lightgbm")
    from honestml.adapters import Reader
    from honestml.adapters.boosting import LIGHTGBM, build_boosting
    from honestml.application.slice import run_slice
    from honestml.composition import build_default_components
    from honestml.core import RunContext

    task = Task(kind="regression")
    x = np.column_stack((np.arange(120.0), np.arange(120.0) % 3))
    dataset = Reader(task).read(x, x[:, 0])
    components = build_default_components(task, models=("linear", "lightgbm"), cv=2, random_state=0)
    estimators = dict(components.estimators)
    estimators["lightgbm"] = lambda: build_boosting(
        LIGHTGBM, task=task, random_state=0, n_estimators=7
    )
    ctx = RunContext()
    result = run_slice(
        dataset,
        task,
        estimators=estimators,
        splitter=components.splitter,
        metric=components.metric,
        policy=components.policy,
        search=SearchConfig(),
        ctx=ctx,
    )
    work = [
        entry
        for entry in ctx.cost_report()["work"]
        if entry["stage"] == "scouting" and entry["model_id"] == "lightgbm"
    ]
    assert len(work) == 3
    assert all(entry["tree_budget"] == 7 for entry in work)
    assert result.search["cost_model"]["lightgbm"]["initial_iteration_cap"] == 7
    parts = list(components.splitter.split(dataset))
    assert result.search["cost_model"]["lightgbm"]["cv_training_rows"] == sum(
        len(f.fit_idx) for f in parts
    )
    assert result.search["cost_model"]["linear"]["cv_training_rows"] == sum(
        len(f.fit_idx) + len(f.es_idx) for f in parts
    )


def test_non_search_finalize_keeps_existing_cache_scope() -> None:
    x = pl.DataFrame({"a": np.arange(120.0), "b": np.arange(120.0) % 3})
    fingerprints = []
    for finalize in (False, True):
        model = AutoML(
            task="regression",
            models=("baseline",),
            cv=CVConfig(n_splits=2, outer_holdout=0.25),
            finalize=finalize,
            significance="off",
        ).fit(x, x["a"].to_numpy())
        fingerprints.append(model.run_report_["run_fingerprint"])
    assert fingerprints[0] == fingerprints[1]


def test_search_fingerprint_includes_refit_rows_with_same_dev_signature() -> None:
    from honestml.application import compute_run_fingerprint
    from honestml.core import RunConfig

    keys = [
        compute_run_fingerprint(
            run_config=RunConfig(search=SearchConfig()),
            task=Task(kind="regression"),
            metric=Metric(),
            data_signature="same-dev",
            estimators=("linear",),
            lib_versions={},
            search_completion={"run_mode": "full", "finalize": True, "refit_rows": rows},
        )
        for rows in [(90, 120), (90, 130)]
    ]
    assert keys[0] != keys[1]


def test_search_reports_effective_fs_proxy_caps() -> None:
    from honestml import FeatureSelectionConfig

    x = pl.DataFrame({"a": np.arange(120.0), "b": np.arange(120.0) % 3})
    model = AutoML(
        task="regression",
        models=("baseline",),
        cv=2,
        feature_selection=FeatureSelectionConfig(strategy="null_importance", n_runs=30),
        search=SearchConfig(max_rows=40, model_iterations=3, max_fs_fits=1),
        significance="off",
    ).fit(x, x["a"].to_numpy())
    execution = model.run_report_["search"]["fs_execution"]
    assert execution["requested"] and execution["approximate"]
    assert execution["n_runs"] == execution["n_probes"] == 3
    assert execution["ranker_iterations"] == 3


def test_failed_factory_is_isolated_from_resource_plan_collection() -> None:
    from honestml.adapters import Reader
    from honestml.application.slice import run_slice
    from honestml.composition import build_default_components
    from honestml.core import Estimator

    task = Task(kind="regression")
    x = np.arange(120.0)[:, None]
    dataset = Reader(task).read(x, x[:, 0])
    components = build_default_components(task, models=("linear",), cv=2, random_state=0)

    def broken() -> Estimator:
        raise ValueError("invalid native constructor")

    result = run_slice(
        dataset,
        task,
        estimators={"broken": broken, "linear": components.estimators["linear"]},
        splitter=components.splitter,
        metric=components.metric,
        policy=components.policy,
        search=SearchConfig(),
    )
    assert result.best_model_id == "linear"
    assert result.search["model_failures"][0][0] == "broken"
    assert "invalid native constructor" in result.search["model_failures"][0][1]
