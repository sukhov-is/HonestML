"""Facade search enforces actual FS work limits and reports approximate execution."""

import numpy as np
import polars as pl
import pytest

from honestml import AutoML, FeatureSelectionConfig, SearchConfig

pytestmark = pytest.mark.unit


def test_single_null_recipe_stops_at_actual_fs_cap_and_keeps_wide_control() -> None:
    rng = np.random.default_rng(12)
    x = pl.DataFrame(rng.normal(size=(120, 5)), schema=[f"x{i}" for i in range(5)])
    y = x["x0"].to_numpy() + rng.normal(scale=0.1, size=120)
    model = AutoML(
        task="regression",
        models=("linear",),
        cv=2,
        feature_selection=FeatureSelectionConfig(strategy="null_importance", n_runs=30),
        search=SearchConfig(max_rows=40, model_iterations=3, max_fs_fits=1),
    ).fit(x, y)
    report = model.run_report_
    assert report["search"]["stop_reason"] == "fs_fit_limit"
    assert report["search"]["fs_fit_count"] == 1
    fs_work = [item for item in report["cost"]["work"] if item["stage"] == "fs"]
    assert len(fs_work) == 1
    assert fs_work[0]["rows"] <= 40
    assert fs_work[0]["tree_budget"] == 3
    assert report["cost"]["fit_counts"]["failed"] == 0
    assert report["feature_selection"] is None
    assert report["search"]["fs_execution"]["evaluation"] == "post_search_dev"


@pytest.mark.slow
def test_probe_fit_rows_match_for_linear_and_early_stopping_model() -> None:
    pytest.importorskip("lightgbm")
    rng = np.random.default_rng(2)
    x = pl.DataFrame(rng.normal(size=(140, 3)), schema=["a", "b", "c"])
    y = x["a"].to_numpy() + rng.normal(scale=0.2, size=140)
    model = AutoML(
        task="regression",
        models=("linear", "lightgbm"),
        cv=2,
        search=SearchConfig(
            max_rows=40, model_iterations=3, confirmation_rows=80, confirmation_iterations=6
        ),
    ).fit(x, y)
    work = [item for item in model.run_report_["cost"]["work"] if item["stage"] == "scouting"]
    assert len(work) == 6
    initial = work[:2]
    assert initial[0]["rows"] == initial[1]["rows"] == 24
    assert model.run_report_["search"]["confirmation_probes"]


def test_infeasible_fs_class_cap_keeps_completed_wide_control() -> None:
    rng = np.random.default_rng(4)
    x = pl.DataFrame(rng.normal(size=(160, 4)), schema=["a", "b", "c", "d"])
    y = np.tile(np.arange(40), 4)
    model = AutoML(
        task="multiclass",
        models=("baseline",),
        cv=2,
        feature_selection=FeatureSelectionConfig(strategy="importance"),
        search=SearchConfig(max_rows=32),
    ).fit(x, y)
    report = model.run_report_
    assert report["search"]["fs_execution_reason"] == "infeasible_training_classes"
    assert not [item for item in report["cost"]["work"] if item["stage"] == "fs"]
    assert report["feature_selection"] is None
    assert report["cost"]["fit_counts"]["failed"] == 0
