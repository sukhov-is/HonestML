"""The selected subset, factory refit budget and predictions survive native artifact delivery."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from honestml import AutoML
from honestml.composition.artifact import load_artifact, save_artifact
from honestml.composition.build import build_default_components
from honestml.core import CVConfig, FeatureSelectionConfig, SearchConfig, Task

pytestmark = pytest.mark.unit


def test_fast_search_native_artifact_preserves_subset_rounds_and_predictions(
    tmp_path: Path,
) -> None:
    pytest.importorskip("lightgbm")
    components = build_default_components(
        Task(kind="regression"), random_state=5, models=("lightgbm",)
    )
    expected_budget = components.estimators["lightgbm"]().iteration_limit(early_stopping=False)
    rng = np.random.default_rng(19)
    x = pd.DataFrame(
        {
            "signal": np.arange(160, dtype=float),
            "noise0": rng.normal(size=160),
            "noise1": rng.normal(size=160),
        }
    )
    model = AutoML(
        task="regression",
        models=("lightgbm",),
        random_state=5,
        cv=CVConfig(scheme="kfold", n_splits=2, outer_holdout=0.25),
        search=SearchConfig(max_rows=48, model_iterations=3, threads=1),
        feature_selection=FeatureSelectionConfig(
            strategy="importance", cutoff="top_k", top_k=1, refine=False
        ),
        significance="off",
    ).fit(x, x["signal"].to_numpy())
    assert model.run_report_["search"]["final_control"] == "selected_subset"
    selected = model.schema_.selected_features
    assert selected is not None and len(selected) == 1
    refits = [row for row in model.run_report_["cost"]["work"] if row["stage"] == "refit"]
    assert len(refits) == 2
    assert all(row["tree_budget"] == expected_budget for row in refits)
    assert model.best_estimator_.iteration_budget == expected_budget
    expected_rounds = model.best_estimator_.fitted_iterations
    assert expected_rounds is not None and 0 < expected_rounds <= expected_budget
    assert model.shipped_on_ == "all"
    art = tmp_path / "native"
    save_artifact(model.fitted_, art, model_format="native")
    loaded = load_artifact(art)
    assert loaded.schema.selected_features == selected
    assert loaded.estimator._booster.current_iteration() == expected_rounds
    assert loaded.shipped_on == "all" and loaded.holdout_score == model.holdout_score_
    unseen = pd.DataFrame(
        {
            "signal": [4.5, 43.5, 85.5, 144.5],
            "noise0": [0.2, -0.1, 0.6, -0.9],
            "noise1": [-0.3, 0.6, -0.2, 0.5],
        }
    )
    np.testing.assert_allclose(loaded.predict(unseen), model.predict(unseen), rtol=0.0, atol=1e-12)
