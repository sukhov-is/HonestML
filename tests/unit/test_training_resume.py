"""Facade integration for checkpoint reuse, DEV refit rounds and cooperative HPO stopping."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

import honestml.composition.build as build_mod
import honestml.composition.facade as facade_mod
from honestml import AutoML
from honestml.adapters import RunBudget
from honestml.composition.build import Components
from honestml.core import BudgetConfig, CVConfig, FeatureSelectionConfig, HPOConfig
from honestml.core.config import SearchConfig

pytestmark = pytest.mark.unit


def _data() -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(19)
    x = pd.DataFrame(
        {"f0": np.arange(96, dtype=float), "f1": rng.normal(size=96), "f2": rng.normal(size=96)}
    )
    return x, x["f0"].to_numpy().copy()


def _options(cache: Path) -> dict[str, Any]:
    return {
        "task": "regression",
        "models": ("linear",),
        "random_state": 5,
        "cv": CVConfig(scheme="kfold", n_splits=3, outer_holdout=0.25),
        "hpo": HPOConfig(n_trials=2, inner_cv=2),
        "feature_selection": FeatureSelectionConfig(
            strategy="importance", cutoff="top_k", top_k=1, refine=False
        ),
        "significance": "off",
        "cache": cache,
    }


def test_completed_selection_reuse_skips_fs_hpo_and_reports_actual_trials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("lightgbm")
    monkeypatch.setattr(
        build_mod,
        "_resolve_tunable",
        lambda estimators, hpo: {
            "lightgbm": {"n_estimators": {"type": "categorical", "choices": [3, 5]}}
        },
    )
    x, y = _data()
    options = _options(tmp_path)
    options["models"] = ("lightgbm",)
    cold = AutoML(**options).fit(x, y)
    warm = AutoML(**options).fit(x, y)
    cold_work = cold.run_report_["cost"]["work"]
    warm_work = warm.run_report_["cost"]["work"]
    assert any(row["stage"] == "fs" for row in cold_work)
    assert any(row["stage"] == "hpo" for row in cold_work)
    assert warm_work and {row["stage"] for row in warm_work} == {"refit"}
    assert len(warm_work) == 2
    assert warm.best_model_id_ == cold.best_model_id_
    assert warm.schema_.selected_features == cold.schema_.selected_features
    np.testing.assert_array_equal(warm.predict(x), cold.predict(x))
    cold_hpo = cold.run_report_["hpo"]["tuned"]["lightgbm"]
    warm_hpo = warm.run_report_["hpo"]["tuned"]["lightgbm"]
    assert cold_hpo["reused_trials"] == 0
    assert warm_hpo["reused_trials"] == warm_hpo["n_trials_run"] == 2
    assert warm_hpo["completed"] is True
    assert warm_hpo["chosen_params"] == cold_hpo["chosen_params"]


@dataclass(frozen=True)
class _Fit:
    rows: int
    columns: int
    tuned: bool
    requested_rounds: int | None
    used_rounds: int


class _RoundEstimator:
    def __init__(
        self, fits: list[_Fit], *, tuned: bool, bias: float, on_tuned_fit: Callable[[], None] | None
    ) -> None:
        self.feature_names: list[str] = []
        self.fitted_iterations: int | None = None
        self.iteration_budget: int | None = None
        self._requested: int | None = None
        self._fits = fits
        self._tuned = tuned
        self._bias = bias
        self._on_tuned_fit = on_tuned_fit

    def set_refit_iterations(self, count: int) -> None:
        self._requested = count

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> _RoundEstimator:
        self.fitted_iterations = (
            self._requested if self._requested is not None else 3 + int(X[:, 0].sum()) % 7
        )
        self.iteration_budget = self._requested if self._requested is not None else 30
        self._fits.append(
            _Fit(len(X), X.shape[1], self._tuned, self._requested, self.fitted_iterations)
        )
        if self._tuned and self._on_tuned_fit is not None:
            self._on_tuned_fit()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X[:, self.feature_names.index("f0")] + self._bias


def _install_round_estimator(
    monkeypatch: pytest.MonkeyPatch,
    *,
    tuned_bias: float = 0.0,
    on_tuned_fit: Callable[[], None] | None = None,
) -> list[_Fit]:
    fits: list[_Fit] = []
    original = facade_mod.build_default_components

    def factory(params: Mapping[str, Any]) -> Callable[[], _RoundEstimator]:
        return lambda: _RoundEstimator(
            fits, tuned=bool(params), bias=tuned_bias if params else 0.0, on_tuned_fit=on_tuned_fit
        )

    def build(*args: Any, **kwargs: Any) -> Components:
        components = original(*args, **kwargs)
        return components._replace(
            estimators={"linear": factory({})},
            make_factory=lambda name, params: factory(params),
            tunable={"linear": {"regularization": {"type": "float", "low": 0.1, "high": 1.0}}},
        )

    monkeypatch.setattr(facade_mod, "build_default_components", build)
    return fits


def test_cached_dev_round_median_survives_both_refits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fits = _install_round_estimator(monkeypatch)
    x, y = _data()
    options = _options(tmp_path)
    options["feature_selection"] = None
    cold = AutoML(**options).fit(x, y)
    fold_counts = [
        row["iterations"] for row in cold.run_report_["cost"]["work"] if row["stage"] == "cv"
    ]
    expected = int(np.median(fold_counts))
    assert len(fold_counts) == 3 and len(set(fold_counts)) > 1
    assert [(fit.rows, fit.requested_rounds) for fit in fits[-2:]] == [
        (72, expected),
        (96, expected),
    ]
    assert cold.best_estimator_.fitted_iterations == expected
    fits.clear()
    warm = AutoML(**options).fit(x, y)
    assert [(fit.rows, fit.requested_rounds) for fit in fits] == [(72, expected), (96, expected)]
    assert warm.holdout_score_ == cold.holdout_score_ == 0.0
    assert warm.best_estimator_.fitted_iterations == expected


def test_hpo_wide_fallback_reuses_original_factory_and_dev_rounds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fits = _install_round_estimator(monkeypatch, tuned_bias=20.0)
    x, y = _data()
    options = _options(tmp_path)
    options["search"] = SearchConfig(max_rows=48, model_iterations=3)
    cold = AutoML(**options).fit(x, y)
    assert cold.run_report_["search"]["final_control"] == "wide_control"
    assert cold.schema_.selected_features is None
    assert cold.run_report_["hpo"]["tuned_on"] == "fs_subset"
    counts = [
        row["iterations"]
        for row in cold.run_report_["cost"]["work"]
        if row["stage"] == "wide_control"
    ]
    expected = int(np.median(counts))
    assert all(
        not fit.tuned and fit.columns == 3 and fit.requested_rounds == expected for fit in fits[-2:]
    )
    np.testing.assert_array_equal(cold.predict(x), y)
    fits.clear()
    warm = AutoML(**options).fit(x, y)
    assert len(fits) == 2 and all(
        not fit.tuned and fit.requested_rounds == expected for fit in fits
    )
    np.testing.assert_array_equal(warm.predict(x), y)


def test_time_reserve_stops_partial_hpo_and_ships_complete_wide_control(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = [0.0]
    tuned_fits = [0]

    def consume_time() -> None:
        tuned_fits[0] += 1
        if tuned_fits[0] == 3:
            clock[0] = 10000.0

    fits = _install_round_estimator(monkeypatch, on_tuned_fit=consume_time)

    def build_budget(self: AutoML, config: BudgetConfig) -> RunBudget:
        return RunBudget(config, clock=lambda: clock[0])

    monkeypatch.setattr(AutoML, "_build_budget", build_budget)
    x, y = _data()
    options = _options(tmp_path)
    options.update(
        feature_selection=None,
        search=SearchConfig(max_rows=48, model_iterations=3),
        budget=BudgetConfig(mode="time", time_budget_s=10000.0),
    )
    model = AutoML(**options).fit(x, y)
    hpo = model.run_report_["hpo"]["tuned"]["linear"]
    assert hpo["n_trials_run"] == 1 and hpo["completed"] is False
    assert model.run_report_["budget"]["exhausted"]
    assert model.run_report_["search"]["stop_reason"] == "completion_reserve"
    assert not list(tmp_path.glob("*/_stages/selection/meta.json"))
    assert tuned_fits[0] == 3
    assert not any(fit.tuned for fit in fits[-2:])
    assert not any(row["status"] == "failed" for row in model.run_report_["cost"]["work"])
    np.testing.assert_array_equal(model.predict(x), y)


def test_all_invalid_hpo_keeps_original_model_and_failure_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempted = [0]

    def fail_tuned_fit() -> None:
        attempted[0] += 1
        raise ValueError("invalid hyperparameter combination")

    _install_round_estimator(monkeypatch, on_tuned_fit=fail_tuned_fit)
    x, y = _data()
    options = _options(tmp_path)
    options.update(
        feature_selection=None,
        hpo=HPOConfig(n_trials=2, inner_cv=2, keep_baseline=True),
        search=SearchConfig(max_rows=48, model_iterations=3),
    )
    cold = AutoML(**options).fit(x, y)
    cold_hpo = cold.run_report_["hpo"]["tuned"]["linear"]
    assert cold_hpo["failed_trials"] == cold_hpo["n_trials_run"] == 2
    assert cold_hpo["successful_trials"] == 0 and not cold_hpo["completed"]
    assert cold_hpo["chosen_params"] == {} and cold_hpo["inner_best_score"] is None
    assert [entry.model_id for entry in cold.leaderboard_] == ["linear"]
    assert not any(row["stage"] == "cv" for row in cold.run_report_["cost"]["work"])
    np.testing.assert_array_equal(cold.predict(x), y)
    assert attempted[0] == 2
    warm = AutoML(**options).fit(x, y)
    warm_hpo = warm.run_report_["hpo"]["tuned"]["linear"]
    assert attempted[0] == 2
    assert warm_hpo["reused_trials"] == warm_hpo["failed_trials"] == 2
    assert not warm_hpo["completed"] and warm_hpo["successful_trials"] == 0
    assert [entry.model_id for entry in warm.leaderboard_] == ["linear"]
    np.testing.assert_array_equal(warm.predict(x), y)
