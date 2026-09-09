"""Conditional HPO cost profiles on deterministic fake estimators, without native training."""

from __future__ import annotations

import copy
import json
from typing import Any

import numpy as np
import pytest

import honestml.application.tuning as tuning
from honestml.adapters import Accuracy
from honestml.core import FEConfig, Fold, RunContext, SearchConfig, SelectionPolicy, Task
from honestml.core.exceptions import BudgetExhaustedError

pytestmark = pytest.mark.unit


class _Schema:
    selected_features = None
    features = ["row", "other"]
    numeric = features
    categorical: list[str] = []
    target_encoding = None


class _Dataset:
    schema = _Schema()

    def __init__(self) -> None:
        self.x = np.column_stack((np.arange(60), np.arange(60) * 2)).astype(float)
        self.y = np.arange(60) % 2

    def target(self) -> np.ndarray:
        return self.y

    def to_numpy(self) -> np.ndarray:
        return self.x

    def categorical_codes(self) -> np.ndarray:
        return np.empty((60, 0), dtype=np.int64)

    @property
    def n_rows(self) -> int:
        return 60


class _Splitter:
    def split(self, dataset: Any) -> list[Fold]:
        rows = np.arange(60)
        return [
            Fold(rows[20:], rows[10:20], rows[:10]),
            Fold(rows[:40], rows[40:50], rows[50:]),
            Fold(rows[20:], rows[:10], rows[10:20]),
        ]


class _Model:
    def __init__(
        self, params: dict[str, Any], clock: list[float], fits: list[dict[str, Any]]
    ) -> None:
        self.params = params
        self.clock = clock
        self.fits = fits
        self.feature_names: list[str] = []
        self.limit = int(params["rounds"])
        self.threads = 0
        self.fitted_iterations: int | None = None

    def iteration_limit(self, *, early_stopping: bool) -> int:
        return self.limit

    def set_refit_iterations(self, count: int) -> None:
        self.limit = count

    def set_threads(self, threads: int) -> None:
        self.threads = threads

    @property
    def iteration_budget(self) -> int:
        return self.limit

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> _Model:
        self.clock[0] += len(y) * self.limit / 100
        self.fitted_iterations = self.limit
        self.fits.append(
            {
                "params": self.params,
                "rows": x[:, 0].copy(),
                "columns": x.shape[1],
                "es_rows": 0 if X_val is None else len(X_val),
                "threads": self.threads,
                "limit": self.limit,
                "weights": None if sample_weight is None else sample_weight.copy(),
            }
        )
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return x[:, 0].astype(int) % 2


class _ESModel(_Model):
    supports_early_stopping = True


SPACE = {
    "rounds": {"type": "int", "low": 100, "high": 500, "step": 50},
    "rate": {"type": "float", "low": 0.01, "high": 1.0, "log": True},
    "depth": {"type": "float", "low": 2.0, "high": 6.0},
    "mode": {"type": "categorical", "choices": ["a", "b", "c"]},
}


@pytest.fixture
def inputs(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    clock = [0.0]
    monkeypatch.setattr("time.perf_counter", lambda: clock[0])
    fits: list[dict[str, Any]] = []
    return {
        "ds": _Dataset(),
        "clock": clock,
        "fits": fits,
        "ctx": RunContext(),
        "search": SearchConfig(
            max_rows=32,
            confirmation_rows=96,
            max_folds=1,
            confirmation_folds=2,
            model_iterations=8,
            confirmation_iterations=16,
            threads=4,
        ),
    }


def _profile(inputs: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    model = overrides.pop("model", _ESModel)
    kwargs = dict(
        tunable={"model": copy.deepcopy(SPACE)},
        make_factory=lambda name, params: (
            lambda: model(dict(params), inputs["clock"], inputs["fits"])
        ),
        metric=Accuracy(),
        policy=SelectionPolicy(greater_is_better=True),
        inner_splitter=_Splitter(),
        n_trials=5,
        timeout_s=0.001,
        random_state=42,
        search=inputs["search"],
        ctx=inputs["ctx"],
    )
    kwargs.update(overrides)
    return tuning.profile_tuning_cost(inputs["ds"], Task(kind="binary"), **kwargs)


@pytest.mark.parametrize("model", [_Model, _ESModel])
def test_profile_preserves_resources_weights_and_two_level_extrapolation(
    inputs: dict[str, Any], model: type
) -> None:
    weights = np.arange(60, dtype=float) + 1
    result = _profile(inputs, model=model, sample_weight=weights)["model"]
    assert result["status"] == "conditional"
    assert result["planned_fit_count"] == 15
    assert result["estimated_s"] == pytest.approx(1800.0 if model is _ESModel else 2250.0)
    assert result["timeout_limit_s"] == 0.001
    assert len(result["profiles"]) == 2
    low, high = result["profiles"]
    assert (
        low["params"]
        == high["params"]
        == {"rounds": 300, "rate": pytest.approx(0.1), "depth": 4.0, "mode": "b"}
    )
    assert low["params_sha256"] == high["params_sha256"]
    assert low["features_sha256"] == high["features_sha256"]
    assert low["iteration_cap"] == 8 and high["iteration_cap"] == 16
    assert low["fit_count"] == 1 and high["fit_count"] == 2
    assert result["validation"]["error_s"] == pytest.approx(0.0)
    assert len(inputs["fits"]) == 3
    for fit in inputs["fits"]:
        assert fit["threads"] == 4
        np.testing.assert_array_equal(fit["weights"], weights[fit["rows"].astype(int)])
        assert bool(fit["es_rows"]) is (model is _ESModel)
    work = inputs["ctx"].cost_report()["work"]
    assert len(work) == 3 and all(item["stage"] == "scouting" for item in work)
    assert inputs["ctx"].timings["run"]["hpo_cost_probe"] > 0
    json.dumps(result, allow_nan=False)


def test_stopped_second_level_keeps_measurement_but_no_estimate(inputs: dict[str, Any]) -> None:
    allowed = iter((True, False))
    result = _profile(inputs, can_start=lambda: next(allowed))["model"]
    assert result["status"] == "unknown" and result["estimated_s"] is None
    assert result["reason"] == "probe_budget"
    assert len(result["profiles"]) == 1 and len(inputs["fits"]) == 1


def test_fit_checkpoint_interrupts_mid_level_without_partial_estimate(
    inputs: dict[str, Any],
) -> None:
    attempts = [0]

    def check(stage: str) -> None:
        attempts[0] += 1
        if attempts[0] == 3:
            raise BudgetExhaustedError("trials", completed=2, skipped=1, failed=0)

    inputs["ctx"].before_fit = check
    result = _profile(inputs)["model"]
    assert result["status"] == "unknown" and result["estimated_s"] is None
    assert len(result["profiles"]) == 1
    assert len(inputs["ctx"].cost_report()["work"]) == 2


def test_invalid_candidate_never_becomes_cost_evidence(inputs: dict[str, Any]) -> None:
    class Broken(_ESModel):
        def predict(self, x: np.ndarray) -> np.ndarray:
            raise ValueError("invalid prediction")

    result = _profile(inputs, model=Broken)["model"]
    assert result["status"] == "unknown" and result["estimated_s"] is None
    assert result["profiles"] == []


def test_empty_and_unsupported_spaces_do_not_fit(inputs: dict[str, Any]) -> None:
    assert _profile(inputs, tunable={}) == {}
    result = _profile(inputs, tunable={"off": {}, "bad": {"opaque": object()}})
    assert result["off"]["status"] == "disabled" and result["off"]["estimated_s"] == 0
    assert result["bad"]["status"] == "unknown" and result["bad"]["estimated_s"] is None
    assert inputs["fits"] == []


def test_missing_iteration_protocol_does_not_fit(inputs: dict[str, Any]) -> None:
    result = _profile(inputs, make_factory=lambda name, params: lambda: object())["model"]
    assert result["status"] == "unknown"
    assert result["reason"] == "unsupported_iteration_protocol"
    assert inputs["fits"] == []


def test_shared_inner_preparation_runs_te_before_feature_projection(
    inputs: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = inputs["ds"]
    dataset.schema = copy.copy(dataset.schema)
    dataset.schema.target_encoding = {"other": object()}
    seen: list[Any] = []

    def encode(x: np.ndarray, *args: Any, **kwargs: Any) -> np.ndarray:
        seen.append((x.shape[1], kwargs["time_ordered"]))
        return x + 100

    monkeypatch.setattr(tuning, "_augment_oof_te", encode)
    prepared = tuning._prepare_objective(
        dataset, Task(kind="binary"), _Splitter(), FEConfig(target_encoding=True), ("other",)
    )
    assert seen == [(2, False)]
    np.testing.assert_array_equal(prepared.x_eval[:, 0], dataset.x[:, 1] + 100)
    assert prepared.feature_names == ["other"]


def test_native_ceiling_clips_both_requested_iteration_levels(inputs: dict[str, Any]) -> None:
    search = inputs["search"].model_copy(
        update={"model_iterations": 512, "confirmation_iterations": 1024}
    )
    result = _profile(inputs, search=search)["model"]
    assert result["status"] == "conditional"
    assert [p["iteration_cap"] for p in result["profiles"]] == [300, 300]
    assert all(fit["limit"] == 300 for fit in inputs["fits"])


def test_identical_resources_are_not_independent_validation(inputs: dict[str, Any]) -> None:
    search = inputs["search"].model_copy(
        update={"confirmation_rows": 32, "confirmation_folds": 1, "confirmation_iterations": 8}
    )
    result = _profile(inputs, search=search)["model"]
    assert result["status"] == "unknown" and result["estimated_s"] is None
    assert result["reason"] == "identical_probe_resources"
    assert len(inputs["fits"]) == 1 and result["validation"] is None
