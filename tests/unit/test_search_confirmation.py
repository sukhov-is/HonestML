"""Confirmation keeps selection bounded and exposes weak DEV evidence."""

from collections.abc import Sequence

import numpy as np
import pytest

from honestml.application.search import probe_models
from honestml.core import Fold, Task
from honestml.core.config import SearchConfig
from honestml.core.selection_policy import Candidate

pytestmark = pytest.mark.unit


class Metric:
    name = "accuracy"
    greater_is_better = True
    needs = "class"
    optimum = 1.0
    average = None
    proper_proba = False

    def score(
        self, y_true: np.ndarray, y_pred: np.ndarray, sample_weight: np.ndarray | None = None
    ) -> float:
        return float(np.average(y_true == y_pred, weights=sample_weight))


def folds() -> list[Fold]:
    return [
        Fold(np.arange(600), np.arange(600, 800), np.arange(800, 1000)),
        Fold(np.arange(1000), np.arange(1000, 1200), np.arange(1200, 1400)),
    ]


def test_resource_confirmation_detects_rank_inversion_with_two_finalists() -> None:
    seen: list[tuple[str, int, int]] = []

    def evaluate(name: str, parts: Sequence[Fold], iterations: int) -> Candidate:
        seen.append(
            (name, iterations, sum(len(f.fit_idx) + len(f.es_idx) + len(f.test_idx) for f in parts))
        )
        scores = {"linear": 0.9, "boost": 0.8, "third": 0.7}
        if iterations == 256:
            scores = {"linear": 0.6, "boost": 0.95}
        return Candidate(name, scores[name], train_time=1.0)

    result = probe_models(
        ["linear", "boost", "third"],
        folds(),
        y=np.arange(1400.0),
        groups=None,
        task=Task(kind="regression"),
        metric=Metric(),
        config=SearchConfig(max_rows=100, confirmation_rows=600),
        seed=1,
        evaluate=evaluate,
    )
    assert result.winner == "boost"
    assert result.rank_changed
    assert [name for name, iterations, _ in seen if iterations == 256] == ["linear", "boost"]
    assert all(rows <= (100 if iterations == 64 else 600) for _, iterations, rows in seen)
    assert result.reason == "confirmation_score"


def test_small_es_and_rare_classes_are_reported_after_confirmation() -> None:
    y = np.zeros(1400)
    y[[0, 600, 800, 1000, 1200]] = 1
    parts = [Fold(np.arange(800), np.array([1000]), np.arange(1200, 1400))]
    result = probe_models(
        ["a", "b"],
        parts,
        y=y,
        groups=np.arange(1400),
        task=Task(kind="binary"),
        metric=Metric(),
        config=SearchConfig(),
        seed=1,
        evaluate=lambda name, parts, iterations: Candidate(name, 1.0),
    )
    assert result.reason == "insufficient_confirmation"
    assert "small_es" in result.issues
    assert "few_test_class_rows" in result.issues
    assert result.diagnostics["confirmation"][0]["es"]["rows"] == 1


def test_partial_confirmation_keeps_first_round_winner_and_reports_uncertainty() -> None:
    calls = 0

    def evaluate(name: str, parts: Sequence[Fold], iterations: int) -> Candidate:
        nonlocal calls
        calls += 1
        return Candidate(name, 0.9 if name == "a" and iterations == 64 else 0.8)

    result = probe_models(
        ["a", "b"],
        folds(),
        y=np.arange(1400.0),
        groups=None,
        task=Task(kind="regression"),
        metric=Metric(),
        config=SearchConfig(),
        seed=0,
        evaluate=evaluate,
        can_start=lambda: calls < 3,
    )
    assert calls == 3
    assert result.winner == "a"
    assert result.reason == "incomplete_confirmation_budget"


class Noninferiority:
    seed = 0
    n_boot = 1000

    def equivalent(self, *args: object, **kwargs: object) -> bool:
        raise AssertionError("failure to reject a difference is not evidence of noninferiority")

    def noninferior(self, *args: object, **kwargs: object) -> bool:
        assert kwargs["margin"] == 0.01
        return True


def test_cost_preference_requires_explicit_margin_and_paired_noninferiority() -> None:
    y = np.tile([0, 1], 700)

    def evaluate(name: str, parts: Sequence[Fold], iterations: int) -> Candidate:
        mask = np.zeros(len(y), dtype=bool)
        for part in parts:
            mask[part.test_idx] = True
        return Candidate(
            name,
            0.9 if name == "slow" else 0.895,
            train_time=20.0 if name == "slow" else 1.0,
            oof_pred=y.copy(),
            oof_mask=mask,
            refit_iterations=iterations,
        )

    result = probe_models(
        ["slow", "cheap"],
        folds(),
        y=y,
        groups=None,
        task=Task(kind="binary"),
        metric=Metric(),
        config=SearchConfig(model_margin=0.01),
        seed=0,
        evaluate=evaluate,
        significance_test=Noninferiority(),
        full_iterations={"slow": 1000, "cheap": 1000},
    )
    assert result.winner == "cheap"
    assert result.reason == "confirmed_cost"
    assert result.cost_estimates["cheap"] < result.cost_estimates["slow"]


def test_zero_margin_does_not_spend_quality_for_speed_without_evidence() -> None:
    result = probe_models(
        ["slow", "cheap"],
        folds(),
        y=np.arange(1400.0),
        groups=None,
        task=Task(kind="regression"),
        metric=Metric(),
        config=SearchConfig(),
        seed=0,
        evaluate=lambda name, parts, iterations: Candidate(
            name, 0.9 if name == "slow" else 0.895, train_time=20 if name == "slow" else 1
        ),
    )
    assert result.winner == "slow"


@pytest.mark.parametrize(
    "field,value",
    [
        ("confirmation_rows", 32),
        ("confirmation_iterations", 1),
        ("confirmation_folds", 1),
    ],
)
def test_confirmation_cannot_reduce_the_first_round_resource(field: str, value: int) -> None:
    with pytest.raises(ValueError, match="confirmation resource"):
        SearchConfig(max_folds=2, **{field: value})
