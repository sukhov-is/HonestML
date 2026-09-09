"""Additive preparation must preserve the actual scorer and its call sequence."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from honestml.adapters.metrics import Accuracy, LogLoss, Mae, Rmse, _prepare_bootstrap_score
from honestml.adapters.significance import BootstrapSignificanceTest
from honestml.core import Metric

pytestmark = pytest.mark.unit

KINDS = (LogLoss, Accuracy, Mae, Rmse)


def _inputs(kind: type[Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.tile(np.array([0, 1]), 8)
    a, b = np.linspace(0.1, 0.9, len(y)), np.linspace(0.8, 0.2, len(y))
    if kind is Accuracy:
        a, b = (a >= 0.5).astype(int), (b >= 0.5).astype(int)
    elif kind in (Mae, Rmse):
        y = y.astype(np.float64)
    return y, a, b


def _reference(
    metric: Metric, y: np.ndarray, a: np.ndarray, b: np.ndarray, weights: np.ndarray | None
) -> np.ndarray:
    rng = np.random.default_rng(42)
    test = BootstrapSignificanceTest(metric, seed=42, n_boot=13)
    return np.asarray(
        [test._delta(rng.integers(0, len(y), len(y)), a, b, y, weights) for _ in range(13)]
    )


@pytest.mark.parametrize("kind", KINDS)
def test_foreign_bound_score_is_not_prepared(kind: type[Any]) -> None:
    metric, other = kind(), kind(positive=0)
    y, a, _ = _inputs(kind)
    with patch.object(kind, "score", other.score):
        assert _prepare_bootstrap_score(metric, y, a, None) is None


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("weighted", [False, True])
def test_stateful_score_keeps_exact_calls_and_distribution(kind: type[Any], weighted: bool) -> None:
    metric = kind()
    y, a, b = _inputs(kind)
    weights = np.linspace(0.1, 2.0, len(y)) if weighted else None
    original = metric.score
    calls = 0

    def custom(
        target: np.ndarray, pred: np.ndarray, sample_weight: np.ndarray | None = None
    ) -> float:
        nonlocal calls
        calls += 1
        return original(target, pred, sample_weight) + calls / 100.0

    with patch.object(metric, "score", side_effect=custom):
        expected = _reference(metric, y, a, b, weights)
        assert calls == 26
        calls = 0
        actual = BootstrapSignificanceTest(metric, seed=42, n_boot=13)._delta_distribution(
            a, b, y, weights, None
        )
        assert calls == 26
        assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize("kind", KINDS)
def test_input_dependent_override_is_not_validated_on_full_input_only(kind: type[Any]) -> None:
    metric = kind()
    y, a, b = _inputs(kind)
    original = metric.score

    def custom(
        target: np.ndarray, pred: np.ndarray, sample_weight: np.ndarray | None = None
    ) -> float:
        return original(target, pred, sample_weight) + float(target[0])

    with patch.object(metric, "score", side_effect=custom):
        assert _prepare_bootstrap_score(metric, y, a, None) is None
        expected = _reference(metric, y, a, b, None)
        actual = BootstrapSignificanceTest(metric, seed=42, n_boot=13)._delta_distribution(
            a, b, y, None, None
        )
        assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize("scope", ["instance", "class", "foreign_bound"])
def test_logloss_helper_override_is_not_prepared(scope: str) -> None:
    metric = LogLoss()
    y = np.array([0, 1, 0, 1])
    pred = np.array([0.2, 0.8, 0.8, 0.2])
    calls = 0

    def orient(target: np.ndarray) -> np.ndarray:
        nonlocal calls
        calls += 1
        return target if calls % 2 else 1 - target

    replacement: Callable[..., np.ndarray] = (
        LogLoss(positive=0)._orient_binary if scope == "foreign_bound" else orient
    )
    owner = metric if scope == "instance" else LogLoss
    with patch.object(owner, "_orient_binary", side_effect=replacement):
        assert _prepare_bootstrap_score(metric, y, pred, None) is None
        assert calls == 0


def test_logloss_stateful_helper_keeps_exact_distribution() -> None:
    metric = LogLoss()
    y, a, b = _inputs(LogLoss)
    calls = 0

    def orient(target: np.ndarray) -> np.ndarray:
        nonlocal calls
        calls += 1
        return target if calls % 2 else 1 - target

    with patch.object(metric, "_orient_binary", side_effect=orient):
        expected = _reference(metric, y, a, b, None)
        assert calls == 26
        calls = 0
        actual = BootstrapSignificanceTest(metric, seed=42, n_boot=13)._delta_distribution(
            a, b, y, None, None
        )
        assert calls == 26
        assert actual.tobytes() == expected.tobytes()
