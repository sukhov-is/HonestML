"""Prepared additive bootstrap keeps the reference draws and scorer semantics."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from honestml.adapters import metrics
from honestml.adapters.metrics import Accuracy, LogLoss, Mae, PrAuc, Rmse, RocAuc
from honestml.adapters.significance import BootstrapSignificanceTest
from honestml.core import Metric

pytestmark = pytest.mark.unit


def _reference(
    metric: Metric,
    a: np.ndarray,
    b: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray | None,
    blocks: np.ndarray | None,
    *,
    seed: int = 9,
    n_boot: int = 80,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    test = BootstrapSignificanceTest(metric, seed=seed, n_boot=n_boot)
    groups = (
        None
        if blocks is None
        else [np.flatnonzero(blocks == key) for key in np.unique(blocks) if key >= 0]
    )
    result = []
    for _ in range(n_boot):
        idx = (
            rng.integers(0, len(y), size=len(y))
            if groups is None
            else np.concatenate([groups[j] for j in rng.integers(0, len(groups), size=len(groups))])
        )
        result.append(test._delta(idx, a, b, y, weights))
    return np.asarray(result)


@pytest.mark.parametrize("name", ["log_loss", "accuracy", "mae", "rmse"])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("blocked", [False, True])
@pytest.mark.parametrize("force_generic", [False, True])
def test_additive_distribution_matches_reference_and_selected_path_calls(
    name: str,
    weighted: bool,
    blocked: bool,
    force_generic: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if force_generic:
        monkeypatch.setattr(metrics.sklearn, "__version__", "0.0.0")
    rng = np.random.default_rng(17)
    y = np.tile([0, 1, 2], 20)
    a = rng.dirichlet([1, 2, 3], len(y))
    b = rng.dirichlet([2, 2, 3], len(y))
    metric = {
        "log_loss": LogLoss(classes=np.arange(3)),
        "accuracy": Accuracy(),
        "mae": Mae(),
        "rmse": Rmse(),
    }[name]
    if name == "accuracy":
        a, b = a.argmax(axis=1), b.argmax(axis=1)
    elif name in {"mae", "rmse"}:
        y, a, b = y.astype(float), a[:, 0], b[:, 0]
    weights = rng.uniform(0.1, 2, len(y)) if weighted else None
    blocks = np.repeat(np.arange(4), 15) if blocked else None
    public_name = {
        "log_loss": "log_loss",
        "accuracy": "accuracy_score",
        "mae": "mean_absolute_error",
        "rmse": "mean_squared_error",
    }[name]
    with (
        patch.object(metric, "score", wraps=metric.score),
        patch.object(metrics, public_name, wraps=getattr(metrics, public_name)) as scorer,
    ):
        expected = BootstrapSignificanceTest(metric, seed=9, n_boot=80)._delta_distribution(
            a, b, y, weights, blocks
        )
        reference_calls = scorer.call_count
    supported = name != "log_loss" or metrics.sklearn.__version__ == "1.7.2"
    prepared = metrics._prepare_bootstrap_score(metric, y, a, weights)
    assert (prepared is not None) == supported
    with patch.object(metrics, public_name, wraps=getattr(metrics, public_name)) as scorer:
        actual = BootstrapSignificanceTest(metric, seed=9, n_boot=80)._delta_distribution(
            a, b, y, weights, blocks
        )
        if supported:
            assert scorer.call_count <= 2
        else:
            assert scorer.call_count == reference_calls
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
@pytest.mark.parametrize("explicit_labels", [False, True])
def test_log_loss_clipping_labels_and_degenerate_resamples(
    dtype: type[np.float16] | type[np.float32] | type[np.float64], explicit_labels: bool
) -> None:
    y = np.array(["low", "high", "high", "high"])
    a = np.array([0, 1, 0.3, 0.999], dtype=dtype)
    b = np.array([1, 0, 0.7, 0.001], dtype=dtype)
    if explicit_labels:
        a, b = np.column_stack((1 - a, a)), np.column_stack((1 - b, b))
        metric = LogLoss(classes=np.array(["high", "low"]))
    else:
        metric = LogLoss(positive="low")
    expected = _reference(metric, a, b, y, None, None)
    actual = BootstrapSignificanceTest(metric, seed=9, n_boot=80)._delta_distribution(
        a, b, y, None, None
    )
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("metric", [RocAuc(), PrAuc()])
def test_ranking_metrics_use_the_general_path(metric: Metric) -> None:
    y = np.tile([0, 1], 20)
    a, b = np.linspace(0.01, 0.99, 40), np.linspace(0.99, 0.01, 40)
    with patch.object(metric, "score", wraps=metric.score) as scorer:
        BootstrapSignificanceTest(metric, n_boot=20)._delta_distribution(a, b, y, None, None)
    assert scorer.call_count == 40


def test_custom_subclass_keeps_its_scorer() -> None:
    class ShiftedLoss(LogLoss):
        def score(
            self,
            y_true: np.ndarray,
            y_pred: np.ndarray,
            sample_weight: np.ndarray | None = None,
        ) -> float:
            return super().score(y_true, y_pred, sample_weight) + float(y_pred[0])

    metric = ShiftedLoss(classes=np.arange(3))
    rng = np.random.default_rng(2)
    y = np.tile(np.arange(3), 10)
    a, b = rng.dirichlet([1, 1, 1], 30), rng.dirichlet([2, 2, 1], 30)
    # the custom scorer is scalar-valued while still order-dependent.
    a, b, y = a[:, 0], b[:, 0], y % 2
    expected = _reference(metric, a, b, y, None, None)
    actual = BootstrapSignificanceTest(metric, seed=9, n_boot=80)._delta_distribution(
        a, b, y, None, None
    )
    np.testing.assert_array_equal(actual, expected)


def test_zero_weight_draw_preserves_the_reference_exception() -> None:
    metric = LogLoss(classes=np.arange(3))
    y = np.arange(3)
    a = np.array([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.2, 0.2, 0.6]])
    b = np.full((3, 3), 1 / 3)
    weights = np.array([1.0, 0.0, 0.0])
    with pytest.raises(ZeroDivisionError):
        _reference(metric, a, b, y, weights, None)
    with pytest.raises(ZeroDivisionError):
        BootstrapSignificanceTest(metric, seed=9, n_boot=80)._delta_distribution(
            a, b, y, weights, None
        )


@pytest.mark.parametrize("metric", [LogLoss(classes=np.arange(3)), Rmse(), Accuracy()])
def test_confidence_boundaries_and_decisions_match_public_scorer(metric: Metric) -> None:
    class ReferenceMetric:
        def __init__(self, base: Metric) -> None:
            self.base = base
            self.name = base.name
            self.greater_is_better = base.greater_is_better
            self.needs = base.needs
            self.optimum = base.optimum
            self.average = base.average

        def score(
            self,
            y_true: np.ndarray,
            y_pred: np.ndarray,
            sample_weight: np.ndarray | None = None,
        ) -> float:
            return self.base.score(y_true, y_pred, sample_weight)

    rng = np.random.default_rng(13)
    y = np.tile(np.arange(3), 30)
    a = rng.dirichlet([2, 2, 2], len(y))
    b = np.roll(a, 1, axis=0)
    if isinstance(metric, Rmse):
        a, b, y = a[:, 0], b[:, 0], y.astype(float)
    elif isinstance(metric, Accuracy):
        a, b = a.argmax(axis=1), b.argmax(axis=1)
    weights = rng.uniform(0.1, 2, len(y))
    fast = BootstrapSignificanceTest(metric, seed=4, n_boot=1000)
    ref = BootstrapSignificanceTest(ReferenceMetric(metric), seed=4, n_boot=1000)
    dfast = fast._delta_distribution(a, b, y, weights, None)
    dref = ref._delta_distribution(a, b, y, weights, None)
    np.testing.assert_array_equal(dfast, dref)
    np.testing.assert_array_equal(
        np.percentile(dfast, [2.5, 5, 97.5]), np.percentile(dref, [2.5, 5, 97.5])
    )
    assert fast.equivalent(a, b, y, alpha=0.05, sample_weight=weights) == ref.equivalent(
        a, b, y, alpha=0.05, sample_weight=weights
    )
    orientation = 1 if metric.greater_is_better else -1
    boundary = -float(np.percentile(-orientation * dref, 5))
    for margin in [boundary - 1e-14, boundary, boundary + 1e-14]:
        assert fast.noninferior(a, b, y, alpha=0.05, margin=margin, sample_weight=weights) == (
            ref.noninferior(a, b, y, alpha=0.05, margin=margin, sample_weight=weights)
        )


@pytest.mark.parametrize("bad", ["nan", "out_of_range", "missing_class"])
def test_log_loss_unsupported_inputs_retain_resample_semantics(bad: str) -> None:
    y = np.array([0, 0, 1, 1])
    a = np.array([0.1, 0.2, 0.8, 0.9])
    b = 1 - a
    if bad == "nan":
        a[0] = np.nan
    elif bad == "out_of_range":
        a[0] = 2.0
    else:
        y[:] = 0
    metric = LogLoss()
    expected = _reference(metric, a, b, y, None, None)
    actual = BootstrapSignificanceTest(metric, seed=9, n_boot=80)._delta_distribution(
        a, b, y, None, None
    )
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("version", ["1.3.0", "1.7.1", "1.8.0", "1.7.2.dev0"])
def test_log_loss_unverified_versions_use_public_scorer(
    version: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    rng = np.random.default_rng(3)
    y = np.resize(np.arange(3), 30)
    a, b = rng.dirichlet([1, 2, 3], 30), rng.dirichlet([2, 3, 4], 30)
    metric = LogLoss(classes=np.arange(3))
    expected = _reference(metric, a, b, y, None, None)
    monkeypatch.setattr("honestml.adapters.metrics.sklearn.__version__", version)
    with patch.object(metric, "score", wraps=metric.score) as scorer:
        actual = BootstrapSignificanceTest(metric, seed=9, n_boot=80)._delta_distribution(
            a, b, y, None, None
        )
    assert scorer.call_count == 160
    np.testing.assert_array_equal(actual, expected)
