"""Ranking bootstrap preserves public scores on prepared and generic version paths."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from honestml.adapters import metrics
from honestml.adapters.metrics import PrAuc, RocAuc, _prepare_bootstrap_score
from honestml.adapters.significance import BootstrapSignificanceTest
from honestml.core import Metric

pytestmark = pytest.mark.unit


class _Reference:
    def __init__(self, metric: Metric) -> None:
        self.metric = metric

    def __getattr__(self, name: str) -> Any:
        return getattr(self.metric, name)

    def score(self, y: np.ndarray, pred: np.ndarray, weight: np.ndarray | None = None) -> float:
        return self.metric.score(y, pred, weight)


def _resample_score(
    metric: Metric, y: np.ndarray, pred: np.ndarray
) -> Callable[[np.ndarray], float]:
    prepared = _prepare_bootstrap_score(metric, y, pred, None)
    if metrics.sklearn.__version__ == "1.7.2":
        assert prepared is not None
        return prepared
    assert prepared is None
    return lambda indices: metric.score(y[indices], pred[indices])


def _assert_scalar_bytes(actual: float, expected: float) -> None:
    assert (
        np.asarray(actual, dtype=np.float64).tobytes()
        == np.asarray(expected, dtype=np.float64).tobytes()
    )


@pytest.mark.parametrize("kind", [RocAuc, PrAuc])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("ties", [False, True])
@pytest.mark.parametrize("labels", ["binary", "signed", "one_two", "positive_zero", "text"])
def test_selected_version_path_matches_every_public_resample(
    kind: type[RocAuc] | type[PrAuc],
    dtype: type[np.float32] | type[np.float64],
    ties: bool,
    labels: str,
) -> None:
    rng = np.random.default_rng(19)
    y = np.resize(np.array([0, 1]), 18)
    positive: object | None = None
    if labels == "signed":
        y = y * 2 - 1
    elif labels == "one_two":
        y = y + 1
    elif labels == "positive_zero":
        positive = 0
    elif labels == "text":
        y = np.where(y, "high", "low")
        positive = "low"
    metric = kind(positive=positive)
    pred = rng.normal(size=len(y)).astype(dtype)
    if ties:
        pred = np.round(pred, 0)
    prepared = _resample_score(metric, y, pred)
    indices = [
        np.arange(len(y)),
        np.arange(len(y))[::-1],
        np.array([0, 0, 0, 1, 7, 7, 11, 11, 11]),
        *(rng.integers(0, len(y), size=len(y)) for _ in range(17)),
    ]
    for selected in indices:
        _assert_scalar_bytes(prepared(selected), metric.score(y[selected], pred[selected]))


@pytest.mark.parametrize("kind", [RocAuc, PrAuc])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("blocked", [False, True])
@pytest.mark.parametrize("seed", [7, 42])
@pytest.mark.parametrize("force_generic", [False, True])
def test_distribution_preserves_draws_and_selected_path_calls(
    kind: type[RocAuc] | type[PrAuc],
    dtype: type[np.float32] | type[np.float64],
    blocked: bool,
    seed: int,
    force_generic: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if force_generic:
        monkeypatch.setattr(metrics.sklearn, "__version__", "0.0.0")
    rng = np.random.default_rng(5)
    y = np.resize(np.array([0, 1]), 32)
    a = np.round(rng.uniform(size=len(y)), 1).astype(dtype)
    b = np.round(rng.uniform(size=len(y)), 1).astype(dtype)
    blocks = np.repeat(np.arange(4), 8) if blocked else None
    metric = kind()
    reference = BootstrapSignificanceTest(_Reference(metric), seed=seed, n_boot=43)
    public_name = "roc_auc_score" if kind is RocAuc else "average_precision_score"
    with patch.object(metrics, public_name, wraps=getattr(metrics, public_name)) as public:
        expected = reference._delta_distribution(a, b, y, None, blocks)
        reference_calls = public.call_count
    with patch.object(metrics, public_name, wraps=getattr(metrics, public_name)) as public:
        actual = BootstrapSignificanceTest(metric, seed=seed, n_boot=43)._delta_distribution(
            a, b, y, None, blocks
        )
        assert public.call_count == (
            2 if metrics.sklearn.__version__ == "1.7.2" else reference_calls
        )
    assert actual.tobytes() == expected.tobytes()
    with (
        patch.object(np, "argsort", wraps=np.argsort) as sort,
        patch.object(metrics, public_name, wraps=getattr(metrics, public_name)) as public,
    ):
        score = _resample_score(metric, y, a)
        initial_sorts, initial_calls = sort.call_count, public.call_count
        for _ in range(12):
            score(rng.integers(0, len(y), size=len(y)))
        if metrics.sklearn.__version__ == "1.7.2":
            assert sort.call_count == initial_sorts
            assert public.call_count == initial_calls
        else:
            assert public.call_count == initial_calls + 12


@pytest.mark.parametrize("kind", [RocAuc, PrAuc])
@pytest.mark.parametrize("positive_only", [False, True])
def test_degenerate_draw_preserves_scalar_and_warning(
    kind: type[RocAuc] | type[PrAuc],
    positive_only: bool,
) -> None:
    y = np.array([0, 1, 0, 1])
    pred = np.array([0.2, 0.8, 0.2, 0.6])
    metric = kind()
    prepared = _resample_score(metric, y, pred)
    selected = np.array([1, 1, 3]) if positive_only else np.array([0, 0, 2])
    with warnings.catch_warnings(record=True) as ref_warnings:
        warnings.simplefilter("always")
        expected = metric.score(y[selected], pred[selected])
    with warnings.catch_warnings(record=True) as actual_warnings:
        warnings.simplefilter("always")
        actual = prepared(selected)
    _assert_scalar_bytes(actual, expected)
    assert [(w.category, str(w.message)) for w in actual_warnings] == [
        (w.category, str(w.message)) for w in ref_warnings
    ]


@pytest.mark.parametrize("kind", [RocAuc, PrAuc])
def test_absent_threshold_groups_and_constant_predictions(kind: type[RocAuc] | type[PrAuc]) -> None:
    y = np.array([0, 1, 0, 1, 0, 1])
    for pred in (np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]), np.ones(6)):
        prepared = _resample_score(kind(), y, pred)
        for selected in (np.array([0, 0, 5, 5]), np.array([1, 2, 2, 4])):
            _assert_scalar_bytes(prepared(selected), kind().score(y[selected], pred[selected]))


@pytest.mark.parametrize("kind", [RocAuc, PrAuc])
@pytest.mark.parametrize("version", ["1.7.1", "1.8.0", "1.7.2.dev0"])
def test_unverified_versions_keep_generic_distribution(
    monkeypatch: pytest.MonkeyPatch,
    kind: type[RocAuc] | type[PrAuc],
    version: str,
) -> None:
    y = np.resize(np.array([0, 1]), 20)
    a, b = np.linspace(0.0, 1.0, 20), np.linspace(1.0, 0.0, 20)
    metric = kind()
    expected = BootstrapSignificanceTest(_Reference(metric), seed=9, n_boot=13)._delta_distribution(
        a, b, y, None, None
    )
    monkeypatch.setattr(metrics.sklearn, "__version__", version)
    assert _prepare_bootstrap_score(metric, y, a, None) is None
    actual = BootstrapSignificanceTest(metric, seed=9, n_boot=13)._delta_distribution(
        a, b, y, None, None
    )
    assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize("kind", [RocAuc, PrAuc])
def test_weights_multiclass_and_invalid_inputs_keep_general_path(
    kind: type[RocAuc] | type[PrAuc],
) -> None:
    y = np.resize(np.array([0, 1]), 12)
    pred = np.linspace(0.0, 1.0, 12)
    for weights in (np.ones(12), np.linspace(0.1, 2.0, 12)):
        assert _prepare_bootstrap_score(kind(), y, pred, weights) is None
        expected = BootstrapSignificanceTest(
            _Reference(kind()), seed=7, n_boot=17
        )._delta_distribution(pred, pred[::-1], y, weights, None)
        actual = BootstrapSignificanceTest(kind(), seed=7, n_boot=17)._delta_distribution(
            pred, pred[::-1], y, weights, None
        )
        assert actual.tobytes() == expected.tobytes()
    assert _prepare_bootstrap_score(kind(), y, np.column_stack([pred, 1 - pred]), None) is None
    assert _prepare_bootstrap_score(kind(), y, pred.astype(np.float16), None) is None
    for bad in (np.nan, np.inf):
        changed = pred.copy()
        changed[0] = bad
        assert _prepare_bootstrap_score(kind(), y, changed, None) is None


@pytest.mark.parametrize("kind", [RocAuc, PrAuc])
@pytest.mark.parametrize(
    "override",
    ["subclass", "instance_score", "class_score", "instance_helper", "class_helper", "proxy"],
)
def test_custom_and_overridden_score_paths_are_not_prepared(
    monkeypatch: pytest.MonkeyPatch,
    kind: type[RocAuc] | type[PrAuc],
    override: str,
) -> None:
    metric = kind()
    y = np.array([0, 1, 0, 1])
    pred = np.array([0.2, 0.7, 0.4, 0.9])
    if override == "subclass":

        class Custom(kind):
            pass

        metric = Custom()
    elif override == "instance_score":
        monkeypatch.setattr(metric, "score", lambda y, pred, sample_weight=None: 0.25)
    elif override == "class_score":
        monkeypatch.setattr(kind, "score", lambda self, y, pred, sample_weight=None: 0.25)
    elif override == "instance_helper":
        monkeypatch.setattr(metric, "_orient_binary", lambda y: 1 - y)
    elif override == "class_helper":
        monkeypatch.setattr(kind, "_orient_binary", lambda self, y: 1 - y)
    else:
        metric = _Reference(metric)
    assert _prepare_bootstrap_score(metric, y, pred, None) is None


@pytest.mark.parametrize("kind", [RocAuc, PrAuc])
def test_stateful_instance_score_is_called_for_every_draw(
    monkeypatch: pytest.MonkeyPatch,
    kind: type[RocAuc] | type[PrAuc],
) -> None:
    metric = kind()
    calls = 0

    def stateful(y: np.ndarray, pred: np.ndarray, sample_weight: np.ndarray | None = None) -> float:
        nonlocal calls
        calls += 1
        return calls * 0.01

    monkeypatch.setattr(metric, "score", stateful)
    y = np.array([0, 1, 0, 1])
    a = np.array([0.2, 0.7, 0.4, 0.9])
    actual = BootstrapSignificanceTest(metric, seed=9, n_boot=11)._delta_distribution(
        a, a, y, None, None
    )
    assert calls == 22
    expected = np.array([(2 * i + 1) * 0.01 - (2 * i + 2) * 0.01 for i in range(11)])
    assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize("kind", [RocAuc, PrAuc])
def test_intervals_and_boundary_decisions_match(kind: type[RocAuc] | type[PrAuc]) -> None:
    rng = np.random.default_rng(101)
    y = np.resize(np.array([0, 1]), 24)
    a = np.round(rng.uniform(size=len(y)), 1)
    b = np.round(rng.uniform(size=len(y)), 1)
    blocks = np.repeat(np.arange(3), 8)
    fast = BootstrapSignificanceTest(kind(), seed=5, n_boot=1000)
    reference = BootstrapSignificanceTest(_Reference(kind()), seed=5, n_boot=1000)
    expected = reference._delta_distribution(a, b, y, None, blocks)
    actual = fast._delta_distribution(a, b, y, None, blocks)
    assert actual.tobytes() == expected.tobytes()
    np.testing.assert_array_equal(
        np.percentile(actual, [2.5, 97.5]), np.percentile(expected, [2.5, 97.5])
    )
    assert fast.equivalent(a, b, y, alpha=0.05, block_index=blocks) == reference.equivalent(
        a, b, y, alpha=0.05, block_index=blocks
    )
    threshold = max(0.0, -float(np.percentile(-expected, 5)))
    for margin in (0.0, max(0.0, threshold - 1e-14), threshold, threshold + 1e-14):
        assert fast.noninferior(
            a, b, y, alpha=0.05, margin=margin, block_index=blocks
        ) == reference.noninferior(a, b, y, alpha=0.05, margin=margin, block_index=blocks)


@pytest.mark.parametrize("kind", [RocAuc, PrAuc])
def test_multiclass_targets_with_explicit_positive_keep_fallback(
    kind: type[RocAuc] | type[PrAuc],
) -> None:
    y = np.resize(np.array([0, 1, 2]), 12)
    pred = np.linspace(0.0, 1.0, 12)
    assert _prepare_bootstrap_score(kind(positive=1), y, pred, None) is None


@pytest.mark.parametrize("kind", [RocAuc, PrAuc])
def test_score_bound_to_another_instance_keeps_fallback(
    kind: type[RocAuc] | type[PrAuc],
) -> None:
    metric = kind()
    other = kind(positive=0)
    y = np.array([0, 1, 0, 1])
    pred = np.array([0.0, 1.0, 1.0, 0.0])
    with patch.object(kind, "score", other.score):
        assert _prepare_bootstrap_score(metric, y, pred, None) is None
        assert metric._prepare_ensemble_score(y, None) is None
