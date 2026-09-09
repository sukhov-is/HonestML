"""Exact-input ensemble memoization preserves the application decision."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from honestml.adapters import metrics
from honestml.adapters.ensembling import CaruanaEnsembler
from honestml.adapters.metrics import Accuracy, LogLoss, Mae, PrAuc, Rmse, RocAuc
from honestml.adapters.significance import BootstrapSignificanceTest
from honestml.application.ensemble import EnsembleOutcome, ensemble_selection
from honestml.core import Candidate, EnsembleRecipe, Metric, SelectionPolicy, Task

pytestmark = pytest.mark.unit


class _ReferenceLogLoss(LogLoss):
    pass


class _TraceCaruana(CaruanaEnsembler):
    def __init__(self) -> None:
        super().__init__(size=7, n_bags=4)
        self.traces: list[np.ndarray] = []
        self.selections: list[np.ndarray] = []
        self.best_counts: list[np.ndarray] = []
        self.metric_calls = 0

    def _greedy(
        self, oof: np.ndarray, score: Callable[[np.ndarray], float], lib: np.ndarray
    ) -> np.ndarray:
        values: list[float] = []

        def observed(pred: np.ndarray) -> float:
            value = score(pred)
            values.append(value)
            return value

        before = getattr(metrics.log_loss, "call_count", 0)
        counts = super()._greedy(oof, observed, lib)
        self.metric_calls += getattr(metrics.log_loss, "call_count", 0) - before
        trace = np.asarray(values).reshape(self.size, len(lib) + 1)
        self.traces.append(trace)
        self.selections.append(lib[np.argmax(trace[:, : len(lib)], axis=1)])
        self.best_counts.append(counts.copy())
        return counts


def _application(
    metric: Metric, seed: int, ties: bool, *, weighted: bool = True
) -> tuple[EnsembleOutcome, _TraceCaruana]:
    rng = np.random.default_rng(seed)
    y = np.resize(np.array([0, 1]), 48)
    proba = rng.uniform(0.05, 0.95, (3, len(y)))
    if ties:
        proba[1] = proba[0]
    weights = rng.uniform(0.1, 2, len(y)) if weighted else None
    candidates = [
        Candidate(
            id=str(i),
            score=0.0,
            oof_pred=(pred >= 0.5).astype(int),
            oof_proba=pred,
            oof_mask=np.ones(len(y), dtype=bool),
        )
        for i, pred in enumerate(proba)
    ]
    trace = _TraceCaruana()
    result = ensemble_selection(
        candidates,
        Task(kind="binary", metric="log_loss"),
        y=y,
        best_model_id="0",
        ensembler=trace,
        metric=metric,
        significance_test=BootstrapSignificanceTest(LogLoss(), seed=11, n_boot=1000),
        policy=SelectionPolicy(greater_is_better=False),
        significance_mode="bootstrap",
        sample_weight=weights,
        random_state=9,
    )
    return result, trace


@pytest.mark.parametrize("seed", [7, 42])
@pytest.mark.parametrize("ties", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
def test_application_reduces_calls_preserving_every_selection_and_gate(
    seed: int, ties: bool, weighted: bool
) -> None:
    with patch.object(metrics, "log_loss", wraps=metrics.log_loss) as score:
        actual, observed = _application(LogLoss(), seed, ties, weighted=weighted)
        actual_calls = score.call_count
    with patch.object(metrics, "log_loss", wraps=metrics.log_loss) as score:
        expected, reference = _application(_ReferenceLogLoss(), seed, ties, weighted=weighted)
        reference_calls = score.call_count
    assert actual == expected
    assert actual_calls < reference_calls
    assert observed.metric_calls < reference.metric_calls
    for left, right in zip(observed.traces, reference.traces):
        assert left.tobytes() == right.tobytes()
    for left, right in zip(observed.selections, reference.selections):
        np.testing.assert_array_equal(left, right)
    for left, right in zip(observed.best_counts, reference.best_counts):
        np.testing.assert_array_equal(left, right)


def _prepared(
    metric: LogLoss, y: np.ndarray, sw: np.ndarray | None = None
) -> Callable[[np.ndarray], float]:
    prepare = getattr(metric, "_prepare_ensemble_score", None)
    assert prepare is not None
    scorer = prepare(y, sw)
    assert scorer is not None
    return scorer


@pytest.mark.parametrize(
    "field,limit", [("_ENSEMBLE_CACHE_MAX_ENTRIES", 1), ("_ENSEMBLE_CACHE_MAX_BYTES", 1)]
)
def test_cache_eviction_or_disabled_capacity_retains_scalar_values(
    monkeypatch: pytest.MonkeyPatch, field: str, limit: int
) -> None:
    monkeypatch.setattr(metrics, field, limit, raising=False)
    y = np.array([0, 1, 0, 1])
    a = np.array([0.2, 0.8, 0.3, 0.7])
    b = 1 - a
    scorer = _prepared(LogLoss(), y)
    with patch.object(metrics, "log_loss", wraps=metrics.log_loss) as public:
        values = [scorer(p) for p in (a, b, a)]
        assert public.call_count == 3
    assert values == [LogLoss().score(y, p) for p in (a, b, a)]


def test_cache_byte_budget_evicts_retained_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(metrics, "_ENSEMBLE_CACHE_MAX_BYTES", 1800, raising=False)
    y = np.resize(np.array([0, 1]), 16)
    scorer = _prepared(LogLoss(), y)
    inputs = [np.full(len(y), 0.1 + i * 0.01) for i in range(20)]
    with patch.object(metrics, "log_loss", wraps=metrics.log_loss) as public:
        first = scorer(inputs[0])
        for pred in inputs[1:]:
            scorer(pred)
        assert scorer(inputs[0]) == first
        assert public.call_count == 21
        assert scorer(inputs[0]) == first
        assert public.call_count == 21


def test_mutating_prediction_buffer_and_layout_are_separate_keys() -> None:
    y = np.array([0, 1, 0, 1])
    base = np.array([0.2, 0.8, 0.3, 0.7])
    pred = base.copy()
    scorer = _prepared(LogLoss(), y)
    with patch.object(metrics, "log_loss", wraps=metrics.log_loss) as public:
        a = scorer(pred)
        assert scorer(pred.copy()) == a
        assert public.call_count == 1
        pred[0] = 0.6
        assert scorer(pred) != a
        assert public.call_count == 2
        pred[:] = base
        assert scorer(pred) == a
        assert public.call_count == 2
        strided = np.repeat(base, 2)[::2]
        assert scorer(strided) == a
        assert public.call_count == 3
        assert scorer(base.astype(np.float32)) == LogLoss().score(y, base.astype(np.float32))
        assert public.call_count == 5


def test_shape_and_dtype_identity_do_not_alias() -> None:
    y = np.array([0, 1, 0, 1])
    pred = np.array([0.2, 0.8, 0.3, 0.7])
    scorer = _prepared(LogLoss(), y)
    with patch.object(metrics, "log_loss", wraps=metrics.log_loss) as public:
        assert scorer(pred) == scorer(pred.reshape(-1, 1))
        assert public.call_count == 2


def test_target_and_weight_snapshots_are_fixed_and_caches_are_local() -> None:
    y = np.array([0, 1, 0, 1])
    sw = np.array([1.0, 2.0, 3.0, 4.0])
    pred = np.array([0.2, 0.8, 0.3, 0.7])
    metric = LogLoss()
    expected = metric.score(y, pred, sw)
    first = _prepared(metric, y, sw)
    second = _prepared(metric, y, sw)
    y[:] = 1
    sw[:] = 0
    with patch.object(metrics, "log_loss", wraps=metrics.log_loss) as public:
        assert first(pred) == expected
        assert first(pred) == expected
        assert second(pred) == expected
        assert public.call_count == 2


@pytest.mark.parametrize("metric", [LogLoss(), Accuracy(), Mae(), Rmse(), RocAuc(), PrAuc()])
def test_audited_exact_builtins_prepare(metric: Metric) -> None:
    prepare = getattr(metric, "_prepare_ensemble_score", None)
    assert prepare is not None
    assert prepare(np.array([0, 1]), None) is not None


class _StatefulLogLoss(LogLoss):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def score(
        self, y_true: np.ndarray, y_pred: np.ndarray, sample_weight: np.ndarray | None = None
    ) -> float:
        self.calls += 1
        return super().score(y_true, y_pred, sample_weight) + self.calls * 1e-8


def test_subclasses_and_instance_overrides_do_not_prepare() -> None:
    y = np.array([0, 1])
    for metric in (_ReferenceLogLoss(), _StatefulLogLoss()):
        prepare = getattr(metric, "_prepare_ensemble_score", None)
        if prepare is not None:
            assert prepare(y, None) is None
    metric = LogLoss()
    with patch.object(metric, "score", wraps=metric.score):
        prepare = getattr(metric, "_prepare_ensemble_score", None)
        if prepare is not None:
            assert prepare(y, None) is None


@dataclass
class _RepeatingEnsembler:
    name: str = "repeating"
    observed: tuple[float, float] | None = None
    mutate: bool = False

    def combine(
        self,
        oof: np.ndarray,
        y: np.ndarray,
        *,
        score: Callable[[np.ndarray], float],
        member_ids: Sequence[str],
        random_state: int,
        sample_weight: np.ndarray | None = None,
    ) -> EnsembleRecipe:
        pred = oof[0].copy()
        first = score(pred)
        if self.mutate:
            pred[0] = 1 - pred[0]
        self.observed = (first, score(pred))
        ids = tuple(member_ids)
        return EnsembleRecipe({mid: 1 / len(ids) for mid in ids}, self.name, ids)


def test_application_keeps_custom_stateful_scorer_calls() -> None:
    metric = _StatefulLogLoss()
    y = np.array([0, 1, 0, 1])
    pred = np.array([0.2, 0.8, 0.3, 0.7])
    members = [Candidate(id=str(i), score=0.0, oof_proba=pred.copy()) for i in range(2)]
    ensembler = _RepeatingEnsembler()
    ensemble_selection(
        members,
        Task(kind="binary"),
        y=y,
        best_model_id="0",
        ensembler=ensembler,
        metric=metric,
        significance_test=BootstrapSignificanceTest(metric, n_boot=5),
        policy=SelectionPolicy(greater_is_better=False),
        significance_mode="off",
    )
    assert ensembler.observed is not None
    assert ensembler.observed[0] != ensembler.observed[1]
    assert metric.calls == 4


def test_application_handles_mutating_prediction_buffer() -> None:
    metric = LogLoss()
    y = np.array([0, 1, 0, 1])
    pred = np.array([0.2, 0.8, 0.3, 0.7])
    members = [Candidate(id=str(i), score=0.0, oof_proba=pred.copy()) for i in range(2)]
    ensembler = _RepeatingEnsembler(mutate=True)
    ensemble_selection(
        members,
        Task(kind="binary"),
        y=y,
        best_model_id="0",
        ensembler=ensembler,
        metric=metric,
        significance_test=BootstrapSignificanceTest(metric, n_boot=1000),
        policy=SelectionPolicy(greater_is_better=False),
        significance_mode="off",
    )
    changed = pred.copy()
    changed[0] = 1 - changed[0]
    assert ensembler.observed == (-metric.score(y, pred), -metric.score(y, changed))


def test_failed_score_is_not_cached() -> None:
    y = np.array([0, 1])
    scorer = _prepared(LogLoss(), y, np.zeros(2))
    with patch.object(metrics, "log_loss", wraps=metrics.log_loss) as public:
        for _ in range(2):
            with pytest.raises(ZeroDivisionError):
                scorer(np.array([0.2, 0.8]))
        assert public.call_count == 2


class _DelegatingMetric:
    def __init__(self) -> None:
        self.inner = LogLoss()
        self.calls = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    def score(
        self, y_true: np.ndarray, y_pred: np.ndarray, sample_weight: np.ndarray | None = None
    ) -> float:
        self.calls += 1
        return self.inner.score(y_true, y_pred, sample_weight) + self.calls * 1e-8


def test_delegating_proxy_does_not_bypass_its_score() -> None:
    metric = _DelegatingMetric()
    y = np.array([0, 1, 0, 1])
    pred = np.array([0.2, 0.8, 0.3, 0.7])
    members = [Candidate(id=str(i), score=0.0, oof_proba=pred.copy()) for i in range(2)]
    ensembler = _RepeatingEnsembler()
    ensemble_selection(
        members,
        Task(kind="binary"),
        y=y,
        best_model_id="0",
        ensembler=ensembler,
        metric=metric,
        significance_test=BootstrapSignificanceTest(LogLoss(), n_boot=1000),
        policy=SelectionPolicy(greater_is_better=False),
        significance_mode="off",
    )
    assert ensembler.observed is not None
    assert ensembler.observed[0] != ensembler.observed[1]
    assert metric.calls == 4


@pytest.mark.parametrize("metric_type", [LogLoss, RocAuc, PrAuc])
@pytest.mark.parametrize("override_scope", ["instance", "class"])
def test_overridden_binary_orientation_keeps_application_scorer_calls(
    monkeypatch: pytest.MonkeyPatch,
    metric_type: type[LogLoss] | type[RocAuc] | type[PrAuc],
    override_scope: str,
) -> None:
    metric = metric_type()
    y = np.array([0, 1, 0, 1])
    pred = np.array([0.2, 0.8, 0.3, 0.7])
    calls = 0

    def orient(target: np.ndarray) -> np.ndarray:
        nonlocal calls
        calls += 1
        return target if calls % 2 else 1 - target

    if override_scope == "instance":
        monkeypatch.setattr(metric, "_orient_binary", orient)
    else:

        def class_orient(self: object, target: np.ndarray) -> np.ndarray:
            return orient(target)

        monkeypatch.setattr(metric_type, "_orient_binary", class_orient)
    members = [Candidate(id=str(i), score=0.0, oof_proba=pred.copy()) for i in range(2)]
    ensembler = _RepeatingEnsembler()
    ensemble_selection(
        members,
        Task(kind="binary"),
        y=y,
        best_model_id="0",
        ensembler=ensembler,
        metric=metric,
        significance_test=BootstrapSignificanceTest(LogLoss(), n_boot=1000),
        policy=SelectionPolicy(greater_is_better=metric.greater_is_better),
        significance_mode="off",
    )
    assert ensembler.observed is not None
    assert ensembler.observed[0] != ensembler.observed[1]
    assert calls == 4
