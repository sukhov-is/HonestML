"""Exact within-run subset reuse and bounded retained predictions."""

from __future__ import annotations

import numpy as np
import pytest

from honestml.adapters.feature_selectors import SequentialSelector
from honestml.application.oof_scorer import OOFSubsetCache, make_oof_scorer, make_oof_vector_scorer
from honestml.core import Fold, Task

pytestmark = pytest.mark.unit


class _Metric:
    name = "mean"
    greater_is_better = True
    needs = "value"
    optimum = float("inf")
    average = None
    proper_proba = False

    def score(
        self, y_true: np.ndarray, y_pred: np.ndarray, sample_weight: np.ndarray | None = None
    ) -> float:
        return float(np.average(y_pred, weights=sample_weight))


class _FitSpy:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(
        self,
        x_tr: np.ndarray,
        y_tr: np.ndarray,
        x_te: np.ndarray,
        sample_weight: np.ndarray | None,
        random_state: int,
    ) -> tuple[None, np.ndarray, None]:
        self.calls += 1
        return None, np.full(len(x_te), float(x_te.shape[1])), None


def _folds(n: int) -> list[Fold]:
    rows = np.arange(n)
    return [
        Fold(rows[::2], np.array([], dtype=int), rows[1::2]),
        Fold(rows[1::2], np.array([], dtype=int), rows[::2]),
    ]


def test_search_trajectory_reuses_exact_oof_without_changing_tie_break() -> None:
    x, y = np.ones((30, 6)), np.ones(30)
    folds, spy, metric, task = _folds(30), _FitSpy(), _Metric(), Task(kind="regression")
    cache = OOFSubsetCache(max_bytes=100_000)
    scalar = make_oof_scorer(
        x, y, folds, fit_predict=spy, metric=metric, task=task, random_state=3, subset_cache=cache
    )
    selector = SequentialSelector(full_descent=False, patience=10)
    trajectory = selector.select(
        x, y, folds, categorical=np.zeros(6, dtype=bool), score_subset=scalar, random_state=3
    )
    assert trajectory == tuple(tuple(range(n)) for n in range(6, 0, -1))
    count = spy.calls
    vector = make_oof_vector_scorer(
        x, y, folds, fit_predict=spy, metric=metric, task=task, random_state=3, subset_cache=cache
    )
    for subset in trajectory:
        score, oof, mask = vector(subset)
        assert score == len(subset)
        np.testing.assert_array_equal(oof, len(subset) * np.ones(30))
        assert mask.all()
    assert spy.calls == count
    assert cache.hits >= len(trajectory)
    assert cache.retained_bytes <= cache.max_bytes


@pytest.mark.parametrize(
    "change",
    [
        "data",
        "column_order",
        "target",
        "folds",
        "seed",
        "weights",
        "model",
        "metric",
        "task",
        "classes",
    ],
)
def test_changed_context_cannot_hit(change: str) -> None:
    x, y = np.arange(120.0).reshape(30, 4), np.ones(30)
    folds, spy, metric, task = _folds(30), _FitSpy(), _Metric(), Task(kind="regression")
    cache = OOFSubsetCache()
    kwargs = dict(fit_predict=spy, metric=metric, task=task, random_state=3, subset_cache=cache)
    make_oof_vector_scorer(x, y, folds, **kwargs)((0, 1))
    if change == "data":
        x[0, 0] += 1
    elif change == "column_order":
        x = x[:, ::-1]
    elif change == "target":
        y[0] += 1
    elif change == "folds":
        folds = list(reversed(folds))
    elif change == "seed":
        kwargs["random_state"] = 5
    elif change == "weights":
        kwargs["sample_weight"] = np.ones(30)
    elif change == "model":
        kwargs["fit_predict"] = _FitSpy()
    elif change == "metric":
        kwargs["metric"] = _Metric()
    elif change == "task":
        kwargs["task"] = Task(kind="regression", metric="mae")
    else:
        kwargs["global_classes"] = np.array([0, 1])
    make_oof_vector_scorer(x, y, folds, **kwargs)((0, 1))
    assert cache.hits == 0
    assert cache.misses == 2


def test_cache_eviction_preserves_result_and_memory_limit() -> None:
    x, y = np.ones((100, 8)), np.ones(100)
    spy, metric, task = _FitSpy(), _Metric(), Task(kind="regression")
    cache = OOFSubsetCache(max_bytes=4000)
    score = make_oof_vector_scorer(
        x,
        y,
        _folds(100),
        fit_predict=spy,
        metric=metric,
        task=task,
        random_state=0,
        subset_cache=cache,
    )
    for n in range(1, 9):
        assert score(tuple(range(n)))[0] == n
        assert cache.retained_bytes <= cache.max_bytes
    assert cache.peak_bytes <= cache.max_bytes
    calls = spy.calls
    assert score((0,))[0] == 1
    assert spy.calls > calls


def test_cached_vectors_are_read_only() -> None:
    x, y = np.ones((12, 2)), np.ones(12)
    score = make_oof_vector_scorer(
        x,
        y,
        _folds(12),
        fit_predict=_FitSpy(),
        metric=_Metric(),
        task=Task(kind="regression"),
        random_state=0,
    )
    _, oof, mask = score((0,))
    assert not oof.flags.writeable
    assert not mask.flags.writeable


def test_object_context_bypasses_reuse() -> None:
    x, y = np.ones((12, 2), dtype=object), np.ones(12)
    spy = _FitSpy()
    cache = OOFSubsetCache()
    score = make_oof_vector_scorer(
        x,
        y,
        _folds(12),
        fit_predict=spy,
        metric=_Metric(),
        task=Task(kind="regression"),
        random_state=0,
        subset_cache=cache,
    )
    score((0,))
    score((0,))
    assert spy.calls == 4
    assert cache.hits == 0
    assert cache.retained_bytes == 0


def test_sequential_cache_retains_only_one_probe_vector_per_width() -> None:
    x, y = np.ones((30, 12)), np.ones(30)
    cache = OOFSubsetCache()
    scalar = make_oof_scorer(
        x,
        y,
        _folds(30),
        fit_predict=_FitSpy(),
        metric=_Metric(),
        task=Task(kind="regression"),
        random_state=0,
        subset_cache=cache,
    )
    SequentialSelector(patience=20).select(
        x, y, _folds(30), categorical=np.zeros(12, dtype=bool), score_subset=scalar, random_state=0
    )
    assert sum(entry.vector is not None for entry in cache._entries.values()) == 12
    assert len(cache._entries) > 60


def test_scalar_repeat_of_losing_probe_reuses_score() -> None:
    x, y = np.ones((12, 3)), np.ones(12)
    spy = _FitSpy()
    scalar = make_oof_scorer(
        x,
        y,
        _folds(12),
        fit_predict=spy,
        metric=_Metric(),
        task=Task(kind="regression"),
        random_state=0,
    )
    assert scalar((0, 1)) == scalar((1, 2)) == scalar((1, 2)) == 2
    assert spy.calls == 4


def test_compare_trajectory_and_gate_share_context() -> None:
    from honestml.application.feature_compare import _select_one, no_selection_gate
    from honestml.core import FeatureSelectionConfig
    from honestml.core.ports.significance import NoSignificanceTest
    from honestml.core.selection_policy import SelectionPolicy

    x, y = np.ones((30, 4)), np.ones(30)
    spy, metric, task = _FitSpy(), _Metric(), Task(kind="regression")
    cache = OOFSubsetCache()
    folds = _folds(30)
    _select_one(
        SequentialSelector(patience=10),
        x,
        y,
        folds,
        categorical=np.zeros(4, dtype=bool),
        config=FeatureSelectionConfig(),
        seed=7,
        sample_weight=None,
        fit_predict=spy,
        metric=metric,
        task=task,
        global_classes=None,
        subset_cache=cache,
    )
    calls = spy.calls
    keep, reason = no_selection_gate(
        x,
        y,
        (0,),
        folds,
        fit_predict=spy,
        metric=metric,
        task=task,
        sample_weight=None,
        significance_test=NoSignificanceTest(),
        policy=SelectionPolicy(),
        random_state=7,
        subset_cache=cache,
    )
    assert not keep
    assert reason == "no_selection_better"
    assert spy.calls == calls
