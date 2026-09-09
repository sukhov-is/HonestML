"""Out-of-fold subset scoring for the feature-compare arbiter (ADR-0046/0052) — pure application logic.

The OOF scorer cluster shared by the wrapper-selector inject point (:func:`make_oof_scorer`) and the
nested / per-fold arbiters (:func:`make_oof_vector_scorer`): one fold loop (:func:`_oof_fold_loop`) and one
projection (:func:`_score_and_band_vector`), so the score the arbiter ranks on and the vector the band tests
on can never diverge. The leakage-critical model fit/predict is the injected ``FitPredict`` adapter — this
module names no adapter (Humble Object, NFR-FSC-2).
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from collections.abc import Callable, Sequence
from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np

from honestml.core import Fold, Metric, RunContext, Task, resolve_positive

from .projection import _scorer_setup, align_proba, project_for_metric

# (proba_or_none, pred, classes_or_none) from a single cheap model fit on a column subset
FitPredict = Callable[
    [np.ndarray, np.ndarray, np.ndarray, "np.ndarray | None", int],
    tuple["np.ndarray | None", np.ndarray, "np.ndarray | None"],
]


@dataclass(frozen=True, eq=False)
class _Identity:
    value: object

    def __hash__(self) -> int:
        return id(self.value)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _Identity) and self.value is other.value


_Context = tuple[object, ...]
_Key = tuple[_Context, str, tuple[int, ...] | int]
_Vector = tuple[float, np.ndarray, np.ndarray]


@dataclass(frozen=True)
class _SubsetEntry:
    indices: tuple[int, ...]
    score: float
    vector: _Vector | None
    size: int


class OOFSubsetCache:
    """Bounded run-local memoization with immutable model and metric adapters.

    Contexts fingerprint numeric/string arrays, ordered folds and class order. Adapter identity
    isolates model/resources/transforms and metric implementations; callers keep these adapters
    fixed for the cache lifetime. Object arrays have unknown content identity and bypass reuse.
    Scalar search retains one winning OOF per width, using the sequential selector's tie order;
    other probes retain only scalar scores. Eviction changes work, never selection semantics.
    ``retained_bytes`` conservatively accounts for array storage, index tuples and entry metadata.
    """

    def __init__(self, max_bytes: int = 32 * 1024 * 1024) -> None:
        self.max_bytes = max(0, max_bytes)
        self.hits = 0
        self.misses = 0
        self.retained_bytes = 0
        self.peak_bytes = 0
        self._entries: OrderedDict[_Key, _SubsetEntry] = OrderedDict()

    def _get(
        self, context: _Context | None, indices: tuple[int, ...], *, vector: bool
    ) -> _SubsetEntry | None:
        if context is not None:
            keys: tuple[_Key, ...] = (
                (context, "subset", indices),
                (context, "width", len(indices)),
            )
            for key in keys:
                entry = self._entries.get(key)
                if (
                    entry is not None
                    and entry.indices == indices
                    and (not vector or entry.vector is not None)
                ):
                    self._entries.move_to_end(key)
                    self.hits += 1
                    return entry
        self.misses += 1
        return None

    def _store(self, key: _Key, indices: tuple[int, ...], result: _Vector, *, vector: bool) -> None:
        # the allowance covers dictionary nodes, context references and ndarray/tuple headers.
        size = 2048 + 36 * len(indices)
        if vector:
            size += result[1].nbytes + result[2].nbytes
        if size > self.max_bytes:
            return
        old = self._entries.pop(key, None)
        if old is not None:
            self.retained_bytes -= old.size
        while self._entries and self.retained_bytes + size > self.max_bytes:
            _, evicted = self._entries.popitem(last=False)
            self.retained_bytes -= evicted.size
        self._entries[key] = _SubsetEntry(indices, result[0], result if vector else None, size)
        self.retained_bytes += size
        self.peak_bytes = max(self.peak_bytes, self.retained_bytes)

    def _put(
        self, context: _Context | None, indices: tuple[int, ...], result: _Vector, *, scalar: bool
    ) -> None:
        if context is None:
            return
        if not scalar:
            self._store((context, "subset", indices), indices, result, vector=True)
            return
        self._store((context, "subset", indices), indices, result, vector=False)
        width_key: _Key = (context, "width", len(indices))
        best = self._entries.get(width_key)
        if (
            best is None
            or result[0] > best.score
            or (result[0] == best.score and indices < best.indices)
        ):
            self._store(width_key, indices, result, vector=True)


def _context(
    x: np.ndarray,
    y: np.ndarray,
    folds: Sequence[Fold],
    *,
    fit_predict: FitPredict,
    metric: Metric,
    task: Task,
    random_state: int,
    sample_weight: np.ndarray | None,
    global_classes: np.ndarray | None,
) -> _Context | None:
    arrays = [x, y, sample_weight, global_classes]
    arrays.extend(idx for fold in folds for idx in (fold.fit_idx, fold.es_idx, fold.test_idx))
    digest = hashlib.blake2b(digest_size=32)
    for arr in arrays:
        if arr is None:
            digest.update(b"none;")
            continue
        if arr.dtype.hasobject:
            return None
        digest.update(str((arr.dtype.str, arr.shape)).encode())
        contiguous = np.ascontiguousarray(arr)
        digest.update(memoryview(contiguous).cast("B"))
    return (
        digest.digest(),
        _Identity(fit_predict),
        _Identity(metric),
        _Identity(task),
        random_state,
    )


def _fold_proba(
    proba: np.ndarray,
    cls: np.ndarray,
    *,
    multiclass: bool,
    global_classes: np.ndarray,
    positive: object,
) -> np.ndarray:
    """One model's proba reindexed to ``global_classes`` (multiclass) or its P(positive) column (binary)."""
    if multiclass:
        return align_proba(proba, cls, global_classes)
    return proba[:, int(np.where(cls == positive)[0][0])]


def _positive_of(task: Task, global_classes: np.ndarray | None) -> object | None:
    return (
        resolve_positive(task, global_classes)
        if task.kind == "binary" and global_classes is not None
        else None
    )


def _oof_fold_loop(
    x: np.ndarray,
    y: np.ndarray,
    folds: Sequence[Fold],
    idx: list[int],
    *,
    fit_predict: FitPredict,
    task: Task,
    random_state: int,
    sample_weight: np.ndarray | None,
    classes: np.ndarray | None,
    need_proba: bool,
    ctx: RunContext | None = None,
    recipe: str | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Pooled OOF over folds for one column subset (shared by both scorers, ADR-0052 §2).

    Fits ``fit_predict`` on each fold's ``fit ⊕ es`` and predicts its ``test`` rows; returns
    ``(oof_pred, oof_proba_or_None, mask)`` aligned to ``y``'s rows. The single fold loop is here so the
    float scorer (sequential) and the vector scorer (nested arbitration/band) can never project differently.
    """
    multiclass = task.kind == "multiclass"
    positive = _positive_of(task, classes)
    n = y.shape[0]
    oof_proba = (
        np.full((n, classes.size), np.nan)
        if multiclass and classes is not None
        else np.full(n, np.nan)
    )
    oof_pred = np.empty(n, dtype=y.dtype)
    mask = np.zeros(n, dtype=bool)
    produced_proba = False
    for fold_id, fold in enumerate(folds):
        test_idx = fold.test_idx
        train_idx = (
            fold.fit_idx if fold.es_idx.size == 0 else np.concatenate([fold.fit_idx, fold.es_idx])
        )
        sw_tr = sample_weight[train_idx] if sample_weight is not None else None
        with ctx.fit_attribution(recipe=recipe, fold=fold_id) if ctx is not None else nullcontext():
            proba, pred, cls = fit_predict(
                x[train_idx][:, idx], y[train_idx], x[test_idx][:, idx], sw_tr, random_state
            )
        oof_pred[test_idx] = pred
        if need_proba and proba is not None and cls is not None and classes is not None:
            oof_proba[test_idx] = _fold_proba(
                proba, cls, multiclass=multiclass, global_classes=classes, positive=positive
            )
            produced_proba = True
        mask[test_idx] = True
    return oof_pred, (oof_proba if produced_proba else None), mask


def _score_and_band_vector(
    y: np.ndarray,
    oof_pred: np.ndarray,
    oof_proba: np.ndarray | None,
    mask: np.ndarray,
    *,
    metric: Metric,
    task: Task,
    sample_weight: np.ndarray | None,
    sign: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Pooled-OOF score + metric-ready band vector from raw pooled OOF (shared: ADR-0052 §2 / ADR-0054 §2).

    Projects the pooled raw OOF once via ``project_for_metric`` and returns ``(score, metric_ready_oof, mask)``:
    proba metrics -> a float vector/``(n,K)`` NaN-filled where uncovered; non-proba -> the class/value OOF
    (``mask`` marks validity). Consumed by both :func:`make_oof_vector_scorer` (fixed subset) and the per-fold
    ``_score_procedure`` (re-selected subset, ADR-0054), so the score the arbiter ranks on and the vector the band
    tests on can never diverge. The float scorer uses the same projection and memoized result.
    """
    proba_arg = oof_proba[mask] if oof_proba is not None else None
    proj = project_for_metric(metric, proba=proba_arg, pred=oof_pred[mask], kind=task.kind)
    sw_valid = sample_weight[mask] if sample_weight is not None else None
    score = sign * metric.score(y[mask], proj, sw_valid)
    if proba_arg is not None:
        proj_arr = np.asarray(proj)
        shape = (y.shape[0], proj_arr.shape[1]) if proj_arr.ndim == 2 else (y.shape[0],)
        oof_ready = np.full(shape, np.nan)
        oof_ready[mask] = proj_arr
        return score, oof_ready, mask
    return score, oof_pred, mask


def make_oof_scorer(
    x: np.ndarray,
    y: np.ndarray,
    folds: Sequence[Fold],
    *,
    fit_predict: FitPredict,
    metric: Metric,
    task: Task,
    random_state: int,
    sample_weight: np.ndarray | None = None,
    global_classes: np.ndarray | None = None,
    subset_cache: OOFSubsetCache | None = None,
    ctx: RunContext | None = None,
    recipe: str | None = None,
) -> Callable[[Sequence[int]], float]:
    """Build the injected ``score_subset`` for a wrapper selector (ADR-0046 §1).

    For a column subset, fits ``fit_predict`` on each fold's ``fit ⊕ es`` and scores the pooled OOF with
    ``metric`` (higher-is-better, sign-flipped for loss metrics so the selector always maximizes). The
    fold loop and projection live here (Humble Object); the adapter only fits one matrix and never sees
    test rows. ``global_classes`` is the full class order to align proba to (the compare driver passes
    the whole-DEV classes so a class missing from a sub-split is still aligned); defaults to ``y``'s.
    """
    vector = make_oof_vector_scorer(
        x,
        y,
        folds,
        fit_predict=fit_predict,
        metric=metric,
        task=task,
        random_state=random_state,
        sample_weight=sample_weight,
        global_classes=global_classes,
        subset_cache=subset_cache,
        _scalar=True,
        ctx=ctx,
        recipe=recipe,
    )

    def score_subset(indices: Sequence[int]) -> float:
        return vector(indices)[0]

    return score_subset


def make_oof_vector_scorer(
    x: np.ndarray,
    y: np.ndarray,
    folds: Sequence[Fold],
    *,
    fit_predict: FitPredict,
    metric: Metric,
    task: Task,
    random_state: int,
    sample_weight: np.ndarray | None = None,
    global_classes: np.ndarray | None = None,
    subset_cache: OOFSubsetCache | None = None,
    _scalar: bool = False,
    ctx: RunContext | None = None,
    recipe: str | None = None,
) -> Callable[[Sequence[int]], tuple[float, np.ndarray, np.ndarray]]:
    """Like :func:`make_oof_scorer` but also returns the metric-ready OOF vector + mask (ADR-0052 §2).

    The nested arbiter needs the pooled-OOF score AND the per-row metric-ready prediction (for the
    significance band, ADR-0053): ``(score, oof_metric_ready, mask)``. Same fold loop/projection as the
    float scorer (shared ``_oof_fold_loop``), so the score the arbiter ranks on and the vector the band
    tests on can never diverge.
    """
    classes, _, need_proba, sign = _scorer_setup(task, metric, y, global_classes=global_classes)

    cache = subset_cache if subset_cache is not None else OOFSubsetCache()
    context = _context(
        x,
        y,
        folds,
        fit_predict=fit_predict,
        metric=metric,
        task=task,
        random_state=random_state,
        sample_weight=sample_weight,
        global_classes=global_classes,
    )

    def score_vector(indices: Sequence[int]) -> tuple[float, np.ndarray, np.ndarray]:
        idx = tuple(int(i) for i in indices)
        cached = cache._get(context, idx, vector=not _scalar)
        if cached is not None:
            if cached.vector is not None:
                return cached.vector
            return cached.score, np.empty(0), np.empty(0, dtype=bool)
        oof_pred, oof_proba, mask = _oof_fold_loop(
            x,
            y,
            folds,
            list(indices),
            fit_predict=fit_predict,
            task=task,
            random_state=random_state,
            sample_weight=sample_weight,
            classes=classes,
            need_proba=need_proba,
            ctx=ctx,
            recipe=recipe,
        )
        result = _score_and_band_vector(
            y,
            oof_pred,
            oof_proba,
            mask,
            metric=metric,
            task=task,
            sample_weight=sample_weight,
            sign=sign,
        )

        result[1].setflags(write=False)
        result[2].setflags(write=False)
        cache._put(context, idx, result, scalar=_scalar)
        return result

    return score_vector
