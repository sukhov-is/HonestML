"""Out-of-fold subset scoring for the feature-compare arbiter (ADR-0046/0052) — pure application logic.

The OOF scorer cluster shared by the wrapper-selector inject point (:func:`make_oof_scorer`) and the
nested / per-fold arbiters (:func:`make_oof_vector_scorer`): one fold loop (:func:`_oof_fold_loop`) and one
projection (:func:`_score_and_band_vector`), so the score the arbiter ranks on and the vector the band tests
on can never diverge. The leakage-critical model fit/predict is the injected ``FitPredict`` adapter — this
module names no adapter (Humble Object, NFR-FSC-2).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from honestml.core import Fold, Metric, Task, resolve_positive

from .projection import _scorer_setup, align_proba, project_for_metric

# (proba_or_none, pred, classes_or_none) from a single cheap model fit on a column subset
FitPredict = Callable[
    [np.ndarray, np.ndarray, np.ndarray, "np.ndarray | None", int],
    tuple["np.ndarray | None", np.ndarray, "np.ndarray | None"],
]


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
    for fold in folds:
        test_idx = fold.test_idx
        train_idx = (
            fold.fit_idx if fold.es_idx.size == 0 else np.concatenate([fold.fit_idx, fold.es_idx])
        )
        sw_tr = sample_weight[train_idx] if sample_weight is not None else None
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
    tests on can never diverge. The float scorer (:func:`make_oof_scorer`) projects once without the band fill and
    is deliberately NOT routed here.
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
) -> Callable[[Sequence[int]], float]:
    """Build the injected ``score_subset`` for a wrapper selector (ADR-0046 §1).

    For a column subset, fits ``fit_predict`` on each fold's ``fit ⊕ es`` and scores the pooled OOF with
    ``metric`` (higher-is-better, sign-flipped for loss metrics so the selector always maximizes). The
    fold loop and projection live here (Humble Object); the adapter only fits one matrix and never sees
    test rows. ``global_classes`` is the full class order to align proba to (the compare driver passes
    the whole-DEV classes so a class missing from a sub-split is still aligned); defaults to ``y``'s.
    """
    classes, _, need_proba, sign = _scorer_setup(task, metric, y, global_classes=global_classes)

    def score_subset(indices: Sequence[int]) -> float:
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
        )
        proba_arg = oof_proba[mask] if oof_proba is not None else None
        proj = project_for_metric(metric, proba=proba_arg, pred=oof_pred[mask], kind=task.kind)
        sw_valid = sample_weight[mask] if sample_weight is not None else None
        return sign * metric.score(y[mask], proj, sw_valid)

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
) -> Callable[[Sequence[int]], tuple[float, np.ndarray, np.ndarray]]:
    """Like :func:`make_oof_scorer` but also returns the metric-ready OOF vector + mask (ADR-0052 §2).

    The nested arbiter needs the pooled-OOF score AND the per-row metric-ready prediction (for the
    significance band, ADR-0053): ``(score, oof_metric_ready, mask)``. Same fold loop/projection as the
    float scorer (shared ``_oof_fold_loop``), so the score the arbiter ranks on and the vector the band
    tests on can never diverge.
    """
    classes, _, need_proba, sign = _scorer_setup(task, metric, y, global_classes=global_classes)

    def score_vector(indices: Sequence[int]) -> tuple[float, np.ndarray, np.ndarray]:
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
        )
        return _score_and_band_vector(
            y,
            oof_pred,
            oof_proba,
            mask,
            metric=metric,
            task=task,
            sample_weight=sample_weight,
            sign=sign,
        )

    return score_vector
