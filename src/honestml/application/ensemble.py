"""The ``ensemble_selection`` use-case + honest ``choose_better`` gate (ADR-0063 §5).

After the honest leaderboard (``run_slice``), blend the candidates' out-of-fold predictions and ship the
ensemble **only if** it is *significantly* better than the best single — reusing the same M4
``SignificanceTest``/``equivalence_band`` machinery, so the ensemble is honest by construction (no
"ensemble for the ensemble's sake", R-ENSOVERFIT). The spine is a Humble Object: it builds the per-model
blend space (``oof``) and an injected higher-is-better ``score`` from the ``Metric`` port, hands them to
an injected :class:`Ensembler`, then gates the resulting recipe. ``refit_members`` refits the surviving
members on full-DEV for shipping, dropping (and renormalizing around) any member whose refit raises
(ADR-0064 §4). Pure numpy + injected callables — names no adapter.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from honestml.core import (
    Candidate,
    Dataset,
    Ensembler,
    Estimator,
    Metric,
    RunContext,
    SelectionPolicy,
    SignificanceTest,
    Task,
    get_logger,
)
from honestml.core.config import SignificanceMode

from .projection import _PROBA_NEEDS, _scorer_setup
from .slice import EstimatorFactory, refit_best

_W_EPS = 1e-6


@dataclass(frozen=True)
class EnsembleOutcome:
    """The ensemble decision (ADR-0063 §5): whether it was applied, the recipe, and why.

    ``gate_reason`` is one of ``significant_improvement``/``equivalent_to_best``/``worse_than_best``/
    ``no_proba_channel``/``single_candidate``/``degenerate_recipe`` (or, set by the facade after a
    member refit dropped the ensemble below 2 members, ``insufficient_members_after_refit``; or, when
    the optional post-selection ensemble stage raised, ``failed: <err>`` — the honest single winner
    ships instead of the fit dying).
    ``oof_delta`` is ``blended - best_single`` in the metric's own orientation, or ``None`` when the
    gate short-circuited before scoring.
    """

    applied: bool
    method: str
    member_ids: tuple[str, ...]
    weights: dict[str, float]
    gate_reason: str
    oof_delta: float | None


def _valid(channel: np.ndarray) -> np.ndarray:
    """Rows with a usable (non-NaN) OOF value — row-wise for the multiclass (n, K) channel."""
    nan = np.isnan(channel)
    return ~(nan.any(axis=1) if nan.ndim == 2 else nan)


def _blend(oof: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted combination along axis 0 ((m,n)->(n,), (m,n,K)->(n,K)); weights are a simplex."""
    return np.tensordot(np.asarray(weights, dtype=np.float64), oof, axes=([0], [0]))


def _metric_projection(
    blended: np.ndarray,
    metric: Metric,
    *,
    kind: str,
    is_clf: bool,
    classes: np.ndarray | None,
    positive: object,
) -> np.ndarray:
    """Project a blended channel to the metric's input type (ADR-0021 §3, ensemble path).

    Proba metrics consume the blend directly. A hard-label metric (``needs='class'``) on
    classification has no estimator to call ``predict`` on, so the blended probabilities are
    projected to class labels first — ``P(pos) >= 0.5`` (binary) / ``argmax`` over the global
    class order (multiclass); otherwise the continuous blend reaches a label metric and raises
    "mix of … and continuous targets". A regression value metric passes the value blend through.
    """
    if metric.needs in _PROBA_NEEDS:
        return blended
    if is_clf:
        assert classes is not None
        if kind == "multiclass":
            return classes[np.argmax(blended, axis=1)]
        negative = classes[classes != positive][0]
        labels = np.empty(blended.shape[0], dtype=classes.dtype)
        pos_mask = blended >= 0.5
        labels[pos_mask] = positive
        labels[~pos_mask] = negative
        return labels
    return blended


def ensemble_selection(
    candidates: Sequence[Candidate],
    task: Task,
    *,
    y: np.ndarray,
    best_model_id: str,
    ensembler: Ensembler,
    metric: Metric,
    significance_test: SignificanceTest,
    policy: SelectionPolicy,
    significance_mode: SignificanceMode,
    block_index: np.ndarray | None = None,
    sample_weight: np.ndarray | None = None,
    random_state: int = 0,
) -> EnsembleOutcome:
    """Blend the candidates and decide whether the blend ships (honest ``choose_better``; ADR-0063 §5)."""
    is_clf = task.is_classification
    kind = task.kind
    # blend channel per candidate: proba for classification (binary 1-D P(pos) / multiclass (n,K)),
    # the value OOF for regression. Linear combination of class LABELS is incorrect -> require proba/value.
    channels = {c.id: (c.oof_proba if is_clf else c.oof_pred) for c in candidates}
    usable = [c for c in candidates if channels[c.id] is not None]
    if not usable:
        return EnsembleOutcome(False, ensembler.name, (), {}, "no_proba_channel", None)
    if len(usable) < 2:
        ids = tuple(c.id for c in usable)
        return EnsembleOutcome(False, ensembler.name, ids, {ids[0]: 1.0}, "single_candidate", None)

    member_ids = tuple(c.id for c in usable)
    n = np.asarray(channels[usable[0].id]).shape[0]
    mask = np.ones(n, dtype=bool)
    for c in usable:
        mask &= _valid(np.asarray(channels[c.id]))
    if int(mask.sum()) < 2:
        # too few rows are jointly covered across members -> no usable blend channel (same gate_reason
        # as a missing channel: there is no scorable common OOF to blend on, ADR-0063 §2)
        return EnsembleOutcome(False, ensembler.name, member_ids, {}, "no_proba_channel", None)

    oof = np.stack([np.asarray(channels[c.id])[mask] for c in usable], axis=0)
    yt = y[mask]
    sw = sample_weight[mask] if sample_weight is not None else None
    bi = block_index[mask] if block_index is not None else None
    classes, positive, _, sign = _scorer_setup(task, metric, y)

    def project(channel: np.ndarray) -> np.ndarray:
        return _metric_projection(
            channel, metric, kind=kind, is_clf=is_clf, classes=classes, positive=positive
        )

    def score(blended: np.ndarray) -> float:
        return sign * float(metric.score(yt, project(blended), sw))

    recipe = ensembler.combine(
        oof, yt, score=score, member_ids=member_ids, random_state=random_state, sample_weight=sw
    )

    # a degenerate recipe (all mass on one member) is not an ensemble -> ship the single (ADR-0063 §5)
    active = [mid for mid in member_ids if recipe.weights[mid] > _W_EPS]
    if len(active) < 2:
        return EnsembleOutcome(
            False, recipe.method, member_ids, recipe.weights, "degenerate_recipe", None
        )

    weights = np.array([recipe.weights[mid] for mid in member_ids], dtype=np.float64)
    blended = _blend(oof, weights)
    blended_proj = project(blended)
    best_channel = _best_single_channel(
        usable, channels, best_model_id, mask, project, metric, yt, sw, sign
    )
    best_proj = project(best_channel)
    raw_blended = float(metric.score(yt, blended_proj, sw))
    raw_best = float(metric.score(yt, best_proj, sw))
    better = raw_blended > raw_best if metric.greater_is_better else raw_blended < raw_best
    oof_delta = raw_blended - raw_best

    if significance_mode == "off":
        # legacy strict-`>` gate (the _try_blend semantics): ship iff the blend beats the best single
        applied, reason = bool(better), "significant_improvement" if better else "worse_than_best"
    else:
        # honest gate: ship iff the blend is NOT statistically equivalent to the best single AND better
        equivalent = significance_test.equivalent(
            best_proj, blended_proj, yt, alpha=policy.alpha, block_index=bi, sample_weight=sw
        )
        if equivalent:
            applied, reason = False, "equivalent_to_best"
        elif better:
            applied, reason = True, "significant_improvement"
        else:
            applied, reason = False, "worse_than_best"
    return EnsembleOutcome(applied, recipe.method, member_ids, recipe.weights, reason, oof_delta)


def _best_single_channel(
    usable: Sequence[Candidate],
    channels: Mapping[str, np.ndarray | None],
    best_model_id: str,
    mask: np.ndarray,
    project: Callable[[np.ndarray], np.ndarray],
    metric: Metric,
    yt: np.ndarray,
    sw: np.ndarray | None,
    sign: float,
) -> np.ndarray:
    """The best single member's masked channel: the run winner if it has one, else the best by metric."""
    by_id = {c.id: c for c in usable}
    best = by_id.get(best_model_id)
    if best is None:
        best = max(
            usable,
            key=lambda c: (
                sign * float(metric.score(yt, project(np.asarray(channels[c.id])[mask]), sw))
            ),
        )
    return np.asarray(channels[best.id])[mask]


def refit_members(
    dataset: Dataset,
    task: Task,
    *,
    member_ids: Sequence[str],
    factories: Mapping[str, EstimatorFactory],
    ctx: RunContext | None = None,
) -> tuple[list[Estimator], tuple[str, ...], tuple[str, ...]]:
    """Refit each member on full-DEV; drop a member whose refit raises (ADR-0064 §4 drop-and-renormalize).

    Returns ``(fitted_members, kept_ids, dropped_ids)``. Unlike the per-fold isolation in ``run_slice``,
    member refit on full-DEV has no fold-level fallback, so a raising member is excluded and a WARNING is
    logged; the caller renormalizes the weights over the survivors (and ships a single model if < 2 remain).
    """
    logger = ctx.logger if ctx is not None else get_logger()
    members: list[Estimator] = []
    kept: list[str] = []
    dropped: list[str] = []
    for mid in member_ids:
        # a missing factory is a composition wiring bug -> fail fast, not silently a "dropped member"
        factory = factories[mid]
        try:
            est = refit_best(dataset, task, factory=factory, ctx=ctx)
        except Exception as exc:  # external model refit failed on full-DEV -> drop-and-renormalize
            logger.warning("ensemble member %r refit failed on full-DEV (%s); dropping", mid, exc)
            dropped.append(mid)
            continue
        members.append(est)
        kept.append(mid)
    return members, tuple(kept), tuple(dropped)
