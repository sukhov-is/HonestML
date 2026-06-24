"""Absolute, reproducible model selection.

Ranking is by the **absolute** primary metric, never a candidate-relative
normalization, so adding or removing a candidate cannot change the ranks of the
others (reproducible across runs). Within a statistical equivalence band around the
single point anchor, ties break lexicographically by compactness → stability →
speed. Under ``NoSignificanceTest`` the band is empty, so ``select_best`` is a
deterministic argmax and the tie-break branch is inert.

``select_best`` is a pure function given a deterministic ``SignificanceTest`` —
the testability boundary.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel, ConfigDict

from .ports.significance import SignificanceTest

_TIE_BREAK_KEYS = ("n_features", "stability", "train_time")


@dataclass(frozen=True)
class Candidate:
    """A leaderboard entry: its absolute score plus secondary, OOF predictions.

    ``oof_pred`` is the **metric-ready** out-of-fold vector the band aligns on:
    ``P(positive)``/``(n, K)`` proba for proba-metrics, else the predicted class/value. ``oof_mask``
    marks which rows actually have an OOF prediction (holdout yields a partial OOF; degenerate
    folds are skipped), so validity is tracked by the mask, never ``np.isnan`` —
    which would crash on int/str class vectors.
    """

    id: str
    score: float
    n_features: int = 0
    stability: float = 0.0
    train_time: float = 0.0
    oof_pred: np.ndarray | None = None
    oof_mask: np.ndarray | None = None
    # raw out-of-fold probabilities for the calibrator (ADR-0030 §1 / ADR-0031 §3): a SEPARATE
    # channel from the metric-ready oof_pred (which may be class/value); the band ignores it.
    oof_proba: np.ndarray | None = None


class SelectionPolicy(BaseModel):
    """Selection rule: absolute primary metric + inert lexicographic tie-break."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    greater_is_better: bool = True
    alpha: float = 0.05
    tie_break: tuple[str, ...] = _TIE_BREAK_KEYS


@dataclass(frozen=True)
class BandResult:
    """The equivalence band's outcome — the honesty observability container.

    ``member_ids`` are the candidates statistically indistinguishable from the absolute
    anchor (anchor included); ``winner`` is the tie-broken pick; ``unstable`` flags an
    anchor-sensitive band (the runner-up is itself in the band, so which model anchors is
    arbitrary); ``width`` is the band size; ``winner_by_tiebreak`` is True when the winner
    is not the anchor.
    """

    member_ids: tuple[str, ...]
    winner: str
    unstable: bool
    width: int
    winner_by_tiebreak: bool


def rank(candidates: Sequence[Candidate], policy: SelectionPolicy) -> list[Candidate]:
    """Order candidates by the absolute primary metric (stable, id tie-break)."""
    sign = -1.0 if policy.greater_is_better else 1.0
    return sorted(candidates, key=lambda c: (sign * c.score, c.id))


def equivalence_band(
    candidates: Sequence[Candidate],
    policy: SelectionPolicy,
    test: SignificanceTest | None = None,
    y_true: np.ndarray | None = None,
    *,
    block_index: np.ndarray | None = None,
    sample_weight: np.ndarray | None = None,
    min_rows: int = 2,
) -> BandResult:
    """Build the equivalence band around the absolute anchor (pure, numpy-only).

    Owns the **single fixed common mask** and all alignment: the mask is the
    intersection of valid-OOF rows over every candidate that carries OOF, computed once BEFORE
    the test (non-circular). The anchor and each candidate are scored on that same row set, so a
    non-additive metric (roc_auc) compares Δ on identical samples; ``block_index``/``sample_weight``
    are sliced to the mask and handed to the test already aligned.
    """
    if not candidates:
        raise ValueError("equivalence_band requires at least one candidate")
    ordered = rank(candidates, policy)
    anchor = ordered[0]

    if test is None or y_true is None or anchor.oof_pred is None:
        return BandResult((anchor.id,), anchor.id, False, 1, False)

    mask = _common_mask(ordered, y_true.shape[0])
    band = _band_members(ordered, mask, policy, test, y_true, block_index, sample_weight, min_rows)
    winner, by_tiebreak = _resolve_winner(band, anchor, policy)
    # anchor-sensitive iff the runner-up is itself in the band (full tie ⇒ runner-up in band)
    unstable = len(ordered) > 1 and ordered[1].id in {c.id for c in band}
    return BandResult(
        member_ids=tuple(c.id for c in band),
        winner=winner.id,
        unstable=unstable,
        width=len(band),
        winner_by_tiebreak=by_tiebreak,
    )


def select_best(
    candidates: Sequence[Candidate],
    policy: SelectionPolicy,
    significance_test: SignificanceTest | None = None,
    y_true: np.ndarray | None = None,
    *,
    block_index: np.ndarray | None = None,
    sample_weight: np.ndarray | None = None,
) -> Candidate:
    """Return the winning candidate: absolute argmax, then equivalence tie-break."""
    result = equivalence_band(
        candidates,
        policy,
        significance_test,
        y_true,
        block_index=block_index,
        sample_weight=sample_weight,
    )
    return next(c for c in candidates if c.id == result.winner)


def _common_mask(candidates: Sequence[Candidate], n: int) -> np.ndarray:
    """Intersection of valid-OOF rows over the fixed set of candidates with OOF."""
    mask = np.ones(n, dtype=bool)
    has_oof = False
    for c in candidates:
        if c.oof_pred is None:
            continue
        mask &= _valid_mask(c, n)
        has_oof = True
    return mask if has_oof else np.zeros(n, dtype=bool)


def _band_members(
    ordered: Sequence[Candidate],
    mask: np.ndarray,
    policy: SelectionPolicy,
    test: SignificanceTest,
    y_true: np.ndarray,
    block_index: np.ndarray | None,
    sample_weight: np.ndarray | None,
    min_rows: int,
) -> list[Candidate]:
    """Candidates equivalent to ``ordered[0]`` on the common mask (anchor always included)."""
    anchor = ordered[0]
    band = [anchor]
    if anchor.oof_pred is None:  # guarded by the caller; narrows the optional for the slice below
        return band
    n_eff = int(mask.sum())
    anchor_pred = anchor.oof_pred[mask]
    yt = y_true[mask]
    bi = block_index[mask] if block_index is not None else None
    sw = sample_weight[mask] if sample_weight is not None else None
    for c in ordered[1:]:
        if c.oof_pred is None:
            continue
        if n_eff < min_rows:  # degenerate common mask -> conservatively include (ADR-0026 §7)
            band.append(c)
            continue
        if test.equivalent(
            c.oof_pred[mask], anchor_pred, yt, alpha=policy.alpha, block_index=bi, sample_weight=sw
        ):
            band.append(c)
    return band


def _resolve_winner(
    band: Sequence[Candidate], anchor: Candidate, policy: SelectionPolicy
) -> tuple[Candidate, bool]:
    """Lexicographic Occam tie-break within the band (compactness → stability → speed)."""
    if len(band) == 1:
        return anchor, False
    winner = min(band, key=lambda c: tuple(getattr(c, key) for key in policy.tie_break))
    return winner, winner.id != anchor.id


def _valid_mask(candidate: Candidate, n: int) -> np.ndarray:
    """Rows with a usable OOF prediction (explicit mask, else non-NaN row-wise, else all)."""
    if candidate.oof_mask is not None:
        return candidate.oof_mask
    pred = candidate.oof_pred
    if pred is not None:
        nan = np.isnan(pred)
        valid: np.ndarray = ~(nan.any(axis=1) if nan.ndim == 2 else nan)
        return valid
    return np.ones(n, dtype=bool)
