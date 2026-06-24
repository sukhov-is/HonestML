"""M1b: absolute, reproducible select_best (ADR-0007) — the C1 fix."""

from __future__ import annotations

import numpy as np
import pytest

from honestml.core import (
    BandResult,
    Candidate,
    NoSignificanceTest,
    SelectionPolicy,
    equivalence_band,
    rank,
    select_best,
)

pytestmark = pytest.mark.unit


class _AllEquivalent:
    seed = 1
    n_boot = 10

    def equivalent(self, pred_a, pred_b, y_true, *, alpha, block_index=None, sample_weight=None):
        return True


def test_argmax_greater_is_better() -> None:
    cands = [Candidate("a", 0.7), Candidate("b", 0.9), Candidate("c", 0.8)]
    assert select_best(cands, SelectionPolicy()).id == "b"


def test_argmin_when_lower_is_better() -> None:
    cands = [Candidate("a", 0.7), Candidate("b", 0.9), Candidate("c", 0.5)]
    policy = SelectionPolicy(greater_is_better=False)
    assert select_best(cands, policy).id == "c"


def test_tie_break_is_inert_in_m1b() -> None:
    """With NoSignificanceTest the band is empty -> pure argmax, ignores n_features."""
    y = np.array([0, 1, 0, 1])
    cands = [
        Candidate("big", 0.90, n_features=100, oof_pred=np.array([0.1, 0.9, 0.2, 0.8])),
        Candidate("small", 0.89, n_features=5, oof_pred=np.array([0.2, 0.7, 0.3, 0.6])),
    ]
    best = select_best(cands, SelectionPolicy(), NoSignificanceTest(), y)
    assert best.id == "big"  # higher score wins; compactness does not tip it (N-3)


def test_tie_break_activates_with_real_significance() -> None:
    """A test that declares equivalence enables the compactness tie-break (M4-style)."""
    y = np.array([0, 1, 0, 1])
    cands = [
        Candidate("big", 0.90, n_features=100, oof_pred=np.array([0.1, 0.9, 0.2, 0.8])),
        Candidate("small", 0.89, n_features=5, oof_pred=np.array([0.2, 0.7, 0.3, 0.6])),
    ]
    best = select_best(cands, SelectionPolicy(), _AllEquivalent(), y)
    assert best.id == "small"  # equivalent to anchor -> prefer fewer features


def test_select_best_is_pure() -> None:
    cands = [Candidate("a", 0.7), Candidate("b", 0.9)]
    policy = SelectionPolicy()
    assert select_best(cands, policy).id == select_best(cands, policy).id


def test_empty_candidates_rejected() -> None:
    with pytest.raises(ValueError, match="at least one"):
        select_best([], SelectionPolicy())


def test_rank_orders_by_absolute_score() -> None:
    cands = [Candidate("a", 0.7), Candidate("b", 0.9), Candidate("c", 0.8)]
    assert [c.id for c in rank(cands, SelectionPolicy())] == ["b", "c", "a"]


def test_select_best_aligns_significance_on_oof_mask() -> None:
    """Partial OOF (holdout/skip): NaN holes are stripped before the significance test."""
    y = np.array([0, 1, 0, 1, 0, 1])
    mask = np.array([False, False, True, True, True, True])  # first two rows unpredicted

    class _AssertClean:
        seed = 1
        n_boot = 5

        def equivalent(
            self, pred_a, pred_b, y_true, *, alpha, block_index=None, sample_weight=None
        ):
            assert not np.isnan(pred_a).any() and not np.isnan(pred_b).any()
            assert len(pred_a) == len(pred_b) == len(y_true)
            return True

    a = Candidate("a", 0.9, oof_pred=np.array([np.nan, np.nan, 0.2, 0.8, 0.3, 0.7]), oof_mask=mask)
    b = Candidate("b", 0.8, oof_pred=np.array([np.nan, np.nan, 0.3, 0.7, 0.4, 0.6]), oof_mask=mask)
    best = select_best([a, b], SelectionPolicy(), _AssertClean(), y)
    assert best.id in {"a", "b"}


# --- M4a1: equivalence_band (common fixed mask, BandResult) -----------------


def test_common_fixed_band_mask() -> None:
    """Every pairwise test runs on ONE common mask = intersection over all OOF (not pairwise)."""
    y = np.array([0, 1, 0, 1, 0, 1])
    seen = []

    class _Recorder:
        seed = 1
        n_boot = 10

        def equivalent(
            self, pred_a, pred_b, y_true, *, alpha, block_index=None, sample_weight=None
        ):
            seen.append((len(pred_a), len(pred_b), len(y_true)))
            assert not np.isnan(pred_a).any() and not np.isnan(pred_b).any()
            return True

    full = np.ones(6, dtype=bool)
    drop_first = np.array([False, True, True, True, True, True])
    drop_last = np.array([True, True, True, True, True, False])
    a = Candidate("a", 0.9, oof_pred=np.full(6, 0.6), oof_mask=full)
    b = Candidate("b", 0.8, oof_pred=np.where(drop_first, 0.6, np.nan), oof_mask=drop_first)
    c = Candidate("c", 0.7, oof_pred=np.where(drop_last, 0.6, np.nan), oof_mask=drop_last)

    res = equivalence_band([a, b, c], SelectionPolicy(), _Recorder(), y)
    # common = full & drop_first & drop_last = rows 1..4 -> 4 rows for EVERY call (not 5 pairwise)
    assert seen == [(4, 4, 4), (4, 4, 4)]
    assert isinstance(res, BandResult)
    assert set(res.member_ids) == {"a", "b", "c"} and res.width == 3


def test_band_tie_break() -> None:
    """Within the band the winner is the most compact (Occam); BandResult reports the tie-break."""
    y = np.array([0, 1, 0, 1])
    full = np.ones(4, dtype=bool)
    big = Candidate(
        "big", 0.90, n_features=100, oof_pred=np.array([0.1, 0.9, 0.2, 0.8]), oof_mask=full
    )
    small = Candidate(
        "small", 0.89, n_features=5, oof_pred=np.array([0.2, 0.7, 0.3, 0.6]), oof_mask=full
    )

    res = equivalence_band([big, small], SelectionPolicy(), _AllEquivalent(), y)
    assert res.winner == "small" and res.winner_by_tiebreak is True
    assert set(res.member_ids) == {"big", "small"} and res.width == 2


def test_band_collapses_to_anchor_without_test() -> None:
    """No test (or no OOF) -> band is the lone anchor, winner is the absolute argmax."""
    cands = [Candidate("a", 0.7), Candidate("b", 0.9)]
    res = equivalence_band(cands, SelectionPolicy())
    assert res.member_ids == ("b",) and res.winner == "b"
    assert res.width == 1 and res.unstable is False and res.winner_by_tiebreak is False


def test_band_invariant_to_worse_candidate() -> None:
    """FR-M4-1/NFR-M4-2: adding a clearly-worse candidate changes neither band nor winner.

    The bootstrap seeds from a fixed seed (not rank position), so each pairwise verdict is
    independent of the pool; the anchor is the absolute argmax (rank-invariant).
    """
    from honestml.adapters import BootstrapSignificanceTest, RocAuc

    rng = np.random.default_rng(0)
    n = 300
    y = rng.integers(0, 2, n)
    sig = np.where(y == 1, 0.75, 0.25)
    full = np.ones(n, dtype=bool)
    a = Candidate("a", 0.90, n_features=10, oof_pred=sig + rng.normal(0, 0.1, n), oof_mask=full)
    b = Candidate("b", 0.895, n_features=5, oof_pred=sig + rng.normal(0, 0.1, n), oof_mask=full)
    worse = Candidate("worse", 0.55, n_features=3, oof_pred=rng.uniform(size=n), oof_mask=full)
    test = BootstrapSignificanceTest(RocAuc(), seed=7, n_boot=1000)

    base = equivalence_band([a, b], SelectionPolicy(), test, y)
    extended = equivalence_band([a, b, worse], SelectionPolicy(), test, y)

    assert (set(base.member_ids) & {"a", "b"}) == (set(extended.member_ids) & {"a", "b"})
    assert base.winner == extended.winner
    assert "worse" not in extended.member_ids
