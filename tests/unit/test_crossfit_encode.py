"""M6a-2: out-of-fold target encoding ``crossfit_encode`` (ADR-0041) — anti-leakage core."""

from __future__ import annotations

import numpy as np
import pytest

from honestml.application import crossfit_encode, crossfit_encode_expanding

pytestmark = pytest.mark.unit


def test_golden_oof_values() -> None:
    # 1 column, 3 folds of 2 rows; each fold encoded by a map fitted on the OTHER folds, k=1.
    codes = np.array([[0], [1], [0], [1], [0], [1]], dtype=np.int64)
    y = np.array([1.0, 1.0, 1.0, 0.0, 1.0, 0.0])
    fold = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
    out = crossfit_encode(codes, y, fold, smoothing=1.0)
    expected = np.array([0.83333333, 0.16666667, 0.91666667, 0.58333333, 0.91666667, 0.58333333])
    assert np.allclose(out[:, 0], expected)


def test_unseen_code_falls_back_to_fold_global_mean() -> None:
    # fold 0's test rows (code 2) never appear in the fit side -> the fold's global mean
    codes = np.array([[2], [0], [0], [1]], dtype=np.int64)
    y = np.array([1.0, 1.0, 0.0, 1.0])
    fold = np.array([0, 1, 1, 1], dtype=np.int64)
    out = crossfit_encode(codes, y, fold, smoothing=5.0)
    fit_mean = y[fold != 0].mean()  # rows 1..3
    assert out[0, 0] == pytest.approx(fit_mean)


def test_target_permutation_within_fold_leaves_that_fold_unchanged() -> None:
    # NFR-FE-1: a row's OOF-TE is fitted on OTHER folds, so permuting y INSIDE fold f cannot change
    # the OOF-TE of fold f's own rows (its map and global mean both come from fold != f).
    rng = np.random.default_rng(0)
    codes = rng.integers(0, 4, size=(60, 2)).astype(np.int64)
    y = rng.integers(0, 2, size=60).astype(np.float64)
    fold = np.repeat(np.arange(5), 12).astype(np.int64)
    base = crossfit_encode(codes, y, fold, smoothing=3.0)
    f = 2
    y_perm = y.copy()
    idx = np.where(fold == f)[0]
    y_perm[idx] = y[rng.permutation(idx)]
    perm = crossfit_encode(codes, y_perm, fold, smoothing=3.0)
    assert np.array_equal(base[fold == f], perm[fold == f])


def test_reserve_codes_collapse_to_global_mean() -> None:
    # ADR-0041 §2: null/unknown codes (>= reserve_from) get the fold global mean, matching the
    # full-train spec — so OOF==full-train==inference for null rows (no null-bucket target bleed).
    codes = np.array([[0], [1], [2], [0], [1], [2]], dtype=np.int64)  # code 2 = reserve (null)
    y = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    fold = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    out = crossfit_encode(codes, y, fold, smoothing=1.0, reserve_from=np.array([2]))
    assert out[2, 0] == 0.0  # fold0 reserve -> fit(fold1) global mean = 0.0
    assert out[5, 0] == 1.0  # fold1 reserve -> fit(fold0) global mean = 1.0


def test_zero_smoothing_no_divide_warning() -> None:
    # NFR-FE-2: k=0 with an unseen code must not leak a numpy divide RuntimeWarning to stderr
    import warnings

    codes = np.array([[2], [0], [0], [1]], dtype=np.int64)
    y = np.array([1.0, 1.0, 0.0, 1.0])
    fold = np.array([0, 1, 1, 1], dtype=np.int64)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any RuntimeWarning becomes a failure
        out = crossfit_encode(codes, y, fold, smoothing=0.0)
    assert out[0, 0] == pytest.approx(y[fold != 0].mean())  # unseen code -> global mean


def test_shape_and_determinism_multi_column() -> None:
    codes = np.array([[0, 1], [1, 0], [0, 1], [1, 0]], dtype=np.int64)
    y = np.array([1.0, 0.0, 0.0, 1.0])
    fold = np.array([0, 0, 1, 1], dtype=np.int64)
    a = crossfit_encode(codes, y, fold, smoothing=2.0)
    b = crossfit_encode(codes, y, fold, smoothing=2.0)
    assert a.shape == (4, 2) and np.array_equal(a, b)


def test_uncovered_block_encoded_on_itself_not_on_test_rows() -> None:
    # F014: rows in no test fold (-1) are train-only; their OOF-TE must be fitted on the -1 block
    # ITSELF (a within-train map), never on the covered test rows. Under a single-fold holdout the -1
    # block is the whole training set, so the old `fit = ~test` fitted train-row encodings from the
    # holdout-test target -> the trained model saw the test answer (leakage, inflated leaderboard).
    # Invariant: permuting the TEST fold's target must NOT change the -1 block's encoding.
    codes = np.array([[0], [1], [0], [1]], dtype=np.int64)
    y = np.array([1.0, 0.0, 1.0, 0.0])
    idx = np.array(
        [-1, -1, 0, 0], dtype=np.int64
    )  # rows 0,1 = train (no test fold); 2,3 = test fold 0
    base = crossfit_encode(codes, y, idx, smoothing=1.0)
    y_perm = y.copy()
    test = idx == 0
    y_perm[test] = y[test][::-1]  # permute only the test fold's target
    perm = crossfit_encode(codes, y_perm, idx, smoothing=1.0)
    assert np.array_equal(
        base[idx == -1], perm[idx == -1]
    )  # -1 block independent of the test target
    assert np.array_equal(
        base[idx == -1], base[idx == 0]
    )  # holdout: train & test share one train-fit map


# --- crossfit_encode_expanding (time-series, ADR-0082) ---------------------


def test_expanding_golden_values() -> None:
    # ADR-0082: fold f is encoded by a map fitted ONLY on strictly earlier folds (idx < f); the -1 block
    # (earliest, no past) collapses to its own base rate. 1 column, k=1; idx = [-1, -1, 0, 0, 1, 1].
    codes = np.array([[0], [1], [0], [1], [0], [1]], dtype=np.int64)
    y = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    idx = np.array([-1, -1, 0, 0, 1, 1], dtype=np.int64)
    out = crossfit_encode_expanding(codes, y, idx, smoothing=1.0)
    # -1 block -> base rate mean([1,0])=0.5; fold0 from rows{0,1}; fold1 from rows{0,1,2,3}
    expected = np.array([0.5, 0.5, 0.75, 0.25, 0.83333333, 0.16666667])
    assert np.allclose(out[:, 0], expected)


def test_expanding_no_lookahead() -> None:
    # ADR-0082 §Проверки: fold f is fitted on idx < f, so permuting the target in fold f AND every LATER
    # fold cannot change fold f's encoding (it never sees its own or any future fold's target).
    rng = np.random.default_rng(1)
    codes = rng.integers(0, 4, size=(80, 2)).astype(np.int64)
    y = rng.integers(0, 2, size=80).astype(np.float64)
    idx = np.repeat(np.array([-1, 0, 1, 2, 3]), 16).astype(np.int64)
    base = crossfit_encode_expanding(codes, y, idx, smoothing=3.0)
    f = 2
    later = np.where(idx >= f)[0]  # fold f AND all later folds
    y_perm = y.copy()
    y_perm[later] = y[rng.permutation(later)]
    perm = crossfit_encode_expanding(codes, y_perm, idx, smoothing=3.0)
    assert np.array_equal(base[idx == f], perm[idx == f])


def test_expanding_reserve_collapses_to_past_global_mean() -> None:
    # ADR-0082 §3 / ADR-0041 §2: a reserve code (>= reserve_from) gets the fold's PAST global mean.
    codes = np.array([[0], [1], [0], [1], [1], [2]], dtype=np.int64)  # code 2 = reserve (null)
    y = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    idx = np.array([-1, -1, -1, -1, 0, 0], dtype=np.int64)  # rows 0..3 = earliest block
    out = crossfit_encode_expanding(codes, y, idx, smoothing=1.0, reserve_from=np.array([2]))
    assert out[5, 0] == pytest.approx(0.5)  # fold0 reserve -> past(-1 block) global mean = 0.5


def test_expanding_earliest_real_fold_collapses_without_minus1_block() -> None:
    # ADR-0082 §2: when the earliest fold is a real fold 0 (no -1 block), idx < 0 is empty -> fold 0
    # collapses to its own base rate; later folds expand on it. Mirror of the golden, labels shifted.
    codes = np.array([[0], [1], [0], [1], [0], [1]], dtype=np.int64)
    y = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    idx = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
    out = crossfit_encode_expanding(codes, y, idx, smoothing=1.0)
    expected = np.array([0.5, 0.5, 0.75, 0.25, 0.83333333, 0.16666667])
    assert np.allclose(out[:, 0], expected)


def test_expanding_single_test_fold() -> None:
    # ADR-0082: the few-folds regime (small n_splits). One test fold + the always-present -1 block:
    # the -1 block is its own base rate, fold 0 is fitted on the -1 block.
    codes = np.array([[0], [1], [0], [1]], dtype=np.int64)
    y = np.array([1.0, 0.0, 1.0, 0.0])
    idx = np.array([-1, -1, 0, 0], dtype=np.int64)
    out = crossfit_encode_expanding(codes, y, idx, smoothing=1.0)
    assert np.allclose(out[:, 0], [0.5, 0.5, 0.75, 0.25])
