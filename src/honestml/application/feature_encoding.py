"""Out-of-fold target encoding (ADR-0041) — pure use-case, mirror of ``crossfit_calibrate``.

For each CV fold ``f`` the smoothed target-mean map is fitted on the rows of the OTHER folds and
looked up on ``f``, so a row's target-encoded feature never sees its own fold's target — the
anti-leakage core of honest target encoding (NFR-FE-1). Pure numpy over integer category codes and
the binary target (no polars/sklearn, NFR-FE-2) — synchronously testable on arrays.
"""

from __future__ import annotations

import numpy as np


def crossfit_encode(
    codes: np.ndarray,
    y: np.ndarray,
    oof_fold_index: np.ndarray,
    *,
    smoothing: float,
    reserve_from: np.ndarray | None = None,
) -> np.ndarray:
    """Out-of-fold target-encoded columns (same ``(n, k)`` shape as ``codes``).

    ``codes`` are integer category codes (``CategoryTable.encode``; non-negative, including the
    null/unknown reserves); ``y`` is the binary positive indicator in ``{0.0, 1.0}``;
    ``oof_fold_index`` assigns each row to a cross-fit fold (the CV fold id, ``-1`` for rows in no
    test fold). Fold ``f`` is encoded by a map fitted on all rows with ``oof_fold_index != f`` — except
    the ``-1`` block (train-only rows, e.g. the whole train set under a single-fold holdout), which is
    fitted on ITSELF so its encoding never sees the covered test folds' target (F014):

        smoothed(code) = (sum_y[code] + k·global_mean_f) / (count[code] + k)

    where ``global_mean_f = mean(y[fold != f])`` and ``k = smoothing``. A code unseen in the fit side
    (``count == 0``) collapses to ``global_mean_f``. ``reserve_from`` (per-column ``null_code``) forces
    every reserve code (null/unknown, ``code >= null_code``) to ``global_mean_f`` too, mirroring the
    full-train fit (ADR-0041 §2) so OOF==full-train==inference for null rows. Computed once per run for
    the whole leaderboard (ADR-0040 §2), not per candidate.
    """
    n, k = codes.shape
    out = np.empty((n, k), dtype=np.float64)
    for f in np.unique(oof_fold_index):
        test = oof_fold_index == f
        # rows in no test fold (f == -1) are train-only: encode the -1 block on ITSELF (a within-train
        # map), never on the covered test rows — otherwise their target-encoded feature would carry the
        # test folds' outcome into model training (leak under a single-fold holdout CV, F014).
        fit = test if f == -1 else ~test
        y_fit = y[fit]
        global_mean = float(y_fit.mean()) if y_fit.size else 0.0
        for j in range(k):
            col = codes[:, j]
            width = int(col.max()) + 1 if col.size else 1  # per-column, not the global max
            count = np.bincount(col[fit], minlength=width).astype(np.float64)
            sum_y = np.bincount(col[fit], weights=y_fit, minlength=width)
            with np.errstate(invalid="ignore", divide="ignore"):  # 0/0 at k==0 -> handled below
                smoothed = (sum_y + smoothing * global_mean) / (count + smoothing)
            smoothed[count == 0] = global_mean  # unseen code on the fit side -> global_mean
            if reserve_from is not None and reserve_from[j] < width:
                smoothed[reserve_from[j] :] = global_mean  # null/unknown reserves -> global_mean
            out[test, j] = smoothed[col[test]]
    return out


def crossfit_encode_expanding(
    codes: np.ndarray,
    y: np.ndarray,
    oof_fold_index: np.ndarray,
    *,
    smoothing: float,
    reserve_from: np.ndarray | None = None,
) -> np.ndarray:
    """Expanding-window out-of-fold target encoding for time-ordered CV (ADR-0082).

    Like :func:`crossfit_encode`, but fold ``f`` is encoded by a map fitted ONLY on strictly EARLIER folds
    (``oof_fold_index < f``) instead of all other folds — so under a time-ordered split (where the fold id
    is monotone in time, ADR-0027) a row's encoding never looks ahead. The earliest block
    (``oof_fold_index == -1``, the initial train period before the first test window — always present and in
    the past of every test window under :class:`TimeSeriesSplitter`) is captured by ``idx < f`` for every
    ``f >= 0``. The earliest fold value itself has no prior (``idx < f`` empty): its rows are train-only
    (never scored on the leaderboard), so they collapse to their own block's base rate ``mean(y[block])`` —
    a documented boundary compromise (ADR-0082 §2), not a per-category signal. Smoothing and reserve
    handling match :func:`crossfit_encode` (ADR-0041 §2); ``global_mean_f = mean(y[idx < f])`` (per-fold,
    on the past). Computed once per run (ADR-0040 §2).
    """
    n, k = codes.shape
    out = np.empty((n, k), dtype=np.float64)
    for f in np.unique(oof_fold_index):
        test = oof_fold_index == f
        fit = (
            oof_fold_index < f
        )  # strictly-earlier folds (the -1 block included, since -1 < f for f >= 0)
        if (
            not fit.any()
        ):  # earliest block has no past -> its own base rate (train-only rows, ADR-0082 §2)
            out[test] = float(y[test].mean()) if test.any() else 0.0
            continue
        y_fit = y[fit]
        global_mean = float(y_fit.mean())
        for j in range(k):
            col = codes[:, j]
            width = int(col.max()) + 1 if col.size else 1  # per-column, not the global max
            count = np.bincount(col[fit], minlength=width).astype(np.float64)
            sum_y = np.bincount(col[fit], weights=y_fit, minlength=width)
            with np.errstate(invalid="ignore", divide="ignore"):  # 0/0 at k==0 -> handled below
                smoothed = (sum_y + smoothing * global_mean) / (count + smoothing)
            smoothed[count == 0] = global_mean  # unseen-in-the-past code -> global_mean
            if reserve_from is not None and reserve_from[j] < width:
                smoothed[reserve_from[j] :] = global_mean  # null/unknown reserves -> global_mean
            out[test, j] = smoothed[col[test]]
    return out
