"""Etap3: period-weighted leaderboard scoring (``_score_weighted``, ADR-0098 §2, FR-6).

Pure leaf helpers over a metric-ready OOF vector + a per-row block index. A class-agnostic ``_MeanScore``
(weighted mean of the prediction, y ignored) makes pooled-vs-period deterministic without sklearn edges;
``_PickyScore`` raises on a single-class block to exercise the NaN-drop (R-6).
"""

from __future__ import annotations

import numpy as np
import pytest

from honestml.application.slice import _period_block_scores, _score_weighted

pytestmark = pytest.mark.unit


class _MeanScore:
    name = "mean"
    greater_is_better = True
    needs = "value"
    optimum = 1.0
    average = None

    def score(self, y_true, y_pred, sample_weight=None):
        return float(np.average(y_pred, weights=sample_weight))


class _PickyScore(_MeanScore):
    def score(self, y_true, y_pred, sample_weight=None):
        if np.unique(y_true).size < 2:
            raise ValueError("metric undefined on a single-class block")
        return super().score(y_true, y_pred, sample_weight)


def test_pooled_is_single_metric_over_mask() -> None:
    # pooled = one metric over all valid rows, independent of the block index (NFR-5 path)
    y = np.zeros(6)
    pred = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    mask = np.ones(6, dtype=bool)
    bi = np.array([0, 0, 0, 1, 1, 1])
    assert _score_weighted(_MeanScore(), y, pred, mask, bi, None, "pooled") == 2.5


def test_period_macro_differs_from_pooled_on_unequal_blocks() -> None:
    # FR-6: period weighs each block equally; pooled is size-weighted -> different numbers on unequal blocks
    y = np.zeros(4)
    pred = np.array([10.0, 0.0, 0.0, 0.0])
    mask = np.ones(4, dtype=bool)
    bi = np.array([0, 1, 1, 1])  # block0: [10] -> 10; block1: [0,0,0] -> 0
    assert _score_weighted(_MeanScore(), y, pred, mask, bi, None, "pooled") == 2.5  # (10+0+0+0)/4
    assert _score_weighted(_MeanScore(), y, pred, mask, bi, None, "period") == 5.0  # mean(10, 0)
    assert _period_block_scores(_MeanScore(), y, pred, mask, bi, None) == [10.0, 0.0]


def test_period_skips_block_with_no_valid_rows() -> None:
    # a block whose rows are all outside the mask contributes nothing (not a NaN, not a crash)
    y = np.zeros(6)
    pred = np.array([4.0, 6.0, 0.0, 0.0, 0.0, 0.0])
    mask = np.array([True, True, False, False, False, False])  # block1 has no valid rows
    bi = np.array([0, 0, 1, 1, 1, 1])
    assert (
        _score_weighted(_MeanScore(), y, pred, mask, bi, None, "period") == 5.0
    )  # only block0 mean
    assert _period_block_scores(_MeanScore(), y, pred, mask, bi, None) == [5.0]


def test_period_drops_block_with_undefined_metric() -> None:
    # R-6: a block whose metric raises (single-class here) is dropped from the macro-average
    y = np.array([0, 1, 0, 0])  # block0 has both classes; block1 is single-class
    pred = np.array([1.0, 3.0, 5.0, 7.0])
    mask = np.ones(4, dtype=bool)
    bi = np.array([0, 0, 1, 1])
    assert _period_block_scores(_PickyScore(), y, pred, mask, bi, None) == [2.0]  # only block0
    assert _score_weighted(_PickyScore(), y, pred, mask, bi, None, "period") == 2.0


def test_period_all_blocks_invalid_is_nan() -> None:
    # all blocks undefined -> nan, mirroring an empty pooled mask (no fabricated score)
    y = np.zeros(3)
    pred = np.zeros(3)
    mask = np.zeros(3, dtype=bool)
    bi = np.array([0, 1, 2])
    assert np.isnan(_score_weighted(_MeanScore(), y, pred, mask, bi, None, "period"))


def test_period_excludes_uncovered_rows() -> None:
    # rows with block id -1 (covered by no fold) never form a block
    y = np.zeros(4)
    pred = np.array([1.0, 3.0, 99.0, 99.0])
    mask = np.ones(4, dtype=bool)
    bi = np.array([0, 0, -1, -1])
    assert _period_block_scores(_MeanScore(), y, pred, mask, bi, None) == [
        2.0
    ]  # only block0 = mean(1,3)
