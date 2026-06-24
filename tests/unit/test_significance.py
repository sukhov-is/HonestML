"""M4a1: BootstrapSignificanceTest — paired bootstrap, two-sided CI-overlap (ADR-0026).

Pure numpy: the injected Metric is the only ML dependency. A ``_MeanScore`` fake gives a
monotone, class-agnostic metric so power/Type-I/weighting are deterministic without sklearn
edge cases; the ranking path is exercised with the real ``RocAuc`` (SPIKE-0002).
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from honestml.adapters import BootstrapSignificanceTest, LogLoss, RocAuc
from honestml.core import ConfigError

pytestmark = pytest.mark.unit


class _MeanScore:
    """Class-agnostic monotone metric: weighted mean of the prediction (higher is better)."""

    name = "mean"
    greater_is_better = True
    needs = "value"
    optimum = 1.0
    average = None

    def score(self, y_true, y_pred, sample_weight=None):
        return float(np.average(y_pred, weights=sample_weight))


def test_delta_distribution_ignores_uncovered_rows() -> None:
    """F104: rows with the uncovered id -1 never form a bootstrap block, so their values cannot
    influence the delta distribution (mirrors the period path's -1 guard)."""
    sig = BootstrapSignificanceTest(metric=_MeanScore(), n_boot=200, seed=0)
    rng = np.random.default_rng(1)
    n = 60
    y = np.zeros(n)
    a, b = rng.normal(size=n), rng.normal(size=n)
    blocks = np.repeat([0, 1, 2], 20)
    blocks[:6] = -1  # mark the first 6 rows uncovered
    base = sig._delta_distribution(a, b, y, None, blocks)
    a2, b2 = a.copy(), b.copy()
    a2[:6], b2[:6] = 1e9, -1e9  # poison the uncovered rows: must not change anything
    assert np.allclose(base, sig._delta_distribution(a2, b2, y, None, blocks))


def test_typeI_equivalent_share_band() -> None:
    """Two equivalent models (same signal, independent noise) -> declared equivalent."""
    rng = np.random.default_rng(0)
    n = 400
    y = np.zeros(n)
    sig = rng.normal(0.0, 1.0, n)
    a = sig + rng.normal(0.0, 0.05, n)
    b = sig + rng.normal(0.0, 0.05, n)  # same mean signal, decorrelated noise
    test = BootstrapSignificanceTest(_MeanScore(), seed=1, n_boot=1000)
    assert test.equivalent(a, b, y, alpha=0.05) is True


def test_power_excludes_worse() -> None:
    """A clearly-worse model (mean far below the anchor) is excluded from the band."""
    rng = np.random.default_rng(1)
    n = 200
    y = np.zeros(n)
    anchor = 0.9 + rng.normal(0.0, 0.05, n)
    worse = 0.1 + rng.normal(0.0, 0.05, n)
    test = BootstrapSignificanceTest(_MeanScore(), seed=1, n_boot=1000)
    assert test.equivalent(worse, anchor, y, alpha=0.05) is False


def test_roc_auc_band() -> None:
    """Ranking metric (default binary): equivalent rankers in band, AUC≈0.5 excluded (SPIKE-0002)."""
    rng = np.random.default_rng(0)
    n = 400
    y = rng.integers(0, 2, size=n)
    sig = np.where(y == 1, 0.8, 0.2)
    strong = sig + rng.normal(0.0, 0.05, n)
    near = sig + rng.normal(0.0, 0.05, n)  # same ranking quality
    weak = rng.uniform(size=n)  # AUC ≈ 0.5
    test = BootstrapSignificanceTest(RocAuc(), seed=1, n_boot=1000)
    assert test.equivalent(near, strong, y, alpha=0.05) is True
    assert test.equivalent(weak, strong, y, alpha=0.05) is False


def test_multiclass_band_2d() -> None:
    """The (n, K) probability path resamples rows and discriminates (log_loss, equivalent vs worse)."""
    rng = np.random.default_rng(0)
    n, k = 300, 3
    y = rng.integers(0, k, size=n)

    def proba(strength: float) -> np.ndarray:
        p = np.full((n, k), (1.0 - strength) / (k - 1))
        p[np.arange(n), y] = strength
        p = np.clip(p + rng.normal(0.0, 0.02, (n, k)), 1e-6, None)
        return p / p.sum(axis=1, keepdims=True)

    metric = LogLoss(classes=np.array([0, 1, 2]))
    test = BootstrapSignificanceTest(metric, seed=1, n_boot=1000)
    strong, near = proba(0.8), proba(0.8)
    weak = np.full((n, k), 1.0 / k)  # uninformative -> clearly worse log_loss
    assert test.equivalent(near, strong, y, alpha=0.05) is True
    assert test.equivalent(weak, strong, y, alpha=0.05) is False


@pytest.mark.filterwarnings("ignore:Only one class")
def test_roc_auc_robust_to_single_class_resample() -> None:
    """Imbalanced binary: single-class bootstrap resamples are dropped, not crashed (ADR-0026 §7)."""
    rng = np.random.default_rng(0)
    n = 24
    y = np.array([1, 1] + [0] * (n - 2))  # heavy imbalance -> many single-class resamples
    a = np.where(y == 1, 0.8, 0.3) + rng.normal(0.0, 0.05, n)
    b = np.where(y == 1, 0.7, 0.4) + rng.normal(0.0, 0.05, n)
    test = BootstrapSignificanceTest(RocAuc(), seed=1, n_boot=1000)
    assert isinstance(test.equivalent(a, b, y, alpha=0.05), bool)  # no ValueError crash


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.5])
def test_alpha_out_of_range_raises(alpha: float) -> None:
    """ADR-0026 §7 boundary: alpha outside (0, 1) -> ConfigError, not a raw numpy error."""
    test = BootstrapSignificanceTest(RocAuc(), seed=1, n_boot=2000)
    with pytest.raises(ConfigError, match="alpha"):
        test.equivalent(np.array([0.1, 0.9]), np.array([0.2, 0.8]), np.array([0, 1]), alpha=alpha)


def test_no_fold_mean_ttest() -> None:
    """NFR-M4-1: sample-level bootstrap = 2*n_boot metric evals, not a (few-fold) fold-mean t-test."""
    calls = {"n": 0}

    class _Counter(_MeanScore):
        def score(self, y_true, y_pred, sample_weight=None):
            calls["n"] += 1
            return super().score(y_true, y_pred, sample_weight)

    n = 50
    y = np.zeros(n)
    a = np.full(n, 0.6)
    b = np.full(n, 0.6)
    test = BootstrapSignificanceTest(_Counter(), seed=1, n_boot=1000)
    test.equivalent(a, b, y, alpha=0.05)
    assert calls["n"] == 2 * 1000  # two scores per resample over n_boot resamples


def test_sample_weighted_resample() -> None:
    """NFR-M4-1: sample_weight is resampled and forwarded into Metric.score on every resample."""
    received: list[np.ndarray | None] = []

    class _WCheck(_MeanScore):
        def score(self, y_true, y_pred, sample_weight=None):
            received.append(sample_weight)
            return super().score(y_true, y_pred, sample_weight)

    n = 20
    y = np.zeros(n)
    a = np.linspace(0.0, 1.0, n)
    b = np.linspace(0.0, 1.0, n)
    w = np.arange(n, dtype=float) + 1.0
    test = BootstrapSignificanceTest(_WCheck(), seed=1, n_boot=1000)
    test.equivalent(a, b, y, alpha=0.05, sample_weight=w)
    assert all(sw is not None for sw in received)
    assert received[0].shape == (n,)  # resampled to the (masked) length, aligned with idx


def test_block_bootstrap_resamples_whole_folds() -> None:
    """Time-series path resamples WHOLE CV folds, not individual rows (ADR-0026 §2)."""
    seen_counts: list[set[int]] = []

    class _FoldRec(_MeanScore):
        def score(self, y_true, y_pred, sample_weight=None):
            _, counts = np.unique(y_pred, return_counts=True)
            seen_counts.append(set(counts.tolist()))
            return super().score(y_true, y_pred, sample_weight)

    n = 30
    block_index = np.repeat(np.arange(3), 10)  # 3 folds of 10 rows
    pred = block_index.astype(float)  # each row's value encodes its fold id
    y = np.zeros(n)
    test = BootstrapSignificanceTest(_FoldRec(), seed=1, n_boot=200)
    test.equivalent(pred, pred.copy(), y, alpha=0.30, block_index=block_index)
    # whole-fold resampling => each present fold contributes a multiple of its size (10)
    assert seen_counts and all(all(c % 10 == 0 for c in counts) for counts in seen_counts)


def test_reproducible_band_same_seed() -> None:
    """NFR-M4-2: same seed -> bit-identical delta distribution and identical verdict."""
    rng = np.random.default_rng(0)
    n = 200
    y = rng.integers(0, 2, n)
    sig = np.where(y == 1, 0.7, 0.3)
    a = sig + rng.normal(0.0, 0.1, n)
    b = sig + rng.normal(0.0, 0.1, n)
    t1 = BootstrapSignificanceTest(RocAuc(), seed=42, n_boot=1000)
    t2 = BootstrapSignificanceTest(RocAuc(), seed=42, n_boot=1000)
    assert t1.equivalent(a, b, y, alpha=0.05) == t2.equivalent(a, b, y, alpha=0.05)
    d1 = t1._delta_distribution(a, b, y, None, None)
    d2 = t2._delta_distribution(a, b, y, None, None)
    assert np.array_equal(d1, d2)


def test_n_boot_alpha_floor_raises() -> None:
    """ADR-0026 §7: n_boot*alpha < 50 -> ConfigError (thin CI tail), not a silent garbage CI."""
    test = BootstrapSignificanceTest(RocAuc(), seed=1, n_boot=100)  # 100*0.05 = 5 < 50
    with pytest.raises(ConfigError, match="too thin"):
        test.equivalent(np.array([0.1, 0.9]), np.array([0.2, 0.8]), np.array([0, 1]), alpha=0.05)


# --- Etap3: aggregate='period' macro-by-period significance (ADR-0098 §3, FR-6) ---


def _unequal_blocks() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """8 period blocks: one big (120 rows) where B≈A, seven tiny (4 rows) where B is much worse."""
    sizes = [120, 4, 4, 4, 4, 4, 4, 4]
    block_index = np.repeat(np.arange(len(sizes)), sizes)
    n = block_index.size
    a = np.full(n, 0.5)
    b = np.where(block_index == 0, 0.6, 0.0)  # block0 Δ=+0.1, small blocks Δ=-0.5
    return a, b, block_index


def test_period_aggregate_differs_from_pooled() -> None:
    # FR-6: pooled is size-weighted (the big block dominates -> equivalent); period weighs each block
    # equally (the seven worse blocks dominate the macro-average -> excluded). Same data, different verdict.
    a, b, bi = _unequal_blocks()
    y = np.zeros(bi.size)
    pooled = BootstrapSignificanceTest(_MeanScore(), seed=1, n_boot=1000, aggregate="pooled")
    period = BootstrapSignificanceTest(_MeanScore(), seed=1, n_boot=1000, aggregate="period")
    assert pooled.equivalent(a, b, y, alpha=0.05, block_index=bi) is True
    assert period.equivalent(a, b, y, alpha=0.05, block_index=bi) is False


def test_period_too_few_blocks_raises() -> None:
    # F2/G3: below _MIN_BLOCKS_HARD (4) the block-bootstrap CI degenerates -> fail fast, not a fake band
    n = 30
    bi = np.repeat(np.arange(3), 10)  # 3 valid blocks < 4
    y = np.zeros(n)
    pred = np.full(n, 0.5)
    test = BootstrapSignificanceTest(_MeanScore(), seed=1, n_boot=1000, aggregate="period")
    with pytest.raises(ConfigError, match="valid period block"):
        test.equivalent(pred, pred.copy(), y, alpha=0.05, block_index=bi)


def test_period_few_blocks_warns_but_runs(caplog: pytest.LogCaptureFixture) -> None:
    # F2: 4 <= n_valid < _MIN_BLOCKS (8) -> the test RUNS but warns the band is wide (visible reaction)
    n = 50
    bi = np.repeat(
        np.arange(5), 10
    )  # 5 valid blocks: passes the hard gate, below the warn threshold
    y = np.zeros(n)
    pred = np.full(n, 0.5)
    test = BootstrapSignificanceTest(_MeanScore(), seed=1, n_boot=1000, aggregate="period")
    with caplog.at_level(logging.WARNING, logger="honestml"):
        result = test.equivalent(pred, pred.copy(), y, alpha=0.05, block_index=bi)
    assert result is True  # identical models -> equivalent
    assert any("period" in r.getMessage() for r in caplog.records)


def test_period_drops_block_undefined_for_either_candidate() -> None:
    # F7 anti-survivorship: a block whose metric is undefined for EITHER candidate is dropped for the
    # pair, so both are compared on the COMMON set of periods (Δ NaN -> dropped, symmetric via _delta).
    class _NegRaises(_MeanScore):
        def score(self, y_true, y_pred, sample_weight=None):
            if float(np.mean(y_pred)) < 0.0:
                raise ValueError("undefined on a negative-mean block")
            return super().score(y_true, y_pred, sample_weight)

    n = 50
    bi = np.repeat(np.arange(5), 10)  # 5 blocks
    y = np.zeros(n)
    a = np.ones(n)
    a[bi == 2] = -1.0  # candidate A is undefined on block 2 only
    b = np.ones(n)  # candidate B is defined on every block
    test = BootstrapSignificanceTest(_NegRaises(), seed=1, n_boot=1000, aggregate="period")
    deltas = test._period_block_deltas(a, b, y, None, bi)
    assert deltas.size == 4  # block 2 dropped for BOTH (common valid set), the other 4 remain


@pytest.mark.filterwarnings("ignore:Only one class")
def test_period_drops_single_class_block(caplog: pytest.LogCaptureFixture) -> None:
    # R-6: a block whose metric is undefined (single-class roc_auc) is dropped from the macro-average,
    # leaving the remaining valid blocks; no crash.
    rng = np.random.default_rng(0)
    parts_y, parts_p = [], []
    for b in range(6):  # blocks 0..4 mixed, block 5 single-class -> dropped, 5 valid remain
        yb = np.zeros(20) if b == 5 else rng.integers(0, 2, size=20)
        parts_y.append(yb)
        parts_p.append(np.where(yb == 1, 0.7, 0.3) + rng.normal(0.0, 0.05, 20))
    y = np.concatenate(parts_y)
    pred = np.concatenate(parts_p)
    block_index = np.repeat(np.arange(6), 20)
    test = BootstrapSignificanceTest(RocAuc(), seed=1, n_boot=1000, aggregate="period")
    with caplog.at_level(logging.WARNING, logger="honestml"):
        verdict = test.equivalent(pred, pred.copy(), y, alpha=0.05, block_index=block_index)
    assert isinstance(verdict, bool)  # the single-class block did not crash the test
