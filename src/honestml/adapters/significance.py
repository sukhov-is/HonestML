"""Bootstrap significance test (ADR-0026, SPIKE-0002) — the honest M4 band test.

Implements the ``SignificanceTest`` port with a **paired bootstrap over OOF**: resample
rows with replacement, recompute the METRIC on each resample for both models (so it works
for ranking metrics like ``roc_auc`` that do not decompose into per-sample losses), and
declare equivalence when the **two-sided ``(1-alpha)`` CI of the metric difference Δ
includes 0** — conservative to the post-hoc argmax anchor (SPIKE-0002: roc_auc Type-I
1.000). For time-series OOF a **fold-level block bootstrap** resamples whole CV test folds
(``block_index``), since i.i.d. row resampling understates variance under autocorrelation.

numpy-only: the injected ``Metric`` carries the sklearn dependency, so the band machinery
keeps ``core`` free of heavy ML libraries (NFR-M4-4/6).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from honestml.core import ConfigError, Metric, get_logger

logger = get_logger("adapters.significance")

# n_boot * alpha floor: fewer than this many resamples land in the CI tail and the
# percentile estimate is noise, not a CI (ADR-0026 §7).
_MIN_TAIL = 50
# block-count floors for aggregate='period' (ADR-0098 §3): a block = a CV test fold/period, so the count
# is ~n_splits (3-6 monthly). Below _MIN_BLOCKS_HARD the percentile CI over so few blocks is a degenerate
# discrete lattice -> fail fast; between hard and _MIN_BLOCKS the test runs but the band is wide -> WARNING.
_MIN_BLOCKS_HARD = 4
_MIN_BLOCKS = 8


@dataclass
class BootstrapSignificanceTest:
    """Paired-bootstrap equivalence test: two-sided CI-overlap of the metric difference Δ.

    Not ``frozen``: the ``SignificanceTest`` port declares ``seed``/``n_boot`` as settable (like
    ``NoSignificanceTest``); the fields are set once at construction and not mutated in practice.
    """

    metric: Metric
    seed: int = 0
    n_boot: int = 2000
    # ADR-0098 §3: "pooled" recomputes the metric on the concatenation of resampled blocks; "period"
    # macro-averages per-block Δ (each period weighs equally, matching weighting='period'). Adapter detail,
    # NOT in the SignificanceTest port (set only on the bootstrap test, never on NoSignificanceTest, F16).
    aggregate: Literal["pooled", "period"] = "pooled"

    def equivalent(
        self,
        pred_a: np.ndarray,
        pred_b: np.ndarray,
        y_true: np.ndarray,
        *,
        alpha: float,
        block_index: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> bool:
        if not 0.0 < alpha < 1.0:
            raise ConfigError(f"alpha={alpha!r} must be in the open interval (0, 1)")
        if self.n_boot * alpha < _MIN_TAIL:
            raise ConfigError(
                f"n_boot*alpha={self.n_boot * alpha:.4g} < {_MIN_TAIL}: the percentile CI tail "
                "is too thin; raise n_boot or alpha"
            )
        if self.aggregate == "period" and block_index is not None:
            deltas = self._period_delta_distribution(
                pred_a, pred_b, y_true, sample_weight, block_index
            )
        else:
            deltas = self._delta_distribution(pred_a, pred_b, y_true, sample_weight, block_index)
        finite = deltas[np.isfinite(deltas)]  # drop resamples the metric could not score (NaN)
        if finite.size == 0 or float(np.ptp(finite)) == 0.0:
            return True  # degenerate / all-NaN CI -> conservatively include (operational §5)
        lo, hi = np.percentile(finite, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)])
        return bool(lo <= 0.0 <= hi)

    def _delta_distribution(
        self,
        pred_a: np.ndarray,
        pred_b: np.ndarray,
        y_true: np.ndarray,
        sample_weight: np.ndarray | None,
        block_index: np.ndarray | None,
    ) -> np.ndarray:
        # reset RNG from the fixed seed on every call -> reproducible AND invariant to the
        # candidate pool (each pairwise test draws the same resamples; NFR-M4-2, ADR-0026 §2)
        rng = np.random.default_rng(self.seed)
        n = y_true.shape[0]
        deltas = np.empty(self.n_boot, dtype=np.float64)
        if block_index is None:
            for i in range(self.n_boot):
                deltas[i] = self._delta(
                    rng.integers(0, n, size=n), pred_a, pred_b, y_true, sample_weight
                )
        else:
            # exclude uncovered rows (id -1); mirrors _period_block_deltas so a stray -1 never forms a block
            blocks = [np.flatnonzero(block_index == b) for b in np.unique(block_index) if b >= 0]
            n_blocks = len(blocks)
            for i in range(self.n_boot):
                chosen = rng.integers(0, n_blocks, size=n_blocks)
                idx = np.concatenate([blocks[j] for j in chosen])
                deltas[i] = self._delta(idx, pred_a, pred_b, y_true, sample_weight)
        return deltas

    def _period_delta_distribution(
        self,
        pred_a: np.ndarray,
        pred_b: np.ndarray,
        y_true: np.ndarray,
        sample_weight: np.ndarray | None,
        block_index: np.ndarray,
    ) -> np.ndarray:
        # macro-by-period (ADR-0098 §3): per-block Δ is fixed (uses ALL of the block's rows), so the
        # block bootstrap resamples the per-block Δ values and averages — each resample weighs periods
        # equally. The valid (finite-metric) block set is fixed BEFORE resampling and is identical for
        # both candidates (a block where either metric is undefined -> _delta NaN -> dropped, F7/R-6).
        per_block = self._period_block_deltas(pred_a, pred_b, y_true, sample_weight, block_index)
        n_valid = per_block.size
        if n_valid < _MIN_BLOCKS_HARD:
            raise ConfigError(
                f"weighting='period' significance has only {n_valid} valid period block(s) "
                f"(< {_MIN_BLOCKS_HARD}); the block-bootstrap CI degenerates into a discrete lattice — "
                "use weighting='pooled' or run with more periods/folds"
            )
        if n_valid < _MIN_BLOCKS:
            logger.warning(
                "weighting='period' significance over only %d period(s) (< %d): the equivalence band "
                "is wide and low-resolution; treat band membership with caution",
                n_valid,
                _MIN_BLOCKS,
            )
        rng = np.random.default_rng(self.seed)  # reproducible + pool-invariant (ADR-0026 §2)
        deltas = np.empty(self.n_boot, dtype=np.float64)
        for i in range(self.n_boot):
            deltas[i] = float(per_block[rng.integers(0, n_valid, size=n_valid)].mean())
        return deltas

    def _period_block_deltas(
        self,
        pred_a: np.ndarray,
        pred_b: np.ndarray,
        y_true: np.ndarray,
        sample_weight: np.ndarray | None,
        block_index: np.ndarray,
    ) -> np.ndarray:
        """Per-block metric Δ (B−A) over the finite-metric blocks; reuses :meth:`_delta` (R-6).

        A block undefined for EITHER candidate drops for both (``_delta`` returns NaN when either
        ``metric.score`` is non-finite — older sklearn raises ValueError, newer returns nan — and the
        ``isfinite`` filter removes it), so the pair is compared on the COMMON set of periods (F7).
        """
        deltas = []
        for b in np.unique(block_index):
            if b < 0:  # uncovered rows carry id -1 and never form a real block
                continue
            d = self._delta(np.flatnonzero(block_index == b), pred_a, pred_b, y_true, sample_weight)
            if np.isfinite(d):
                deltas.append(d)
        return np.asarray(deltas, dtype=np.float64)

    def _delta(
        self,
        idx: np.ndarray,
        pred_a: np.ndarray,
        pred_b: np.ndarray,
        y_true: np.ndarray,
        sample_weight: np.ndarray | None,
    ) -> float:
        yt = y_true[idx]
        sw = sample_weight[idx] if sample_weight is not None else None
        # paired: ONE resample drives both models; metric recomputed so ranking metrics are valid.
        # A degenerate resample (e.g. single-class for roc_auc) makes the metric raise -> NaN, which
        # is dropped from the CI rather than crashing the whole selection (ADR-0026 §7, operational §5).
        try:
            return float(
                self.metric.score(yt, pred_b[idx], sw) - self.metric.score(yt, pred_a[idx], sw)
            )
        except ValueError:
            return float("nan")
