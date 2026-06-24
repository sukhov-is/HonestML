"""The ``SignificanceTest`` port.

Separates the statistical equivalence test from the selection policy. ``seed`` and
``n_boot`` are part of the contract, not implementation details: a p-value near the
band boundary depends on them, so leaving them free would re-introduce
non-reproducibility. :class:`NoSignificanceTest` is the explicit opt-out — an empty
equivalence band, so ``select_best`` is a pure argmax.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class SignificanceTest(Protocol):
    """Decides whether two prediction vectors are statistically equivalent.

    ``block_index``/``sample_weight`` are **additive**: the core
    :func:`honestml.core.equivalence_band` passes them already aligned to the common
    dense mask (``block_index[mask]``/``sample_weight[mask]``), so the adapter never
    re-indexes. ``block_index`` (one fold id per OOF row) selects fold-level block
    bootstrap for time-series; ``None`` means i.i.d. row bootstrap. ``sample_weight``
    is forwarded into ``Metric.score`` on every resample so the band matches the
    weighted selection criterion.
    """

    seed: int
    n_boot: int

    def equivalent(
        self,
        pred_a: np.ndarray,
        pred_b: np.ndarray,
        y_true: np.ndarray,
        *,
        alpha: float,
        block_index: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> bool: ...


class NoSignificanceTest:
    """The significance opt-out: nothing is equivalent, so the band is empty."""

    seed = 0
    n_boot = 0

    def equivalent(
        self,
        pred_a: np.ndarray,
        pred_b: np.ndarray,
        y_true: np.ndarray,
        *,
        alpha: float = 0.05,
        block_index: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> bool:
        return False
