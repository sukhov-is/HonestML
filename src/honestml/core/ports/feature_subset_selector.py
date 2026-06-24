"""The ``FeatureSubsetSelector`` port — wrapper-style feature selection.

A ranker scores one matrix (:class:`FeatureRanker`); a *wrapper* method instead drives a greedy
estimator+scorer loop that yields a subset directly (e.g. ``sequential`` backward-elimination), which
does not fit ``rank()->scores``. This port owns only the **policy** (which feature to drop, when to
stop); the leakage-critical OOF scoring is an injected ``score_subset`` callable supplied by the
application, so the adapter receives only a scalar per candidate subset and never sees raw test rows
(Humble Object). Pure numpy/Callable signature — the domain stays free of sklearn/polars.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol, runtime_checkable

import numpy as np

from .splitter import Fold


@runtime_checkable
class FeatureSubsetSelector(Protocol):
    """Select a feature subset by a greedy wrapper policy over an injected OOF scorer."""

    name: str

    def select(
        self,
        x: np.ndarray,
        y: np.ndarray,
        folds: Sequence[Fold],
        *,
        categorical: np.ndarray,
        score_subset: Callable[[Sequence[int]], float],
        random_state: int,
        sample_weight: np.ndarray | None = None,
    ) -> tuple[tuple[int, ...], ...]:
        """Return the greedy backward-elimination **trajectory** (ADR-0083 §1).

        The trajectory is an ordered tuple of visited subsets of **strictly decreasing size** — the
        full set first, then each committed subset down to the floor (band) or the patience stop
        (argmax). Each subset is a sorted tuple of column indices, non-empty (floor ``>= 1``). The
        application picks the final subset from this trajectory (argmax, or significance-band + Occam);
        the adapter owns only the greedy exploration policy.

        ``x`` is the selection matrix (numeric block ⊕ categorical codes); ``categorical`` is the
        per-column bool mask; ``folds`` are the evaluation folds. ``score_subset(indices)`` is an
        application-provided pure scorer: it fits on each fold's ``fit ⊕ es`` and scores the ``test``
        part on the columns ``indices`` (leakage-safe by construction), returning a higher-is-better
        scalar. The adapter calls it with **column indices only** — it must not access raw rows. The
        trajectory is deterministic given ``random_state``.
        """
        ...
