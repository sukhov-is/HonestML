"""The ``FeatureRanker`` port — per-feature importance for one training matrix.

A strategy scores features given ONE training matrix; the application spine (``select_features``)
drives the anti-leakage per-fold loop, aggregates and cuts. The ranker never sees test
rows and never decides the cutoff — the leakage-critical mechanics stay in one place. Pure signature
(numpy), so the domain stays free of polars/sklearn (import-linter ``core-independence``).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class FeatureRanker(Protocol):
    """Score features by importance for one training matrix (higher = more important)."""

    name: str

    def rank(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        categorical: np.ndarray,
        random_state: int,
        sample_weight: np.ndarray | None = None,
        groups: np.ndarray | None = None,
    ) -> np.ndarray:
        """Per-feature importance, shape ``(n_features,)`` in column order of ``x``.

        ``x`` is ONE fold's training matrix (numeric block ⊕ categorical codes); ``categorical`` is the
        per-column bool mask. The ranker fits on these rows only (the caller never passes test rows),
        uses ``random_state`` as its sole randomness source and may weight rows by ``sample_weight``.
        Output invariants: length ``== x.shape[1]``, no NaN/inf (else ``ValueError``); a non-negative
        ``importance`` or a signed ``random_probe`` margin; an empty ``x`` (``n_fit == 0``) raises.
        Scale-comparability across folds is the spine's job (it normalizes per fold).

        ``groups`` is an optional per-row structure label (time-block / group index) for
        structure-aware strategies (``null_importance`` permutes the target WITHIN each label).
        ``None`` (default) keeps the i.i.d. behavior; importance-style rankers ignore it. The
        spine prepares and slices it to the fold's training rows, so the ranker still never sees
        test rows.
        """
        ...

    def auto_threshold(self, n_features: int) -> float:
        """Threshold for ``cutoff='auto'``: keep features whose aggregate score exceeds it.

        ``importance`` returns the uniform share ``1/n_features`` (above-average importance);
        ``random_probe`` returns ``0.0`` (mean margin beats the random-probe baseline).
        """
        ...
