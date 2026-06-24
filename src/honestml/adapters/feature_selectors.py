"""Wrapper feature-subset-selector adapters (ADR-0047 §2) — behind the ``FeatureSubsetSelector`` port.

``sequential`` is greedy backward elimination: at each step drop the feature whose removal best keeps
the injected OOF ``score_subset``. It returns the whole **trajectory** of visited subsets (ADR-0083 §1)
— the application picks the final subset (argmax, or significance-band + Occam). ``full_descent`` (set by
composition under an active band, ADR-0084) runs to the floor; otherwise the descent early-stops after
``patience`` non-improving steps (legacy argmax path). The adapter holds only the **policy** — the
leakage-critical OOF scoring is the injected ``score_subset`` callable (it sees column indices only,
never raw test rows), so anti-leakage stays in the application (Humble Object, ADR-0046 §1). Pure
numpy/Callable; no sklearn here.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from honestml.core import Fold

_EPS = 1e-12


class SequentialSelector:
    """Greedy backward-elimination subset selector over an injected OOF scorer (ADR-0047 §2)."""

    name = "sequential"

    def __init__(
        self, min_features: int = 1, patience: int = 2, full_descent: bool = False
    ) -> None:
        self._min_features = max(1, min_features)
        self._patience = max(1, patience)
        self._full_descent = full_descent

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
        """Greedily drop the least-harmful feature each step; return the visited trajectory (ADR-0083 §1).

        At each step ``max((score_subset(keep\\{j}), j) for j in keep)`` picks the best post-drop score,
        breaking ties toward the larger index (deterministic). The trajectory is the full set then each
        committed subset (strictly decreasing size, each a sorted index tuple). ``full_descent`` runs to
        the ``>= min_features`` floor (ADR-0084); otherwise the descent early-stops after ``patience``
        non-improving steps. Never empty — the full set is always the first point (FR-FSC-3).
        """
        keep = list(range(x.shape[1]))
        trajectory: list[tuple[int, ...]] = [tuple(keep)]
        best_score = -np.inf if self._full_descent else score_subset(keep)
        stale = 0
        while len(keep) > self._min_features:
            # the least-harmful single drop this step (deterministic tie-break: larger index)
            gain, drop = max((score_subset([c for c in keep if c != j]), j) for j in keep)
            keep.remove(drop)
            trajectory.append(tuple(keep))  # keep stays ascending -> already sorted
            if self._full_descent:
                continue
            if gain > best_score + _EPS:
                best_score, stale = gain, 0
            else:
                stale += 1
                if stale >= self._patience:
                    break
        return tuple(trajectory)
