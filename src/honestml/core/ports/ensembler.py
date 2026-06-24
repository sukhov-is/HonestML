"""The ``Ensembler`` port + ``EnsembleRecipe``.

Ensembling behind a port: the application builds the per-model blend space (``oof``) and an injected
``score(blended) -> float`` (higher-is-better wrapper of :class:`~honestml.core.Metric`); a concrete
:class:`Ensembler` (Caruana / weighted) searches the weight simplex over it. The port is a **Humble
Object** (like :class:`~honestml.core.Tuner`): the adapter sees only scalars and the ``oof``
array — never raw rows/folds — so the domain stays free of scipy/sklearn.ensemble and the search is
leakage-safe by construction. The honesty of the ensemble (ship it only if *significantly* better
than the best single) is enforced separately by the ``SignificanceTest`` gate in the application.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

_SUM_TOL = 1e-6
_NEG_TOL = -1e-9


@dataclass(frozen=True)
class EnsembleRecipe:
    """A blend recipe: per-member weights, the method that produced it, the member order.

    ``weights`` is a **simplex** (each ``>= 0``, ``sum ~= 1``) keyed by ``member_ids``; the
    ``__post_init__`` validates it and coerces every weight to a python-native ``float`` (not
    ``np.float64``) so report/manifest emission is byte-stable. A degenerate recipe (all mass
    on one member) is still a valid simplex — the application decides whether to ship it.
    """

    weights: dict[str, float]
    method: str
    member_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if set(self.weights) != set(self.member_ids):
            raise ValueError("recipe weights keys must equal member_ids")
        native = {k: float(v) for k, v in self.weights.items()}
        if any(not math.isfinite(v) or v < _NEG_TOL for v in native.values()):
            raise ValueError("recipe weights must be finite and non-negative")
        total = math.fsum(native.values())
        if not math.isclose(total, 1.0, abs_tol=_SUM_TOL):
            raise ValueError(f"recipe weights must sum to 1 (got {total})")
        object.__setattr__(self, "weights", native)


@runtime_checkable
class Ensembler(Protocol):
    """Search the weight simplex over the member OOF to maximize an injected score (Humble Object)."""

    name: str

    def combine(
        self,
        oof: np.ndarray,
        y: np.ndarray,
        *,
        score: Callable[[np.ndarray], float],
        member_ids: Sequence[str],
        random_state: int,
        sample_weight: np.ndarray | None = None,
    ) -> EnsembleRecipe:
        """Return the blend recipe maximizing ``score``.

        ``oof`` is ``(n_models, n_rows[, K])`` — the per-model blend space on the common dense mask
        (binary ``P(positive)`` / regression value: 2-D; multiclass proba: 3-D), already aligned by the
        application. ``score(blended)`` is an application-provided pure scorer returning **higher-is-
        better** (the application has already oriented the metric and projects ``blended`` to it), so the
        adapter is metric-agnostic and never sees raw rows. ``member_ids`` labels axis 0; the adapter must
        be deterministic given ``random_state`` (seeded bagging / fixed SLSQP start).
        """
        ...


__all__ = ["EnsembleRecipe", "Ensembler"]
