"""Ensembler adapters + the ``BlendedEstimator`` artifact (ADR-0063 §3 / ADR-0064 §1).

Two backend-neutral weight searches over the member OOF blend space (the application injects a pure
higher-is-better ``score`` and the ``oof`` array, so these adapters are metric-agnostic and never see
raw rows): :class:`CaruanaEnsembler` (the default — greedy selection with replacement + seeded bagging,
deterministic smallest-index tie-break) and :class:`WeightedEnsembler` (SLSQP over the simplex, ported
from the legacy ``_optimize_blend_weights``). :class:`BlendedEstimator` is the shipped artifact: it
implements the ``Estimator``/``ProbabilisticEstimator`` protocol structurally, so the inference path and
the single ``model.joblib`` serialization are unchanged (ADR-0064 §1/§2). ``scipy`` (SLSQP) rides in
with sklearn; Caruana and the blend are pure numpy.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from honestml.core import EnsembleRecipe, Estimator

_PROBA_EPS = 1e-6  # mirror of application.align_proba: adapters cannot import the application layer


def _blend(oof: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted combination of member predictions along axis 0 ((m,n)->(n,), (m,n,K)->(n,K)).

    Mirror of ``application.ensemble._blend``: duplicated (one numpy line) because an adapter cannot
    import the application layer (the dependency rule), like ``_align_member_proba`` below.
    """
    return np.tensordot(np.asarray(weights, dtype=np.float64), oof, axes=([0], [0]))


class CaruanaEnsembler:
    """Greedy ensemble selection with replacement + seeded bagging (Caruana 2004; ADR-0063 §3).

    Each bag runs a greedy hill-climb over a seeded library subset, keeping the best running ensemble
    seen (so the result is never worse than the best single member); the final weights are the bagged
    selection frequencies. Ties in the incremental score break to the **smallest member index**
    (deterministic), and bagging is seeded from ``random_state`` (NFR-M7-2).
    """

    name = "caruana"

    def __init__(self, *, size: int = 50, n_bags: int = 20) -> None:
        self.size = size
        self.n_bags = n_bags

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
        ids = tuple(member_ids)
        m = oof.shape[0]
        if m == 1:
            return EnsembleRecipe({ids[0]: 1.0}, self.name, ids)
        rng = np.random.default_rng(random_state)
        counts = np.zeros(m, dtype=np.float64)
        for _ in range(self.n_bags):
            lib = self._bag(rng, m)
            counts += self._greedy(oof, score, lib)
        total = counts.sum()
        weights = counts / total if total > 0 else np.full(m, 1.0 / m)
        return EnsembleRecipe({mid: float(w) for mid, w in zip(ids, weights)}, self.name, ids)

    def _bag(self, rng: np.random.Generator, m: int) -> np.ndarray:
        """A sorted library subset for one bag; >= 2 models so a bag can blend (ADR-0063 §3)."""
        if self.n_bags == 1:
            return np.arange(m)
        k = min(m, max(2, int(round(0.5 * m))))
        return np.sort(rng.choice(m, size=k, replace=False))

    def _greedy(
        self, oof: np.ndarray, score: Callable[[np.ndarray], float], lib: np.ndarray
    ) -> np.ndarray:
        """Selection counts of the best running ensemble over ``size`` greedy steps (best-step kept)."""
        m = oof.shape[0]
        counts = np.zeros(m, dtype=np.float64)
        best_counts = np.zeros(m, dtype=np.float64)
        best_score = -np.inf
        running: np.ndarray | None = None
        n_sel = 0
        for _ in range(self.size):
            # smallest-index tie-break: lib is sorted and argmax returns the first max (ADR-0063 §3)
            cand = [
                score(oof[j] if running is None else (running + oof[j]) / (n_sel + 1)) for j in lib
            ]
            pick = int(lib[int(np.argmax(cand))])
            running = oof[pick].astype(np.float64) if running is None else running + oof[pick]
            n_sel += 1
            counts[pick] += 1
            s = score(running / n_sel)
            if s > best_score:
                best_score = s
                best_counts = counts.copy()
        return best_counts


class WeightedEnsembler:
    """SLSQP over the weight simplex (ported from legacy ``_optimize_blend_weights``; ADR-0063 §3).

    Deterministic start (``x0 = 1/m``); BLAS/scipy float drift across environments makes this a
    time-mode-level (non-load-bearing) source of non-determinism — the default path is Caruana (§3).
    """

    name = "weighted"

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
        from scipy.optimize import minimize

        ids = tuple(member_ids)
        m = oof.shape[0]
        if m == 1:
            return EnsembleRecipe({ids[0]: 1.0}, self.name, ids)
        x0 = np.full(m, 1.0 / m)
        result = minimize(
            lambda w: -score(_blend(oof, w)),
            x0=x0,
            method="SLSQP",
            bounds=[(0.0, 1.0)] * m,
            constraints={"type": "eq", "fun": lambda w: w.sum() - 1.0},
        )
        w = np.clip(np.asarray(result.x, dtype=np.float64), 0.0, None)
        total = w.sum()
        w = w / total if total > 0 else x0
        return EnsembleRecipe({mid: float(wi) for mid, wi in zip(ids, w)}, self.name, ids)


def _align_member_proba(
    raw: np.ndarray, est_classes: np.ndarray, classes: np.ndarray
) -> np.ndarray:
    """Reindex a member's ``predict_proba`` to the global class order (mirror of application.align_proba).

    Duplicated here (numpy-only) because an adapter cannot import the application layer (the dependency
    rule); a class absent from a member gets ``_PROBA_EPS`` mass then the row is renormalized so blending
    stays a valid distribution (ADR-0064 §1).
    """
    n = raw.shape[0]
    aligned = np.full((n, classes.size), _PROBA_EPS, dtype=np.float64)
    positions = {label: j for j, label in enumerate(classes.tolist())}
    for src, label in enumerate(est_classes.tolist()):
        target = positions.get(label)
        if target is not None:
            aligned[:, target] = raw[:, src]
    aligned /= aligned.sum(axis=1, keepdims=True)
    return aligned


class BlendedEstimator:
    """A weighted blend of fitted members, opaque to the inference path (``Estimator``; ADR-0064 §1).

    ``classes_`` MUST equal the global class order (``FittedModel.classes``), so the artifact's
    ``_aligned_proba``/``_positive_index`` stay identity-reindex (ADR-0064 §1). ``predict_proba`` returns
    the full ``(n, K)`` (binary ``(n, 2)``, both columns, rows sum to 1) because the artifact indexes
    ``proba[:, pos]``. ``predict`` is argmax of the blended proba (classification) / a weighted mean of
    the members' ``predict`` (regression). Members are independent; ``fit`` refits each on the same data.
    """

    def __init__(
        self,
        members: Sequence[Estimator],
        weights: np.ndarray,
        classes: np.ndarray | None,
    ) -> None:
        self.members = list(members)
        self.weights = np.asarray(weights, dtype=np.float64)
        self.classes_ = classes if classes is None else np.asarray(classes)
        self.feature_names: list[str] = list(self.members[0].feature_names) if self.members else []

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> BlendedEstimator:
        for est in self.members:
            est.feature_names = self.feature_names
            est.fit(X, y, sample_weight=sample_weight)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.classes_ is None:  # regression: weighted mean of member point predictions
            preds = np.stack([est.predict(X) for est in self.members], axis=0)
            return _blend(preds, self.weights)
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.classes_ is None:
            raise ValueError("regression BlendedEstimator has no probabilities")
        acc = np.zeros((X.shape[0], self.classes_.size), dtype=np.float64)
        for est, w in zip(self.members, self.weights):
            raw = np.asarray(est.predict_proba(X), dtype=np.float64)  # type: ignore[attr-defined]
            acc += w * _align_member_proba(raw, np.asarray(est.classes_), self.classes_)  # type: ignore[attr-defined]
        return acc / acc.sum(axis=1, keepdims=True)
