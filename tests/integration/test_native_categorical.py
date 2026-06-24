"""WS-B honesty regression: native categorical handling must not leak the target (ADR-0090, NFR-1/4).

On a high-cardinality, weak-signal, small-sample dataset (the regime where leakage/overfit would show),
the honest per-fold native CatBoost OOF must stay near a no-categorical baseline and well below an
"omniscient" target-encoding pre-computed on ALL rows before the split (which DOES leak). Mirrors the
throwaway SPIKE-0004 measurement as a standing guard.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from honestml.adapters.boosting import CATBOOST, build_boosting
from honestml.core import Task

pytestmark = pytest.mark.slow


def _data(n: int = 300, levels: int = 200, seed: int = 0):
    rng = np.random.default_rng(seed)
    cat = rng.integers(0, levels, size=n)  # ~1.5 rows/level: extreme cardinality
    num = rng.normal(size=(n, 3))
    level_effect = rng.normal(size=levels) * 0.4
    signal = 1.2 * num[:, 0] + level_effect[cat] + 0.5 * rng.normal(size=n)
    y = (signal > np.median(signal)).astype(int)
    X = np.hstack([num, cat.astype(np.float64).reshape(-1, 1)])  # cat at column 3
    return X, y, n


def _folds(n: int, k: int = 5):
    return [np.arange(i, n, k) for i in range(k)]


def _oof_native(X, y, n) -> float:
    oof = np.full(n, np.nan)
    for test in _folds(n):
        train = np.setdiff1d(np.arange(n), test)
        est = build_boosting(CATBOOST, task=Task(kind="binary"), random_state=0)
        est.feature_names = ["n0", "n1", "n2", "c0"]
        est.categorical_indices = [3]
        est.fit(X[train], y[train])  # CTR computed strictly from this fold's train rows
        oof[test] = est.predict_proba(X[test])[:, 1]
    return roc_auc_score(y, oof)


def _oof_codes_only(X, y, n) -> float:
    """Numeric-only baseline (categorical dropped) — an honest model that cannot use the category."""
    oof = np.full(n, np.nan)
    for test in _folds(n):
        train = np.setdiff1d(np.arange(n), test)
        est = build_boosting(CATBOOST, task=Task(kind="binary"), random_state=0)
        est.feature_names = ["n0", "n1", "n2"]
        est.fit(
            X[train, :3], y[train]
        )  # no categorical_indices -> codes path, here cat column removed
        oof[test] = est.predict_proba(X[test, :3])[:, 1]
    return roc_auc_score(y, oof)


def _oof_omniscient(X, y, n) -> float:
    """LEAKY baseline: per-level target mean computed on ALL rows (pre-split), used as a feature."""
    cat = X[:, 3].astype(int)
    means = {c: y[cat == c].mean() for c in np.unique(cat)}
    te = np.array([means[c] for c in cat])
    Xl = np.column_stack([X[:, :3], te])
    oof = np.full(n, np.nan)
    for test in _folds(n):
        train = np.setdiff1d(np.arange(n), test)
        est = build_boosting(CATBOOST, task=Task(kind="binary"), random_state=0)
        est.feature_names = ["n0", "n1", "n2", "te"]
        est.fit(Xl[train], y[train])
        oof[test] = est.predict_proba(Xl[test])[:, 1]
    return roc_auc_score(y, oof)


def test_native_categorical_per_fold_does_not_leak_or_overfit() -> None:
    pytest.importorskip("catboost")
    X, y, n = _data()
    honest = _oof_native(X, y, n)
    codes = _oof_codes_only(X, y, n)
    leaky = _oof_omniscient(X, y, n)
    # NFR-1: the honest per-fold native OOF is far below the omniscient pre-split target-encoding leak
    assert honest < leaky - 0.03
    # NFR-4: native handling does not hallucinate signal from the noise-cardinal category (no overfit-inflate)
    assert honest <= codes + 0.05
