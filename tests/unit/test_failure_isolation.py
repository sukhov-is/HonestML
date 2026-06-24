"""M3a (ADR-0022): per-candidate failure isolation in ``run_slice``.

Uses hand-built fake ports (Humble Object, NFR-3): a failing model is isolated,
survivors share OOF coverage, all-fail is a loud error, and a bug in *our* code
(outside the narrow except) propagates instead of being masked as "failed".
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from honestml.adapters import Accuracy, RocAuc
from honestml.application import run_slice
from honestml.core import FitFailedError, Fold, SelectionPolicy, Task

pytestmark = pytest.mark.unit


class _Schema:
    selected_features = None

    def __init__(self) -> None:
        self._group = None

    @property
    def features(self):
        return ["rid"]

    @property
    def numeric(self):
        return ["rid"]

    @property
    def categorical(self):
        return []

    @property
    def group(self):
        return None


class _Dataset:
    def __init__(self, n, y) -> None:
        self._num = np.arange(n, dtype=float).reshape(-1, 1)
        self._codes = np.empty((n, 0), dtype=np.int64)
        self._y = y
        self._schema = _Schema()

    @property
    def schema(self):
        return self._schema

    @property
    def n_rows(self):
        return self._num.shape[0]

    def to_numpy(self):
        return self._num

    def categorical_codes(self):
        return self._codes

    def target(self):
        return self._y

    def sample_weight(self):
        return None


class _Splitter:
    def __init__(self, folds) -> None:
        self._folds = folds

    def split(self, dataset):
        return iter(self._folds)


def _kfold(n, k=5):
    idx = np.arange(n)
    folds = []
    for i in range(k):
        test = idx[i::k]
        train = np.setdiff1d(idx, test)
        folds.append(Fold(fit_idx=train, es_idx=np.empty(0, dtype=np.int64), test_idx=test))
    return _Splitter(folds)


class _Good:
    """A working rid-indexed probabilistic model (fully controlled OOF)."""

    def __init__(self, proba_table) -> None:
        self.feature_names: list[str] = []
        self._proba = proba_table
        self.classes_ = None

    def fit(self, X, y, X_val=None, y_val=None, sample_weight=None):
        self.classes_ = np.unique(y)
        return self

    def _rid(self, X):
        return X[:, 0].astype(int)

    def predict(self, X):
        return (self._proba[self._rid(X)] > 0.5).astype(int)

    def predict_proba(self, X):
        p = self._proba[self._rid(X)]
        return np.column_stack([1.0 - p, p])


class _BoomFit:
    def __init__(self) -> None:
        self.feature_names: list[str] = []
        self.classes_ = None

    def fit(self, X, y, X_val=None, y_val=None, sample_weight=None):
        raise RuntimeError("fit blew up")

    def predict(self, X):  # pragma: no cover - never reached
        return np.zeros(len(X), dtype=int)

    def predict_proba(self, X):  # pragma: no cover
        return np.zeros((len(X), 2))


class _BadPredictShape(_Good):
    """Fits fine; predict returns a non-broadcastable length -> *our* assignment fails."""

    def predict(self, X):
        return np.zeros(len(X) + 1, dtype=int)  # length n+1 cannot broadcast into n slots

    def predict_proba(self, X):
        return np.zeros((len(X) + 1, 2))


def _good(y):
    return lambda: _Good(np.where(y == 1, 0.8, 0.2))


def test_one_candidate_fails_run_completes(caplog) -> None:
    n = 20
    y = np.array([0, 1] * (n // 2))
    estimators = {"bad": _BoomFit, "good": _good(y)}
    with caplog.at_level(logging.WARNING, logger="honestml"):
        res = run_slice(
            _Dataset(n, y),
            Task(kind="binary"),
            estimators=estimators,
            splitter=_kfold(n),
            metric=RocAuc(),
            policy=SelectionPolicy(),
        )
    assert [e.model_id for e in res.leaderboard] == ["good"]
    assert res.best_model_id == "good"
    assert [f.id for f in res.failed] == ["bad"]
    assert "fit blew up" in res.failed[0].reason
    assert any("bad" in r.getMessage() for r in caplog.records)


def test_all_candidates_fail_raises_fitfailed() -> None:
    n = 12
    y = np.array([0, 1] * (n // 2))
    with pytest.raises(FitFailedError, match="all 2 candidate"):
        run_slice(
            _Dataset(n, y),
            Task(kind="binary"),
            estimators={"bad1": _BoomFit, "bad2": _BoomFit},
            splitter=_kfold(n),
            metric=RocAuc(),
            policy=SelectionPolicy(),
        )


def test_survivors_share_oof_mask() -> None:
    n = 20
    y = np.array([0, 1] * (n // 2))
    estimators = {"a": _good(y), "b": _good(y)}
    res = run_slice(
        _Dataset(n, y),
        Task(kind="binary"),
        estimators=estimators,
        splitter=_kfold(n),
        metric=RocAuc(),
        policy=SelectionPolicy(),
    )
    masks = [c.oof_mask for c in res.candidates]
    assert len(masks) == 2
    assert np.array_equal(masks[0], masks[1])  # identical coverage -> fair comparison


def test_our_code_bug_propagates_not_masked() -> None:
    """A wrong-shape predict makes *our* OOF assignment fail; it must NOT be a 'failed' candidate."""
    n = 20
    y = np.array([0, 1] * (n // 2))
    with pytest.raises(
        (ValueError, IndexError)
    ):  # surfaces; not swallowed into FitFailedError/failed
        run_slice(
            _Dataset(n, y),
            Task(kind="binary"),
            estimators={"bad_shape": lambda: _BadPredictShape(np.where(y == 1, 0.8, 0.2))},
            splitter=_kfold(n),
            metric=Accuracy(),
            policy=SelectionPolicy(),
        )


def test_successful_run_has_empty_failed() -> None:
    n = 16
    y = np.array([0, 1] * (n // 2))
    res = run_slice(
        _Dataset(n, y),
        Task(kind="binary"),
        estimators={"good": _good(y)},
        splitter=_kfold(n),
        metric=RocAuc(),
        policy=SelectionPolicy(),
    )
    assert res.failed == []
