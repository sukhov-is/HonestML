"""M4d-2: refinement-based selection (ADR-0031) — run_slice on fake ports, no sklearn estimators.

The fakes return per-row OOF probabilities by row id, so a candidate that is strongly
discriminative but over-confident (high RAW log-loss, low CALIBRATED log-loss) can be pitted
against a weakly discriminative but well-calibrated one — the case where ranking by raw loss
and ranking by refinement disagree.
"""

from __future__ import annotations

import numpy as np
import pytest

from honestml.adapters import IsotonicCalibrator, LogLoss, RocAuc, SigmoidCalibrator
from honestml.application import run_slice
from honestml.core import Fold, SelectionPolicy, Task

pytestmark = pytest.mark.unit


class _Schema:
    def __init__(self) -> None:
        self.features = ["rid"]
        self.numeric = ["rid"]
        self.categorical: list[str] = []
        self.group = None
        self.selected_features = None


class _Dataset:
    def __init__(self, n, y, *, time=None) -> None:
        self._num = np.arange(n, dtype=float).reshape(-1, 1)
        self._codes = np.empty((n, 0), dtype=np.int64)
        self._y = y
        self._time = time
        self._schema = _Schema()

    @property
    def schema(self):
        return self._schema

    def to_numpy(self):
        return self._num

    def categorical_codes(self):
        return self._codes

    def target(self):
        return self._y

    def sample_weight(self):
        return None

    def groups(self):
        return None

    def time(self):
        return self._time


class _RidEstimator:
    """Binary probabilistic fake: P(pos) per row id is read from a table."""

    def __init__(self, proba_table) -> None:
        self.feature_names: list[str] = []
        self._p = proba_table
        self.classes_ = None

    def fit(self, X, y, X_val=None, y_val=None, sample_weight=None):
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return (self._p[X[:, 0].astype(int)] >= 0.5).astype(int)

    def predict_proba(self, X):
        p = self._p[X[:, 0].astype(int)]
        return np.column_stack([1.0 - p, p])


class _Splitter:
    def __init__(self, folds) -> None:
        self._folds = folds


class _KFold(_Splitter):
    def split(self, dataset):
        return iter(self._folds)


class _TSSplitter(_Splitter):
    time_ordered = True

    def split(self, dataset):
        return iter(self._folds)


def _kfold(n, k=5):
    idx = np.arange(n)
    folds = [
        Fold(fit_idx=np.setdiff1d(idx, idx[i::k]), es_idx=np.empty(0, np.int64), test_idx=idx[i::k])
        for i in range(k)
    ]
    return _KFold(folds)


def _scenario(n=400):
    """A = discriminative but over-confident; B = weak but calibrated (raw vs refinement disagree)."""
    rng = np.random.default_rng(0)
    y = np.array([0, 1] * (n // 2))
    a_correct = rng.random(n) < 0.80
    a_proba = np.where(np.where(a_correct, y == 1, y == 0), 0.99, 0.01)
    b_correct = rng.random(n) < 0.65
    b_proba = np.where(np.where(b_correct, y == 1, y == 0), 0.65, 0.35)
    return y, a_proba, b_proba


def _run(
    y,
    estimators,
    *,
    selection="raw",
    calibrator=SigmoidCalibrator,
    min_oof=10,
    k=5,
    significance=None,
    splitter=None,
):
    n = y.shape[0]
    return run_slice(
        _Dataset(n, y),
        Task(kind="binary"),
        estimators=estimators,
        splitter=splitter or _kfold(n, k),
        metric=LogLoss(classes=np.array([0, 1])),
        policy=SelectionPolicy(greater_is_better=False),
        significance_test=significance,
        calibrator_factory=calibrator,
        selection=selection,
        refinement_min_oof=min_oof,
    )


def test_refinement_picks_better_refinement_not_raw() -> None:
    """A wins on calibrated log-loss (refinement), B wins on raw log-loss — modes disagree."""
    y, a, b = _scenario()
    est = {"A": (lambda: _RidEstimator(a)), "B": (lambda: _RidEstimator(b))}
    raw = _run(y, est, selection="raw")
    ref = _run(y, est, selection="refinement")
    assert raw.best_model_id == "B" and raw.selection_mode == "raw"
    assert ref.best_model_id == "A"
    assert ref.selection_mode == "refinement" and ref.score_space == "calibrated_oof"


def test_crossfit_no_insample_optimism() -> None:
    """The refinement score is the OUT-OF-FOLD calibrated loss, not the optimistic in-sample fit."""
    y, a, b = _scenario()
    est = {"A": (lambda: _RidEstimator(a)), "B": (lambda: _RidEstimator(b))}
    ref = _run(y, est, selection="refinement", calibrator=IsotonicCalibrator)
    # in-sample isotonic on extreme proba would near-perfectly memorize -> ~0 loss; cross-fit does not
    a_score = next(e.score for e in ref.leaderboard if e.model_id == "A")
    assert a_score > 0.2  # honest OOF refinement error, not an overfit ~0


def test_ranking_metric_noop_via_gate() -> None:
    """A non-proper metric (roc_auc) never triggers refinement — no-op by the proper_proba gate."""
    y, a, b = _scenario()
    est = {"A": (lambda: _RidEstimator(a)), "B": (lambda: _RidEstimator(b))}
    res = run_slice(
        _Dataset(y.shape[0], y),
        Task(kind="binary"),
        estimators=est,
        splitter=_kfold(y.shape[0]),
        metric=RocAuc(),
        policy=SelectionPolicy(),
        calibrator_factory=SigmoidCalibrator,
        selection="refinement",
        refinement_min_oof=10,
    )
    assert res.selection_mode == "raw" and res.score_space == "raw_oof"


def test_band_scheme_unchanged_non_ts() -> None:
    """B1: the calib cross-fit blocks are NEVER handed to the band (no bootstrap-scheme switch)."""
    y, a, b = _scenario()
    est = {"A": (lambda: _RidEstimator(a)), "B": (lambda: _RidEstimator(b))}
    seen: list[object] = []

    class _RecSig:
        seed = 0
        n_boot = 1

        def equivalent(self, pa, pb, yt, *, alpha, block_index=None, sample_weight=None):
            seen.append(block_index)
            return True

    _run(y, est, selection="refinement", significance=_RecSig())
    assert seen and all(b is None for b in seen)  # i.i.d. row bootstrap, never fold-block


def test_timeseries_disables_refinement() -> None:
    """B2: refinement is disabled under a time-ordered splitter (M4) -> raw fallback."""
    n = 100
    y = np.array([0, 1] * (n // 2))
    a = np.where(y == 1, 0.99, 0.01)
    folds = [
        Fold(fit_idx=np.arange(60), es_idx=np.empty(0, np.int64), test_idx=np.arange(60, 80)),
        Fold(fit_idx=np.arange(80), es_idx=np.empty(0, np.int64), test_idx=np.arange(80, 100)),
    ]
    # the dataset exposes time() for the time-ordered validate_fold; refinement must self-disable
    res = run_slice(
        _Dataset(n, y, time=np.arange(n, dtype=float)),
        Task(kind="binary"),
        estimators={"A": (lambda: _RidEstimator(a)), "B": (lambda: _RidEstimator(np.full(n, 0.6)))},
        splitter=_TSSplitter(folds),
        metric=LogLoss(classes=np.array([0, 1])),
        policy=SelectionPolicy(greater_is_better=False),
        calibrator_factory=SigmoidCalibrator,
        selection="refinement",
        refinement_min_oof=10,
    )
    assert res.selection_mode == "raw"


def test_fallback_signal_min_oof() -> None:
    """ADR-0031 §4b: too few OOF rows for a reliable refinement signal -> raw fallback."""
    y, a, b = _scenario()
    est = {"A": (lambda: _RidEstimator(a)), "B": (lambda: _RidEstimator(b))}
    res = _run(y, est, selection="refinement", min_oof=100_000)
    assert res.selection_mode == "raw"


def test_fallback_per_block_not_viable() -> None:
    """ADR-0031 §4a: a per-block calibration train side below the floor -> raw fallback."""
    n = 60  # 5-fold -> each block's train side is 48 < MIN_CALIB_N(50)
    y = np.array([0, 1] * (n // 2))
    a = np.where(y == 1, 0.99, 0.01)
    est = {"A": (lambda: _RidEstimator(a)), "B": (lambda: _RidEstimator(np.full(n, 0.6)))}
    res = _run(y, est, selection="refinement", min_oof=10)
    assert res.selection_mode == "raw"


def test_isotonic_crossfit_finite_score() -> None:
    """Isotonic on separable proba would emit 0/1 -> inf log-loss; the clip keeps scores finite."""
    n = 400
    y = np.array([0, 1] * (n // 2))
    sep = np.where(y == 1, 1.0, 0.0)  # perfectly separable -> isotonic maps to exactly 0/1
    est = {"A": (lambda: _RidEstimator(sep)), "B": (lambda: _RidEstimator(np.full(n, 0.6)))}
    res = _run(y, est, selection="refinement", calibrator=IsotonicCalibrator)
    assert res.selection_mode == "refinement"
    assert all(np.isfinite(e.score) for e in res.leaderboard)
