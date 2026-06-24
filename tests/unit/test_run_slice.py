"""M2-4: the run_slice use-case on fake ports (ADR-0010, NFR-3).

The use-case is a Humble Object: it is exercised with hand-built fake ports and a
fake Dataset, no sklearn/polars, no I/O. Fakes read a row-id feature column so
per-row OOF outputs are fully controlled and the tests are deterministic.
"""

from __future__ import annotations

import numpy as np
import pytest

from honestml.adapters import Accuracy, LogLoss, RocAuc
from honestml.application import (
    FeatureSelectionBundle,
    align_proba,
    design_matrix,
    project_for_metric,
    refit_best,
    run_slice,
)
from honestml.core import (
    BudgetExhaustedError,
    CategoryTable,
    ConfigError,
    FeatureSelectionConfig,
    FEConfig,
    FitFailedError,
    Fold,
    NoSignificanceTest,
    SchemaValidationError,
    SelectionPolicy,
    TargetEncodingSpec,
    Task,
)
from honestml.core.schema import native_routable

pytestmark = pytest.mark.unit


class _Schema:
    selected_features = None

    def __init__(self, numeric, categorical, group=None, *, cat_cardinality=3) -> None:
        self._numeric = numeric
        self._categorical = categorical
        self._group = group
        # category tables so the native-cardinality gate (native_routable) can read len(categories);
        # low-card by default (below any cap) -> the fakes' categoricals route natively as before.
        self.categories = {
            c: CategoryTable.fit([str(i) for i in range(cat_cardinality)]) for c in categorical
        }

    @property
    def features(self):
        return self._numeric + self._categorical

    @property
    def numeric(self):
        return self._numeric

    @property
    def categorical(self):
        return self._categorical

    def categorical_indices(self, cap=None):
        feats = self.features  # the fakes never carry a selected subset on the schema itself
        cat = set(native_routable(self, cap))  # mirror the real schema: gate by cardinality cap
        return [i for i, f in enumerate(feats) if f in cat]

    @property
    def group(self):
        return self._group


class _Dataset:
    """Minimal Dataset whose only numeric feature is the row id (column 0)."""

    def __init__(self, n, y, *, sample_weight=None, group=None, groups=None) -> None:
        self._num = np.arange(n, dtype=float).reshape(-1, 1)
        self._codes = np.empty((n, 0), dtype=np.int64)
        self._y = y
        self._sw = sample_weight
        self._groups = groups
        self._schema = _Schema(["rid"], [], group)

    @property
    def schema(self):
        return self._schema

    @property
    def n_rows(self):
        return self._num.shape[0]

    def take(self, idx):
        idx = np.asarray(idx, dtype=int)
        d = _Dataset.__new__(_Dataset)
        d._num = self._num[idx]
        d._codes = self._codes[idx]
        d._y = self._y[idx] if self._y is not None else None
        d._sw = self._sw[idx] if self._sw is not None else None
        d._groups = self._groups[idx] if self._groups is not None else None
        d._schema = self._schema
        return d

    def to_numpy(self):
        return self._num

    def categorical_codes(self):
        return self._codes

    def target(self):
        return self._y

    def sample_weight(self):
        return self._sw

    def groups(self):
        return self._groups


class _RidEstimator:
    """Returns per-row outputs by row id, so OOF is fully controlled."""

    def __init__(self, proba_table, class_table) -> None:
        self.feature_names: list[str] = []
        self._proba = proba_table
        self._class = class_table
        self.fit_rows = 0
        self.classes_ = None

    def fit(self, X, y, X_val=None, y_val=None, sample_weight=None):
        self.fit_rows = X.shape[0]
        self.classes_ = np.unique(y)
        return self

    def _rid(self, X):
        return X[:, 0].astype(int)

    def predict(self, X):
        return self._class[self._rid(X)]

    def predict_proba(self, X):
        p = self._proba[self._rid(X)]
        return np.column_stack([1.0 - p, p])


class _ClassOnly:
    def __init__(self) -> None:
        self.feature_names: list[str] = []
        self.classes_ = None

    def fit(self, X, y, X_val=None, y_val=None, sample_weight=None):
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return np.zeros(X.shape[0], dtype=int)


class _MulticlassEstimator:
    """Probabilistic K-class fake: per-row proba over classes_, controlled for OOF."""

    def __init__(self) -> None:
        self.feature_names: list[str] = []
        self.classes_ = None

    def fit(self, X, y, X_val=None, y_val=None, sample_weight=None):
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def predict_proba(self, X):
        rid = X[:, 0].astype(int)
        k = len(self.classes_)
        p = np.full((len(rid), k), 0.1)
        p[np.arange(len(rid)), rid % k] = 1.0
        return p / p.sum(axis=1, keepdims=True)


class _Splitter:
    def __init__(self, folds) -> None:
        self._folds = folds

    def split(self, dataset):
        return iter(self._folds)


class _GroupSplitter(_Splitter):
    """A group-aware fake: marks itself so run_slice runs the group-leakage check."""

    group_aware = True


def _kfold(n, k=5):
    idx = np.arange(n)
    folds = []
    for i in range(k):
        test = idx[i::k]
        train = np.setdiff1d(idx, test)
        folds.append(Fold(fit_idx=train, es_idx=np.empty(0, dtype=np.int64), test_idx=test))
    return _Splitter(folds)


# --- project_for_metric ----------------------------------------------------


def test_project_for_metric_branches() -> None:
    proba = np.array([0.2, 0.8])
    pred = np.array([0, 1])
    assert np.array_equal(project_for_metric(RocAuc(), proba=proba, pred=pred), proba)
    assert np.array_equal(project_for_metric(Accuracy(), proba=proba, pred=pred), pred)
    with pytest.raises(ConfigError, match="needs probabilities"):
        project_for_metric(RocAuc(), proba=None, pred=pred)


def test_project_for_metric_multiclass_requires_2d() -> None:
    with pytest.raises(ConfigError, match="2-D"):
        project_for_metric(
            RocAuc(), proba=np.array([0.2, 0.8]), pred=np.array([0, 1]), kind="multiclass"
        )


# --- align_proba (pure numpy, ADR-0021 §2) ---------------------------------


def test_align_proba_reorders_to_global() -> None:
    # est_classes [2, 0] -> global [0, 1, 2]: class 1 absent in this fold -> ε mass
    out = align_proba(np.array([[0.3, 0.7]]), np.array([2, 0]), np.array([0, 1, 2]))
    assert out.shape == (1, 3)
    assert np.isclose(out.sum(), 1.0)
    assert out[0, 0] > out[0, 2] > out[0, 1]  # class0=0.7, class2=0.3, class1=ε


def test_align_proba_missing_class_smoothed_rows_sum_one() -> None:
    out = align_proba(np.array([[0.5, 0.5], [0.9, 0.1]]), np.array([0, 1]), np.array([0, 1, 2]))
    assert out.shape == (2, 3)
    assert np.allclose(out.sum(axis=1), 1.0)  # valid distributions, not 0-columns
    assert (out[:, 2] > 0).all() and (out[:, 2] < 1e-3).all()  # tiny ε, never literal 0


# --- run_slice orchestration ----------------------------------------------


def test_metric_changes_winner() -> None:
    """RocAuc picks the good-ranking model; Accuracy picks the good-classifying one."""
    n = 20
    y = np.array([0, 1] * (n // 2))
    perfect_rank = np.where(y == 1, 0.9, 0.1)  # AUC 1.0, but...
    wrong_class = np.zeros(n, dtype=int)  # ...predicts all-0 -> 50% accuracy
    flat_rank = np.full(n, 0.5)  # AUC 0.5...
    perfect_class = y.copy()  # ...but 100% accuracy
    estimators = {
        "ranker": (lambda: _RidEstimator(perfect_rank, wrong_class)),
        "classifier": (lambda: _RidEstimator(flat_rank, perfect_class)),
    }
    ds = _Dataset(n, y)
    auc_res = run_slice(
        ds,
        Task(kind="binary"),
        estimators=estimators,
        splitter=_kfold(n),
        metric=RocAuc(),
        policy=SelectionPolicy(),
    )
    acc_res = run_slice(
        ds,
        Task(kind="binary"),
        estimators=estimators,
        splitter=_kfold(n),
        metric=Accuracy(),
        policy=SelectionPolicy(),
    )
    assert auc_res.best_model_id == "ranker"
    assert acc_res.best_model_id == "classifier"


def test_multiclass_oof_shape_and_score() -> None:
    n = 30
    y = np.tile([0, 1, 2], n // 3)  # k=5 striding over tile-3 -> every fold sees all classes
    est = {"m": _MulticlassEstimator}
    res = run_slice(
        _Dataset(n, y),
        Task(kind="multiclass"),
        estimators=est,
        splitter=_kfold(n, k=5),
        metric=LogLoss(classes=np.array([0, 1, 2])),
        policy=SelectionPolicy(greater_is_better=False),
    )
    cand = res.candidates[0]
    assert cand.oof_pred is not None and cand.oof_pred.shape == (n, 3)  # (n, K) OOF
    assert np.allclose(cand.oof_pred[cand.oof_mask].sum(axis=1), 1.0)
    assert isinstance(res.leaderboard[0].score, float)


def test_deterministic_leaderboard() -> None:
    n = 20
    y = np.array([0, 1] * (n // 2))
    est = {"m": (lambda: _RidEstimator(np.where(y == 1, 0.8, 0.2), y.copy()))}
    ds = _Dataset(n, y)
    kw = dict(estimators=est, splitter=_kfold(n), metric=RocAuc(), policy=SelectionPolicy())
    a = run_slice(ds, Task(kind="binary"), **kw)
    b = run_slice(ds, Task(kind="binary"), **kw)
    assert [e.score for e in a.leaderboard] == [e.score for e in b.leaderboard]
    assert a.leaderboard[0].rank == 1


def test_leaderboard_entry_fields() -> None:
    n = 20
    y = np.array([0, 1] * (n // 2))
    est = {"m": (lambda: _RidEstimator(np.where(y == 1, 0.8, 0.2), y.copy()))}
    res = run_slice(
        _Dataset(n, y),
        Task(kind="binary"),
        estimators=est,
        splitter=_kfold(n),
        metric=RocAuc(),
        policy=SelectionPolicy(),
        significance_test=NoSignificanceTest(),
    )
    entry = res.leaderboard[0]
    assert entry.model_id == "m"
    assert entry.metric == "roc_auc"
    assert entry.n_features == 1


def test_fit_uses_fit_plus_es_tail() -> None:
    """Non-ES models train on fit_idx ∪ es_idx — the es tail is not lost (ADR-0010 §6)."""
    n = 6
    y = np.array([0, 1, 0, 1, 0, 1])
    created = []

    def factory():
        e = _RidEstimator(np.where(y == 1, 0.8, 0.2), y.copy())
        created.append(e)
        return e

    fold = Fold(fit_idx=np.array([0, 1]), es_idx=np.array([2, 3]), test_idx=np.array([4, 5]))
    run_slice(
        _Dataset(n, y),
        Task(kind="binary"),
        estimators={"m": factory},
        splitter=_Splitter([fold]),
        metric=RocAuc(),
        policy=SelectionPolicy(),
    )
    assert created[0].fit_rows == 4  # 2 fit + 2 es


def test_es_capable_model_holds_out_es_tail() -> None:
    """ADR-0080: an ES-capable model trains on fit only and gets the es tail as X_val."""
    n = 8
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    captured: dict[str, int | None] = {}

    class _ESEstimator(_RidEstimator):
        supports_early_stopping = True

        def fit(self, X, y, X_val=None, y_val=None, sample_weight=None):
            captured["fit_rows"] = X.shape[0]
            captured["val_rows"] = None if X_val is None else X_val.shape[0]
            return super().fit(X, y, X_val=X_val, y_val=y_val, sample_weight=sample_weight)

    fold = Fold(fit_idx=np.array([0, 1, 2, 3]), es_idx=np.array([4, 5]), test_idx=np.array([6, 7]))
    run_slice(
        _Dataset(n, y),
        Task(kind="binary"),
        estimators={"m": (lambda: _ESEstimator(np.where(y == 1, 0.8, 0.2), y.copy()))},
        splitter=_Splitter([fold]),
        metric=Accuracy(),
        policy=SelectionPolicy(),
    )
    assert captured["fit_rows"] == 4  # fit only — es NOT merged into training
    assert captured["val_rows"] == 2  # es tail routed as the validation set


def test_non_probabilistic_with_proba_metric_raises() -> None:
    n = 20
    y = np.array([0, 1] * (n // 2))
    with pytest.raises(ConfigError, match="needs probabilities"):
        run_slice(
            _Dataset(n, y),
            Task(kind="binary"),
            estimators={"c": _ClassOnly},
            splitter=_kfold(n),
            metric=RocAuc(),
            policy=SelectionPolicy(),
        )


def test_group_dataset_runs_with_group_aware_splitter() -> None:
    # ADR-0023: the M2 group-rejection is lifted; a group-aware splitter completes
    n = 6
    y = np.array([0, 1, 0, 1, 0, 1])
    groups = np.array([0, 0, 1, 1, 2, 2])
    ds = _Dataset(n, y, groups=groups)
    fold = Fold(  # group-disjoint: fit groups {0,1}, test group {2}
        fit_idx=np.array([0, 1, 2, 3]),
        es_idx=np.empty(0, dtype=np.int64),
        test_idx=np.array([4, 5]),
    )
    res = run_slice(
        ds,
        Task(kind="binary"),
        estimators={"m": (lambda: _RidEstimator(np.where(y == 1, 0.8, 0.2), y.copy()))},
        splitter=_GroupSplitter([fold]),
        metric=Accuracy(),
        policy=SelectionPolicy(),
    )
    assert res.best_model_id == "m"


def test_group_leakage_in_fold_is_rejected() -> None:
    # a group spanning fit and test is caught by validate_fold(groups=...) for group-aware splitters
    n = 6
    y = np.array([0, 1, 0, 1, 0, 1])
    groups = np.array([0, 0, 1, 1, 2, 2])
    ds = _Dataset(n, y, groups=groups)
    leaky = Fold(  # group 0 in both fit (idx 0) and test (idx 1)
        fit_idx=np.array([0, 2, 3]),
        es_idx=np.empty(0, dtype=np.int64),
        test_idx=np.array([1, 4, 5]),
    )
    with pytest.raises(SchemaValidationError, match="group leakage"):
        run_slice(
            ds,
            Task(kind="binary"),
            estimators={"m": _ClassOnly},
            splitter=_GroupSplitter([leaky]),
            metric=Accuracy(),
            policy=SelectionPolicy(),
        )


def test_single_class_target_rejected() -> None:
    n = 8
    y = np.zeros(n, dtype=int)
    with pytest.raises(SchemaValidationError, match="2 classes"):
        run_slice(
            _Dataset(n, y),
            Task(kind="binary"),
            estimators={"m": _ClassOnly},
            splitter=_kfold(n, k=2),
            metric=Accuracy(),
            policy=SelectionPolicy(),
        )


def test_refit_best_uses_all_rows() -> None:
    n = 10
    y = np.array([0, 1] * 5)
    created = []

    def factory():
        e = _RidEstimator(np.where(y == 1, 0.8, 0.2), y.copy())
        created.append(e)
        return e

    est = refit_best(_Dataset(n, y), Task(kind="binary"), factory=factory)
    assert created[-1].fit_rows == n
    assert est is created[-1]


def test_oof_mask_populated_for_proba_metric() -> None:
    n = 20
    y = np.array([0, 1] * (n // 2))
    est = {"m": (lambda: _RidEstimator(np.where(y == 1, 0.8, 0.2), y.copy()))}
    res = run_slice(
        _Dataset(n, y),
        Task(kind="binary"),
        estimators=est,
        splitter=_kfold(n),
        metric=RocAuc(),
        policy=SelectionPolicy(),
    )
    cand = res.candidates[0]
    assert cand.oof_pred is not None and cand.oof_mask is not None
    assert cand.oof_mask.all()  # full KFold partition -> every row has an OOF prediction
    assert not np.isnan(cand.oof_pred[cand.oof_mask]).any()


def test_oof_pred_skipped_for_class_metric() -> None:
    """No significance test + a class metric -> proba (oof_pred) is not computed."""
    n = 20
    y = np.array([0, 1] * (n // 2))
    est = {"m": (lambda: _RidEstimator(np.where(y == 1, 0.8, 0.2), y.copy()))}
    res = run_slice(
        _Dataset(n, y),
        Task(kind="binary"),
        estimators=est,
        splitter=_kfold(n),
        metric=Accuracy(),
        policy=SelectionPolicy(),
    )
    cand = res.candidates[0]
    assert cand.oof_pred is None and cand.oof_mask is None


def test_design_matrix_requires_features() -> None:
    class _Empty(_Dataset):
        def __init__(self):
            super().__init__(4, np.array([0, 1, 0, 1]))
            self._num = np.empty((4, 0))

    with pytest.raises(SchemaValidationError, match="no model features"):
        design_matrix(_Empty())


# --- M4a1: OOF capture for non-proba metrics under an active band (FR-M4-2) -


class _RidRegressor:
    """Returns per-row continuous values by row id, so the value-OOF is controlled."""

    def __init__(self, table) -> None:
        self.feature_names: list[str] = []
        self._t = table
        self.classes_ = None

    def fit(self, X, y, X_val=None, y_val=None, sample_weight=None):
        return self

    def predict(self, X):
        return self._t[X[:, 0].astype(int)]


def test_band_active_on_regression_rmse() -> None:
    """An active test captures the value OOF for a regression (RMSE) slice and runs the band."""
    from honestml.adapters import BootstrapSignificanceTest, Rmse

    n = 40
    y = np.linspace(0.0, 1.0, n)
    est = {"m1": (lambda: _RidRegressor(y + 0.01)), "m2": (lambda: _RidRegressor(y - 0.01))}
    res = run_slice(
        _Dataset(n, y),
        Task(kind="regression"),
        estimators=est,
        splitter=_kfold(n),
        metric=Rmse(),
        policy=SelectionPolicy(greater_is_better=False),
        significance_test=BootstrapSignificanceTest(Rmse(), seed=1, n_boot=1000),
    )
    cand = res.candidates[0]
    assert cand.oof_pred is not None and cand.oof_mask is not None
    assert cand.oof_pred.dtype.kind == "f"  # value vector, not proba
    assert not np.isnan(cand.oof_pred[cand.oof_mask]).any()


def test_oof_captured_for_accuracy() -> None:
    """An active test captures the class OOF (int labels) for accuracy; validity by mask, not isnan."""
    from honestml.adapters import Accuracy, BootstrapSignificanceTest

    n = 40
    y = np.array([0, 1] * (n // 2))
    est = {
        "m1": (lambda: _RidEstimator(np.where(y == 1, 0.8, 0.2), y.copy())),
        "m2": (lambda: _RidEstimator(np.where(y == 1, 0.7, 0.3), y.copy())),
    }
    res = run_slice(
        _Dataset(n, y),
        Task(kind="binary"),
        estimators=est,
        splitter=_kfold(n),
        metric=Accuracy(),
        policy=SelectionPolicy(),
        significance_test=BootstrapSignificanceTest(Accuracy(), seed=1, n_boot=1000),
    )
    cand = res.candidates[0]
    assert cand.oof_pred is not None and cand.oof_mask is not None
    assert cand.oof_pred.dtype.kind in ("i", "u")  # class labels, not proba floats
    assert cand.oof_mask.all()  # full KFold partition; no isnan on int labels


# --- M4a2: instability flag emitted via SliceResult (FR-M4-3) ----------------


def _proba_model(table: np.ndarray):
    return lambda: _RidEstimator(np.clip(table, 0.01, 0.99), np.zeros(len(table), dtype=int))


def test_instability_flag_on_near_tie() -> None:
    """Two near-equivalent models -> runner-up in band -> band_unstable=True."""
    from honestml.adapters import BootstrapSignificanceTest, RocAuc

    rng = np.random.default_rng(0)
    n = 40
    y = np.array([0, 1] * (n // 2))
    base = np.where(y == 1, 0.7, 0.3)
    est = {
        "a": _proba_model(base + rng.normal(0, 0.05, n)),
        "b": _proba_model(base + rng.normal(0, 0.05, n)),
    }
    res = run_slice(
        _Dataset(n, y),
        Task(kind="binary"),
        estimators=est,
        splitter=_kfold(n),
        metric=RocAuc(),
        policy=SelectionPolicy(),
        significance_test=BootstrapSignificanceTest(RocAuc(), seed=1, n_boot=1000),
    )
    assert res.band_unstable is True
    assert set(res.band_member_ids) == {"a", "b"} and res.band_width == 2


def test_no_flag_on_clear_winner() -> None:
    """A clear winner (runner-up AUC≈0.5) -> runner-up excluded -> band_unstable=False."""
    from honestml.adapters import BootstrapSignificanceTest, RocAuc

    rng = np.random.default_rng(0)
    n = 40
    y = np.array([0, 1] * (n // 2))
    strong = np.where(y == 1, 0.7, 0.3) + rng.normal(0, 0.08, n)
    weak = 0.5 + rng.normal(0, 0.02, n)
    est = {"strong": _proba_model(strong), "weak": _proba_model(weak)}
    res = run_slice(
        _Dataset(n, y),
        Task(kind="binary"),
        estimators=est,
        splitter=_kfold(n),
        metric=RocAuc(),
        policy=SelectionPolicy(),
        significance_test=BootstrapSignificanceTest(RocAuc(), seed=1, n_boot=1000),
    )
    assert res.band_unstable is False
    assert res.band_member_ids == ("strong",) and res.band_width == 1
    assert res.best_model_id == "strong"


# --- M5a-engine: cooperative budget gate + graceful degradation (FR-M5-1/2/6, ADR-0032) ---


class _TrialBudget:
    """Fake Budget: exhausts after ``n`` completed (consume) trials; seconds ignored."""

    mode = "trials"

    def __init__(self, n: int) -> None:
        self._n = n
        self._done = 0

    def time_left(self) -> float:
        return float("inf")

    def consume(self, seconds: float) -> None:
        self._done += 1

    @property
    def exhausted(self) -> bool:
        return self._done >= self._n

    @property
    def exhausted_reason(self) -> str | None:
        return "trials" if self._done >= self._n else None

    def memory_left(self) -> float | None:
        return None


class _ExhaustAfter:
    """Fake Budget: returns not-exhausted for the first ``checks`` reads, then exhausted."""

    mode = "time"

    def __init__(self, checks: int) -> None:
        self._left = checks

    def time_left(self) -> float:
        return 0.0

    def consume(self, seconds: float) -> None:
        pass

    @property
    def exhausted(self) -> bool:
        if self._left <= 0:
            return True
        self._left -= 1
        return False

    @property
    def exhausted_reason(self) -> str | None:
        return "time" if self._left <= 0 else None

    def memory_left(self) -> float | None:
        return None


class _MemoryExhaustAfter:
    """Fake Budget exhausting by MEMORY after ``checks`` not-exhausted reads (psutil-free).

    Mirrors ``_ExhaustAfter`` but composes under ``mode="none"`` and reports the memory axis — the
    orthogonal memory limit (ADR-0039). ``exhausted_reason`` is side-effect free (read after the loop).
    """

    mode = "none"

    def __init__(self, checks: int) -> None:
        self._left = checks

    def time_left(self) -> float:
        return float("inf")

    def consume(self, seconds: float) -> None:
        pass

    @property
    def exhausted(self) -> bool:
        if self._left <= 0:
            return True
        self._left -= 1
        return False

    @property
    def exhausted_reason(self) -> str | None:
        return "memory" if self._left <= 0 else None

    def memory_left(self) -> float | None:
        return None


class _Boom:
    """An estimator whose fit always raises (a per-candidate failure, ADR-0022)."""

    def __init__(self) -> None:
        self.feature_names: list[str] = []
        self.classes_ = None

    def fit(self, X, y, X_val=None, y_val=None, sample_weight=None):
        raise ValueError("boom")

    def predict(self, X):  # pragma: no cover - never reached (fit raises)
        return np.zeros(X.shape[0], dtype=int)


def _three_estimators(y):
    def make(table):
        return lambda: _RidEstimator(table, y.copy())

    return {
        "a": make(np.where(y == 1, 0.9, 0.1)),
        "b": make(np.where(y == 1, 0.8, 0.2)),
        "c": make(np.where(y == 1, 0.7, 0.3)),
    }


def test_trials_budget_caps_candidates() -> None:
    n = 20
    y = np.array([0, 1] * (n // 2))
    res = run_slice(
        _Dataset(n, y),
        Task(kind="binary"),
        estimators=_three_estimators(y),
        splitter=_kfold(n),
        metric=Accuracy(),
        policy=SelectionPolicy(),
        budget=_TrialBudget(2),
    )
    assert len(res.leaderboard) == 2  # exactly 2 completed
    assert res.budget.exhausted is True
    assert res.budget.skipped == ("c",)  # the 3rd is skipped, never started
    assert res.budget.exhausted_by == "trials"  # truthful axis (port exhausted_reason, ADR-0039 §5)


def test_no_budget_runs_all() -> None:
    n = 20
    y = np.array([0, 1] * (n // 2))
    res = run_slice(
        _Dataset(n, y),
        Task(kind="binary"),
        estimators=_three_estimators(y),
        splitter=_kfold(n),
        metric=Accuracy(),
        policy=SelectionPolicy(),
        budget=None,
    )
    assert len(res.leaderboard) == 3
    assert res.budget.exhausted is False
    assert res.budget.skipped == ()
    assert res.budget.exhausted_by is None  # within budget -> no exhausted axis


def test_overshoot_bounded_one_candidate() -> None:
    """Once exhausted, no further candidate starts: only the in-flight one completes."""
    n = 20
    y = np.array([0, 1] * (n // 2))
    res = run_slice(
        _Dataset(n, y),
        Task(kind="binary"),
        estimators=_three_estimators(y),
        splitter=_kfold(n),
        metric=Accuracy(),
        policy=SelectionPolicy(),
        budget=_ExhaustAfter(1),  # first gate passes, then exhausted
    )
    assert len(res.leaderboard) == 1
    assert set(res.budget.skipped) == {"b", "c"}
    assert res.budget.exhausted is True


def test_failed_candidate_does_not_consume_trial() -> None:
    """A failing candidate does not burn a trial (a trial = a completed candidate)."""
    n = 20
    y = np.array([0, 1] * (n // 2))
    estimators = {
        "good1": (lambda: _RidEstimator(np.where(y == 1, 0.9, 0.1), y.copy())),
        "bad": _Boom,
        "good2": (lambda: _RidEstimator(np.where(y == 1, 0.8, 0.2), y.copy())),
        "good3": (lambda: _RidEstimator(np.where(y == 1, 0.7, 0.3), y.copy())),
    }
    res = run_slice(
        _Dataset(n, y),
        Task(kind="binary"),
        estimators=estimators,
        splitter=_kfold(n),
        metric=Accuracy(),
        policy=SelectionPolicy(),
        budget=_TrialBudget(2),
    )
    # good1 (trial 1), bad (failed, no trial), good2 (trial 2), good3 skipped
    assert {e.model_id for e in res.leaderboard} == {"good1", "good2"}
    assert res.budget.skipped == ("good3",)
    assert res.budget.exhausted is True


def test_degraded_returns_best_so_far() -> None:
    """>=1 completed on exhaustion -> success, winner from the completed subset (no raise)."""
    n = 20
    y = np.array([0, 1] * (n // 2))
    res = run_slice(
        _Dataset(n, y),
        Task(kind="binary"),
        estimators=_three_estimators(y),
        splitter=_kfold(n),
        metric=Accuracy(),
        policy=SelectionPolicy(),
        budget=_TrialBudget(2),
    )
    assert res.best_model_id in {"a", "b"}  # never "c" (skipped)


def test_mixed_zero_completed_raises_budget_exhausted() -> None:
    """0 completed with skipped candidates -> BudgetExhaustedError, not FitFailedError."""
    n = 20
    y = np.array([0, 1] * (n // 2))
    estimators = {
        "bad": _Boom,
        "m2": (lambda: _RidEstimator(np.where(y == 1, 0.8, 0.2), y.copy())),
    }
    with pytest.raises(BudgetExhaustedError, match="time"):
        run_slice(
            _Dataset(n, y),
            Task(kind="binary"),
            estimators=estimators,
            splitter=_kfold(n),
            metric=Accuracy(),
            policy=SelectionPolicy(),
            budget=_ExhaustAfter(1),  # bad starts (fails, no consume), m2 gated -> 0 completed
        )


def test_zero_completed_all_failed_raises_fitfailed() -> None:
    """0 completed with NO budget skips (all failed) -> FitFailedError (M3 behavior preserved)."""
    n = 20
    y = np.array([0, 1] * (n // 2))
    with pytest.raises(FitFailedError):
        run_slice(
            _Dataset(n, y),
            Task(kind="binary"),
            estimators={"bad1": _Boom, "bad2": _Boom},
            splitter=_kfold(n),
            metric=Accuracy(),
            policy=SelectionPolicy(),
            budget=_TrialBudget(5),
        )


def test_trials_reproducible_same_seed() -> None:
    """Trials budget is fully deterministic: same inputs -> identical leaderboard/winner."""
    n = 20
    y = np.array([0, 1] * (n // 2))
    kw = dict(
        estimators=_three_estimators(y),
        splitter=_kfold(n),
        metric=Accuracy(),
        policy=SelectionPolicy(),
    )
    a = run_slice(_Dataset(n, y), Task(kind="binary"), budget=_TrialBudget(2), **kw)
    b = run_slice(_Dataset(n, y), Task(kind="binary"), budget=_TrialBudget(2), **kw)
    assert [e.model_id for e in a.leaderboard] == [e.model_id for e in b.leaderboard]
    assert a.best_model_id == b.best_model_id
    assert a.budget.skipped == b.budget.skipped


# --- M5-resume RC-c: per-candidate stage-cache skip-on-hit + resume (FR-RC-2/3, ADR-0036 §3) ---


class _DictCache:
    """In-memory fake CandidateCache (a dict keyed by candidate id) — no disk, Humble Object."""

    def __init__(self, preset=None) -> None:
        self._store = dict(preset or {})
        self.put_calls: list[str] = []

    def get(self, candidate_id):
        return self._store.get(candidate_id)

    def put(self, candidate_id, candidate) -> None:
        self._store[candidate_id] = candidate
        self.put_calls.append(candidate_id)


def _accuracy_run(y, estimators, n, **kw):
    return run_slice(
        _Dataset(n, y),
        Task(kind="binary"),
        estimators=estimators,
        splitter=_kfold(n),
        metric=Accuracy(),
        policy=SelectionPolicy(),
        **kw,
    )


def test_cache_hit_skips_run_candidate() -> None:
    """A cached candidate is reused without retraining (FR-RC-2): its factory is never called."""
    n = 20
    y = np.array([0, 1] * (n // 2))
    first = _accuracy_run(
        y, {"m": (lambda: _RidEstimator(np.where(y == 1, 0.8, 0.2), y.copy()))}, n
    )
    cache = _DictCache({c.id: c for c in first.candidates})
    invoked: list[int] = []

    def spy():
        invoked.append(1)
        return _RidEstimator(np.where(y == 1, 0.8, 0.2), y.copy())

    res = _accuracy_run(y, {"m": spy}, n, cache=cache)
    assert invoked == []  # cache-hit short-circuits _run_candidate
    assert res.reused == ("m",) and res.computed == ()
    assert res.best_model_id == "m"


def test_cache_miss_computes_and_puts_durably() -> None:
    n = 20
    y = np.array([0, 1] * (n // 2))
    cache = _DictCache()
    res = _accuracy_run(
        y, {"m": (lambda: _RidEstimator(np.where(y == 1, 0.8, 0.2), y.copy()))}, n, cache=cache
    )
    assert res.computed == ("m",) and res.reused == ()
    assert cache.put_calls == ["m"]  # durable write on completion (resume-ready)


def test_resume_computes_remaining_only() -> None:
    """k cached entries -> only the N-k remaining are recomputed (FR-RC-3)."""
    n = 20
    y = np.array([0, 1] * (n // 2))
    cache = _DictCache()
    _accuracy_run(y, _three_estimators(y), n, cache=cache)  # cache a, b, c
    del cache._store["c"]  # simulate an interruption after a, b were persisted
    invoked: list[str] = []

    def spy(name, table):
        def make():
            invoked.append(name)
            return _RidEstimator(table, y.copy())

        return make

    est2 = {
        "a": spy("a", np.where(y == 1, 0.9, 0.1)),
        "b": spy("b", np.where(y == 1, 0.8, 0.2)),
        "c": spy("c", np.where(y == 1, 0.7, 0.3)),
    }
    res = _accuracy_run(y, est2, n, cache=cache)
    assert set(res.reused) == {"a", "b"} and res.computed == ("c",)
    assert set(invoked) == {"c"}  # only the missing candidate retrained


def test_cache_hit_consumes_trial() -> None:
    """A cache-hit consumes a budget trial like a completed candidate (trials determinism, NFR-RC-2)."""
    n = 20
    y = np.array([0, 1] * (n // 2))
    cache = _DictCache()
    _accuracy_run(y, _three_estimators(y), n, cache=cache)  # cache all three
    res = _accuracy_run(y, _three_estimators(y), n, cache=cache, budget=_TrialBudget(2))
    assert len(res.reused) == 2  # two hits consumed two trials
    assert res.budget.exhausted is True
    assert len(res.budget.skipped) == 1


def test_single_candidate_cache_hit() -> None:
    """N=1 hit (FR boundary): reuse short-circuits even a would-raise factory; lone-anchor band."""
    n = 20
    y = np.array([0, 1] * (n // 2))
    first = _accuracy_run(
        y, {"m": (lambda: _RidEstimator(np.where(y == 1, 0.8, 0.2), y.copy()))}, n
    )
    cache = _DictCache({c.id: c for c in first.candidates})
    res = _accuracy_run(y, {"m": _Boom}, n, cache=cache)  # _Boom would raise if computed
    assert res.reused == ("m",) and res.best_model_id == "m" and res.band_width == 1


def test_all_failed_nothing_cached() -> None:
    """All candidates fail -> cache is not written; a later fit honestly retries (ADR-0036 §3)."""
    n = 20
    y = np.array([0, 1] * (n // 2))
    cache = _DictCache()
    with pytest.raises(FitFailedError):
        _accuracy_run(y, {"bad1": _Boom, "bad2": _Boom}, n, cache=cache)
    assert cache.put_calls == []  # failures carry no OOF -> never cached


# --- M5 memory-enforce: orthogonal gate reuses graceful degradation (FR-MEM-1/2/3, ADR-0039) ---


def test_memory_degraded_best_so_far() -> None:
    """Memory exceeded after the 1st candidate -> the rest are skipped, best-so-far is returned."""
    n = 20
    y = np.array([0, 1] * (n // 2))
    res = run_slice(
        _Dataset(n, y),
        Task(kind="binary"),
        estimators=_three_estimators(y),
        splitter=_kfold(n),
        metric=Accuracy(),
        policy=SelectionPolicy(),
        budget=_MemoryExhaustAfter(1),  # candidate a passes the gate, then memory is out
    )
    assert len(res.leaderboard) == 1 and res.best_model_id == "a"
    assert res.budget.exhausted is True
    assert set(res.budget.skipped) == {"b", "c"}
    assert res.budget.exhausted_by == "memory"  # truthful axis, not "none" (the mode)


def test_memory_zero_completed_raises_memory() -> None:
    """0 completed because memory was out before the first candidate -> BudgetExhaustedError(memory)."""
    n = 20
    y = np.array([0, 1] * (n // 2))
    with pytest.raises(BudgetExhaustedError, match="memory"):
        run_slice(
            _Dataset(n, y),
            Task(kind="binary"),
            estimators=_three_estimators(y),
            splitter=_kfold(n),
            metric=Accuracy(),
            policy=SelectionPolicy(),
            budget=_MemoryExhaustAfter(0),  # exhausted before any candidate starts
        )


def test_memory_cache_hits_skipped_after_exceedance() -> None:
    """memory x cache (ADR-0039 §5): RSS is not freed, so after exceedance even cache-HITS are skipped."""
    n = 20
    y = np.array([0, 1] * (n // 2))
    cache = _DictCache()
    _accuracy_run(y, _three_estimators(y), n, cache=cache)  # prefill a, b, c
    res = _accuracy_run(y, _three_estimators(y), n, cache=cache, budget=_MemoryExhaustAfter(1))
    assert res.reused == ("a",)  # only the 1st hit is taken before memory is out
    assert set(res.budget.skipped) == {"b", "c"}  # subsequent hits skipped (memory not freed)
    assert res.budget.exhausted is True and res.budget.exhausted_by == "memory"


# --- M6a-4: OOF target-encoding augmentation (ADR-0040 §2, ADR-0041) -------


class _TESchema:
    """Fake schema carrying a TE spec; features = [rid, cat_te, cat] (numeric block then categorical)."""

    group = None
    selected_features = None

    def __init__(self) -> None:
        self.target_encoding = TargetEncodingSpec(
            encodings={"cat": {"0": 0.9, "1": 0.1}}, global_mean=0.5, smoothing=1.0
        )
        self.categories = {"cat": CategoryTable.fit(["a", "b"])}  # null_code = 2

    @property
    def numeric(self) -> list[str]:
        return ["rid", "cat_te"]

    @property
    def categorical(self) -> list[str]:
        return ["cat"]

    @property
    def features(self) -> list[str]:
        return self.numeric + self.categorical


class _TEDataset:
    """Dataset whose ``cat_te`` numeric column holds the full-train TE placeholder (0.5)."""

    def __init__(self, n: int, y: np.ndarray, codes: np.ndarray) -> None:
        self._rid = np.arange(n, dtype=float).reshape(-1, 1)
        self._full_te = np.full(
            (n, 1), 0.5
        )  # full-train TE column (overwritten OOF for evaluation)
        self._codes = codes.reshape(-1, 1)
        self._y = y
        self._schema = _TESchema()

    @property
    def schema(self) -> _TESchema:
        return self._schema

    @property
    def n_rows(self) -> int:
        return len(self._y)

    def to_numpy(self) -> np.ndarray:
        return np.hstack([self._rid, self._full_te])

    def categorical_codes(self) -> np.ndarray:
        return self._codes

    def target(self) -> np.ndarray:
        return self._y

    def sample_weight(self) -> None:
        return None

    def groups(self) -> None:
        return None


class _TEColumnSpy:
    """Records the ``cat_te`` (column index 1) values it is trained on."""

    seen: list[np.ndarray] = []

    def __init__(self) -> None:
        self.feature_names: list[str] = []
        self.classes_ = None

    def fit(self, X, y, sample_weight=None):  # noqa: ANN001
        self.classes_ = np.unique(y)
        _TEColumnSpy.seen.append(X[:, 1].copy())
        return self

    def predict(self, X):  # noqa: ANN001
        return np.zeros(X.shape[0], dtype=int)


def _te_dataset(n: int = 20):
    y = np.array([0, 1] * (n // 2))
    codes = np.array([0, 1] * (n // 2), dtype=np.int64)
    return _TEDataset(n, y, codes), y


class _TETimeDataset(_TEDataset):
    """``_TEDataset`` plus a time axis, for the time-series expanding-window TE path (ADR-0082)."""

    def time(self) -> np.ndarray:
        return np.arange(self.n_rows, dtype=float)


class _TimeSplitter(_Splitter):
    """A time-ordered fake: marks itself so run_slice runs the value-based fold check + expanding TE."""

    time_ordered = True


def _ts_folds(n: int) -> list[Fold]:
    idx = np.arange(n)
    half, mid = n // 2, n // 2 + (n - n // 2) // 2
    return [
        Fold(
            fit_idx=idx[:half].astype(np.int64),
            es_idx=np.empty(0, np.int64),
            test_idx=idx[half:mid].astype(np.int64),
        ),
        Fold(
            fit_idx=idx[:mid].astype(np.int64),
            es_idx=np.empty(0, np.int64),
            test_idx=idx[mid:].astype(np.int64),
        ),
    ]


def test_te_expanding_under_time_ordered_splitter() -> None:
    # ADR-0082: under a time-ordered split run_slice runs the EXPANDING-window OOF encoder (each fold from
    # strictly earlier folds, no look-ahead), so the cross-fit index IS built and candidates train on
    # encoded values, not the full-train 0.5 placeholder. Replaces the old defensive skip (C1).
    _TEColumnSpy.seen = []
    n = 20
    y = np.array([0, 1] * (n // 2))
    codes = np.array([0, 1] * (n // 2), dtype=np.int64)
    ds = _TETimeDataset(n, y, codes)
    res = run_slice(
        ds,
        Task(kind="binary"),
        estimators={"m": _TEColumnSpy},
        splitter=_TimeSplitter(_ts_folds(n)),
        metric=Accuracy(),
        policy=SelectionPolicy(),
        fe=FEConfig(target_encoding=True),
    )
    assert res.oof_fold_index is not None  # expanding cross-fit index IS built (no skip)
    # encoder-discriminating: under EXPANDING the earliest (-1) block keeps its base rate (here 0.5), so
    # fold0's train (== the -1 block, rows 0..9) is all 0.5 while fold1's train carries expanded fold0-test
    # values. The leaky IID encoder would give the -1 block per-category values -> seen[0] would NOT be 0.5,
    # so this pins the time-ordered routing (not just "TE ran").
    assert len(_TEColumnSpy.seen) == 2
    assert (
        _TEColumnSpy.seen[0] == 0.5
    ).all()  # fold0 train == -1 block -> base rate (expanding only)
    assert not (_TEColumnSpy.seen[1] == 0.5).all()  # fold1 train includes expanded fold0-test rows


def test_oof_fold_index_built_when_target_encoding_without_calibration() -> None:
    # ADR-0041 §1 gate: TE requires the cross-fit fold index even with no calibration/refinement
    ds, _ = _te_dataset()
    res = run_slice(
        ds,
        Task(kind="binary"),
        estimators={"m": _ClassOnly},
        splitter=_kfold(20),
        metric=Accuracy(),
        policy=SelectionPolicy(),
        fe=FEConfig(target_encoding=True),
    )
    assert res.oof_fold_index is not None


def test_te_augmentation_runs_once_and_feeds_oof_into_x_full(monkeypatch) -> None:  # noqa: ANN001
    # the OOF cross-fit is computed ONCE per run (ADR-0040 §2) and its output replaces the cat_te
    # column the candidates train on (ADR-0041 §1) — not the full-train 0.5 placeholder.
    import honestml.application.slice as slice_mod

    calls = {"n": 0}

    def fake_encode(codes, y, fold, *, smoothing, reserve_from=None):  # noqa: ANN001
        calls["n"] += 1
        return np.full((codes.shape[0], codes.shape[1]), 7.0)

    monkeypatch.setattr(slice_mod, "crossfit_encode", fake_encode)
    _TEColumnSpy.seen = []
    ds, _ = _te_dataset()
    run_slice(
        ds,
        Task(kind="binary"),
        estimators={"m": _TEColumnSpy},
        splitter=_kfold(20),
        metric=Accuracy(),
        policy=SelectionPolicy(),
        fe=FEConfig(target_encoding=True),
    )
    assert calls["n"] == 1  # once per run, not per candidate/fold
    assert _TEColumnSpy.seen and all((col == 7.0).all() for col in _TEColumnSpy.seen)
    # the dataset's full-train TE column is untouched (refit/inference use it, ADR-0041 §3)
    assert (ds.to_numpy()[:, 1] == 0.5).all()


def test_no_augmentation_when_target_encoding_off() -> None:
    _TEColumnSpy.seen = []
    ds, _ = _te_dataset()
    res = run_slice(
        ds,
        Task(kind="binary"),
        estimators={"m": _TEColumnSpy},
        splitter=_kfold(20),
        metric=Accuracy(),
        policy=SelectionPolicy(),
        fe=FEConfig(target_encoding=False),
    )
    assert res.oof_fold_index is None  # no gate trigger
    assert all((col == 0.5).all() for col in _TEColumnSpy.seen)  # candidates saw full-train column


def test_te_crossfit_uses_the_single_eval_fold_index(monkeypatch) -> None:  # noqa: ANN001
    # R-FE-FOLD-ALIGN (NFR-FE-1 #2): the array fed to crossfit_encode IS result.oof_fold_index — the
    # same single index that assigns OOF rows for scoring, so TE folds and eval folds cannot diverge.
    import honestml.application.slice as slice_mod

    real = slice_mod.crossfit_encode
    captured = {}

    def spy(codes, y, fold, **kw):  # noqa: ANN001
        captured["fold"] = fold.copy()
        return real(codes, y, fold, **kw)

    monkeypatch.setattr(slice_mod, "crossfit_encode", spy)
    ds, _ = _te_dataset()
    res = run_slice(
        ds,
        Task(kind="binary"),
        estimators={"m": _ClassOnly},
        splitter=_kfold(20),
        metric=Accuracy(),
        policy=SelectionPolicy(),
        fe=FEConfig(target_encoding=True),
    )
    assert np.array_equal(captured["fold"], res.oof_fold_index)
    # and the fold index is exactly the per-fold test_idx assignment used for scoring
    expected = np.full(20, -1, dtype=np.int64)
    for fid, fold in enumerate(_kfold(20)._folds):
        expected[fold.test_idx] = fid
    assert np.array_equal(res.oof_fold_index, expected)


def test_holdout_te_train_block_encoded_on_its_own_target_not_the_holdout() -> None:
    # F014 end-to-end: under a single-fold holdout the train rows are uncovered (-1) in oof_fold_index, so
    # their OOF target encoding must be fitted on the train target, NOT on the holdout test target. Train
    # and holdout carry INVERTED category->target relations (train: code==y; holdout: code!=y), so a leak
    # would flip the train block's encoding. Pins the run_slice wiring of the F014 crossfit_encode fix.
    _TEColumnSpy.seen = []
    n = 20
    codes = np.array([0, 1] * (n // 2), dtype=np.int64)
    y = np.array(
        [0, 1] * 5 + [1, 0] * 5
    )  # rows 0..9 (train) code==y; rows 10..19 (holdout) code!=y
    ds = _TEDataset(n, y, codes)
    fold = Fold(
        fit_idx=np.arange(10, dtype=np.int64),
        es_idx=np.empty(0, dtype=np.int64),
        test_idx=np.arange(10, 20, dtype=np.int64),
    )
    run_slice(
        ds,
        Task(kind="binary"),
        estimators={"m": _TEColumnSpy},
        splitter=_Splitter([fold]),
        metric=Accuracy(),
        policy=SelectionPolicy(),
        fe=FEConfig(target_encoding=True),
    )
    train_te = _TEColumnSpy.seen[
        0
    ]  # the cat_te column the candidate trained on == the -1 (train) block
    te_code0 = train_te[codes[:10] == 0].mean()
    te_code1 = train_te[codes[:10] == 1].mean()
    # train's own target gives code1 a higher rate than code0; a holdout leak would invert this
    assert te_code1 > te_code0 + 0.1


# --- M6b feature selection integration (ADR-0044, FR-FS-3/7, NFR-FS-5) -------


class _MultiNumeric(_Dataset):
    """`_Dataset` with extra numeric columns (col 0 stays the row id for the fakes)."""

    def __init__(self, n, y, n_extra=3, **kw) -> None:
        super().__init__(n, y, **kw)
        extra = np.random.default_rng(0).random((n, n_extra))
        self._num = np.column_stack([self._num, extra])
        self._schema = _Schema(["rid"] + [f"f{i}" for i in range(n_extra)], [])


class _ColRanker:
    """Estimator-independent ranker: returns fixed per-column weights; counts fold calls."""

    name = "col"

    def __init__(self, weights) -> None:
        self.weights = np.asarray(weights, dtype=float)
        self.calls = 0

    def rank(self, x, y, *, categorical, random_state, sample_weight=None, groups=None):
        self.calls += 1
        return self.weights.copy()

    def auto_threshold(self, n_features):
        return 1.0 / n_features


def _fs_run(ds, y, est, ranker, cfg, n):
    return run_slice(
        ds,
        Task(kind="binary"),
        estimators=est,
        splitter=_kfold(n),
        metric=RocAuc(),
        policy=SelectionPolicy(),
        features=FeatureSelectionBundle(config=cfg, ranker=ranker),
    )


def test_selection_projects_and_reports_subset() -> None:
    # FR-FS-3: selection keeps top-k (rid kept so the rid-fake still predicts); n_features reflects it
    n = 20
    y = np.array([0, 1] * (n // 2))
    ds = _MultiNumeric(n, y, n_extra=3)  # features: rid, f0, f1, f2
    ranker = _ColRanker([0.5, 0.9, 0.05, 0.05])  # top-2 -> rid(0.5), f0(0.9)
    est = {"m": (lambda: _RidEstimator(np.where(y == 1, 0.8, 0.2), y.copy()))}
    res = _fs_run(ds, y, est, ranker, FeatureSelectionConfig(cutoff="top_k", top_k=2), n)
    assert res.feature_selection is not None
    assert res.feature_selection.selected_features == (
        "rid",
        "f0",
    )  # schema.features order preserved
    assert res.leaderboard[0].n_features == 2


def test_selection_computed_once_not_per_candidate() -> None:
    # NFR-FS-5: the ranker runs once per fold for the whole run, not once per candidate
    n = 20
    y = np.array([0, 1] * (n // 2))
    ds = _MultiNumeric(n, y, n_extra=3)
    ranker = _ColRanker([0.5, 0.9, 0.05, 0.05])
    est = {
        "m1": (lambda: _RidEstimator(np.where(y == 1, 0.8, 0.2), y.copy())),
        "m2": (lambda: _RidEstimator(np.where(y == 1, 0.7, 0.3), y.copy())),
    }
    _fs_run(ds, y, est, ranker, FeatureSelectionConfig(cutoff="top_k", top_k=2), n)
    assert ranker.calls == 5  # n_folds, independent of the 2 candidates


def test_subset_independent_of_candidate_set() -> None:
    # R-FS-RANKER-MODEL: a fixed ranker yields the same subset regardless of which candidates run
    n = 20
    y = np.array([0, 1] * (n // 2))
    cfg = FeatureSelectionConfig(cutoff="top_k", top_k=2)
    a = _fs_run(
        _MultiNumeric(n, y, 3),
        y,
        {"m": (lambda: _RidEstimator(np.where(y == 1, 0.8, 0.2), y.copy()))},
        _ColRanker([0.5, 0.9, 0.05, 0.05]),
        cfg,
        n,
    )
    b = _fs_run(
        _MultiNumeric(n, y, 3),
        y,
        {"z": (lambda: _RidEstimator(np.where(y == 1, 0.6, 0.4), y.copy()))},
        _ColRanker([0.5, 0.9, 0.05, 0.05]),
        cfg,
        n,
    )
    assert a.feature_selection is not None and b.feature_selection is not None
    assert a.feature_selection.selected_features == b.feature_selection.selected_features


def test_selection_off_leaves_result_unselected() -> None:
    n = 20
    y = np.array([0, 1] * (n // 2))
    est = {"m": (lambda: _RidEstimator(np.where(y == 1, 0.8, 0.2), y.copy()))}
    res = run_slice(
        _Dataset(n, y),
        Task(kind="binary"),
        estimators=est,
        splitter=_kfold(n),
        metric=RocAuc(),
        policy=SelectionPolicy(),
    )
    assert res.feature_selection is None


# --- WS-A native categorical routing (ADR-0087/0088, FR-1) ------------------


class _CatDataset(_Dataset):
    """`_Dataset` with one categorical code column; col 0 stays the rid the fakes read."""

    def __init__(self, n, y, **kw) -> None:
        super().__init__(n, y, **kw)
        self._codes = (np.arange(n) % 3).reshape(-1, 1).astype(np.int64)
        self._schema = _Schema(["rid"], ["c1"])  # design_matrix columns: [rid, c1]


class _NativeCatEstimator(_RidEstimator):
    """Native-capable fake (SupportsNativeCategorical): records the indices injected before fit."""

    supports_native_categorical = True

    def __init__(self, proba_table, class_table, sink) -> None:
        super().__init__(proba_table, class_table)
        self.categorical_indices: list[int] = []
        self._sink = sink

    def fit(self, X, y, X_val=None, y_val=None, sample_weight=None):
        self._sink.append(list(self.categorical_indices))
        return super().fit(X, y, X_val, y_val, sample_weight)


def test_native_model_receives_categorical_indices_per_fold() -> None:
    # FR-1: a SupportsNativeCategorical candidate is handed the CATEGORICAL-column positions before fit
    n = 20
    y = np.array([0, 1] * (n // 2))
    seen: list[list[int]] = []
    est = {"native": (lambda: _NativeCatEstimator(np.where(y == 1, 0.8, 0.2), y.copy(), seen))}
    run_slice(
        _CatDataset(n, y),
        Task(kind="binary"),
        estimators=est,
        splitter=_kfold(n),
        metric=RocAuc(),
        policy=SelectionPolicy(),
    )
    assert seen and all(idx == [1] for idx in seen)  # c1 is column 1 (after the rid numeric)


def test_plain_model_is_not_touched_by_routing() -> None:
    # FR-1: a non-native candidate never gets categorical_indices set (behaves exactly as before)
    n = 20
    y = np.array([0, 1] * (n // 2))
    created: list[_RidEstimator] = []

    def factory():
        e = _RidEstimator(np.where(y == 1, 0.8, 0.2), y.copy())
        created.append(e)
        return e

    run_slice(
        _CatDataset(n, y),
        Task(kind="binary"),
        estimators={"plain": factory},
        splitter=_kfold(n),
        metric=RocAuc(),
        policy=SelectionPolicy(),
    )
    assert created and all(not hasattr(e, "categorical_indices") for e in created)


def test_refit_best_injects_categorical_indices_for_native_model() -> None:
    # FR-1/FR-4: the shipped (refit) native model trains with the same projected indices the CV path used
    n = 10
    y = np.array([0, 1] * 5)
    seen: list[list[int]] = []
    est = refit_best(
        _CatDataset(n, y),
        Task(kind="binary"),
        factory=lambda: _NativeCatEstimator(np.where(y == 1, 0.8, 0.2), y.copy(), seen),
    )
    assert seen == [[1]]  # one refit fit, indices injected beforehand
    assert est.categorical_indices == [1]


# --- native cardinality gate wiring (ADR-0092/0095, FR-2/FR-4/FR-5) ----------


class _HighCardCatDataset(_Dataset):
    """`_Dataset` with one HIGH-cardinality categorical (200 categories) -> the gate demotes it."""

    def __init__(self, n, y, **kw) -> None:
        super().__init__(n, y, **kw)
        self._codes = (
            (np.arange(n) % 7).reshape(-1, 1).astype(np.int64)
        )  # code values are immaterial
        self._schema = _Schema(["rid"], ["c1"], cat_cardinality=200)  # 200 > any tens-scale cap


def test_native_gate_demotes_high_card_in_cv_refit_and_reports(caplog) -> None:
    # FR-2/FR-5: a categorical above native_cat_max_unique is demoted to codes in BOTH the per-fold CV
    # routing and the refit (no drift), the verdict is recorded, and the demotion is logged (never silent).
    import logging

    n = 20
    y = np.array([0, 1] * (n // 2))
    task = Task(kind="binary")  # default tens cap; c1 has 200 categories -> demoted
    seen: list[list[int]] = []
    with caplog.at_level(logging.WARNING):
        res = run_slice(
            _HighCardCatDataset(n, y),
            task,
            estimators={
                "native": lambda: _NativeCatEstimator(np.where(y == 1, 0.8, 0.2), y.copy(), seen)
            },
            splitter=_kfold(n),
            metric=RocAuc(),
            policy=SelectionPolicy(),
        )
    assert seen and all(idx == [] for idx in seen)  # demoted -> no native indices in CV
    assert res.native_routing == {"c1": "high_cardinality"}  # FR-5: verdict carried
    assert any("demoted" in r.message for r in caplog.records)  # FR-5: never silent
    refit = refit_best(
        _HighCardCatDataset(n, y),
        task,
        factory=lambda: _NativeCatEstimator(np.where(y == 1, 0.8, 0.2), y.copy(), []),
    )
    assert refit.categorical_indices == []  # FR-2: refit demotes identically, no drift vs CV


def test_native_gate_opt_out_routes_all_native() -> None:
    # FR-4/NFR-3: native_cat_max_unique=None disables the gate -> the high-card categorical routes native
    n = 20
    y = np.array([0, 1] * (n // 2))
    seen: list[list[int]] = []
    res = run_slice(
        _HighCardCatDataset(n, y),
        Task(kind="binary", native_cat_max_unique=None),
        estimators={
            "native": lambda: _NativeCatEstimator(np.where(y == 1, 0.8, 0.2), y.copy(), seen)
        },
        splitter=_kfold(n),
        metric=RocAuc(),
        policy=SelectionPolicy(),
    )
    assert seen and all(idx == [1] for idx in seen)  # gate off -> c1 native again
    assert res.native_routing is None  # nothing demoted -> no report block, no warning
