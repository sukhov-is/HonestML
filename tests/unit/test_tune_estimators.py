"""M7a-D: the tune_estimators spine (ADR-0062 §2) on fake ports — inner-CV, TE, weighting, budget."""

from __future__ import annotations

import numpy as np
import pytest

import honestml.application.tuning as tuning_mod
from honestml.adapters import Accuracy
from honestml.application import tune_estimators
from honestml.core import CategoryTable, Fold, SelectionPolicy, Task, TuneOutcome

pytestmark = pytest.mark.unit


class _Schema:
    selected_features = None

    def __init__(self, numeric, categorical=(), target_encoding=None, *, cat_cardinality=3) -> None:
        self.numeric = list(numeric)
        self.categorical = list(categorical)
        self.target_encoding = target_encoding
        # category tables so the native-cardinality gate reads len(categories); low-card by default
        self.categories = {
            c: CategoryTable.fit([str(i) for i in range(cat_cardinality)]) for c in self.categorical
        }

    @property
    def features(self):
        return self.numeric + self.categorical


class _Dataset:
    def __init__(self, num, y, *, sample_weight=None, schema=None) -> None:
        self._num = num
        self._y = y
        self._sw = sample_weight
        self.schema = schema or _Schema([f"f{i}" for i in range(num.shape[1])])

    @property
    def n_rows(self):
        return self._num.shape[0]

    def to_numpy(self):
        return self._num

    def categorical_codes(self):
        return np.empty((self._num.shape[0], 0), dtype=np.int64)

    def target(self):
        return self._y

    def sample_weight(self):
        return self._sw


# capture every estimator the (fake) make_factory builds, to inspect fit args
_BUILT: list = []


class _RecordingEstimator:
    def __init__(self, params) -> None:
        self.params = dict(params)
        self.feature_names: list[str] = []
        self.classes_ = None
        self.fit_width = None
        self.fit_rows = None
        self.fit_sw_is_none = True

    def fit(self, X, y, X_val=None, y_val=None, sample_weight=None):
        self.fit_width = X.shape[1]
        self.fit_rows = X.shape[0]
        self.fit_sw_is_none = sample_weight is None
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        # accuracy improves when params["good"] is True -> the search has a real optimum
        rid = X[:, 0].astype(int)
        return rid % 2 if self.params.get("good") else np.zeros(X.shape[0], dtype=int)


def _make_factory(name, params):
    def build():
        est = _RecordingEstimator(params)
        _BUILT.append(est)
        return est

    return build


class _Splitter:
    def __init__(self, folds) -> None:
        self._folds = folds

    def split(self, dataset):
        return iter(self._folds)


def _kfold(n, k=3):
    idx = np.arange(n)
    folds = [
        Fold(
            fit_idx=np.setdiff1d(idx, idx[i::k]),
            es_idx=np.empty(0, dtype=np.int64),
            test_idx=idx[i::k],
        )
        for i in range(k)
    ]
    return _Splitter(folds)


class _FakeTuner:
    name = "fake"

    def __init__(self, param_sets) -> None:
        self.param_sets = param_sets
        self.last_max_trials = None
        self.last_timeout = None
        self.last_seed = None
        self.timeouts: list = []  # per-model timeout_s, in call order (fair-share check)

    def tune(self, space, score, *, max_trials, timeout_s, greater_is_better, random_state):
        self.last_max_trials, self.last_timeout, self.last_seed = (
            max_trials,
            timeout_s,
            random_state,
        )
        self.timeouts.append(timeout_s)
        pick = max if greater_is_better else min
        scored = [(score(p), p) for p in self.param_sets]
        best_score, best = pick(scored, key=lambda t: t[0])
        return TuneOutcome(
            best_params=best, n_trials_run=len(self.param_sets), best_score=best_score
        )


class _FakeBudget:
    def __init__(self, *, exhausted=False, time_left=float("inf")) -> None:
        self._exhausted = exhausted
        self._time_left = time_left
        self.consumed = 0.0

    def time_left(self):
        return self._time_left

    def consume(self, seconds):
        self.consumed += seconds

    @property
    def exhausted(self):
        return self._exhausted

    @property
    def exhausted_reason(self):
        return "time" if self._exhausted else None

    def memory_left(self):
        return None


_SPACE = {"good": {"type": "categorical", "choices": [False, True]}}


def _ds(n=30, width=3, seed=0, sample_weight=None):
    rng = np.random.default_rng(seed)
    num = np.column_stack(
        [np.arange(n, dtype=float)] + [rng.normal(size=n) for _ in range(width - 1)]
    )
    y = np.array([0, 1] * (n // 2))
    return _Dataset(num, y, sample_weight=sample_weight)


class _TimeSplitter(_Splitter):
    """Time-ordered fake inner splitter: marks itself so tune_estimators routes TE to the expanding encoder."""

    time_ordered = True


def _run(
    tunable,
    *,
    sample_weight=None,
    budget=None,
    fe=None,
    ds=None,
    random_state=7,
    inner_splitter=None,
):
    _BUILT.clear()
    ds = ds or _ds(sample_weight=sample_weight)
    tuner = _FakeTuner([{"good": False}, {"good": True}])
    outcomes = tune_estimators(
        ds,
        Task(kind="binary"),
        tunable=tunable,
        make_factory=_make_factory,
        tuner=tuner,
        metric=Accuracy(),
        policy=SelectionPolicy(greater_is_better=True),
        inner_splitter=inner_splitter or _kfold(ds.n_rows),
        n_trials=2,
        timeout_s=None,
        random_state=random_state,
        fe=fe,
        sample_weight=sample_weight,
        budget=budget,
    )
    return outcomes, tuner


def test_inner_objective_sees_full_feature_width() -> None:
    # the inner objective fits on the FULL DEV feature space (no FS projection, ADR-0062 §2a)
    _run({"lightgbm": _SPACE}, ds=_ds(width=4))
    assert _BUILT and all(e.fit_width == 4 for e in _BUILT)


def test_inner_objective_fits_on_inner_train_only() -> None:
    # each inner fit trains on fit⊕es of an inner fold (< n rows), not the whole DEV (anti-leakage)
    _run({"lightgbm": _SPACE}, ds=_ds(n=30))
    assert _BUILT and all(e.fit_rows is not None and e.fit_rows < 30 for e in _BUILT)


def test_inner_objective_weighted() -> None:
    sw = np.linspace(0.5, 2.0, 30)
    _run({"lightgbm": _SPACE}, sample_weight=sw)
    assert _BUILT and all(not e.fit_sw_is_none for e in _BUILT)


def test_finds_best_params_and_passes_budget_scalars() -> None:
    outcomes, tuner = _run({"lightgbm": _SPACE})
    assert outcomes["lightgbm"].best_params == {"good": True}  # higher accuracy
    assert tuner.last_max_trials == 2 and tuner.last_seed == 7 and tuner.last_timeout is None


def test_empty_search_space_skips_model() -> None:
    outcomes, _ = _run({"baseline": {}})
    assert outcomes == {} and _BUILT == []  # nothing tuned, baseline stays


def test_zero_trials_falls_back_to_baseline() -> None:
    # budget already exhausted -> the model is not tuned (no outcome); facade keeps the baseline
    outcomes, tuner = _run({"lightgbm": _SPACE}, budget=_FakeBudget(exhausted=True))
    assert outcomes == {} and tuner.last_max_trials is None


def test_tiny_budget_caps_timeout() -> None:
    # one model, time_left=5 -> per-model cap = 5/1 (fair share); HPO does NOT consume a candidate trial
    budget = _FakeBudget(time_left=5.0)
    outcomes, tuner = _run({"lightgbm": _SPACE}, budget=budget)
    assert tuner.last_timeout == 5.0 and "lightgbm" in outcomes
    assert budget.consumed == 0.0  # ADR-0062 §6: HPO does not burn the candidate-loop trial counter


def test_per_model_time_fair_share() -> None:
    # two tunable models under time_left=10 -> caps are 10/2 then 10/1 (no first-model starvation, §5)
    budget = _FakeBudget(time_left=10.0)
    _, tuner = _run({"m1": _SPACE, "m2": _SPACE}, budget=budget)
    assert tuner.timeouts == [5.0, 10.0]


def test_seed_zero_passed_through() -> None:
    # an explicit tuning seed of 0 reaches the tuner verbatim (no falsy-0 swap, review fix)
    _, tuner = _run({"lightgbm": _SPACE}, random_state=0)
    assert tuner.last_seed == 0


def test_inner_cv_no_te_leak(monkeypatch) -> None:
    # when target-encoding is on, the spine itself re-derives OOF-TE against the INNER fold index
    # (ADR-0062 §2) — _run_candidate does none. Spy the augmentation and assert the inner index.
    seen = {}

    def _spy(
        x_full,
        dataset,
        y,
        positive,
        oof_fold_index,
        smoothing,
        feature_names,
        *,
        time_ordered=False,
    ):
        seen["index"] = oof_fold_index.copy()
        return x_full

    monkeypatch.setattr(tuning_mod, "_augment_oof_te", _spy)

    class _TESpec:
        encodings = {"c": object()}

    from honestml.core import FEConfig

    ds = _ds(n=30)
    ds.schema.target_encoding = _TESpec()
    _run({"lightgbm": _SPACE}, fe=FEConfig(target_encoding=True), ds=ds)
    idx = seen["index"]
    # the inner fold index marks every row with its inner-fold id (0..k-1), never -1 left over here
    expected = np.full(30, -1, dtype=np.int64)
    for fid, fold in enumerate(_kfold(30)._folds):
        expected[fold.test_idx] = fid
    assert np.array_equal(idx, expected)


def test_inner_cv_te_routes_to_expanding_under_time_ordered(monkeypatch) -> None:
    # ADR-0082: a time-ordered inner CV must route the inner OOF-TE to the EXPANDING encoder
    # (time_ordered=True), not the leaking IID cross-fit. Pins the routing on the tuning spine.
    seen = {}

    def _spy(
        x_full,
        dataset,
        y,
        positive,
        oof_fold_index,
        smoothing,
        feature_names,
        *,
        time_ordered=False,
    ):
        seen["time_ordered"] = time_ordered
        return x_full

    monkeypatch.setattr(tuning_mod, "_augment_oof_te", _spy)

    class _TESpec:
        encodings = {"c": object()}

    from honestml.core import FEConfig

    ds = _ds(n=30)
    ds.schema.target_encoding = _TESpec()
    _run(
        {"lightgbm": _SPACE},
        fe=FEConfig(target_encoding=True),
        ds=ds,
        inner_splitter=_TimeSplitter(_kfold(30)._folds),
    )
    assert seen["time_ordered"] is True


# --- WS-A native categorical routing in the HPO inner objective (ADR-0088, FR-1) ---


class _NativeRecordingEstimator(_RecordingEstimator):
    """Native-capable recording fake: records the indices the inner objective injects before fit."""

    supports_native_categorical = True

    def __init__(self, params) -> None:
        super().__init__(params)
        self.categorical_indices: list[int] = []
        self.seen_indices: list[int] | None = None

    def fit(self, X, y, X_val=None, y_val=None, sample_weight=None):
        self.seen_indices = list(self.categorical_indices)
        return super().fit(X, y, X_val, y_val, sample_weight)


class _CatDataset(_Dataset):
    """DEV dataset with one categorical code column; design_matrix columns are [f0, c0]."""

    def __init__(self, n: int = 30) -> None:
        num = np.arange(n, dtype=float).reshape(-1, 1)
        y = np.array([0, 1] * (n // 2))
        super().__init__(num, y, schema=_Schema(["f0"], ["c0"]))
        self._codes = (np.arange(n) % 3).reshape(-1, 1).astype(np.int64)

    def categorical_codes(self):
        return self._codes


def test_native_model_in_hpo_objective_receives_categorical_indices() -> None:
    # FR-1: the HPO inner objective hands the cat-column positions to a native candidate before fit
    built: list[_NativeRecordingEstimator] = []

    def make_factory(name, params):
        def build():
            est = _NativeRecordingEstimator(params)
            built.append(est)
            return est

        return build

    tune_estimators(
        _CatDataset(),
        Task(kind="binary"),
        tunable={"lightgbm": _SPACE},
        make_factory=make_factory,
        tuner=_FakeTuner([{"good": False}, {"good": True}]),
        metric=Accuracy(),
        policy=SelectionPolicy(greater_is_better=True),
        inner_splitter=_kfold(30),
        n_trials=2,
        timeout_s=None,
        random_state=7,
    )
    assert built and all(e.seen_indices == [1] for e in built)  # c0 is column 1 (after f0)


def test_native_gate_demotes_high_card_in_hpo_objective() -> None:
    # FR-2: the HPO inner objective applies the SAME cardinality gate as run_slice/refit_best — a
    # high-card categorical is demoted to codes (no native index), so CV/refit/HPO cannot drift.
    built: list[_NativeRecordingEstimator] = []

    def make_factory(name, params):
        def build():
            est = _NativeRecordingEstimator(params)
            built.append(est)
            return est

        return build

    ds = _CatDataset()
    ds.schema = _Schema(["f0"], ["c0"], cat_cardinality=200)  # 200 > default tens cap -> demoted
    tune_estimators(
        ds,
        Task(kind="binary"),
        tunable={"lightgbm": _SPACE},
        make_factory=make_factory,
        tuner=_FakeTuner([{"good": False}, {"good": True}]),
        metric=Accuracy(),
        policy=SelectionPolicy(greater_is_better=True),
        inner_splitter=_kfold(30),
        n_trials=2,
        timeout_s=None,
        random_state=7,
    )
    assert built and all(
        e.seen_indices == [] for e in built
    )  # high-card c0 demoted, no native index
