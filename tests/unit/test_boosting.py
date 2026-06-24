"""M3b (ADR-0020): boosting zoo — lazy build, missing-extra, wrapper on a fake backend.

The real catboost/lightgbm/xgboost extras are not installed in the lightweight test env, so
the wrapper *logic* (seed/n_estimators wiring, predict/proba/importances, the no-ES warning)
is exercised against an injected fake backend module; the install-gated paths are tested for
``MissingDependencyError`` and laziness. The "participates end-to-end" path is ``importorskip``
so CI with extras exercises it.
"""

from __future__ import annotations

import logging
import subprocess
import sys
import types
from dataclasses import replace

import numpy as np
import pytest

import honestml.adapters.boosting as boosting_mod
from honestml.adapters.boosting import (
    _N_ESTIMATORS_ES,
    CATBOOST,
    LIGHTGBM,
    XGBOOST,
    _Backend,
    _BoostingClassifier,
    build_boosting,
)
from honestml.composition.registry import (
    ComponentDescriptor,
    ComponentRegistry,
    available_models,
)
from honestml.core import (
    Capabilities,
    MissingDependencyError,
    ModelSpec,
    ProbabilisticEstimator,
    SchemaValidationError,
    SupportsFeatureImportance,
    SupportsNativeCategorical,
    Task,
)

pytestmark = pytest.mark.unit


class _FakeClf:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.classes_ = None
        self.feature_importances_ = None

    def fit(self, X, y, sample_weight=None):
        self.classes_ = np.unique(y)
        self.feature_importances_ = np.ones(X.shape[1])
        return self

    def predict(self, X):
        return np.zeros(X.shape[0], dtype=int)

    def predict_proba(self, X):
        k = len(self.classes_)
        return np.full((X.shape[0], k), 1.0 / k)


class _FakeReg:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.feature_importances_ = None

    def fit(self, X, y, sample_weight=None):
        self.feature_importances_ = np.ones(X.shape[1])
        return self

    def predict(self, X):
        return np.zeros(X.shape[0])


class _StrictIntClf(_FakeClf):
    """Fake xgboost 3.x: rejects any classification target that is not contiguous ``0..K-1`` (ADR-0081)."""

    def fit(self, X, y, sample_weight=None, eval_set=None, **kwargs):
        for arr in [y, *([pair[1] for pair in eval_set] if eval_set else [])]:
            uniq = np.unique(arr)
            if not np.array_equal(uniq, np.arange(uniq.size)):
                raise ValueError(f"Invalid classes inferred from unique values of `y`: {uniq}")
        return super().fit(X, y, sample_weight)


def _fake_backend(monkeypatch) -> _Backend:
    mod = types.ModuleType("_fakeboost")
    mod.FakeClf = _FakeClf  # type: ignore[attr-defined]
    mod.FakeReg = _FakeReg  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "_fakeboost", mod)
    boosting_mod._warned_backends.discard("_fakeboost")  # the no-ES warning dedups per backend
    return _Backend(
        module="_fakeboost",
        clf_attr="FakeClf",
        reg_attr="FakeReg",
        seed_kwarg="seed",
        n_estimators_kwarg="n_estimators",
        extra_kwargs={"opt": 1},
    )


def _int_label_backend(monkeypatch) -> _Backend:
    mod = types.ModuleType("_fakeint")
    mod.StrictIntClf = _StrictIntClf  # type: ignore[attr-defined]
    mod.FakeReg = _FakeReg  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "_fakeint", mod)
    boosting_mod._warned_backends.discard("_fakeint")
    return _Backend(
        module="_fakeint",
        clf_attr="StrictIntClf",
        reg_attr="FakeReg",
        seed_kwarg="seed",
        n_estimators_kwarg="n_estimators",
        requires_int_labels=True,
    )


# --- wrapper logic on a fake backend ---------------------------------------


def _cat_xy(n: int = 600, seed: int = 0):
    """design_matrix-like float64 input: 2 numeric ⊕ one 4-level categorical (codes) at column 2.

    The per-level effect is NON-ordinal ([2, -2, 2, -2]), so a categorical split (by identity) separates
    the classes while a numeric split on the ordered codes cannot — native vs codes handling must differ.
    """
    rng = np.random.default_rng(seed)
    cat = rng.integers(0, 4, size=n)
    num = rng.normal(size=(n, 2))
    eff = np.array([2.0, -2.0, 2.0, -2.0])
    y = (0.5 * num[:, 0] + eff[cat] + 0.5 * rng.normal(size=n) > 0).astype(int)
    X = np.hstack([num, cat.astype(np.float64).reshape(-1, 1)])
    return X, y, [2]


def _native(backend, cat_idx, X, y, *, X_val=None, y_val=None):
    est = build_boosting(backend, task=Task(kind="binary"), random_state=0)
    est.feature_names = ["n0", "n1", "c0"]
    est.categorical_indices = list(cat_idx)
    est.fit(X, y, X_val=X_val, y_val=y_val)
    return est


def _roundtrip(est, tmp_path):
    import joblib

    p = tmp_path / "est.joblib"
    joblib.dump(est, p)
    return joblib.load(p)


def test_catboost_native_materialization_and_parity(tmp_path) -> None:
    pytest.importorskip("catboost")
    X, y, cat_idx = _cat_xy()
    est = _native(CATBOOST, cat_idx, X, y)
    assert est.native_model().get_cat_feature_indices() == cat_idx  # FR-1: cat declared natively
    p = est.predict_proba(X)
    assert (
        np.max(np.abs(p - est.predict_proba(X.copy()))) <= 1e-6
    )  # FR-4: int-cast is deterministic
    loaded = _roundtrip(
        est, tmp_path
    )  # joblib preserves categorical_indices -> inference int-casts
    assert np.max(np.abs(p - loaded.predict_proba(X))) <= 1e-6


def test_catboost_codes_path_unchanged_without_indices() -> None:
    pytest.importorskip("catboost")
    X, y, _ = _cat_xy()
    est = build_boosting(CATBOOST, task=Task(kind="binary"), random_state=0)
    est.feature_names = ["n0", "n1", "c0"]  # no categorical_indices injected -> codes path (NFR-3)
    est.fit(X, y)
    assert est.native_model().get_cat_feature_indices() == []  # no native cat declared, as before


def _recording_clf():
    """A fake native estimator that records the kwargs the wrapper passes to ``fit``."""
    seen: dict = {}

    class _Rec(_FakeClf):
        def fit(self, X, y, sample_weight=None, **kwargs):
            seen.update(kwargs)
            return super().fit(X, y, sample_weight)

    return _Rec, seen


def _lgbm_backend() -> _Backend:
    return _Backend(
        module="lightgbm",
        clf_attr="C",
        reg_attr="R",
        seed_kwarg="random_state",
        n_estimators_kwarg="n_estimators",
        handles_categorical=True,
    )


def test_lightgbm_branch_passes_categorical_feature() -> None:
    # FR-1: the LightGBM native branch forwards categorical_feature to fit; spy-fake, no real lib needed
    rec, seen = _recording_clf()
    est = _BoostingClassifier(_lgbm_backend(), rec, 0)
    est.feature_names = ["a", "b", "c"]
    est.categorical_indices = [2]
    est.fit(np.zeros((6, 3)), np.array([0, 1] * 3))
    assert seen.get("categorical_feature") == [2]


def test_lightgbm_codes_branch_omits_categorical_feature() -> None:
    # NFR-3: without injected indices the codes path fits exactly as before (no categorical_feature kwarg)
    rec, seen = _recording_clf()
    est = _BoostingClassifier(_lgbm_backend(), rec, 0)
    est.feature_names = ["a", "b", "c"]  # categorical_indices stays []
    est.fit(np.zeros((6, 3)), np.array([0, 1] * 3))
    assert "categorical_feature" not in seen


def test_lightgbm_native_parity(tmp_path) -> None:
    pytest.importorskip("lightgbm")
    X, y, cat_idx = _cat_xy()
    est = _native(LIGHTGBM, cat_idx, X, y)
    p = est.predict_proba(X)
    assert np.max(np.abs(p - est.predict_proba(X.copy()))) <= 1e-6  # FR-4
    loaded = _roundtrip(est, tmp_path)
    assert np.max(np.abs(p - loaded.predict_proba(X))) <= 1e-6


@pytest.mark.parametrize("backend", [CATBOOST, LIGHTGBM], ids=["catboost", "lightgbm"])
def test_native_early_stopping_converges(backend) -> None:
    pytest.importorskip(backend.module)
    X, y, cat_idx = _cat_xy()
    est = _native(backend, cat_idx, X[:450], y[:450], X_val=X[450:], y_val=y[450:])  # es tail
    assert est.predict_proba(X[:10]).shape == (10, 2)  # NFR-8: native ES path fits and predicts


def test_categorical_regularizers_in_search_space() -> None:
    # FR-7: each native backend exposes tunable categorical regularizers with a non-empty range
    lgb_space = LIGHTGBM.search_space
    for key in ("min_data_per_group", "cat_smooth"):
        assert key in lgb_space and lgb_space[key]["low"] < lgb_space[key]["high"]
    assert (
        "one_hot_max_size" in CATBOOST.search_space
        and CATBOOST.search_space["one_hot_max_size"]["low"]
        < CATBOOST.search_space["one_hot_max_size"]["high"]
    )


def test_native_categorical_marker_only_on_capable_backend(monkeypatch) -> None:
    # ADR-0088: the SupportsNativeCategorical marker is present ONLY on a backend that handles
    # categories natively; categorical_indices defaults to [] (a no-op until the use-case injects).
    capable = build_boosting(
        replace(_fake_backend(monkeypatch), handles_categorical=True),
        task=Task(kind="binary"),
        random_state=0,
    )
    assert isinstance(capable, SupportsNativeCategorical)
    assert capable.categorical_indices == []

    plain = build_boosting(_fake_backend(monkeypatch), task=Task(kind="binary"), random_state=0)
    assert not isinstance(plain, SupportsNativeCategorical)


def test_classifier_wrapper_shapes_and_kwargs(monkeypatch, caplog) -> None:
    backend = _fake_backend(monkeypatch)
    X = np.zeros((10, 3))
    y = np.array([0, 1] * 5)
    est = build_boosting(backend, task=Task(kind="binary"), random_state=7)
    assert isinstance(est, ProbabilisticEstimator)
    with caplog.at_level(logging.WARNING, logger="honestml"):
        est.fit(X, y)
    assert any("early stopping" in r.getMessage() for r in caplog.records)  # ADR-0020 §2
    # the wrapper forwards the module default tree-count (whatever it is) -> reference the constant, not a
    # literal, so the test is robust to the prod default and the test-session speed cap (tests/conftest.py)
    assert est._model.kwargs == {
        "n_estimators": boosting_mod._N_ESTIMATORS,
        "seed": 7,
        "opt": 1,
    }  # seed→native kwarg
    assert est.predict(X).shape == (10,)
    assert est.predict_proba(X).shape == (10, 2)
    assert set(est.classes_.tolist()) == {0, 1}
    assert isinstance(est, SupportsFeatureImportance)
    assert est.feature_importances.shape == (3,)  # 1-D


@pytest.mark.parametrize(
    "y",
    [
        np.array(["b", "a", "c", "a", "b", "c"]),  # strings
        np.array([1, 2, 1, 2, 1, 2]),  # non-{0,1} ints
        np.array([10, 20, 30, 10, 20, 30]),  # non-contiguous ints
    ],
    ids=["strings", "one-two", "non-contiguous"],
)
def test_int_label_backend_codes_and_decodes(monkeypatch, y) -> None:
    # ADR-0081: a backend that requires contiguous 0..K-1 (xgboost) must still accept arbitrary labels.
    # The wrapper codes them for the native fit (else _StrictIntClf raises) and decodes predict back; the
    # native proba columns are already in sorted-class order, so classes_ stays the user's labels.
    backend = _int_label_backend(monkeypatch)
    X = np.zeros((y.size, 3))
    kind = "multiclass" if np.unique(y).size > 2 else "binary"
    est = build_boosting(backend, task=Task(kind=kind), random_state=0)
    est.fit(X, y)  # must NOT raise despite labels that are not 0..K-1
    assert np.array_equal(est.classes_, np.unique(y))  # original labels, not the 0..K-1 codes
    pred = est.predict(X)
    assert set(np.unique(pred).tolist()) <= set(
        np.unique(y).tolist()
    )  # decoded to the user's space
    assert est.predict_proba(X).shape == (y.size, np.unique(y).size)


def test_int_label_backend_codes_the_es_tail(monkeypatch) -> None:
    # the es-validation tail is coded with the SAME label map (else the strict-int native fit raises).
    backend = _int_label_backend(monkeypatch)
    y = np.array(["a", "b"] * 6)
    X = np.zeros((12, 2))
    est = build_boosting(backend, task=Task(kind="binary"), random_state=0)
    est.fit(X[:9], y[:9], X_val=X[9:], y_val=y[9:])  # must not raise
    assert np.array_equal(est.classes_, np.array(["a", "b"]))


def test_int_label_backend_rejects_es_class_absent_from_fit(monkeypatch) -> None:
    # F112: an es tail carrying a class absent from the fit fold must fail loudly (a clear domain error
    # the candidate isolation records), not silently mis-code the validation labels.
    backend = _int_label_backend(monkeypatch)
    X = np.zeros((10, 2))
    est = build_boosting(backend, task=Task(kind="binary"), random_state=0)
    with pytest.raises(SchemaValidationError, match="absent from the fit fold"):
        est.fit(X[:8], np.array(["a", "b"] * 4), X_val=X[8:], y_val=np.array(["a", "c"]))


def test_native_backend_passes_labels_through(monkeypatch) -> None:
    # the fix is opt-in by backend flag: catboost/lightgbm (requires_int_labels=False) keep consuming
    # labels natively — classes_ comes from the native model, _label_index stays None (no coding).
    backend = _fake_backend(monkeypatch)  # requires_int_labels defaults False
    est = build_boosting(backend, task=Task(kind="binary"), random_state=0)
    est.fit(np.zeros((6, 2)), np.array(["x", "y"] * 3))
    assert est._label_index is None
    assert set(est.classes_.tolist()) == {"x", "y"}


def test_regressor_wrapper_has_no_proba(monkeypatch) -> None:
    backend = _fake_backend(monkeypatch)
    X = np.zeros((8, 4))
    y = np.arange(8, dtype=float)
    est = build_boosting(backend, task=Task(kind="regression"), random_state=0)
    assert not isinstance(est, ProbabilisticEstimator)
    est.fit(X, y)
    assert est.predict(X).shape == (8,)
    assert est.feature_importances.shape == (4,)
    assert not hasattr(est, "predict_proba")


# --- M7a HPO: tuned params forwarding (ADR-0061 §4) -------------------------


def test_tuned_params_forwarded_and_override(monkeypatch) -> None:
    # tuned hyperparameters reach the native ctor; the tuned tree count OVERRIDES the fixed 300
    # (params last in _make), and an extra tuned key is forwarded (ADR-0061 §4).
    backend = _fake_backend(monkeypatch)  # n_estimators_kwarg="n_estimators"
    est = build_boosting(
        backend, task=Task(kind="binary"), random_state=7, n_estimators=77, learning_rate=0.05
    )
    est.fit(np.zeros((6, 2)), np.array([0, 1] * 3))
    assert est._model.kwargs["n_estimators"] == 77  # not the default 300
    assert est._model.kwargs["learning_rate"] == 0.05
    assert est._model.kwargs["seed"] == 7


def test_tuned_n_estimators_overrides_default_catboost_style(monkeypatch) -> None:
    # catboost's tree-count kwarg is `iterations`; a tuned `iterations` must override 300 on the
    # same key (the R2 collision fix relies on the search_space key == n_estimators_kwarg).
    mod = types.ModuleType("_fakecat")
    mod.FakeClf = _FakeClf  # type: ignore[attr-defined]
    mod.FakeReg = _FakeReg  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "_fakecat", mod)
    boosting_mod._warned_backends.discard("_fakecat")
    backend = _Backend(
        module="_fakecat",
        clf_attr="FakeClf",
        reg_attr="FakeReg",
        seed_kwarg="random_seed",
        n_estimators_kwarg="iterations",
    )
    est = build_boosting(backend, task=Task(kind="binary"), random_state=1, iterations=88)
    est.fit(np.zeros((6, 2)), np.array([0, 1] * 3))
    assert est._model.kwargs["iterations"] == 88  # not 300; single key, no collision
    assert "n_estimators" not in est._model.kwargs


def test_es_ceiling_overrides_tuned_n_estimators(monkeypatch) -> None:
    # F029: on the ES path the caller passes the generous _N_ESTIMATORS_ES ceiling (ADR-0080), which
    # must WIN over a tuned n_estimators — `_make(n_estimators=...)` is applied after **params, so a
    # tuned count no longer silently caps the ES tree budget. The no-ES path keeps the tuned override.
    backend = _fake_backend(monkeypatch)  # n_estimators_kwarg="n_estimators"
    est = build_boosting(backend, task=Task(kind="binary"), random_state=7, n_estimators=77)
    assert est._make(n_estimators=_N_ESTIMATORS_ES).kwargs["n_estimators"] == _N_ESTIMATORS_ES
    assert (
        est._make().kwargs["n_estimators"] == 77
    )  # no-ES path: the tuned count still overrides 300


def test_catboost_subsample_pairs_bernoulli_for_multiclass_only() -> None:
    # CatBoost's multiclass default bootstrap (Bayesian) rejects `subsample`; the wrapper pairs a tuned
    # subsample with Bernoulli for multiclass ONLY (binary/regression default to MVS, which accepts it).
    cat = _Backend(
        module="catboost",
        clf_attr="FakeClf",
        reg_attr="FakeReg",
        seed_kwarg="random_seed",
        n_estimators_kwarg="iterations",
    )
    mc = boosting_mod._BoostingClassifier(cat, _FakeClf, 0, {"subsample": 0.7})
    mc._encode_targets_fit(np.array([0, 1, 2, 0, 1, 2]))  # >2 classes -> multiclass
    assert mc._make().kwargs["bootstrap_type"] == "Bernoulli"

    binary = boosting_mod._BoostingClassifier(cat, _FakeClf, 0, {"subsample": 0.7})
    binary._encode_targets_fit(np.array([0, 1, 0, 1]))  # 2 classes -> binary, untouched
    assert "bootstrap_type" not in binary._make().kwargs

    reg = boosting_mod._BoostingRegressor(cat, _FakeReg, 0, {"subsample": 0.7})
    assert "bootstrap_type" not in reg._make().kwargs  # regressor never multiclass

    no_sub = boosting_mod._BoostingClassifier(cat, _FakeClf, 0, {})
    no_sub._encode_targets_fit(np.array([0, 1, 2]))
    assert "bootstrap_type" not in no_sub._make().kwargs  # only paired WITH subsample


# --- install-gated paths ----------------------------------------------------


def test_build_maps_importerror_to_missing_dependency() -> None:
    # the registry maps a build-time ImportError (missing extra) to a clear, actionable
    # error — tested deterministically, independent of which boosting libs are installed.
    def _needs_absent(**kwargs):
        raise ImportError("No module named 'absent'")

    registry = ComponentRegistry(
        "honestml.models",
        [
            ComponentDescriptor(
                name="needs_absent",
                spec=ModelSpec(name="needs_absent", capabilities=Capabilities(tasks=("binary",))),
                build=_needs_absent,
            )
        ],
    )
    with pytest.raises(MissingDependencyError, match="needs_absent"):
        registry.build("needs_absent", task=Task(kind="binary"), random_state=0)


def test_boosting_listed_in_available_models() -> None:
    listing = available_models()
    assert {"catboost", "lightgbm", "xgboost"} <= set(listing)
    assert listing["catboost"].handles_missing is True  # ADR-0020 §2


def test_lazy_no_heavy_import_in_subprocess() -> None:
    """A clean interpreter: import + descriptor discovery pulls no boosting lib (NFR-2)."""
    code = (
        "import sys; from honestml.composition.registry import model_registry; "
        "model_registry().descriptors(); "
        "print(','.join(m for m in ('catboost','lightgbm','xgboost') if m in sys.modules))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == ""


# --- participates end-to-end when the extra is installed (CI with extras) ----


def test_lightgbm_predict_silences_feature_name_warning() -> None:
    # finding #4: lightgbm auto-names numpy columns at fit, then warns at predict on the same nameless
    # numpy — the wrapper filters exactly that cosmetic message so it cannot flood real-run logs.
    pytest.importorskip("lightgbm")
    import warnings

    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(60, 4)), np.array([0, 1] * 30)
    est = build_boosting(LIGHTGBM, task=Task(kind="binary"), random_state=0)
    est.fit(X, y)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        est.predict(X)
        est.predict_proba(X)
    assert not any("valid feature names" in str(r.message) for r in rec)


def _best_iteration(model, module: str) -> int:
    if module == "lightgbm":
        return int(model.best_iteration_)
    if module == "xgboost":
        return int(model.best_iteration)
    return int(model.get_best_iteration())  # catboost


@pytest.mark.parametrize("backend", [LIGHTGBM, CATBOOST, XGBOOST])
def test_early_stopping_cuts_tree_count(backend: _Backend) -> None:
    # ADR-0080: with a carved es tail the fit stops well before the generous _N_ESTIMATORS_ES ceiling.
    pytest.importorskip(backend.module)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 5))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    Xv = rng.normal(size=(80, 5))
    yv = (Xv[:, 0] + Xv[:, 1] > 0).astype(int)
    est = build_boosting(backend, task=Task(kind="binary"), random_state=0)
    est.fit(X, y, X_val=Xv, y_val=yv)
    assert _best_iteration(est._model, backend.module) < _N_ESTIMATORS_ES
    assert est.predict(X).shape == (200,)
    assert est.predict_proba(X).shape == (200, 2)


def test_no_es_warning_skipped_when_es_tail_present(monkeypatch, caplog) -> None:
    # the "without early stopping" advisory must NOT fire when an es tail is supplied (real lightgbm path).
    pytest.importorskip("lightgbm")
    boosting_mod._warned_backends.discard("lightgbm")
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(120, 4)), (rng.normal(size=120) > 0).astype(int)
    est = build_boosting(LIGHTGBM, task=Task(kind="binary"), random_state=0)
    with caplog.at_level(logging.WARNING, logger="honestml"):
        est.fit(X[:90], y[:90], X_val=X[90:], y_val=y[90:])
    assert not any("early stopping" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize("lib", ["catboost", "lightgbm", "xgboost"])
def test_boosting_participates_when_installed(lib) -> None:
    pytest.importorskip(lib)
    from sklearn.datasets import make_classification

    from honestml import AutoML

    X, y = make_classification(n_samples=80, n_features=6, n_informative=4, random_state=0)
    model = AutoML(task="binary", models=(lib, "linear"), random_state=0).fit(X, y)
    assert lib in [e.model_id for e in model.leaderboard_]


@pytest.mark.parametrize(
    "y_map",
    [
        lambda y: np.array([f"Class_{v}" for v in y]),  # string multiclass labels
        lambda y: y + 1,  # non-{0,1,2} contiguous-from-1 ints
    ],
    ids=["str-labels", "from-one"],
)
def test_xgboost_participates_multiclass_non01_labels(y_map) -> None:
    # ADR-0081: xgboost 3.x rejects any target that is not contiguous 0..K-1, so before the wrapper
    # coded labels it silently dropped from every classification run whose labels were not {0..K-1}
    # (caught only here because the suite previously fed it 0..K-1 labels). It must now participate.
    pytest.importorskip("xgboost")
    from sklearn.datasets import make_classification

    from honestml import AutoML

    X, y = make_classification(
        n_samples=150, n_features=6, n_informative=4, n_redundant=0, n_classes=3, random_state=0
    )
    y = y_map(y)
    model = AutoML(
        task="multiclass", metric="log_loss", models=("xgboost", "baseline"), random_state=0
    ).fit(X, y)
    assert "xgboost" in [e.model_id for e in model.leaderboard_]
    assert set(np.unique(model.predict(X)).tolist()) <= set(np.unique(y).tolist())


def test_catboost_multiclass_subsample_fits() -> None:
    # regression for the 05-otto rough edge: a tuned `subsample` on multiclass catboost must fit —
    # the multiclass default bootstrap (Bayesian) rejects subsample, so the wrapper pairs it with
    # Bernoulli. Without the fix this raises "default bootstrap type (bayesian) doesn't support subsample".
    pytest.importorskip("catboost")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(150, 5))
    y = rng.integers(0, 3, 150)
    est = build_boosting(
        CATBOOST, task=Task(kind="multiclass"), random_state=0, subsample=0.7, iterations=20
    )
    est.fit(X, y)  # must NOT raise
    assert est.predict_proba(X).shape == (150, 3)
    assert est._model.get_all_params()["bootstrap_type"] == "Bernoulli"
