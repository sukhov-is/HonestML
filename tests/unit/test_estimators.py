"""M3b (ADR-0020): per-kind estimator adapters on the numpy boundary."""

from __future__ import annotations

import io

import joblib
import numpy as np
import pytest
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from honestml.adapters import (
    BaselineClassifier,
    BaselineRegressor,
    LinearClassifier,
    LinearRegressor,
)
from honestml.core import (
    Estimator,
    NotFittedError,
    ProbabilisticEstimator,
    SupportsFeatureImportance,
)

pytestmark = pytest.mark.unit

_CLASSIFIERS = [BaselineClassifier, lambda: LinearClassifier(random_state=0)]
_REGRESSORS = [BaselineRegressor, lambda: LinearRegressor(random_state=0)]


def _xy_binary(n: int = 60, d: int = 4, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = (X[:, 0] + 0.3 * rng.normal(size=n) > 0).astype(int)
    return X, y


def _xy_multiclass(n: int = 90, d: int = 4, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = rng.integers(0, 3, size=n)
    return X, y


def _xy_regression(n: int = 80, d: int = 4, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = 2.0 * X[:, 0] - X[:, 1] + 0.1 * rng.normal(size=n)
    return X, y


@pytest.mark.parametrize("factory", _CLASSIFIERS)
def test_classifiers_implement_ports(factory) -> None:
    est = factory()
    assert isinstance(est, Estimator)
    assert isinstance(est, ProbabilisticEstimator)
    assert {"binary", "multiclass"} <= set(est.capabilities.tasks)
    assert est.capabilities.probabilistic is True


@pytest.mark.parametrize("factory", _REGRESSORS)
def test_regressors_implement_ports(factory) -> None:
    est = factory()
    assert isinstance(est, Estimator)
    assert not isinstance(est, ProbabilisticEstimator)
    assert est.capabilities.tasks == ("regression",)
    assert est.capabilities.probabilistic is False


@pytest.mark.parametrize("factory", _CLASSIFIERS)
def test_binary_fit_predict_shapes(factory) -> None:
    X, y = _xy_binary()
    est = factory().fit(X, y)
    assert est.predict(X).shape == (X.shape[0],)
    proba = est.predict_proba(X)
    assert proba.shape == (X.shape[0], 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert set(est.classes_.tolist()) == {0, 1}


@pytest.mark.parametrize("factory", _CLASSIFIERS)
def test_multiclass_fit_predict_shapes(factory) -> None:
    X, y = _xy_multiclass()
    est = factory().fit(X, y)
    proba = est.predict_proba(X)
    assert proba.shape == (X.shape[0], 3)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert set(est.classes_.tolist()) == {0, 1, 2}


@pytest.mark.parametrize("factory", _REGRESSORS)
def test_regression_fit_predict(factory) -> None:
    X, y = _xy_regression()
    est = factory().fit(X, y)
    assert est.predict(X).shape == (X.shape[0],)
    assert not hasattr(est, "predict_proba")


def test_linear_classifier_feature_importance_is_1d() -> None:
    Xb, yb = _xy_binary()
    Xm, ym = _xy_multiclass()
    eb = LinearClassifier(random_state=0).fit(Xb, yb)
    em = LinearClassifier(random_state=0).fit(Xm, ym)
    assert isinstance(eb, SupportsFeatureImportance)
    assert eb.feature_importances.shape == (Xb.shape[1],)  # binary -> 1-D
    assert em.feature_importances.shape == (Xm.shape[1],)  # multiclass -> 1-D aggregate


def test_linear_regressor_feature_importance_is_1d() -> None:
    X, y = _xy_regression()
    est = LinearRegressor(random_state=0).fit(X, y)
    assert isinstance(est, SupportsFeatureImportance)
    assert est.feature_importances.shape == (X.shape[1],)


def test_linear_classifier_learns_signal() -> None:
    X, y = _xy_binary(n=200)
    est = LinearClassifier(random_state=0).fit(X, y)
    assert (est.predict(X) == y).mean() > 0.7


def test_linear_regressor_learns_signal() -> None:
    X, y = _xy_regression(n=200)
    est = LinearRegressor(random_state=0).fit(X, y)
    # the ridge model should track the linear target far better than the mean baseline
    err = np.mean((est.predict(X) - y) ** 2)
    base = np.mean((y.mean() - y) ** 2)
    assert err < 0.5 * base


def test_baseline_ignores_x_and_matches_prior() -> None:
    # the baseline bypasses X entirely (constant stand-in, no imputer pass over the full matrix);
    # prior probabilities depend only on y — pin them to the exact class frequencies
    X, y = _xy_binary(n=80)
    proba = BaselineClassifier().fit(X, y).predict_proba(X)
    prior = np.bincount(y) / len(y)
    assert np.array_equal(proba, np.tile(prior, (len(y), 1)))


@pytest.mark.parametrize("factory", _CLASSIFIERS + _REGRESSORS)
def test_estimators_handle_nan_via_imputer(factory) -> None:
    # ADR-0078: a median SimpleImputer prefixes the Pipeline, so raw NaN no longer crashes the fit.
    X, y = _xy_binary(n=80)
    X = X.copy()
    X[::5, 0] = np.nan  # scatter missingness through one column
    est = factory().fit(X, y)
    pred = est.predict(X)
    assert pred.shape == (X.shape[0],)
    assert np.isfinite(pred.astype(float)).all()
    assert est.capabilities.handles_missing is True


def test_imputer_is_fit_per_fold_not_global() -> None:
    # the imputed value comes from the rows passed to fit (the fold's train), never from unseen rows.
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 2))
    X[0, 1] = np.nan
    train = np.arange(20)  # the median of X[train, 1] differs from the global median
    target = X[train, 0]
    est = LinearRegressor(random_state=0).fit(X[train], target)
    imputer = est._fitted().named_steps["imputer"]
    expected = float(np.nanmedian(X[train, 1]))
    assert np.isclose(imputer.statistics_[1], expected)


def test_ignores_validation_split() -> None:
    X, y = _xy_binary()
    est = LinearClassifier(random_state=0)
    fitted = est.fit(X, y, X_val=X[:5], y_val=y[:5])
    assert fitted is est


@pytest.mark.parametrize("factory", [BaselineClassifier, BaselineRegressor])
def test_predict_before_fit_raises(factory) -> None:
    with pytest.raises(NotFittedError):
        factory().predict(np.zeros((2, 3)))


def test_not_fitted_message_is_method_neutral() -> None:
    # F030: _fitted() is reached from predict / predict_proba / feature_importances — the message must
    # not falsely blame `predict` when the caller was predict_proba or feature_importances.
    est = LinearClassifier(random_state=0)
    with pytest.raises(NotFittedError, match="is not fitted"):
        est.predict_proba(np.zeros((2, 4)))
    with pytest.raises(NotFittedError, match="is not fitted"):
        _ = est.feature_importances


def _legacy_baseline_artifact(kind: str, X: np.ndarray, y: np.ndarray) -> object:
    """A 1.0.0 baseline artifact: the CURRENT class holding the legacy ``Pipeline([SimpleImputer, Dummy])``
    ``_model`` state (exactly what unpickling a 1.0.0 dump restores), round-tripped through joblib like a load."""
    dummy = DummyClassifier(strategy="prior") if kind == "clf" else DummyRegressor(strategy="mean")
    legacy = Pipeline([("imputer", SimpleImputer(strategy="median")), ("dummy", dummy)])
    legacy.fit(X, y, dummy__sample_weight=None)
    est = BaselineClassifier() if kind == "clf" else BaselineRegressor()
    est._model = legacy  # type: ignore[assignment]
    if kind == "clf":
        est.classes_ = legacy.classes_  # type: ignore[attr-defined]
    buf = io.BytesIO()
    joblib.dump(est, buf)
    buf.seek(0)
    return joblib.load(buf)


def test_legacy_1_0_0_baseline_artifact_still_predicts() -> None:
    # F122: a 1.0.0 artifact stored _model as Pipeline([SimpleImputer, Dummy]); ADR-0101 switched 1.1 to a
    # bare Dummy fed a constant column, so a loaded legacy pickle's imputer got 1 feature and predict raised a
    # raw sklearn ValueError. The legacy Pipeline must be fed the real X (imputer runs, Dummy prior/mean unchanged).
    Xc, yc = _xy_binary(n=80)
    clf = _legacy_baseline_artifact("clf", Xc, yc)
    prior = np.bincount(yc) / len(yc)
    assert np.array_equal(clf.predict_proba(Xc), np.tile(prior, (len(yc), 1)))
    assert clf.predict(Xc).shape == (Xc.shape[0],)

    Xr, yr = _xy_regression(n=80)
    reg = _legacy_baseline_artifact("reg", Xr, yr)
    assert np.allclose(reg.predict(Xr), yr.mean())
