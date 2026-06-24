"""Estimator adapters (ADR-0013, generalized ADR-0020).

Lightweight, core-only sklearn models implementing the ``Estimator`` /
``ProbabilisticEstimator`` ports on the numpy boundary, per task kind:
``Baseline{Classifier,Regressor}`` (the ``FR-2`` baseline) and
``Linear{Classifier,Regressor}`` (the ``FR-2`` linear model). Categorical codes arrive as
plain numeric columns (native cat handling is a follow-up). ``X_val``/``y_val`` are ignored:
these models have no early stopping, so the use-case trains them on ``fit ∪ es`` and the
``es`` tail is not lost (ADR-0010 §6). The boosting zoo (behind extras) lives in
:mod:`honestml.adapters.boosting`.
"""

from __future__ import annotations

import numpy as np
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from honestml.core.exceptions import NotFittedError
from honestml.core.ports.model_spec import Capabilities

# handles_cat=False: codes are fed as numeric columns (native cat is a follow-up).
# handles_missing=True: a median SimpleImputer prefixes the Pipeline, fit per-fold and leak-free
# (ADR-0078) — NaN no longer evicts the simple candidates from the zoo (finding #6).
_CLF_CAPS = Capabilities(
    tasks=("binary", "multiclass"), probabilistic=True, handles_cat=False, handles_missing=True
)
_REG_CAPS = Capabilities(
    tasks=("regression",), probabilistic=False, handles_cat=False, handles_missing=True
)


class BaselineClassifier:
    """Prior-frequency baseline (sklearn ``DummyClassifier(strategy="prior")``)."""

    capabilities = _CLF_CAPS

    def __init__(self) -> None:
        self.feature_names: list[str] = []
        self._model: Pipeline | None = None
        self.classes_: np.ndarray | None = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> BaselineClassifier:
        # the imputer makes the prior baseline NaN-safe (ADR-0078); Dummy ignores X but sklearn
        # still validates it, so raw NaN would raise without it.
        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("dummy", DummyClassifier(strategy="prior")),
            ]
        )
        model.fit(X, y, dummy__sample_weight=sample_weight)
        self._model = model
        self.classes_ = model.classes_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._fitted().predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._fitted().predict_proba(X)

    def _fitted(self) -> Pipeline:
        if self._model is None:
            raise NotFittedError(f"{type(self).__name__} is not fitted; call fit() first")
        return self._model


class BaselineRegressor:
    """Mean baseline (sklearn ``DummyRegressor(strategy="mean")``)."""

    capabilities = _REG_CAPS

    def __init__(self) -> None:
        self.feature_names: list[str] = []
        self._model: Pipeline | None = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> BaselineRegressor:
        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("dummy", DummyRegressor(strategy="mean")),
            ]
        )
        model.fit(X, y, dummy__sample_weight=sample_weight)
        self._model = model
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._fitted().predict(X)

    def _fitted(self) -> Pipeline:
        if self._model is None:
            raise NotFittedError(f"{type(self).__name__} is not fitted; call fit() first")
        return self._model


class LinearClassifier:
    """Logistic regression (binary + multiclass); ``feature_importances`` via ``coef_``."""

    capabilities = _CLF_CAPS

    def __init__(self, *, random_state: int = 42, max_iter: int = 1000, C: float = 1.0) -> None:
        self.random_state = random_state
        self.max_iter = max_iter
        self.C = C
        self.feature_names: list[str] = []
        self._model: Pipeline | None = None
        self.classes_: np.ndarray | None = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> LinearClassifier:
        # impute (ADR-0078) then standardize (finding #5) before the lbfgs solver: NaN would crash the
        # scaler, and unscaled features with disparate ranges leave it short of convergence. Both steps
        # live INSIDE the model as a Pipeline so joblib, ONNX (skl2onnx converts the whole Pipeline) and
        # the parity gate all carry them, fit per-fold (leak-free) (ADR-0013).
        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "lr",
                    LogisticRegression(
                        random_state=self.random_state, max_iter=self.max_iter, C=self.C
                    ),
                ),
            ]
        )
        model.fit(X, y, lr__sample_weight=sample_weight)
        self._model = model
        self.classes_ = model.classes_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._fitted().predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._fitted().predict_proba(X)

    @property
    def feature_importances(self) -> np.ndarray:
        # 1-D, uniform for binary & multiclass: mean |coef| across classes (ADR-0020 §2). On the scaled
        # features the coefficients are directly comparable across columns (a better importance, finding #5).
        return np.abs(self._fitted()[-1].coef_).mean(axis=0)

    def _fitted(self) -> Pipeline:
        if self._model is None:
            raise NotFittedError(f"{type(self).__name__} is not fitted; call fit() first")
        return self._model


class LinearRegressor:
    """Ridge regression (regression); ``feature_importances`` via ``coef_``."""

    capabilities = _REG_CAPS

    def __init__(self, *, random_state: int = 42, alpha: float = 1.0) -> None:
        self.random_state = random_state
        self.alpha = alpha
        self.feature_names: list[str] = []
        self._model: Pipeline | None = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> LinearRegressor:
        # impute then standardize inside the model (ADR-0078 + finding #5), symmetric with
        # LinearClassifier — keeps both steps in the Pipeline so every serialization/parity path carries them.
        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=self.alpha, random_state=self.random_state)),
            ]
        )
        model.fit(X, y, ridge__sample_weight=sample_weight)
        self._model = model
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._fitted().predict(X)

    @property
    def feature_importances(self) -> np.ndarray:
        return np.abs(self._fitted()[-1].coef_).ravel()

    def _fitted(self) -> Pipeline:
        if self._model is None:
            raise NotFittedError(f"{type(self).__name__} is not fitted; call fit() first")
        return self._model
