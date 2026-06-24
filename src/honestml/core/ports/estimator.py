"""The ``Estimator`` port and role-interfaces.

The base ``Estimator`` is only ``fit``/``predict`` (interchangeable for every
``Task.kind``, regression included). Optional abilities are separate role-
interfaces so the use-case depends on exactly the slice it needs, without
``hasattr`` branching (ISP/LSP): ``predict_proba`` is classification-only,
importance and SHAP are opt-in. The boundary is numpy; ``sample_weight``
is a first-class optional argument.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Estimator(Protocol):
    """Minimal model contract: fit on numpy, predict numpy."""

    feature_names: list[str]

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> Estimator: ...

    def predict(self, X: np.ndarray) -> np.ndarray: ...


@runtime_checkable
class ProbabilisticEstimator(Estimator, Protocol):
    """Classification models that expose class probabilities.

    ``classes_`` gives the column order of ``predict_proba`` (sklearn convention),
    so the use-case can select the positive-class column.
    """

    classes_: np.ndarray

    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...


@runtime_checkable
class SupportsFeatureImportance(Protocol):
    """Models exposing per-feature importances (role-interface, ISP)."""

    @property
    def feature_importances(self) -> np.ndarray: ...


@runtime_checkable
class SupportsShap(Protocol):
    """Models exposing SHAP values (role-interface, optional)."""

    def shap_values(self, X: np.ndarray) -> np.ndarray: ...


@runtime_checkable
class SupportsEarlyStopping(Protocol):
    """Models that early-stop on a held-out ``es`` tail (role-interface, ADR-0080).

    The marker (mirrors ``Capabilities.supports_early_stopping``) is set ``True`` by models that consume
    the fold's ``es`` rows as a validation tail; for them the use-case routes ``fit(..., X_val=, y_val=)``,
    others merge ``fit ∪ es``. ``isinstance`` treats the marker's presence as the capability — it is set
    only when supported — so a model without it is simply not early-stopping (no ``hasattr`` branching).
    """

    supports_early_stopping: bool


@runtime_checkable
class SupportsNativeCategorical(Protocol):
    """Models that consume categorical columns natively (role-interface, ADR-0088).

    Set ``True`` per-backend by wrappers with native categorical handling (CatBoost ``cat_features`` /
    LightGBM ``categorical_feature``); the marker is present **only** on capable wrappers, so
    ``isinstance`` is the capability (no ``hasattr`` branching, mirroring ``SupportsEarlyStopping``).
    ``categorical_indices`` are the positions of CATEGORICAL columns in the design matrix, **injected
    by the use-case before fit** (like ``feature_names``); ``[]`` is a legitimate no-op (native-capable
    model on a dataset without categories — equivalent to the codes path).
    """

    supports_native_categorical: bool
    categorical_indices: list[int]


@runtime_checkable
class SupportsNativeModel(Protocol):
    """Models exposing the underlying native library object (role-interface).

    Implemented only by the boosting wrappers: a native serializer matches via ``isinstance``
    and persists ``native_model()`` through the library's documented-stable save API
    instead of pickling the wrapper. ``native_format`` names the backing library
    (``"xgboost"``/``"catboost"``/``"lightgbm"``) and doubles as the artifact ``model_type``.
    """

    @property
    def native_format(self) -> str: ...

    def native_model(self) -> Any: ...
