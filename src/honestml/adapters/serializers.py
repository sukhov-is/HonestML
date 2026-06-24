"""Model-body serializers — the ``ModelSerializer`` adapters (ADR-0069/0070).

``JoblibSerializer`` is the catch-all default (the M8 behavior, unchanged). The native
serializers persist a boosting body through each library's documented-stable API —
XGBoost UBJSON, CatBoost cbm, LightGBM text — instead of pickle: the round-trip is exact
(SPIKE-0003) and the body survives library upgrades that break pickle (NFR-SER-5).
Libraries are imported lazily inside ``save``/``load``; a missing runtime surfaces as
``MissingDependencyError`` before any deserialization (ADR-0070 §6).
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from importlib.util import find_spec
from pathlib import Path
from typing import Any, cast

import numpy as np

from honestml.adapters.boosting import (
    CATBOOST,
    XGBOOST,
    _Backend,
    _BoostingClassifier,
    _BoostingRegressor,
)
from honestml.core import (
    Estimator,
    MissingDependencyError,
    ModelFiles,
    SchemaValidationError,
    SupportsNativeModel,
)


def _is_classification(manifest: Mapping[str, Any]) -> bool:
    return manifest["task"]["kind"] in ("binary", "multiclass")


def _body_name(manifest: Mapping[str, Any], default: str) -> str:
    # .name confines the file to the artifact directory (ADR-0067 §2 anti-traversal)
    return Path(manifest.get("model_file", default)).name


def _check_classes(restored: np.ndarray | None, manifest: Mapping[str, Any]) -> None:
    """ADR-0070 §2: the natively-restored class order must match the manifest's global order."""
    declared = np.asarray(manifest["classes"])
    if restored is None or not np.array_equal(np.asarray(restored), declared):
        raise SchemaValidationError(
            f"native body classes {restored!r} do not match manifest classes {declared.tolist()!r}"
        )


class JoblibSerializer:
    """Pickle body via joblib — the M8 default and the catch-all fallback (ADR-0069 §2)."""

    model_type = "joblib"
    _file = "model.joblib"

    def can_serialize(self, estimator: Estimator) -> bool:
        return True

    def save(self, estimator: Estimator, directory: Path) -> ModelFiles:
        import joblib

        joblib.dump(estimator, directory / self._file)
        return ModelFiles(files=(self._file,))

    def load(self, directory: Path, manifest: Mapping[str, Any]) -> Estimator:
        import joblib

        return cast(Estimator, joblib.load(directory / _body_name(manifest, self._file)))


class _NativeBoostingSerializer:
    """Shared native shape: match the wrapper via ``SupportsNativeModel``, re-wrap on load.

    ``load`` re-creates the library's sklearn-style estimator from the ``_Backend`` naming
    (the same data ``build_boosting`` uses) and re-wraps it via ``from_native`` (ADR-0070 §5);
    LightGBM overrides it — its text body reloads as a raw ``Booster`` (ADR-0070 §4).
    """

    model_type: str
    _backend: _Backend
    _file: str

    def can_serialize(self, estimator: Estimator) -> bool:
        return (
            isinstance(estimator, SupportsNativeModel)
            and estimator.native_format == self.model_type
        )

    def load(self, directory: Path, manifest: Mapping[str, Any]) -> Estimator:
        self._require_runtime()
        module = importlib.import_module(self._backend.module)
        is_clf = _is_classification(manifest)
        attr = self._backend.clf_attr if is_clf else self._backend.reg_attr
        model = getattr(module, attr)()
        model.load_model(str(directory / _body_name(manifest, self._file)))
        wrapper_cls = _BoostingClassifier if is_clf else _BoostingRegressor
        # int-coded backends (xgboost, ADR-0081) restore their original label order from the manifest;
        # the native body holds only 0..K-1 codes. Native backends ignore it (labels are in the body).
        classes = np.asarray(manifest["classes"]) if is_clf else None
        wrapper = wrapper_cls.from_native(self._backend, model, classes=classes)
        # restore native categorical indices so CatBoost int-casts the cat block on predict (FR-5, ADR-0091);
        # absent (old/non-native artifact) -> [] -> codes path. LightGBM bakes categories into its booster
        # and loads as a thin adapter (overridden load), so it needs no index here.
        wrapper.categorical_indices = list(manifest.get("categorical_indices") or [])
        if is_clf:
            _check_classes(cast(_BoostingClassifier, wrapper).classes_, manifest)
        return wrapper

    def _require_runtime(self) -> None:
        if find_spec(self.model_type) is None:
            raise MissingDependencyError(self.model_type)


class XGBoostSerializer(_NativeBoostingSerializer):
    """XGBoost UBJSON body — ``save_model``/``load_model`` guarantee backward-compat (ADR-0070 §2)."""

    model_type = "xgboost"
    _backend = XGBOOST
    _file = "model.ubj"

    def save(self, estimator: Estimator, directory: Path) -> ModelFiles:
        cast(SupportsNativeModel, estimator).native_model().save_model(str(directory / self._file))
        return ModelFiles(files=(self._file,), required_extra=self.model_type)


class CatBoostSerializer(_NativeBoostingSerializer):
    """CatBoost cbm body — the library's portable native format (ADR-0070 §3)."""

    model_type = "catboost"
    _backend = CATBOOST
    _file = "model.cbm"

    def save(self, estimator: Estimator, directory: Path) -> ModelFiles:
        cast(SupportsNativeModel, estimator).native_model().save_model(
            str(directory / self._file), format="cbm"
        )
        return ModelFiles(files=(self._file,), required_extra=self.model_type)


class LightGbmSerializer(_NativeBoostingSerializer):
    """LightGBM text body; classification re-wraps the raw ``Booster`` (ADR-0070 §4).

    ``lgb.Booster`` carries no sklearn API, so the load path returns thin adapters: the
    regressor passes ``predict`` through, the classifier synthesizes ``predict_proba``/
    ``classes_`` from the manifest's global class order — values are exact (SPIKE-0003),
    this is API-shape recovery, not a numeric approximation.
    """

    model_type = "lightgbm"
    _file = "model.txt"

    def save(self, estimator: Estimator, directory: Path) -> ModelFiles:
        booster = cast(SupportsNativeModel, estimator).native_model().booster_
        booster.save_model(str(directory / self._file))
        return ModelFiles(files=(self._file,), required_extra=self.model_type)

    def load(self, directory: Path, manifest: Mapping[str, Any]) -> Estimator:
        self._require_runtime()
        import lightgbm as lgb

        booster = lgb.Booster(model_file=str(directory / _body_name(manifest, self._file)))
        if not _is_classification(manifest):
            return _NativeLgbmRegressor(booster)
        return _NativeLgbmClassifier(booster, np.asarray(manifest["classes"]))


class _NativeLgbmClassifier:
    """Thin ``ProbabilisticEstimator`` over a text-loaded LightGBM ``Booster`` (ADR-0070 §4).

    ``Booster.predict`` returns P(``classes_[1]``) 1-D for binary and ``(n, K)`` in class-index
    order for multiclass — exactly the sklearn wrapper's ``predict_proba`` columns (SPIKE-0003,
    max|Δ|=0.0). ``classes_`` comes from the manifest's global class order (the sklearn-sorted
    labels the booster was trained against).
    """

    def __init__(self, booster: Any, classes: np.ndarray) -> None:
        self._booster = booster
        self.classes_: np.ndarray = np.asarray(classes)
        self.feature_names: list[str] = []

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> _NativeLgbmClassifier:
        raise NotImplementedError(
            "a natively-loaded LightGBM model is inference-only; train a new model to refit"
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raw = np.asarray(self._booster.predict(X))
        if raw.ndim == 1:  # binary: P(classes_[1]) -> sklearn's [1 - p, p] columns
            return np.column_stack([1.0 - raw, raw])
        return raw

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class _NativeLgbmRegressor:
    """Thin ``Estimator`` over a text-loaded LightGBM ``Booster`` — ``predict`` passthrough."""

    def __init__(self, booster: Any) -> None:
        self._booster = booster
        self.feature_names: list[str] = []

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> _NativeLgbmRegressor:
        raise NotImplementedError(
            "a natively-loaded LightGBM model is inference-only; train a new model to refit"
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self._booster.predict(X))
