"""Boosting estimator adapters (ADR-0020) — lazy, behind optional extras.

CatBoost / LightGBM / XGBoost wrappers implementing the ``Estimator`` /
``ProbabilisticEstimator`` ports. The heavy library is imported **only inside**
:func:`build_boosting` (NFR-2 laziness): the module top-level stays import-light, so
``import honestml`` and registry discovery never pull a boosting package. One ``_Backend`` per
library spans all task kinds; ``build`` picks the classifier (binary/multiclass) or regressor
(regression) branch by ``task.kind``.

**Early stopping** (ADR-0080): when ``run_slice`` passes a carved ``X_val``/``y_val`` tail, the
fit raises the tree count to a generous ceiling and stops on the validation metric (each library's
native API); without a tail it falls back to the conservative fixed ``n_estimators`` and logs the
"no early stopping" advisory (ADR-0020 §2). When ``categorical_indices`` is injected (native-capable
wrapper, ADR-0088/0089), CatBoost/LightGBM consume those columns natively (CatBoost int-cast Pool,
LightGBM ``categorical_feature``); otherwise codes are fed as numeric. ``random_state`` maps to each
library's native seed kwarg for reproducibility (NFR-4).
"""

from __future__ import annotations

import importlib
import warnings
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from honestml.core import Task, get_logger
from honestml.core.exceptions import NotFittedError, SchemaValidationError

logger = get_logger("adapters.boosting")

# conservative fixed budget for the NO-early-stopping path: 1000 trees would overfit (ADR-0020 §2)
_N_ESTIMATORS = 300
# with early stopping the tree count is a CEILING that ES cuts per fold (ADR-0080), so it is generous
_N_ESTIMATORS_ES = 1000
# early-stopping patience (rounds without val-metric improvement); a fixed default like _N_ESTIMATORS
_ES_ROUNDS = 50


@contextmanager
def _quiet_feature_names() -> Iterator[None]:
    """Silence the cosmetic 'X does not have valid feature names' warning (finding #4).

    The model boundary is numpy by design (ADR-0013): lightgbm auto-generates column names when fit on
    a numpy array, then warns at predict on the same nameless numpy — a false positive that floods
    real-run logs and drowns the library's own honesty-relevant warnings. Prediction is correct; only
    this exact message is filtered, so genuine warnings still surface.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="X does not have valid feature names")
        yield


# the "no early stopping" notice is a run-level advisory (ADR-0020 §2), durably recorded in the
# manifest (early_stopping=False); dedupe the log line so a 5-fold run does not emit it 5× per model
_warned_backends: set[str] = set()


@dataclass(frozen=True)
class _Backend:
    """How to construct one boosting library's sklearn-style estimators.

    ``search_space`` is the backend-neutral HPO declaration carried to ``ModelSpec.search_space``
    (ADR-0061 §4); its tree-count key MUST equal ``n_estimators_kwarg`` so a tuned tree count
    overrides the fixed default rather than colliding (ADR-0061 §4, R2 fix).
    """

    module: str
    clf_attr: str
    reg_attr: str
    seed_kwarg: str
    n_estimators_kwarg: str
    # xgboost 3.x dropped its internal label encoder and rejects any classification target that is not
    # contiguous 0..K-1; the classifier wrapper codes/decodes labels for such backends (ADR-0081).
    requires_int_labels: bool = False
    # native categorical handling (ADR-0087): catboost cat_features / lightgbm categorical_feature. The
    # single source of truth for both the registry capability (handles_cat) and the wrapper's
    # SupportsNativeCategorical marker, so the static capability cannot drift from the runtime marker.
    handles_categorical: bool = False
    extra_kwargs: dict[str, Any] = field(default_factory=dict)
    search_space: dict[str, Any] = field(default_factory=dict)


# HPO search spaces ported from the legacy `_suggest_*` (ADR-0061 §4). The tree-count key is the
# backend's own `n_estimators_kwarg` (`iterations` for catboost, `n_estimators` for lgbm/xgb).
CATBOOST = _Backend(
    module="catboost",
    clf_attr="CatBoostClassifier",
    reg_attr="CatBoostRegressor",
    seed_kwarg="random_seed",
    n_estimators_kwarg="iterations",
    handles_categorical=True,
    extra_kwargs={"verbose": False},
    search_space={
        "depth": {"type": "int", "low": 4, "high": 10},
        "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
        "iterations": {"type": "int", "low": 50, "high": 500, "step": 50},
        "l2_leaf_reg": {"type": "float", "low": 1.0, "high": 10.0, "log": True},
        "subsample": {"type": "float", "low": 0.6, "high": 1.0},
        # categorical overfit control (ADR-0090 §A, FR-7): one_hot↔CTR boundary; others stay on defaults
        "one_hot_max_size": {"type": "int", "low": 2, "high": 64},
    },
)
LIGHTGBM = _Backend(
    module="lightgbm",
    clf_attr="LGBMClassifier",
    reg_attr="LGBMRegressor",
    seed_kwarg="random_state",
    n_estimators_kwarg="n_estimators",
    handles_categorical=True,
    extra_kwargs={"verbosity": -1},
    search_space={
        "max_depth": {"type": "int", "low": 3, "high": 10},
        "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
        "n_estimators": {"type": "int", "low": 50, "high": 500, "step": 50},
        "reg_lambda": {"type": "float", "low": 0.0, "high": 10.0},
        "subsample": {"type": "float", "low": 0.6, "high": 1.0},
        "colsample_bytree": {"type": "float", "low": 0.5, "high": 1.0},
        # categorical overfit control (ADR-0090 §A, FR-7): the two strongest per the LightGBM docs
        "min_data_per_group": {"type": "int", "low": 10, "high": 300},
        "cat_smooth": {"type": "float", "low": 1.0, "high": 50.0},
    },
)
XGBOOST = _Backend(
    module="xgboost",
    clf_attr="XGBClassifier",
    reg_attr="XGBRegressor",
    seed_kwarg="random_state",
    n_estimators_kwarg="n_estimators",
    requires_int_labels=True,
    extra_kwargs={"verbosity": 0},
    search_space={
        "max_depth": {"type": "int", "low": 3, "high": 10},
        "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
        "n_estimators": {"type": "int", "low": 50, "high": 500, "step": 50},
        "reg_lambda": {"type": "float", "low": 0.0, "high": 10.0},
        "subsample": {"type": "float", "low": 0.6, "high": 1.0},
        "colsample_bytree": {"type": "float", "low": 0.5, "high": 1.0},
    },
)


class _BoostingBase:
    """Shared fit/predict/importance over a constructed native estimator."""

    # boostings consume a held-out es tail for early stopping (ADR-0080); run_slice reads this to
    # route the fold's es_idx as a validation set instead of merging it into the training rows.
    supports_early_stopping = True

    def __init__(
        self,
        backend: _Backend,
        ctor: Any,
        random_state: int,
        params: Mapping[str, Any] | None = None,
    ) -> None:
        self._backend = backend
        self._ctor = ctor
        self._random_state = random_state
        # tuned hyperparameters (ADR-0061 §4); empty in the untuned M3 path
        self._params = dict(params or {})
        self.feature_names: list[str] = []
        self._model: Any | None = None
        # original labels for int-coded backends (ADR-0081), set by the classifier branch; None = the
        # native estimator consumes the labels as-is (catboost/lightgbm, and every regressor).
        self._label_index: np.ndarray | None = None
        # set by the classifier branch at fit (>2 classes); gates the CatBoost subsample fixup in _make.
        self._multiclass: bool = False
        # native categorical routing (ADR-0088/0089): categorical_indices is injected by the use-case
        # before fit for a native-capable wrapper (else stays [] = codes-path no-op) and read in fit/predict
        # to materialize cat_features/categorical_feature. The SupportsNativeCategorical marker (isinstance)
        # is the capability — set ONLY for native backends — so a non-native wrapper is never routed.
        self.categorical_indices: list[int] = []
        if backend.handles_categorical:
            self.supports_native_categorical = True

    def _encode_targets_fit(self, y: np.ndarray) -> np.ndarray:
        """Map the fit target for the native estimator; identity unless a subclass codes labels."""
        return y

    def _encode_targets_apply(self, y: np.ndarray) -> np.ndarray:
        """Identity on the base path; overridden by ``_BoostingClassifier`` to apply the label map set by ``_encode_targets_fit``."""
        return y

    def _make(
        self, extra: Mapping[str, Any] | None = None, *, n_estimators: int | None = None
    ) -> Any:
        # tuned `params` override the fixed defaults (ADR-0061 §4). The tree count is special: on the ES
        # path the caller passes the generous ceiling, which must win over a tuned count (ADR-0080), so it
        # is applied LAST; on the no-ES path a tuned count overrides the default (`setdefault`).
        kwargs: dict[str, Any] = {
            self._backend.seed_kwarg: self._random_state,
            **self._backend.extra_kwargs,
            **(extra or {}),
            **self._params,
        }
        if n_estimators is not None:
            kwargs[self._backend.n_estimators_kwarg] = (
                n_estimators  # ES ceiling wins over a tuned count
            )
        else:
            kwargs.setdefault(
                self._backend.n_estimators_kwarg, _N_ESTIMATORS
            )  # tuned count overrides default
        # CatBoost's multiclass default bootstrap (Bayesian) rejects `subsample`; binary/regression
        # default to MVS, which accepts it. Pair a tuned `subsample` with Bernoulli so the shared HPO
        # knob works for multiclass too, leaving the binary/regression default untouched.
        if self._backend.module == "catboost" and self._multiclass and "subsample" in kwargs:
            kwargs.setdefault("bootstrap_type", "Bernoulli")
        return self._ctor(**kwargs)

    # -- native categorical materialization (ADR-0089, validated by SPIKE-0004) ---------------------
    # The slice↔adapter boundary stays float64 numpy (ADR-0005); these rebuild the native input INSIDE
    # the adapter only when categorical_indices is non-empty (native-capable wrapper with categories).

    def _cb_frame(self, X: np.ndarray) -> Any:
        """CatBoost rejects float in cat columns -> a DataFrame whose cat columns are int, numeric float64.

        The codes are non-negative integers stored as float64 (no NaN in the cat block — null/unknown are
        reserved codes), so the int cast is exact and gives bit-identical predictions (SPIKE-0004). A
        DataFrame keeps the numeric block as a native float64 block (an object array would box every numeric
        cell too — needless memory on wide inputs); cat_features are positional indices over its columns.
        """
        import pandas as pd

        df = pd.DataFrame(X)  # RangeIndex columns 0..n-1; labels coincide with positional indices
        for j in self.categorical_indices:
            df[j] = df[j].astype(np.int64)
        return df

    def _cb_pool(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None) -> Any:
        from catboost import Pool

        return Pool(
            self._cb_frame(X), label=y, cat_features=self.categorical_indices, weight=sample_weight
        )

    def _predict_input(self, X: np.ndarray) -> Any:
        """Predict-time input: CatBoost needs the same int cat block as fit (parity); others unchanged."""
        if self._backend.module == "catboost" and self.categorical_indices:
            return self._cb_frame(X)
        return X

    @classmethod
    def from_native(
        cls, backend: _Backend, model: Any, classes: np.ndarray | None = None
    ) -> _BoostingBase:
        """Re-wrap an already-fitted native estimator loaded from a native artifact (ADR-0070 §5).

        The fit-driven ``__init__`` captures the model only inside ``fit``; this is the load-path
        constructor used by the native serializers (ADR-0069), restoring the same wrapper the
        facade ships so ``predict_proba``/``classes_``/``feature_importances`` keep working.
        The wrapper is meant for inference: tuned hyperparameters are not round-tripped by the
        native body, so a ``fit`` on it would retrain with the backend defaults.

        ``classes`` (the manifest's global label order) is restored for int-coded backends (ADR-0081):
        the native body holds only the ``0..K-1`` codes, so the original labels come from the manifest
        to decode ``predict`` and satisfy the ``classes_`` guard. Native backends pass it through unused.
        """
        wrapper = cls(backend, type(model), 0)
        wrapper._model = model
        if classes is not None and backend.requires_int_labels:
            wrapper._label_index = np.asarray(classes)
        wrapper._post_fit(model)
        return wrapper

    @property
    def native_format(self) -> str:
        """The backing library name — ``SupportsNativeModel`` role, doubles as ``model_type``."""
        return self._backend.module

    def native_model(self) -> Any:
        """The underlying fitted native estimator (``SupportsNativeModel``, ADR-0069 §3)."""
        return self._fitted()

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> _BoostingBase:
        # code arbitrary labels to 0..K-1 for backends that require it (xgboost, ADR-0081); the es tail
        # is coded with the same map. Identity for catboost/lightgbm and every regressor.
        y = self._encode_targets_fit(y)
        if y_val is not None:
            y_val = self._encode_targets_apply(y_val)
        if X_val is not None and y_val is not None:
            model = self._es_fit(X, y, X_val, y_val, sample_weight)
        else:
            if self._backend.module not in _warned_backends:
                _warned_backends.add(self._backend.module)
                logger.warning(
                    "boosting %r trained without early stopping; leaderboard comparison "
                    "may favor overfit settings",
                    self._backend.module,
                )
            model = self._make()
            self._plain_fit(model, X, y, sample_weight)
        self._model = model
        self._post_fit(model)
        return self

    def _plain_fit(
        self, model: Any, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None
    ) -> None:
        """No-ES fit, materializing the native categorical input when routed (ADR-0089); else unchanged."""
        cat = self.categorical_indices
        if cat and self._backend.module == "catboost":
            model.fit(self._cb_pool(X, y, sample_weight))
        elif cat and self._backend.module == "lightgbm":
            with _quiet_feature_names():
                model.fit(X, y, sample_weight=sample_weight, categorical_feature=cat)
        else:
            model.fit(X, y, sample_weight=sample_weight)

    def _es_fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        sample_weight: np.ndarray | None,
    ) -> Any:
        """Fit with early stopping on the carved es tail (ADR-0080), wiring each library's native API.

        The tree count is the generous ``_N_ESTIMATORS_ES`` ceiling; ES cuts it per fold. The es tail
        is validation only (it never enters the training rows), so OOF honesty is untouched. Validation
        sample weights are not threaded (the port carries no ``val_sample_weight`` slot) — ES is a
        stopping heuristic, not the scored metric, and ``sample_weight`` is ``None`` on the default path.
        """
        module = self._backend.module
        cat = self.categorical_indices
        with _quiet_feature_names():
            if module == "lightgbm":
                import lightgbm as lgb

                model = self._make(n_estimators=_N_ESTIMATORS_ES)
                # categorical_feature over the same float64 X (lgbm rounds the codes to int32); [] when no cats
                model.fit(
                    X,
                    y,
                    sample_weight=sample_weight,
                    eval_set=[(X_val, y_val)],
                    callbacks=[
                        lgb.early_stopping(_ES_ROUNDS, verbose=False),
                        lgb.log_evaluation(0),
                    ],
                    **({"categorical_feature": cat} if cat else {}),
                )
            elif module == "catboost":
                model = self._make(
                    {"early_stopping_rounds": _ES_ROUNDS}, n_estimators=_N_ESTIMATORS_ES
                )
                if cat:
                    # native cat: int-cast Pool for fit; es Pool is UNWEIGHTED (a stop heuristic, ADR-0089)
                    model.fit(
                        self._cb_pool(X, y, sample_weight),
                        eval_set=self._cb_pool(X_val, y_val, None),
                    )
                else:
                    model.fit(X, y, sample_weight=sample_weight, eval_set=(X_val, y_val))
            else:  # xgboost: early_stopping_rounds is a constructor arg in 2.x+ (removed from fit)
                model = self._make(
                    {"early_stopping_rounds": _ES_ROUNDS}, n_estimators=_N_ESTIMATORS_ES
                )
                model.fit(
                    X, y, sample_weight=sample_weight, eval_set=[(X_val, y_val)], verbose=False
                )
        return model

    def _post_fit(self, model: Any) -> None:
        """Hook for the classifier branch to capture ``classes_``."""

    def predict(self, X: np.ndarray) -> np.ndarray:
        # CatBoost returns a column vector (n, 1) for multiclass; flatten to 1-D labels.
        with _quiet_feature_names():
            return np.asarray(self._fitted().predict(self._predict_input(X))).ravel()

    @property
    def feature_importances(self) -> np.ndarray:
        return np.asarray(self._fitted().feature_importances_, dtype=np.float64).ravel()

    def _fitted(self) -> Any:
        if self._model is None:
            raise NotFittedError(f"{type(self).__name__}.predict called before fit")
        return self._model


class _BoostingClassifier(_BoostingBase):
    """Classifier branch (binary/multiclass): exposes ``predict_proba``/``classes_``.

    For ``requires_int_labels`` backends (xgboost 3.x, which dropped its internal label encoder and
    rejects any target that is not contiguous ``0..K-1``), the wrapper codes arbitrary labels —
    strings, ``{1, 2}``, non-contiguous ints — to ``0..K-1`` for the native fit and decodes back, so
    ``classes_``/``predict`` stay in the user's original label space (ADR-0081). The native proba
    columns are already in ``0..K-1`` == sorted-original-class order, so ``predict_proba`` is
    unchanged. Other backends (catboost/lightgbm) consume labels natively (``_label_index`` stays None).
    """

    def __init__(
        self,
        backend: _Backend,
        ctor: Any,
        random_state: int,
        params: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(backend, ctor, random_state, params)
        self.classes_: np.ndarray | None = None

    def _encode_targets_fit(self, y: np.ndarray) -> np.ndarray:
        # >2 classes ⇒ the backend uses its multiclass loss (CatBoost then needs the subsample fixup
        # in _make); tracked for every classifier backend, harmless where unused.
        self._multiclass = np.unique(y).size > 2
        if not self._backend.requires_int_labels:
            return y
        self._label_index = np.unique(
            y
        )  # sorted originals == the sklearn class order the codes index
        return np.searchsorted(self._label_index, y)

    def _encode_targets_apply(self, y: np.ndarray) -> np.ndarray:
        if self._label_index is None:
            return y
        unseen = np.setdiff1d(np.unique(y), self._label_index)
        if unseen.size:
            # the es tail carries a class absent from the fit fold (e.g. a rare class the carve put
            # only in es): coding it via searchsorted would silently mis-map (label below range) or
            # crash the native fit (label >= K). Fail loudly so the candidate is isolated with a clear
            # reason instead of early-stopping on a class it never trained on (ADR-0081, F112).
            raise SchemaValidationError(
                f"early-stopping tail has label(s) {unseen.tolist()} absent from the fit fold; "
                "the model cannot early-stop on a class it never trained on"
            )
        return np.searchsorted(self._label_index, y)

    def _post_fit(self, model: Any) -> None:
        # int-coded backends expose the ORIGINAL labels; the native classes_ are the 0..K-1 codes
        self.classes_ = (
            self._label_index if self._label_index is not None else np.asarray(model.classes_)
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        pred = super().predict(X)
        if self._label_index is None:
            return pred
        return self._label_index[np.asarray(pred, dtype=np.intp)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        with _quiet_feature_names():
            return np.asarray(self._fitted().predict_proba(self._predict_input(X)))


class _BoostingRegressor(_BoostingBase):
    """Regressor branch (regression): ``predict`` only, not probabilistic."""


def build_boosting(
    backend: _Backend, *, task: Task, random_state: int, **params: Any
) -> _BoostingBase:
    """Lazily import *backend* and build its classifier/regressor for *task* (ADR-0020 §2).

    The import lives here, not at module load: a missing extra surfaces as ``ImportError``,
    which the registry maps to ``MissingDependencyError`` (ADR-0019 §3). ``**params`` carries tuned
    hyperparameters (ADR-0061 §4); empty in the untuned path keeps the M3 fixed budget.
    """
    module = importlib.import_module(backend.module)
    if task.is_classification:
        return _BoostingClassifier(backend, getattr(module, backend.clf_attr), random_state, params)
    return _BoostingRegressor(backend, getattr(module, backend.reg_attr), random_state, params)
