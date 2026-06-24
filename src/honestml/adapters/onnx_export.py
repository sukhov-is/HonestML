"""ONNX conversion adapters (ADR-0071) — per-family converters + by-name session reading.

The onnx tooling is imported lazily inside each function (extra ``onnx``), and this module
itself is imported only inside ``export_onnx``'s body — neither ``import honestml`` nor the
predict cone ever touch it (NFR-SER-3). Output tensor NAMES differ per converter
(ADR-0071 §2); readers go strictly by the converter's own expected name, with **no positional
fallback** (§5): a missing name (converter-version drift, R-SER-VERSION) is an explicit error,
never a silent wrong-tensor comparison.
"""

from __future__ import annotations

import tempfile
from importlib.util import find_spec
from pathlib import Path
from typing import cast

import numpy as np

from honestml.adapters.ensembling import BlendedEstimator
from honestml.adapters.estimators import (
    BaselineClassifier,
    BaselineRegressor,
    LinearClassifier,
    LinearRegressor,
)
from honestml.core import (
    Estimator,
    MissingDependencyError,
    NativeCategoricalONNXUnsupportedError,
    SchemaValidationError,
    SupportsNativeCategorical,
    SupportsNativeModel,
)

# ADR-0071 §2: per-converter output tensor names (classification proba / regression prediction),
# verified against the real converters by test_per_converter_output_names (skl2onnx 1.20 emits
# label/probabilities with zipmap=False — not the output_* names sometimes documented)
OUTPUT_NAMES = {
    "skl2onnx": {"proba": "probabilities", "reg": "variable"},
    "lightgbm": {"proba": "probabilities", "reg": "variable"},
    "xgboost": {"proba": "probabilities", "reg": "variable"},
    "catboost": {"proba": "probabilities", "reg": "predictions"},
}


def _require(package: str) -> None:
    if find_spec(package) is None:
        raise MissingDependencyError("onnx", package=package)


def convert(estimator: Estimator, X32: np.ndarray) -> tuple[bytes, str]:
    """Convert *estimator* to ONNX bytes; returns ``(onnx_bytes, method)``.

    ``method`` keys :data:`OUTPUT_NAMES`. The unsupported cases are explicit rejections
    (ADR-0071 §2): baseline has no skl2onnx converter, an ensemble has no converter for a
    heterogeneous weighted blend (per-member export is Day-2).
    """
    # FR-6 gate (ADR-0091, SPIKE-0005): a natively-categorical model cannot export to ONNX (CatBoost
    # impossible per catboost#863; LightGBM deferred) — reject BEFORE any converter, so there is never a
    # silently-wrong graph. Pure check on the marker, so it fires without the onnx extra installed.
    if isinstance(estimator, SupportsNativeCategorical) and estimator.categorical_indices:
        raise NativeCategoricalONNXUnsupportedError(
            "native categorical models are not ONNX-exportable on v1 "
            "(CatBoost: catboost#863; LightGBM: deferred to ONNX re-spike); use joblib or native format"
        )
    if isinstance(estimator, (BaselineClassifier, BaselineRegressor)):
        raise SchemaValidationError(
            "baseline is not ONNX-exportable (skl2onnx has no Dummy* converter)"
        )
    if isinstance(estimator, BlendedEstimator):
        raise SchemaValidationError(
            "ensemble is not ONNX-exportable (no converter for a weighted blend; "
            "export members individually or ship the artifact)"
        )
    if isinstance(estimator, SupportsNativeModel):
        return _convert_boosting(estimator, X32.shape[1]), estimator.native_format
    if isinstance(estimator, (LinearClassifier, LinearRegressor)):
        return _convert_sklearn(estimator, X32), "skl2onnx"
    raise SchemaValidationError(f"{type(estimator).__name__} is not ONNX-exportable")


def _convert_sklearn(estimator: LinearClassifier | LinearRegressor, X32: np.ndarray) -> bytes:
    _require("skl2onnx")
    from skl2onnx import to_onnx

    options = {"zipmap": False} if isinstance(estimator, LinearClassifier) else None
    onx = to_onnx(estimator._fitted(), X32, options=options)
    return cast(bytes, onx.SerializeToString())


def _convert_boosting(estimator: SupportsNativeModel, n_features: int) -> bytes:
    if estimator.native_format == "catboost":
        return _convert_catboost(estimator)
    _require("onnxmltools")
    from onnxmltools import convert_lightgbm, convert_xgboost
    from onnxmltools.convert.common.data_types import FloatTensorType

    initial_types = [("input", FloatTensorType([None, n_features]))]
    model = estimator.native_model()
    if estimator.native_format == "lightgbm":
        onx = convert_lightgbm(model, initial_types=initial_types, zipmap=False)
    else:
        onx = convert_xgboost(model, initial_types=initial_types)
    return cast(bytes, onx.SerializeToString())


def _convert_catboost(estimator: SupportsNativeModel) -> bytes:
    # catboost exports ONNX natively and only to a file path; export-only (ADR-0071 §2)
    _require("onnx")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.onnx"
        estimator.native_model().save_model(str(path), format="onnx")
        return path.read_bytes()


def run_onnx(onnx_bytes: bytes, X32: np.ndarray, output: str) -> np.ndarray:
    """Run the graph on ``X32`` and read ONE output strictly by *output* name (ADR-0071 §5).

    A ZipMap-style output (sequence of {class: proba} maps — CatBoost-native emits one) is
    normalized to a dense ``(n, K)`` matrix in sorted-class order; the declared tensor shape is
    never trusted (the CatBoost shape quirk, §5).
    """
    _require("onnxruntime")
    import onnxruntime as ort

    session = ort.InferenceSession(onnx_bytes, providers=["CPUExecutionProvider"])
    names = [o.name for o in session.get_outputs()]
    if output not in names:
        raise SchemaValidationError(
            f"unexpected ONNX output schema: {output!r} missing (graph outputs {names}; "
            "converter version drift?)"
        )
    result = session.run([output], {session.get_inputs()[0].name: X32})[0]
    if isinstance(result, list) and result and isinstance(result[0], dict):
        # assumption: python-sorted ZipMap keys == the sklearn-sorted classes_ column order (holds
        # for int and homogeneous-string labels); a mismatch cannot ship — the parity gate compares
        # against native predict_proba BEFORE any write and fails closed on a reordered column
        return np.array([[row[key] for key in sorted(row)] for row in result])
    return np.asarray(result)


def tool_versions(method: str) -> dict[str, str]:
    """Versions of the tooling involved in *method* — recorded in ``onnx_manifest`` (NFR-SER-6)."""
    import importlib.metadata as metadata

    packages = {
        "skl2onnx": ("skl2onnx", "onnx", "onnxruntime"),
        "lightgbm": ("onnxmltools", "onnx", "onnxruntime"),
        "xgboost": ("onnxmltools", "onnx", "onnxruntime"),
        "catboost": ("catboost", "onnx", "onnxruntime"),
    }[method]
    return {package: metadata.version(package) for package in packages}
