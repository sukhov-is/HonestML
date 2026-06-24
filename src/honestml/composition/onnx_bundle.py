"""ONNX export bundle — export-only channel for external runtimes.

``export_onnx(model, directory, *, sample)`` converts the supported estimator subset
({linear, lightgbm, xgboost, catboost}) to a standalone bundle: ``model.onnx``
(the RAW-estimator graph over the numeric design matrix) + ``schema.json`` (the preprocessing
contract) + ``onnx_manifest.json`` + ``README.md``. The graph contains NEITHER preprocessing
(the consumer reproduces ``design_matrix`` from the schema) NOR calibration
(disclosed in the manifest). Parity vs the native estimator is validated on ``sample``
BEFORE any file is written (never ship a silently diverging graph). ONNX is not a
``model_type`` — load-back stays native/joblib.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np

from honestml.application import design_matrix
from honestml.core import ProbabilisticEstimator, SchemaValidationError, get_logger

from .artifact import FittedModel, _sha256_file

ONNX_MANIFEST_VERSION = 1
# parity-gate thresholds: SPIKE-0003 at the shipped config (300 trees / depth 10), margin ~18-70x
_PROBA_ATOL = 1e-5
_REG_ATOL = 1e-4
_REG_RTOL = 1e-3
# benign near-tie window for label flips (ADR-0071 §3): a disagreement is tolerated (WARNING)
# only when the native top-2 gap sits within float32 noise; a larger gap is real divergence
_TIE_BAND = 2 * _PROBA_ATOL


def export_onnx(model: FittedModel, directory: str | Path, *, sample: object) -> dict[str, Any]:
    """Export *model* to a standalone ONNX bundle in *directory*; returns the parity report.

    ``sample`` (raw rows, anything the model can predict on) is REQUIRED: the model retains no
    training matrix, and without data the honesty gate cannot run — there is no
    silent skip. The gate compares the converted graph (float32, onnxruntime) against the
    native estimator's RAW output and raises :class:`SchemaValidationError` on a breach;
    a benign near-tie label flip (top-2 gap within the float32 noise band) is downgraded to a
    WARNING and recorded in ``onnx_manifest.json``. Requires the ``onnx`` extra.
    """
    from honestml.adapters import (
        onnx_export,  # lazy: onnx tooling only when exporting (ADR-0071 §7)
    )

    matrix = design_matrix(model._read(sample))
    x32 = matrix.astype(np.float32)
    onnx_bytes, method = onnx_export.convert(model.estimator, x32)
    parity = _parity_gate(model, matrix, x32, onnx_bytes, method)
    _write_bundle(model, Path(directory), onnx_bytes, method, parity)
    return parity


def _parity_gate(
    model: FittedModel,
    matrix: np.ndarray,
    x32: np.ndarray,
    onnx_bytes: bytes,
    method: str,
) -> dict[str, Any]:
    """ONNX vs native-RAW parity on the sample, BEFORE the bundle is written."""
    from honestml.adapters import onnx_export

    names = onnx_export.OUTPUT_NAMES[method]
    report: dict[str, Any] = {
        "proba_max_abs": None,
        "reg_max_abs": None,
        "label_verdict": None,
        "n_tie_warnings": 0,
        "n_validation_rows": int(len(x32)),
    }
    if not model.task.is_classification:
        onnx_pred = onnx_export.run_onnx(onnx_bytes, x32, names["reg"]).ravel()
        native = np.asarray(model.estimator.predict(matrix), dtype=np.float64).ravel()
        reg_max = float(np.max(np.abs(onnx_pred - native)))
        if not np.allclose(onnx_pred, native, rtol=_REG_RTOL, atol=_REG_ATOL):
            raise SchemaValidationError(
                f"ONNX parity breach: max|Δpred|={reg_max:.3g} exceeds "
                f"atol={_REG_ATOL}/rtol={_REG_RTOL} — not shipping a diverging graph"
            )
        report["reg_max_abs"] = reg_max
        return report

    onnx_proba = np.asarray(onnx_export.run_onnx(onnx_bytes, x32, names["proba"]), dtype=np.float64)
    estimator = cast(ProbabilisticEstimator, model.estimator)
    native_proba = np.asarray(estimator.predict_proba(matrix), dtype=np.float64)
    if onnx_proba.shape != native_proba.shape:
        raise SchemaValidationError(
            f"unexpected ONNX proba shape {onnx_proba.shape} (native {native_proba.shape})"
        )
    proba_max = float(np.max(np.abs(onnx_proba - native_proba)))
    if proba_max > _PROBA_ATOL:
        raise SchemaValidationError(
            f"ONNX parity breach: max|Δproba|={proba_max:.3g} > {_PROBA_ATOL} — "
            "not shipping a diverging graph"
        )
    report["proba_max_abs"] = proba_max
    # boundary-aware label rule (ADR-0071 §3): defined over argmax of the probas, so it is
    # immune to label-tensor dtype quirks; flips are benign only inside the float32 noise band
    disagree = np.argmax(onnx_proba, axis=1) != np.argmax(native_proba, axis=1)
    n_flips = int(disagree.sum())
    report["label_verdict"] = "ok"
    if n_flips:
        top2 = np.sort(native_proba[disagree], axis=1)
        max_gap = float((top2[:, -1] - top2[:, -2]).max())
        if max_gap > _TIE_BAND:
            raise SchemaValidationError(
                f"ONNX parity breach: {n_flips} label flip(s) with top-2 gap "
                f"{max_gap:.3g} > {_TIE_BAND:.3g} — real divergence, not float32 noise"
            )
        report["label_verdict"] = "warning"
        report["n_tie_warnings"] = n_flips
        # the human-visible channel for a standalone export (the run report is fit-time only)
        get_logger().warning(
            "ONNX export: %d benign near-tie label flip(s) within the float32 noise band "
            "(top-2 gap <= %.0e); recorded in onnx_manifest.json",
            n_flips,
            _TIE_BAND,
        )
    return report


def _write_bundle(
    model: FittedModel,
    path: Path,
    onnx_bytes: bytes,
    method: str,
    parity: dict[str, Any],
) -> None:
    from honestml.adapters import onnx_export

    features = model.schema.features
    selected = model.schema.selected_features
    selected_set = None if selected is None else set(selected)
    # the graph input width and column order are the PROJECTED design matrix — the same rule as
    # design_matrix's choke-point (slice.py: project in schema.features order, FR-FS-7/ADR-0071 §4)
    order = features if selected_set is None else [f for f in features if f in selected_set]
    categorical = set(model.schema.categorical)
    calibration = model.calibration or {}
    classes = _graph_classes(model)  # raises BEFORE anything is written (DM-B4)
    path.mkdir(parents=True, exist_ok=True)
    (path / "model.onnx").write_bytes(onnx_bytes)
    (path / "schema.json").write_text(model.schema.model_dump_json(indent=2), encoding="utf-8")
    manifest = {
        "onnx_manifest_version": ONNX_MANIFEST_VERSION,
        "schema_ref": "schema.json",
        "feature_order": order,
        "columns": [
            {"name": name, "dtype": "float32", "ordinal": name in categorical} for name in order
        ],
        "classes": classes,
        "conversion": {"method": method, "tool_versions": onnx_export.tool_versions(method)},
        "calibration": {
            "applied": model.calibrator is not None,
            "method": calibration.get("method"),
        },
        "parity": parity,
        # tamper/mixed-rewrite detection for the standalone bundle (NFR-SER-2): digests of the
        # on-disk bytes, written last — a partial re-export over an old bundle becomes visible
        "checksums": {name: _sha256_file(path / name) for name in ("model.onnx", "schema.json")},
    }
    (path / "onnx_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (path / "README.md").write_text(_readme(manifest), encoding="utf-8")


def _graph_classes(model: FittedModel) -> list | None:
    """The class order the GRAPH emits — the estimator's own, verified against the global order.

    A shipped model may have seen fewer classes than the global order (a class
    living entirely in the holdout): the graph would then emit K-1 columns while the manifest
    declared K — a silently wrong class map for the consumer. Refuse instead.
    """
    if model.classes is None:
        return None
    estimator_classes = np.asarray(getattr(model.estimator, "classes_", model.classes))
    if not np.array_equal(estimator_classes, np.asarray(model.classes)):
        raise SchemaValidationError(
            f"estimator classes {estimator_classes.tolist()!r} differ from the global class "
            f"order {model.classes.tolist()!r}: the graph would emit a different class space "
            "(e.g. a class unseen by the shipped model) — not exporting a misleading bundle"
        )
    return cast(list, model.classes.tolist())


def _readme(manifest: dict[str, Any]) -> str:
    n_cols = len(manifest["feature_order"])
    calibration_warning = (
        [
            "- CALIBRATION WARNING: the source model serves CALIBRATED probabilities, but this "
            "graph outputs RAW ones — re-apply the calibration mapping downstream."
        ]
        if manifest["calibration"]["applied"]
        else []
    )
    lines = [
        "# ONNX serving bundle (export-only)",
        "",
        f"- `model.onnx` — the RAW estimator graph; input: float32 matrix of {n_cols} columns.",
        "- The graph contains NO preprocessing: reproduce the design matrix from `schema.json` —",
        "  columns in `feature_order` order, categorical columns fed as ordinal codes",
        "  (`onnx_manifest.json: columns[].ordinal`).",
        *calibration_warning,
        "- Parity vs the native estimator was validated before writing (`onnx_manifest.json: parity`).",
        "- This bundle is NOT loadable back into honestml; for load-back use the artifact "
        "(`save_artifact`).",
    ]
    return "\n".join(lines) + "\n"
