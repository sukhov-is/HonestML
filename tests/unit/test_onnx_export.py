"""M8b-2: ONNX export bundle (ADR-0071, FR-SER-3, NFR-SER-4/6) — export-only, parity-gated.

The supported subset ({linear, lightgbm, xgboost, catboost}) converts with a parity gate run on
the REQUIRED ``sample`` before any file is written: proba/regression hard-fail thresholds from
SPIKE-0003, labels boundary-aware (a near-tie flip inside the float32 noise band is a WARNING,
a larger gap is a refusal). Baseline and ensembles are explicit rejections. Tests needing onnx
tooling gate on ``importorskip`` (CI runs them under the ``onnx`` extra); the rejection and
gate-rule tests run everywhere.
"""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace

import numpy as np
import pytest

from honestml import AutoML
from honestml.application import design_matrix
from honestml.composition import onnx_bundle
from honestml.composition.artifact import save_artifact
from honestml.composition.onnx_bundle import export_onnx
from honestml.core import NativeCategoricalONNXUnsupportedError, SchemaValidationError, Task

pytestmark = pytest.mark.unit

_ONNX_TOOLING = {
    "linear": ("skl2onnx",),
    "lightgbm": ("onnxmltools",),
    "xgboost": ("onnxmltools",),
    "catboost": ("onnx",),
}


def _data(task: str = "binary", n: int = 80):
    if task == "regression":
        from sklearn.datasets import make_regression

        return make_regression(n_samples=n, n_features=6, n_informative=4, random_state=0)
    from sklearn.datasets import make_classification

    return make_classification(
        n_samples=n, n_features=6, n_informative=4, n_redundant=0, random_state=0
    )


def _fit(lib: str, task: str = "binary", **kw):
    if lib != "linear":
        pytest.importorskip(lib)
    pytest.importorskip("onnxruntime")
    for tool in _ONNX_TOOLING[lib]:
        pytest.importorskip(tool)
    X, y = _data(task)
    return AutoML(task=task, models=(lib,), random_state=0, **kw).fit(X, y), X


def _manifest(bundle) -> dict:
    return json.loads((bundle / "onnx_manifest.json").read_text(encoding="utf-8"))


# --- parity within tolerance per family (FR-SER-3, NFR-SER-4; SPIKE-0003) ---------------------------


@pytest.mark.parametrize("lib", ["linear", "lightgbm", "xgboost", "catboost"])
def test_onnx_parity_within_tol(tmp_path, lib) -> None:
    from honestml.adapters import onnx_export

    model, X = _fit(lib)
    report = export_onnx(model.fitted_, tmp_path / "bundle", sample=X)
    assert report["label_verdict"] in ("ok", "warning")
    # independent re-check OUTSIDE the gate's code path: the WRITTEN graph vs the raw estimator
    matrix = design_matrix(model.fitted_._read(X))
    written = onnx_export.run_onnx(
        (tmp_path / "bundle" / "model.onnx").read_bytes(),
        matrix.astype(np.float32),
        "probabilities",
    )
    native = model.fitted_.estimator.predict_proba(matrix)
    assert np.allclose(written, native, atol=onnx_bundle._PROBA_ATOL)


@pytest.mark.parametrize("lib", ["linear", "lightgbm", "xgboost", "catboost"])
def test_onnx_regression_parity_within_tol(tmp_path, lib) -> None:
    model, X = _fit(lib, task="regression")
    report = export_onnx(model.fitted_, tmp_path / "bundle", sample=X)
    # the gate contract is atol+rtol (targets here are O(100), so the rtol term dominates)
    bound = onnx_bundle._REG_ATOL + onnx_bundle._REG_RTOL * float(np.abs(model.predict(X)).max())
    assert report["reg_max_abs"] <= bound
    assert report["label_verdict"] is None


def test_parity_breach_raises(tmp_path, monkeypatch) -> None:
    """The gate refuses to write a diverging bundle (DM-B4): with a zero tolerance, even the
    benign float32 conversion noise must trip the hard-fail BEFORE any file appears."""
    model, X = _fit("linear")
    monkeypatch.setattr(onnx_bundle, "_PROBA_ATOL", 0.0)
    with pytest.raises(SchemaValidationError, match="parity breach"):
        export_onnx(model.fitted_, tmp_path / "bundle", sample=X)
    assert not (tmp_path / "bundle").exists()  # nothing written on refusal


# --- outputs read strictly by the converter's name (ADR-0071 §2/§5) --------------------------------


@pytest.mark.parametrize("lib", ["linear", "lightgbm", "xgboost", "catboost"])
def test_per_converter_output_names(tmp_path, lib) -> None:
    """The §2 name table matches what each converter actually emits — drift (R-SER-VERSION)
    would surface here deterministically, not as a silent wrong-tensor comparison."""
    from honestml.adapters import onnx_export

    model, X = _fit(lib)
    matrix = design_matrix(model.fitted_._read(X)).astype(np.float32)
    onnx_bytes, method = onnx_export.convert(model.fitted_.estimator, matrix)
    proba = onnx_export.run_onnx(onnx_bytes, matrix, onnx_export.OUTPUT_NAMES[method]["proba"])
    assert proba.shape == (len(X), 2)
    with pytest.raises(SchemaValidationError, match="unexpected ONNX output schema"):
        onnx_export.run_onnx(onnx_bytes, matrix, "no_such_output")


# --- explicit rejections: baseline / ensemble / unknown (ADR-0071 §2) — no onnx tooling needed ------


def test_baseline_rejected(tmp_path) -> None:
    X, y = _data()
    model = AutoML(task="binary", models=("baseline",), random_state=0).fit(X, y)
    with pytest.raises(SchemaValidationError, match="baseline is not ONNX-exportable"):
        export_onnx(model.fitted_, tmp_path / "bundle", sample=X)


def test_ensemble_onnx_rejected(tmp_path) -> None:
    from honestml.adapters.ensembling import BlendedEstimator
    from honestml.adapters.estimators import LinearClassifier

    X, y = _data()
    model = AutoML(task="binary", models=("linear",), random_state=0).fit(X, y)
    fm = model.fitted_
    member = LinearClassifier().fit(X.astype(np.float64), y)
    fm.estimator = BlendedEstimator([member], np.array([1.0]), classes=np.array([0, 1]))
    with pytest.raises(SchemaValidationError, match="ensemble is not ONNX-exportable"):
        export_onnx(fm, tmp_path / "bundle", sample=X)


def _cat_frame(n: int = 200, seed: int = 0):
    import pandas as pd

    rng = np.random.default_rng(seed)
    cat = rng.choice(["a", "b", "c", "d"], size=n)
    x1 = rng.normal(size=n)
    eff = {"a": 1.5, "b": -1.5, "c": 1.0, "d": -1.0}
    y = (0.8 * x1 + np.array([eff[c] for c in cat]) + 0.4 * rng.normal(size=n) > 0).astype(int)
    return pd.DataFrame({"x1": x1, "x2": rng.normal(size=n), "cat": cat}), y


def test_native_categorical_onnx_rejected(tmp_path) -> None:
    # FR-6 (ADR-0091): a native-cat boosting model is rejected for ONNX BEFORE the converter, so the
    # gate fires even without the onnx extra installed (no importorskip of onnx tooling here).
    pytest.importorskip("catboost")
    df, y = _cat_frame()
    model = AutoML(task="binary", models=("catboost",), random_state=0).fit(df, y)
    with pytest.raises(NativeCategoricalONNXUnsupportedError, match="native categorical.*ONNX"):
        export_onnx(model.fitted_, tmp_path / "bundle", sample=df)


def test_export_onnx_requires_sample(tmp_path) -> None:
    X, y = _data()
    model = AutoML(task="binary", models=("linear",), random_state=0).fit(X, y)
    with pytest.raises(TypeError):  # keyword-only, no default: the gate cannot run without data
        export_onnx(model.fitted_, tmp_path / "bundle")  # type: ignore[call-arg]


def test_classes_mismatch_rejected(tmp_path) -> None:
    """DM-B4: a shipped model that saw fewer classes than the global order (ADR-0068 §3,
    dev-unseen class) would emit a graph in a DIFFERENT class space — refuse, don't mislead."""
    pytest.importorskip("onnxruntime")
    pytest.importorskip("skl2onnx")
    model, X = _fit("linear")
    fm = model.fitted_
    fm.classes = np.array([0, 1, 2])  # global order claims a class the estimator never saw
    with pytest.raises(SchemaValidationError, match="class space"):
        export_onnx(fm, tmp_path / "bundle", sample=X)
    assert not (tmp_path / "bundle").exists()


# --- boundary-aware label rule (ADR-0071 §3, R1-AD1) — unit-level, no onnx tooling -----------------


class _StubProba:
    """The gate's actual dependency surface is exactly ``predict_proba`` (argmax-based rule)."""

    def __init__(self, proba: np.ndarray) -> None:
        self._proba = proba

    def predict_proba(self, X) -> np.ndarray:
        return self._proba


def _run_gate(monkeypatch, native: np.ndarray, onnx: np.ndarray, **consts) -> dict:
    import honestml.adapters.onnx_export as onnx_export

    for name, value in consts.items():
        monkeypatch.setattr(onnx_bundle, name, value)
    monkeypatch.setattr(onnx_export, "run_onnx", lambda b, x, name: onnx)
    model = SimpleNamespace(task=Task(kind="binary"), estimator=_StubProba(native))
    x = np.zeros((len(native), 2), dtype=np.float64)
    return onnx_bundle._parity_gate(model, x, x.astype(np.float32), b"", "xgboost")


def test_label_disagreement_boundary_aware_warns_on_tie(monkeypatch, caplog) -> None:
    """A flip whose native top-2 gap sits inside the float32 noise band -> WARNING, not refusal."""
    native = np.array([[0.500004, 0.499996], [0.9, 0.1]])
    onnx = np.array([[0.499996, 0.500004], [0.9, 0.1]])  # row 0 flips, gap 8e-6 <= 2e-5
    with caplog.at_level(logging.WARNING):
        report = _run_gate(monkeypatch, native, onnx)
    assert report["label_verdict"] == "warning" and report["n_tie_warnings"] == 1
    assert any("near-tie" in record.message for record in caplog.records)


def test_label_disagreement_boundary_aware_fails_on_real_gap(monkeypatch) -> None:
    """A flip with a LARGE native gap is real divergence -> refusal even when proba passes."""
    native = np.array([[0.6, 0.4]])
    onnx = np.array([[0.4, 0.6]])
    with pytest.raises(SchemaValidationError, match="label flip"):
        _run_gate(monkeypatch, native, onnx, _PROBA_ATOL=1.0, _TIE_BAND=1e-4)


# --- bundle contract: onnx_manifest schema / preprocessing / calibration / FS width (R2) -----------


def test_onnx_manifest_schema_fields(tmp_path) -> None:
    model, X = _fit("linear")
    export_onnx(model.fitted_, tmp_path / "bundle", sample=X)
    manifest = _manifest(tmp_path / "bundle")
    assert manifest["onnx_manifest_version"] == 1
    assert set(manifest) == {
        "onnx_manifest_version",
        "schema_ref",
        "feature_order",
        "columns",
        "classes",
        "conversion",
        "calibration",
        "parity",
        "checksums",
    }
    assert manifest["schema_ref"] == "schema.json"
    assert manifest["classes"] == [0, 1]
    assert set(manifest["parity"]) == {
        "proba_max_abs",
        "reg_max_abs",
        "label_verdict",
        "n_tie_warnings",
        "n_validation_rows",
    }
    assert manifest["conversion"]["method"] == "skl2onnx"
    assert "skl2onnx" in manifest["conversion"]["tool_versions"]
    # bundle integrity (NFR-SER-2): the manifest digests the files it was written after
    import hashlib

    body = (tmp_path / "bundle" / "model.onnx").read_bytes()
    assert manifest["checksums"]["model.onnx"] == hashlib.sha256(body).hexdigest()


def test_bundle_has_preprocessing_contract(tmp_path) -> None:
    model, X = _fit("linear")
    export_onnx(model.fitted_, tmp_path / "bundle", sample=X)
    bundle = tmp_path / "bundle"
    assert (bundle / "schema.json").is_file()  # the single preprocessing source of truth
    manifest = _manifest(bundle)
    assert manifest["feature_order"] == model.fitted_.schema.features
    assert all({"name", "dtype", "ordinal"} == set(c) for c in manifest["columns"])
    readme = (bundle / "README.md").read_text(encoding="utf-8")
    assert "NO preprocessing" in readme and "feature_order" in readme


def test_onnx_export_discloses_calibration(tmp_path) -> None:
    """ADR-0071 §4: the graph is PRE-calibration; the bundle must disclose it, not silently
    diverge from the artifact's calibrated predict_proba."""
    model, X = _fit("linear")
    fm = model.fitted_
    fm.calibrator = _fitted_calibrator()
    fm.calibration = {"applied": True, "method": "sigmoid"}
    export_onnx(fm, tmp_path / "bundle", sample=X)  # gate compares RAW estimator -> still passes
    manifest = _manifest(tmp_path / "bundle")
    assert manifest["calibration"] == {"applied": True, "method": "sigmoid"}
    assert "CALIBRATION WARNING" in (tmp_path / "bundle" / "README.md").read_text(encoding="utf-8")


def _fitted_calibrator():
    from honestml.adapters import SigmoidCalibrator

    rng = np.random.default_rng(0)
    p = rng.uniform(0.05, 0.95, 200)
    y = (rng.uniform(size=200) < p**2).astype(int)
    calibrator = SigmoidCalibrator()
    calibrator.fit(p, y)
    return calibrator


def test_bundle_columns_match_projected_width(tmp_path) -> None:
    """ADR-0071 §4: under feature selection the graph input width and the manifest columns are
    the PROJECTED design matrix, not the original feature set."""
    from honestml.core import CVConfig, FeatureSelectionConfig

    pytest.importorskip("onnxruntime")
    pytest.importorskip("skl2onnx")
    X, y = _data()
    fs = FeatureSelectionConfig(strategy="random_probe", cutoff="top_k", top_k=3)
    model = AutoML(
        task="binary",
        models=("linear",),
        cv=CVConfig(scheme="holdout", outer_holdout=0.3),
        feature_selection=fs,
        random_state=0,
    ).fit(X, y)
    export_onnx(model.fitted_, tmp_path / "bundle", sample=X)
    manifest = _manifest(tmp_path / "bundle")
    assert len(manifest["feature_order"]) == len(manifest["columns"]) == 3

    import onnxruntime as ort

    session = ort.InferenceSession(
        (tmp_path / "bundle" / "model.onnx").read_bytes(), providers=["CPUExecutionProvider"]
    )
    assert session.get_inputs()[0].shape[1] == 3


# --- reproducibility on top of the parity gate (NFR-SER-6) -----------------------------------------


def test_double_export_within_spike_tol(tmp_path) -> None:
    from honestml.adapters import onnx_export

    model, X = _fit("xgboost")
    export_onnx(model.fitted_, tmp_path / "a", sample=X)
    export_onnx(model.fitted_, tmp_path / "b", sample=X)
    matrix = design_matrix(model.fitted_._read(X)).astype(np.float32)
    out = [
        onnx_export.run_onnx((tmp_path / d / "model.onnx").read_bytes(), matrix, "probabilities")
        for d in ("a", "b")
    ]
    assert np.allclose(out[0], out[1], atol=onnx_bundle._PROBA_ATOL)


def test_artifact_save_unaffected_by_onnx_channel(tmp_path) -> None:
    """ONNX is an export channel, not a model_type (ADR-0071 §1): the artifact stays joblib/native."""
    X, y = _data()
    model = AutoML(task="binary", models=("linear",), random_state=0).fit(X, y)
    save_artifact(model.fitted_, tmp_path / "art")
    manifest = json.loads((tmp_path / "art" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["model_type"] == "joblib"  # no "onnx" model_type exists
