"""M8-1: artifact integrity (sha256 checksums) + model_type dispatch + version gate (ADR-0067/0065).

Integrity detects corruption / naive substitution before ``joblib.load``; ``ARTIFACT_VERSION`` stays 1
(the ``checksums`` block is additive); a legacy artifact without checksums still loads (strict mode rejects).
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from honestml import ArtifactIntegrityError, AutoML
from honestml.composition.artifact import (
    ARTIFACT_VERSION,
    _manifest_digest,
    load_artifact,
    save_artifact,
)
from honestml.core import SchemaValidationError

pytestmark = pytest.mark.unit


def _fit(task: str = "binary", n: int = 60):
    if task == "regression":
        from sklearn.datasets import make_regression

        X, y = make_regression(n_samples=n, n_features=6, n_informative=4, random_state=0)
    else:
        from sklearn.datasets import make_classification

        X, y = make_classification(
            n_samples=n, n_features=6, n_informative=4, n_redundant=0, random_state=0
        )
    return AutoML(task=task, models=("baseline", "linear"), random_state=0).fit(X, y), X


def _save(tmp_path, *, task: str = "binary", with_calibrator: bool = False, sign=None):
    model, X = _fit(task)
    fm = model.fitted_
    if with_calibrator:
        fm.calibrator = _fitted_calibrator()
    art = tmp_path / "art"
    save_artifact(fm, art, sign=sign)
    return art, X, model


def _fitted_calibrator():
    from honestml.adapters import SigmoidCalibrator

    rng = np.random.default_rng(0)
    p = rng.uniform(0.05, 0.95, 200)
    y = (rng.uniform(size=200) < p**2).astype(int)
    cal = SigmoidCalibrator()
    cal.fit(p, y)
    return cal


def _manifest(art) -> dict:
    return json.loads((art / "manifest.json").read_text(encoding="utf-8"))


def _rewrite(art, manifest) -> None:
    (art / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


# --- checksums written + intact round-trip (ADR-0067 §1) -------------------------------------------


def test_checksums_written_and_intact_loads(tmp_path) -> None:
    art, X, model = _save(tmp_path)
    checks = _manifest(art)["checksums"]
    assert checks["algo"] == "sha256"
    assert set(checks["files"]) == {"schema.json", "leaderboard.json", "model.joblib"}
    assert isinstance(checks["manifest"], str) and len(checks["manifest"]) == 64

    loaded = load_artifact(art)  # intact -> verifies and loads
    assert np.array_equal(loaded.predict(X), model.predict(X))


def test_artifact_version_is_one(tmp_path) -> None:
    """The checksums block is additive — ARTIFACT_VERSION stays 1 (NFR-SRV-2)."""
    art, _, _ = _save(tmp_path)
    assert _manifest(art)["artifact_version"] == ARTIFACT_VERSION == 1


def test_calibrator_in_checksums_when_present(tmp_path) -> None:
    art, _, _ = _save(tmp_path, with_calibrator=True)
    assert (art / "calibrator.joblib").exists()
    assert "calibrator.joblib" in _manifest(art)["checksums"]["files"]


def test_calibrator_absent_from_checksums_when_none(tmp_path) -> None:
    art, _, _ = _save(tmp_path)
    assert "calibrator.joblib" not in _manifest(art)["checksums"]["files"]


# --- tamper detection (ADR-0067 §2/§3 digest_mismatch) ---------------------------------------------


def test_tampered_model_raises_digest_mismatch(tmp_path) -> None:
    art, _, _ = _save(tmp_path)
    mp = art / "model.joblib"
    raw = bytearray(mp.read_bytes())
    raw[-1] ^= 0xFF
    mp.write_bytes(bytes(raw))
    with pytest.raises(ArtifactIntegrityError) as exc:
        load_artifact(art)
    assert exc.value.reason == "digest_mismatch" and exc.value.file == "model.joblib"


def test_tampered_schema_raises_digest_mismatch(tmp_path) -> None:
    art, _, _ = _save(tmp_path)
    sp = art / "schema.json"
    sp.write_text(sp.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ArtifactIntegrityError) as exc:
        load_artifact(art)
    assert exc.value.reason == "digest_mismatch" and exc.value.file == "schema.json"


def test_tampered_manifest_data_raises_digest_mismatch(tmp_path) -> None:
    """Editing a manifest payload key (leaving the stale digest) is detected (ADR-0067 §1)."""
    art, _, _ = _save(tmp_path)
    manifest = _manifest(art)
    manifest["best_model_id"] = "forged"
    _rewrite(art, manifest)
    with pytest.raises(ArtifactIntegrityError) as exc:
        load_artifact(art)
    assert exc.value.reason == "digest_mismatch" and exc.value.file == "manifest.json"


def test_missing_checksummed_file_raises(tmp_path) -> None:
    art, _, _ = _save(tmp_path)
    (art / "leaderboard.json").unlink()
    with pytest.raises(ArtifactIntegrityError) as exc:
        load_artifact(art)
    assert exc.value.reason == "missing_file" and exc.value.file == "leaderboard.json"


def test_checksum_file_outside_dir_rejected(tmp_path) -> None:
    """A non-basename entry in checksums.files (path traversal on read) is rejected (ADR-0067 §2).

    checksums.files is covered by the manifest digest, so a sophisticated tamper recomputes the digest;
    the anti-traversal guard is the last line that rejects the escaping name.
    """
    art, _, _ = _save(tmp_path)
    manifest = _manifest(art)
    manifest["checksums"]["files"]["../evil.bin"] = "00" * 32
    manifest["checksums"]["manifest"] = _manifest_digest(
        manifest
    )  # recompute so the guard is reached
    _rewrite(art, manifest)
    with pytest.raises(ArtifactIntegrityError) as exc:
        load_artifact(art)
    assert exc.value.reason == "missing_file" and exc.value.file == "../evil.bin"


# --- legacy without checksums + strict mode (ADR-0067 §3) ------------------------------------------


def test_legacy_without_checksums_loads(tmp_path) -> None:
    art, X, model = _save(tmp_path)
    manifest = _manifest(art)
    manifest.pop("checksums", None)  # a pre-M8 artifact
    _rewrite(art, manifest)
    loaded = load_artifact(art)  # default require_integrity=False -> warns, proceeds
    assert np.array_equal(loaded.predict(X), model.predict(X))


def test_require_integrity_strict_missing_checksums(tmp_path) -> None:
    art, _, _ = _save(tmp_path)
    manifest = _manifest(art)
    manifest.pop("checksums", None)
    _rewrite(art, manifest)
    with pytest.raises(ArtifactIntegrityError) as exc:
        load_artifact(art, require_integrity=True)
    assert exc.value.reason == "missing_checksums"


# --- version gate + model_type dispatch (ADR-0065) ------------------------------------------------


def test_cross_major_version_rejected(tmp_path) -> None:
    art, _, _ = _save(tmp_path)
    manifest = _manifest(art)
    manifest["artifact_version"] = 2  # version-gate runs before integrity
    _rewrite(art, manifest)
    with pytest.raises(SchemaValidationError, match="artifact_version"):
        load_artifact(art)


def test_unknown_model_type_rejected(tmp_path) -> None:
    art, _, _ = _save(tmp_path)
    manifest = _manifest(art)
    manifest.pop("checksums", None)  # legacy path so dispatch (not integrity) is exercised
    manifest["model_type"] = "onnx"
    _rewrite(art, manifest)
    with pytest.raises(SchemaValidationError, match="model_type"):
        load_artifact(art)


# --- regression artifact (classes=None, no calibrator) under integrity (FR-SRV-3) -----------------


def test_regression_artifact_intact_loads(tmp_path) -> None:
    art, X, model = _save(tmp_path, task="regression")
    assert _manifest(art)["classes"] is None
    assert "calibrator.joblib" not in _manifest(art)["checksums"]["files"]
    loaded = load_artifact(art)
    assert np.allclose(loaded.predict(X), model.predict(X))


# --- back-compat of the public signatures (FR-SRV-5, ADR-0067 §3) ---------------------------------


def test_backcompat_positional_signatures(tmp_path) -> None:
    """Old positional calls still work; the new params are keyword-only (cannot be passed positionally)."""
    model, X = _fit()
    art = tmp_path / "art"
    save_artifact(model.fitted_, art)  # positional (model, directory) — unchanged
    loaded = load_artifact(art)  # positional (directory) — unchanged
    assert np.array_equal(loaded.predict(X), model.predict(X))
    with pytest.raises(TypeError):
        load_artifact(art, True)  # require_integrity is keyword-only -> no accidental positional


# --- optional signature hook (ADR-0067 §4) --------------------------------------------------------


def test_signature_roundtrips_and_rejects_mismatch(tmp_path) -> None:
    art, X, model = _save(tmp_path, sign=lambda digest: digest[::-1])
    assert (art / "signature").exists()
    ok = load_artifact(art, verify=lambda sig, digest: sig == digest[::-1])
    assert np.array_equal(ok.predict(X), model.predict(X))
    with pytest.raises(ArtifactIntegrityError) as exc:
        load_artifact(art, verify=lambda sig, digest: False)
    assert exc.value.reason == "signature_mismatch"


def test_signature_covers_files_against_recomputed_tamper(tmp_path) -> None:
    """The signature authenticates per-file digests transitively (ADR-0067 §5): a file swap that
    recomputes its checksums.files entry AND the manifest digest still fails the original signature."""
    art, _, _ = _save(tmp_path, sign=lambda digest: digest[::-1])
    forged = b"forged-but-valid-looking"
    (art / "model.joblib").write_bytes(forged)
    manifest = _manifest(art)
    manifest["checksums"]["files"]["model.joblib"] = hashlib.sha256(forged).hexdigest()
    manifest["checksums"]["manifest"] = _manifest_digest(manifest)  # attacker cannot re-sign
    _rewrite(art, manifest)
    with pytest.raises(ArtifactIntegrityError) as exc:
        load_artifact(art, verify=lambda sig, digest: sig == digest[::-1])
    assert exc.value.reason == "signature_mismatch"


# --- native bodies under the same integrity protocol (M8b-1, NFR-SER-2) ----------------------------


def _save_native(tmp_path):
    pytest.importorskip("xgboost")
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=60, n_features=6, n_informative=4, n_redundant=0, random_state=0
    )
    model = AutoML(task="binary", models=("xgboost",), random_state=0).fit(X, y)
    art = tmp_path / "art"
    save_artifact(model.fitted_, art, model_format="native")
    return art, X, model


def test_native_intact_loads(tmp_path) -> None:
    art, X, model = _save_native(tmp_path)
    assert "model.ubj" in _manifest(art)["checksums"]["files"]
    loaded = load_artifact(art)
    assert np.array_equal(loaded.predict_proba(X), model.predict_proba(X))


def test_native_file_tamper_detected(tmp_path) -> None:
    art, _, _ = _save_native(tmp_path)
    body = art / "model.ubj"
    raw = bytearray(body.read_bytes())
    raw[-1] ^= 0xFF
    body.write_bytes(bytes(raw))
    with pytest.raises(ArtifactIntegrityError) as exc:
        load_artifact(art)
    assert exc.value.reason == "digest_mismatch" and exc.value.file == "model.ubj"


def test_anti_traversal_native_key(tmp_path) -> None:
    art, _, _ = _save_native(tmp_path)
    manifest = _manifest(art)
    manifest["checksums"]["files"]["..\\model.ubj"] = "00" * 32
    manifest["checksums"]["manifest"] = _manifest_digest(manifest)
    _rewrite(art, manifest)
    with pytest.raises(ArtifactIntegrityError) as exc:
        load_artifact(art)
    assert exc.value.reason == "missing_file" and exc.value.file == "..\\model.ubj"
