"""M4a2: honesty-band metadata round-trips through the artifact manifest (ADR-0026 §6, NFR-M4-5).

Band membership lives in additive manifest keys, never in the frozen ``LeaderboardEntry`` — so
``leaderboard.json`` is unchanged and forward/backward compatibility is symmetric. ``ARTIFACT_VERSION``
stays 1.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from honestml import AutoML
from honestml.composition.artifact import ARTIFACT_VERSION, load_artifact, save_artifact

pytestmark = pytest.mark.unit


def _fit(n: int = 80):
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=n, n_features=6, n_informative=4, n_redundant=0, random_state=0
    )
    return AutoML(task="binary", random_state=0).fit(X, y), X


def test_manifest_band_keys(tmp_path) -> None:
    """Non-default band metadata is serialized to the manifest and read back; version unchanged."""
    model, X = _fit()
    fm = model.fitted_
    fm.band_member_ids = ("baseline", "linear")
    fm.band_unstable = True
    fm.band_width = 2
    fm.winner_by_tiebreak = True
    art = tmp_path / "art"
    save_artifact(fm, art)

    manifest = json.loads((art / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["artifact_version"] == ARTIFACT_VERSION == 1
    assert manifest["band_member_ids"] == ["baseline", "linear"]
    assert manifest["band_unstable"] is True
    assert manifest["band_width"] == 2
    assert manifest["winner_by_tiebreak"] is True

    loaded = load_artifact(art)
    assert loaded.band_member_ids == ("baseline", "linear")
    assert loaded.band_unstable is True and loaded.band_width == 2
    assert loaded.winner_by_tiebreak is True
    assert np.array_equal(loaded.predict(X), model.predict(X))


def test_band_not_in_leaderboard_json(tmp_path) -> None:
    """NFR-M4-5: band membership is a manifest key, not a LeaderboardEntry field (file unchanged)."""
    model, _ = _fit()
    art = tmp_path / "art"
    save_artifact(model.fitted_, art)
    entries = json.loads((art / "leaderboard.json").read_text(encoding="utf-8"))
    assert entries and all("in_band" not in e and "band_member_ids" not in e for e in entries)


def test_legacy_manifest_without_band_keys_loads(tmp_path) -> None:
    """A pre-M4a2 manifest (no band keys) loads with lone-anchor defaults and still predicts."""
    model, X = _fit()
    art = tmp_path / "art"
    save_artifact(model.fitted_, art)
    mp = art / "manifest.json"
    manifest = json.loads(mp.read_text(encoding="utf-8"))
    for key in ("band_member_ids", "band_unstable", "band_width", "winner_by_tiebreak"):
        manifest.pop(key, None)
    manifest.pop("checksums", None)  # a pre-M8 manifest also lacks integrity checksums
    mp.write_text(json.dumps(manifest), encoding="utf-8")

    loaded = load_artifact(art)
    assert loaded.band_member_ids == () and loaded.band_unstable is False
    assert loaded.band_width == 1 and loaded.winner_by_tiebreak is False
    assert np.array_equal(loaded.predict(X), model.predict(X))


# --- M4d: calibrator + selection_mode/score_space round-trip (ADR-0030 §4 / ADR-0031 §6) ----


def _fitted_calibrator():
    """A non-identity fitted sigmoid calibrator (true freq = p^2, so P(pos) is miscalibrated)."""
    from honestml.adapters import SigmoidCalibrator

    rng = np.random.default_rng(0)
    p = rng.uniform(0.05, 0.95, 400)
    y = (rng.uniform(size=400) < p**2).astype(int)
    cal = SigmoidCalibrator()
    cal.fit(p, y)
    return cal


def test_calibrator_roundtrips_and_recalibrates(tmp_path) -> None:
    """A calibrator + report round-trips; predict_proba is recalibrated and bit-identical on load."""
    model, X = _fit()
    fm = model.fitted_
    fm.calibrator = _fitted_calibrator()
    fm.calibration = {
        "method": "sigmoid",
        "applied": True,
        "brier_raw": 0.2,
        "brier_calibrated": 0.18,
        "ece_raw": 0.1,
        "ece_calibrated": 0.05,
        "reliability": {"prob_true": [0.4, 0.7], "prob_pred": [0.45, 0.72]},
    }
    fm.selection_mode = "refinement"
    fm.score_space = "calibrated_oof"
    calibrated = fm.predict_proba(X)
    fm.calibrator = None
    raw = fm.predict_proba(X)
    fm.calibrator = _fitted_calibrator()
    assert not np.allclose(calibrated, raw)  # calibration changed the probabilities
    assert np.allclose(calibrated.sum(axis=1), 1.0)  # still valid distributions

    art = tmp_path / "art"
    save_artifact(fm, art)
    assert (art / "calibrator.joblib").exists()
    manifest = json.loads((art / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["artifact_version"] == 1  # additive, version unchanged
    assert manifest["calibration"]["applied"] is True
    assert (
        manifest["selection_mode"] == "refinement" and manifest["score_space"] == "calibrated_oof"
    )

    loaded = load_artifact(art)
    assert loaded.calibrator is not None
    assert np.allclose(loaded.predict_proba(X), calibrated)  # recalibrated, round-tripped
    assert loaded.selection_mode == "refinement" and loaded.calibration["applied"] is True


def test_legacy_manifest_without_calibration_loads(tmp_path) -> None:
    """A pre-M4d manifest (no calibration keys) loads calibrator-less with raw-selection defaults."""
    model, X = _fit()
    art = tmp_path / "art"
    save_artifact(model.fitted_, art)
    mp = art / "manifest.json"
    manifest = json.loads(mp.read_text(encoding="utf-8"))
    for key in ("calibration", "selection_mode", "score_space", "calibrator_file"):
        manifest.pop(key, None)
    manifest.pop("checksums", None)  # a pre-M8 manifest also lacks integrity checksums
    mp.write_text(json.dumps(manifest), encoding="utf-8")

    loaded = load_artifact(art)
    assert loaded.calibrator is None and loaded.calibration is None
    assert loaded.selection_mode == "raw" and loaded.score_space == "raw_oof"
    assert np.array_equal(loaded.predict(X), model.predict(X))


# --- M4c: holdout_score round-trips through the manifest (ADR-0029 §3, NFR-M4-5/7) ----


def test_holdout_score_roundtrips(tmp_path) -> None:
    """An attached holdout_score is serialized to the manifest and read back; version unchanged."""
    model, X = _fit()
    fm = model.fitted_
    fm.holdout_score = 0.873
    art = tmp_path / "art"
    save_artifact(fm, art)
    manifest = json.loads((art / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["artifact_version"] == 1  # additive, version unchanged
    assert manifest["holdout_score"] == 0.873

    loaded = load_artifact(art)
    assert loaded.holdout_score == 0.873
    assert np.array_equal(loaded.predict(X), model.predict(X))


def test_legacy_manifest_without_holdout_loads(tmp_path) -> None:
    """A pre-M4c manifest (no holdout_score) loads with None (off) and still predicts."""
    model, X = _fit()
    art = tmp_path / "art"
    save_artifact(model.fitted_, art)
    mp = art / "manifest.json"
    manifest = json.loads(mp.read_text(encoding="utf-8"))
    manifest.pop("holdout_score", None)
    manifest.pop("checksums", None)  # a pre-M8 manifest also lacks integrity checksums
    mp.write_text(json.dumps(manifest), encoding="utf-8")

    loaded = load_artifact(art)
    assert loaded.holdout_score is None
    assert np.array_equal(loaded.predict(X), model.predict(X))


# --- M7b: ensemble artifact (BlendedEstimator + additive manifest block, ADR-0064) ----


def _blended_fitted(n: int = 120):
    """A FittedModel shipping a BlendedEstimator over two real fitted members (ADR-0064 §1)."""
    from sklearn.datasets import make_classification

    from honestml.adapters import BlendedEstimator

    X, y = make_classification(
        n_samples=n, n_features=6, n_informative=4, n_redundant=0, random_state=0
    )
    m0 = AutoML(task="binary", random_state=0).fit(X, y)
    m1 = AutoML(task="binary", random_state=1).fit(X, y)
    fm = m0.fitted_
    members = [m0.best_estimator_, m1.best_estimator_]
    blended = BlendedEstimator(members, np.array([0.6, 0.4]), fm.classes)
    blended.feature_names = list(members[0].feature_names)
    fm.estimator = blended
    fm.best_model_id = (
        m0.best_model_id_
    )  # the REAL winner id stays on the in-memory model (ADR-0064 §3)
    fm.ensemble = {
        "applied": True,
        "method": "caruana",
        "member_ids": [m0.best_model_id_, m1.best_model_id_],
        "weights": {"member_0": 0.6, "member_1": 0.4},
        "gate_reason": "significant_improvement",
    }
    return fm, X


def test_ensemble_roundtrip_reproduces_predictions(tmp_path) -> None:
    """FR-ENS-5: a BlendedEstimator artifact round-trips; predict/predict_proba reproduce bit-for-bit."""
    fm, X = _blended_fitted()
    before_pred, before_proba = fm.predict(X), fm.predict_proba(X)
    art = tmp_path / "ens"
    save_artifact(fm, art)

    manifest = json.loads((art / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["artifact_version"] == ARTIFACT_VERSION == 1
    assert manifest["best_model_id"] == "ensemble"  # synthetic id on save (ADR-0064 §3)
    assert manifest["ensemble"]["applied"] is True and manifest["ensemble"]["method"] == "caruana"

    loaded = load_artifact(art)
    assert loaded.ensemble["applied"] is True
    assert np.array_equal(loaded.predict(X), before_pred)
    assert np.allclose(loaded.predict_proba(X), before_proba)


def test_artifact_version_is_one_with_ensemble(tmp_path) -> None:
    """FR-ENS-5/NFR-M7-4: the ensemble block is additive — ARTIFACT_VERSION stays 1."""
    fm, _ = _blended_fitted()
    save_artifact(fm, tmp_path / "ens")
    manifest = json.loads((tmp_path / "ens" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["artifact_version"] == 1


def test_legacy_single_model_loads_unchanged(tmp_path) -> None:
    """A pre-M7b single-model artifact (no ensemble key) loads with ensemble=None and predicts (NFR-M7-4)."""
    model, X = _fit()
    art = tmp_path / "single"
    save_artifact(model.fitted_, art)
    mp = art / "manifest.json"
    manifest = json.loads(mp.read_text(encoding="utf-8"))
    assert manifest["ensemble"] is None  # single model writes a null ensemble block
    manifest.pop("ensemble", None)  # a truly legacy manifest lacks the key entirely
    manifest.pop("checksums", None)  # a pre-M8 manifest also lacks integrity checksums
    mp.write_text(json.dumps(manifest), encoding="utf-8")

    loaded = load_artifact(art)
    assert loaded.ensemble is None
    assert np.array_equal(loaded.predict(X), model.predict(X))


def test_binary_ensemble_calibrate_roundtrip(tmp_path) -> None:
    """ADR-0064 §1: a calibrator over the blended (n,2) proba round-trips and stays a valid distribution."""
    fm, X = _blended_fitted()
    fm.calibrator = _fitted_calibrator()
    calibrated = fm.predict_proba(X)
    assert calibrated.shape[1] == 2 and np.allclose(calibrated.sum(axis=1), 1.0)
    art = tmp_path / "ens_cal"
    save_artifact(fm, art)
    loaded = load_artifact(art)
    assert loaded.calibrator is not None and loaded.ensemble["applied"] is True
    assert np.allclose(loaded.predict_proba(X), calibrated)
