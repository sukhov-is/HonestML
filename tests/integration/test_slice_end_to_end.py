"""M2-7/M2-8: end-to-end binary slice on synthetic data (ADR-0010/0011/0012).

Drives the full path through real components — Reader → composition →
``run_slice`` → refit → facade → versioned artifact — including a categorical
column to prove the schema-owned ``CategoryTable`` makes train==inference
preprocessing (ADR-0005). Determinism (NFR-4) and the artifact version-check are
asserted as golden behaviour.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from honestml import AutoML, load_artifact, save_artifact
from honestml.core import CVConfig, SchemaValidationError

pytestmark = pytest.mark.slow


def _frame(n: int = 200, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    cat = rng.choice(["a", "b", "c"], size=n)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    logit = 1.5 * x1 + (cat == "a") * 1.0 - (cat == "c") * 1.0
    y = (logit + 0.5 * rng.normal(size=n) > 0).astype(int)
    return pd.DataFrame({"x1": x1, "x2": x2, "cat": cat}), y


def _frame_multiclass(n: int = 240, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    cat = rng.choice(["a", "b", "c"], size=n)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    score = x1 + (cat == "a") * 1.0 - (cat == "c") * 1.0
    y = np.digitize(score, np.quantile(score, [1 / 3, 2 / 3]))  # 3 balanced classes
    return pd.DataFrame({"x1": x1, "x2": x2, "cat": cat}), y


def _frame_regression(n: int = 200, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    cat = rng.choice(["a", "b", "c"], size=n)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = 2.0 * x1 - x2 + (cat == "a") * 1.0 + 0.1 * rng.normal(size=n)
    return pd.DataFrame({"x1": x1, "x2": x2, "cat": cat}), y


def test_fit_predict_save_load_roundtrip(tmp_path) -> None:
    df, y = _frame()
    model = AutoML(task="binary", random_state=0).fit(df, y)
    pred_before = model.predict(df)
    proba_before = model.predict_proba(df)

    save_artifact(model.fitted_, tmp_path / "art")
    loaded = load_artifact(tmp_path / "art")

    # train==inference: standalone load reproduces the facade exactly
    assert np.array_equal(pred_before, loaded.predict(df))
    assert np.allclose(proba_before, loaded.predict_proba(df))
    assert loaded.best_model_id == model.best_model_id_
    assert [e.model_id for e in loaded.leaderboard] == [e.model_id for e in model.leaderboard_]


def test_artifact_files_and_manifest(tmp_path) -> None:
    df, y = _frame()
    model = AutoML(task="binary", random_state=0).fit(df, y)
    art = tmp_path / "art"
    save_artifact(model.fitted_, art)
    for name in ("manifest.json", "schema.json", "leaderboard.json", "model.joblib"):
        assert (art / name).exists()
    manifest = json.loads((art / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["artifact_version"] == 1
    assert manifest["task"]["kind"] == "binary"


def test_version_check_rejects_mismatch(tmp_path) -> None:
    df, y = _frame()
    model = AutoML(task="binary", random_state=0).fit(df, y)
    art = tmp_path / "art"
    save_artifact(model.fitted_, art)
    manifest_path = art / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_version"] = 999
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(SchemaValidationError, match="artifact_version"):
        load_artifact(art)


def test_unseen_category_at_inference_is_handled(tmp_path) -> None:
    df, y = _frame()
    model = AutoML(task="binary", random_state=0).fit(df, y)
    # a category unseen at train must map to the reserved unknown code, not crash
    infer = df.head(5).copy()
    infer.loc[:, "cat"] = "ZZZ"
    assert model.predict(infer).shape == (5,)


def test_missing_manifest_key_raises_clean_error(tmp_path) -> None:
    df, y = _frame()
    model = AutoML(task="binary", random_state=0).fit(df, y)
    art = tmp_path / "art"
    save_artifact(model.fitted_, art)
    mp = art / "manifest.json"
    manifest = json.loads(mp.read_text(encoding="utf-8"))
    del manifest["best_model_id"]
    manifest.pop(
        "checksums", None
    )  # a pre-M8 manifest lacks checksums (else integrity fires first)
    mp.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(SchemaValidationError, match="missing required key"):
        load_artifact(art)


def test_model_file_traversal_is_confined(tmp_path) -> None:
    df, y = _frame()
    model = AutoML(task="binary", random_state=0).fit(df, y)
    art = tmp_path / "art"
    save_artifact(model.fitted_, art)
    mp = art / "manifest.json"
    manifest = json.loads(mp.read_text(encoding="utf-8"))
    manifest["model_file"] = "../model.joblib"  # traversal attempt
    manifest.pop("checksums", None)  # legacy manifest: reach model_file dispatch, not integrity
    mp.write_text(json.dumps(manifest), encoding="utf-8")
    # basename confinement neutralizes the prefix -> the in-dir model still loads
    loaded = load_artifact(art)
    assert loaded.predict(df).shape == (len(y),)


# --- M3b: multiclass + regression end-to-end save/load (ADR-0020/0021/0024) ----


def test_multiclass_fit_save_load_roundtrip(tmp_path) -> None:
    df, y = _frame_multiclass()
    model = AutoML(task="multiclass", random_state=0).fit(df, y)
    pred_before = model.predict(df)
    proba_before = model.predict_proba(df)
    assert proba_before.shape == (len(y), 3)

    save_artifact(model.fitted_, tmp_path / "art")
    loaded = load_artifact(tmp_path / "art")

    assert np.array_equal(pred_before, loaded.predict(df))
    assert np.allclose(proba_before, loaded.predict_proba(df))
    manifest = json.loads((tmp_path / "art" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["task"]["kind"] == "multiclass"
    assert manifest["classes"] == [0, 1, 2]


def test_regression_fit_save_load_roundtrip(tmp_path) -> None:
    df, y = _frame_regression()
    # regression default CV is kfold (M3c); use holdout for the M3b end-to-end path
    model = AutoML(task="regression", cv=CVConfig(scheme="holdout"), random_state=0).fit(df, y)
    pred_before = model.predict(df)

    save_artifact(model.fitted_, tmp_path / "art")
    loaded = load_artifact(tmp_path / "art")  # must not crash on a model without classes_

    assert np.allclose(pred_before, loaded.predict(df))
    assert isinstance(loaded.score(df, y), float)
    with pytest.raises(SchemaValidationError, match="no probabilities"):
        loaded.predict_proba(df)
    manifest = json.loads((tmp_path / "art" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["task"]["kind"] == "regression"
    assert manifest["classes"] is None


def test_legacy_binary_artifact_without_new_keys_loads(tmp_path) -> None:
    # a pre-M3b manifest (no classes/metric_average/early_stopping) still loads: classes
    # fall back to the classifier's classes_ (ADR-0024 §4 back-compat).
    df, y = _frame()
    model = AutoML(task="binary", random_state=0).fit(df, y)
    art = tmp_path / "art"
    save_artifact(model.fitted_, art)
    mp = art / "manifest.json"
    manifest = json.loads(mp.read_text(encoding="utf-8"))
    for key in ("classes", "metric_average", "early_stopping"):
        manifest.pop(key, None)
    manifest.pop("checksums", None)  # a pre-M8 manifest also lacks integrity checksums
    mp.write_text(json.dumps(manifest), encoding="utf-8")

    loaded = load_artifact(art)
    assert np.array_equal(model.predict(df), loaded.predict(df))
    assert np.allclose(model.predict_proba(df), loaded.predict_proba(df))


@pytest.mark.golden
def test_determinism_with_fixed_seed() -> None:
    df, y = _frame()
    a = AutoML(task="binary", random_state=0).fit(df, y)
    b = AutoML(task="binary", random_state=0).fit(df, y)
    assert [(e.model_id, e.score) for e in a.leaderboard_] == [
        (e.model_id, e.score) for e in b.leaderboard_
    ]
    assert np.array_equal(a.predict(df), b.predict(df))
