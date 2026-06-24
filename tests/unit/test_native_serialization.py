"""M8b-1: native boosting serialization (ADR-0070, FR-SER-2) — exact round-trip via stable formats.

A boosting body persists through the library's documented-stable API (xgb ubj / cat cbm / lgbm text)
instead of pickle; the load path re-wraps it into the same ``Estimator`` the facade ships, so the
``FittedModel`` inference path is unchanged and predictions are EXACT (SPIKE-0003, max|Δ|=0.0).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pytest

from honestml import AutoML
from honestml.composition.artifact import load_artifact, save_artifact
from honestml.core import MissingDependencyError

pytestmark = pytest.mark.unit

_BODY = {"xgboost": "model.ubj", "catboost": "model.cbm", "lightgbm": "model.txt"}


def _data(task: str = "binary", n: int = 80, classes: int = 2):
    if task == "regression":
        from sklearn.datasets import make_regression

        return make_regression(n_samples=n, n_features=6, n_informative=4, random_state=0)
    from sklearn.datasets import make_classification

    return make_classification(
        n_samples=n,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        n_classes=classes,
        random_state=0,
    )


def _fit(lib: str, task: str = "binary", y_map=None, classes: int = 2):
    pytest.importorskip(lib)
    X, y = _data(task, classes=classes)
    if y_map is not None:
        y = y_map(y)
    return AutoML(task=task, models=(lib,), random_state=0).fit(X, y), X


def _roundtrip(tmp_path, lib: str, task: str = "binary", y_map=None, classes: int = 2):
    model, X = _fit(lib, task, y_map=y_map, classes=classes)
    art = tmp_path / "art"
    save_artifact(model.fitted_, art, model_format="native")
    return model, X, art, load_artifact(art)


def _manifest(art) -> dict:
    return json.loads((art / "manifest.json").read_text(encoding="utf-8"))


# --- exact native round-trip per family (ADR-0070 §2-§4, NFR-SER-4 native==exact) ------------------


@pytest.mark.parametrize("lib", ["xgboost", "catboost", "lightgbm"])
def test_native_roundtrip_parity(tmp_path, lib) -> None:
    model, X, art, loaded = _roundtrip(tmp_path, lib)
    manifest = _manifest(art)
    assert manifest["model_type"] == lib
    assert (art / _BODY[lib]).is_file() and not (art / "model.joblib").exists()
    # the body is the same booster -> bit-exact, not approximate (SPIKE-0003)
    assert np.array_equal(loaded.predict_proba(X), model.predict_proba(X))
    assert np.array_equal(loaded.predict(X), model.predict(X))


@pytest.mark.parametrize("lib", ["xgboost", "catboost", "lightgbm"])
def test_native_regression_roundtrip_parity(tmp_path, lib) -> None:
    model, X, art, loaded = _roundtrip(tmp_path, lib, task="regression")
    assert _manifest(art)["model_type"] == lib
    assert np.array_equal(loaded.predict(X), model.predict(X))


# --- LightGBM classification: sklearn-API recovery over the raw Booster (ADR-0070 §4) --------------


def test_lgbm_clf_recovers_predict_proba(tmp_path) -> None:
    model, X, _, loaded = _roundtrip(tmp_path, "lightgbm")
    est = loaded.estimator
    assert np.array_equal(np.asarray(est.classes_), np.asarray(model.fitted_.estimator.classes_))
    proba = loaded.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


@pytest.mark.parametrize(
    ("task", "classes", "y_map"),
    [
        ("binary", 2, lambda y: np.where(y == 1, "yes", "no")),
        ("binary", 2, lambda y: y * 2 - 1),
        ("multiclass", 3, None),
    ],
    ids=["str-labels", "minus1-plus1", "three-class"],
)
def test_lgbm_clf_full_path_non01_labels(tmp_path, task, classes, y_map) -> None:
    """R1-AD2: the FULL FittedModel path (classes_ from the manifest + align_proba), not raw Booster."""
    model, X, _, loaded = _roundtrip(tmp_path, "lightgbm", task=task, y_map=y_map, classes=classes)
    assert np.array_equal(loaded.predict_proba(X), model.predict_proba(X))
    assert np.array_equal(loaded.predict(X), model.predict(X))


@pytest.mark.parametrize("lib", ["xgboost", "catboost", "lightgbm"])
@pytest.mark.parametrize(
    ("task", "classes", "y_map"),
    [
        ("binary", 2, lambda y: np.where(y == 1, "yes", "no")),
        ("binary", 2, lambda y: y * 2 - 1),
        ("multiclass", 3, lambda y: np.array([f"C{v}" for v in y])),
    ],
    ids=["str-labels", "minus1-plus1", "three-class-str"],
)
def test_native_roundtrip_non01_labels(tmp_path, lib, task, classes, y_map) -> None:
    """ADR-0081 regression: the native round-trip preserves predictions for non-0..K-1 labels across
    ALL boosting families. xgboost codes labels to 0..K-1 internally, so the native body holds only
    codes; the manifest's global class order is restored on load. This is the coverage that was
    missing — the xgboost round-trip tests only ever used 0..K-1 labels, hiding the label bug."""
    model, X, _, loaded = _roundtrip(tmp_path, lib, task=task, y_map=y_map, classes=classes)
    assert np.array_equal(loaded.predict_proba(X), model.predict_proba(X))
    assert np.array_equal(loaded.predict(X), model.predict(X))


# --- WS-C native categorical: round-trip restores indices + additive manifest keys (ADR-0091) ------


def _cat_frame(n: int = 200, seed: int = 0):
    import pandas as pd

    rng = np.random.default_rng(seed)
    cat = rng.choice(["a", "b", "c", "d"], size=n)
    x1 = rng.normal(size=n)
    eff = {"a": 1.5, "b": -1.5, "c": 1.0, "d": -1.0}
    y = (0.8 * x1 + np.array([eff[c] for c in cat]) + 0.4 * rng.normal(size=n) > 0).astype(int)
    return pd.DataFrame({"x1": x1, "x2": rng.normal(size=n), "cat": cat}), y


def test_catboost_native_categorical_roundtrip(tmp_path) -> None:
    # FR-5/NFR-6: native (.cbm) round-trip of a categorical CatBoost — the manifest records the routing
    # and restores categorical_indices on load, so predict int-casts identically (exact, SPIKE-0004).
    pytest.importorskip("catboost")
    df, y = _cat_frame()
    model = AutoML(task="binary", models=("catboost",), random_state=0).fit(df, y)
    art = tmp_path / "art"
    save_artifact(model.fitted_, art, model_format="native")
    manifest = _manifest(art)
    assert manifest["categorical_indices"]  # NFR-6: non-empty routing record in the artifact
    assert manifest["native_categorical"] == {
        "backend": "catboost",
        "n_cat": len(manifest["categorical_indices"]),
    }
    loaded = load_artifact(art)
    assert np.array_equal(loaded.predict_proba(df), model.predict_proba(df))  # FR-5: exact
    assert np.array_equal(loaded.predict(df), model.predict(df))


def _mixed_card_frame(n: int = 400, hi_levels: int = 150, seed: int = 0):
    import pandas as pd

    rng = np.random.default_rng(seed)
    lo = rng.choice(["a", "b", "c", "d"], size=n)  # 4 levels -> native
    hi = np.array([f"v{i}" for i in rng.integers(0, hi_levels, size=n)])  # 150 levels -> demoted
    x1 = rng.normal(size=n)
    eff = {"a": 1.4, "b": -1.4, "c": 0.9, "d": -0.9}
    y = (0.7 * x1 + np.array([eff[c] for c in lo]) + 0.4 * rng.normal(size=n) > 0).astype(int)
    return pd.DataFrame({"x1": x1, "lo": lo, "hi": hi}), y


def test_native_categorical_gate_roundtrip_with_demotion(tmp_path) -> None:
    # FR-3/FR-5: a high-card categorical demoted by the cardinality gate rides the codes path; the native
    # round-trip stays EXACT and the manifest n_cat counts only the natively-routed (post-gate) columns.
    pytest.importorskip("catboost")
    df, y = _mixed_card_frame()
    model = AutoML(task="binary", models=("catboost",), random_state=0).fit(df, y)
    art = tmp_path / "art"
    save_artifact(model.fitted_, art, model_format="native")
    manifest = _manifest(art)
    # only the low-card 'lo' routes natively; the 150-level 'hi' is demoted to the codes path
    assert manifest["native_categorical"]["n_cat"] == 1
    assert len(manifest["categorical_indices"]) == 1
    loaded = load_artifact(art)
    assert np.array_equal(
        loaded.predict_proba(df), model.predict_proba(df)
    )  # exact at the demotion
    assert np.array_equal(loaded.predict(df), model.predict(df))


def test_numeric_native_artifact_omits_categorical_keys(tmp_path) -> None:
    # NFR-7: a non-categorical native model writes no categorical keys -> codes-path load, unchanged
    model, X, art, loaded = _roundtrip(tmp_path, "catboost")  # numeric make_classification
    manifest = _manifest(art)
    assert "categorical_indices" not in manifest and "native_categorical" not in manifest
    assert loaded.estimator.categorical_indices == []  # codes path restored
    assert np.array_equal(loaded.predict_proba(X), model.predict_proba(X))


# --- runtime self-description + MissingDependencyError (ADR-0070 §6, FR-SER-2) ---------------------


def test_manifest_records_runtime(tmp_path) -> None:
    _, _, art, _ = _roundtrip(tmp_path, "xgboost")
    manifest = _manifest(art)
    assert manifest["required_extra"] == "xgboost"
    assert manifest["model_file"] == "model.ubj"


def test_missing_lib_raises_missing_dependency(tmp_path, monkeypatch) -> None:
    _, _, art, _ = _roundtrip(tmp_path, "xgboost")
    import sys

    import honestml.adapters.serializers as serializers

    monkeypatch.setattr(serializers, "find_spec", lambda name: None)
    # block the import itself too, so a regression moving the gate AFTER `import xgboost`
    # surfaces as an ImportError here instead of passing on the installed library
    monkeypatch.setitem(sys.modules, "xgboost", None)
    with pytest.raises(MissingDependencyError) as exc:
        load_artifact(art)
    assert exc.value.extra == "xgboost"


# --- fallbacks: sklearn (documented) and ensemble (disclosed) (ADR-0069 §2 / ADR-0070 §7) ----------


def test_sklearn_falls_back_to_joblib(tmp_path) -> None:
    X, y = _data()
    model = AutoML(task="binary", models=("baseline", "linear"), random_state=0).fit(X, y)
    art = tmp_path / "art"
    save_artifact(model.fitted_, art, model_format="native")
    manifest = _manifest(art)
    assert manifest["model_type"] == "joblib"
    assert "required_extra" not in manifest
    loaded = load_artifact(art)
    assert np.array_equal(loaded.predict_proba(X), model.predict_proba(X))


def test_ensemble_native_warns_joblib_body(tmp_path, caplog) -> None:
    """ADR-0070 §7: a shipped ensemble has no native path -> joblib body, disclosed (not silent)."""
    X, y = _data()
    model = AutoML(task="binary", models=("baseline", "linear"), random_state=0).fit(X, y)
    fm = model.fitted_
    fm.ensemble = {"applied": True, "method": "weighted", "member_ids": ["baseline", "linear"]}
    with caplog.at_level(logging.WARNING):
        save_artifact(fm, tmp_path / "art", model_format="native")
    assert _manifest(tmp_path / "art")["model_type"] == "joblib"
    assert any("ensemble" in record.message for record in caplog.records)


# --- the body is the library's format, not a pickle stream (NFR-SER-5) -----------------------------


def test_body_is_native_not_pickle(tmp_path) -> None:
    _, _, art, _ = _roundtrip(tmp_path, "lightgbm")
    body = (art / "model.txt").read_text(encoding="utf-8")
    assert body.startswith("tree")  # lightgbm text model, readable, no pickle opcode stream


# --- cross-version durability: byte-frozen fixtures (NFR-SER-5, R2) --------------------------------

_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "native_artifacts"


@pytest.mark.parametrize(
    ("name", "lib"),
    [
        ("xgboost", "xgboost"),
        ("catboost", "catboost"),
        ("lightgbm_clf", "lightgbm"),
        ("lightgbm_reg", "lightgbm"),
    ],
)
def test_byte_frozen_fixture_loads(name, lib) -> None:
    """A committed artifact generated under a RECORDED library version (fixtures README) still loads
    and reproduces its recorded predictions — a library bump that breaks native load fails CI."""
    pytest.importorskip(lib)
    fixture = _FIXTURES / name
    expected = json.loads((fixture / "expected.json").read_text(encoding="utf-8"))
    X = np.asarray(expected["X"])
    loaded = load_artifact(fixture / "artifact")
    if "proba" in expected:
        assert np.allclose(loaded.predict_proba(X), np.asarray(expected["proba"]), atol=1e-6)
    else:
        assert np.allclose(loaded.predict(X), np.asarray(expected["pred"]), rtol=1e-5, atol=1e-6)
