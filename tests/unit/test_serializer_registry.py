"""M8b-1: the ModelSerializer boundary + registry dispatch (ADR-0069, FR-SER-1/FR-SER-4).

The artifact core dispatches the model body by ``manifest["model_type"]`` through an ordered,
data-driven registry — a new format is a new adapter plus a registry entry, never an edit of the
``save_artifact``/``load_artifact`` cores (OCP). The joblib default is the M8 behavior, unchanged.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from honestml import AutoML
from honestml.composition import artifact as artifact_module
from honestml.composition.artifact import load_artifact, save_artifact
from honestml.core import SchemaValidationError

pytestmark = pytest.mark.unit


def _save_default(tmp_path):
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=60, n_features=6, n_informative=4, n_redundant=0, random_state=0
    )
    model = AutoML(task="binary", models=("baseline", "linear"), random_state=0).fit(X, y)
    art = tmp_path / "art"
    save_artifact(model.fitted_, art)
    return art, X, model


def _manifest(art) -> dict:
    return json.loads((art / "manifest.json").read_text(encoding="utf-8"))


def _rewrite(art, manifest) -> None:
    (art / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


# --- dispatch is keyed by model_type (ADR-0065 §3 / ADR-0069 §2) ------------------------------------


def test_dispatch_by_model_type() -> None:
    for model_type in ("joblib", "xgboost", "catboost", "lightgbm"):
        assert artifact_module._serializer_for_type(model_type).model_type == model_type


def test_unknown_type_raises_schema(tmp_path) -> None:
    art, _, _ = _save_default(tmp_path)
    manifest = _manifest(art)
    manifest.pop("checksums", None)  # legacy path so dispatch (not integrity) is exercised
    manifest["model_type"] = "alien"
    _rewrite(art, manifest)
    with pytest.raises(SchemaValidationError, match="model_type 'alien'"):
        load_artifact(art)


def test_unknown_model_format_rejected(tmp_path) -> None:
    _, _, model = _save_default(tmp_path)
    with pytest.raises(SchemaValidationError, match="model_format"):
        save_artifact(model.fitted_, tmp_path / "other", model_format="onnx")


# --- the joblib default is the M8 artifact, unchanged (FR-SER-4, NFR-SER-1) -------------------------


def test_joblib_default_unchanged(tmp_path) -> None:
    art, X, model = _save_default(tmp_path)
    manifest = _manifest(art)
    assert manifest["model_type"] == "joblib"
    assert manifest["model_file"] == "model.joblib"
    assert "required_extra" not in manifest
    assert set(manifest["checksums"]["files"]) == {
        "schema.json",
        "leaderboard.json",
        "model.joblib",
    }
    loaded = load_artifact(art)
    assert np.array_equal(loaded.predict_proba(X), model.predict_proba(X))


# --- a new model_type is a registry entry, not a core edit (FR-SER-1, OCP) --------------------------


class _StubEstimator:
    feature_names: list[str] = []

    def fit(self, X, y, X_val=None, y_val=None, sample_weight=None) -> _StubEstimator:
        return self

    def predict(self, X) -> np.ndarray:
        return np.zeros(len(X))


class _StubSerializer:
    model_type = "stub"

    def can_serialize(self, estimator) -> bool:
        return False

    def save(self, estimator, directory):
        raise AssertionError("save is not exercised by the dispatch test")

    def load(self, directory, manifest) -> _StubEstimator:
        return _StubEstimator()


def test_register_adds_loader_without_core_edit(tmp_path, monkeypatch) -> None:
    art, _, _ = _save_default(tmp_path)
    manifest = _manifest(art)
    manifest.pop("checksums", None)
    manifest["model_type"] = "stub"
    _rewrite(art, manifest)
    monkeypatch.setattr(
        artifact_module, "_SERIALIZERS", [*artifact_module._SERIALIZERS, _StubSerializer()]
    )
    loaded = load_artifact(art)  # the load core dispatched to the new entry untouched
    assert isinstance(loaded.estimator, _StubEstimator)
