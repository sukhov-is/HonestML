"""Completed stage scopes include the versions of active optional search backends."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from honestml import AutoML
from honestml.adapters import JoblibCandidateCache, RocAuc
from honestml.core import FeatureSelectionConfig, HPOConfig, RunConfig, Task

pytestmark = pytest.mark.unit


def _fingerprint(config: RunConfig, *, tunable: bool = True) -> str:
    components: Any = SimpleNamespace(
        estimators={"linear": None},
        metric=RocAuc(),
        tunable={"linear": {"x": {"type": "int", "low": 1, "high": 2}}} if tunable else {},
    )
    return AutoML()._run_fingerprint(config, Task(kind="binary"), components, None, hpo=config.hpo)


@pytest.mark.parametrize(
    ("package", "config"),
    [
        ("optuna", RunConfig(hpo=HPOConfig(n_trials=1))),
        ("shap", RunConfig(fs=FeatureSelectionConfig(strategy="shap", refine=False))),
        ("shap", RunConfig(fs=FeatureSelectionConfig(compare=("importance", "shap")))),
    ],
    ids=["hpo", "shap-single", "shap-compared-cascade"],
)
def test_backend_upgrade_invalidates_prepared_and_completed_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, package: str, config: RunConfig
) -> None:
    versions = {package: "1.0"}
    monkeypatch.setattr("honestml.application.dataset_signature", lambda dataset: "same-dev")
    monkeypatch.setattr(
        "honestml.application.run_report.version", lambda name: versions.get(name, "fixed")
    )
    original = _fingerprint(config)
    stages = ("prepared_features_v1", "selection")
    for stage in stages:
        JoblibCandidateCache(tmp_path, original).put_stage(stage, {"completed": True})
        assert JoblibCandidateCache(tmp_path, _fingerprint(config)).get_stage(stage) is not None
    versions[package] = "2.0"
    updated = _fingerprint(config)
    assert updated != original
    for stage in stages:
        assert JoblibCandidateCache(tmp_path, updated).get_stage(stage) is None


@pytest.mark.parametrize(
    ("package", "config", "tunable"),
    [
        ("optuna", RunConfig(), True),
        ("optuna", RunConfig(hpo=HPOConfig(n_trials=1)), False),
        ("shap", RunConfig(fs=FeatureSelectionConfig(strategy="importance")), True),
        (
            "shap",
            RunConfig(fs=FeatureSelectionConfig(strategy="shap", compare=("importance",))),
            True,
        ),
    ],
    ids=["hpo-off", "nothing-tunable", "shap-off", "inactive-strategy-field"],
)
def test_inactive_optional_backend_version_does_not_invalidate(
    monkeypatch: pytest.MonkeyPatch, package: str, config: RunConfig, tunable: bool
) -> None:
    versions = {package: "1.0"}
    monkeypatch.setattr("honestml.application.dataset_signature", lambda dataset: "same-dev")
    monkeypatch.setattr(
        "honestml.application.run_report.version", lambda name: versions.get(name, "fixed")
    )
    original = _fingerprint(config, tunable=tunable)
    versions[package] = "2.0"
    assert _fingerprint(config, tunable=tunable) == original
