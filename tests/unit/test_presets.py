"""M9-1: declarative facade presets (ADR-0074, FR-DLV-1/2, NFR-DLV-2).

Presets are data over the None-default surface; an explicit value always wins;
honesty parameters are not presettable by construction; the fingerprint carries the
RESOLVED parameters (a preset and its explicit equivalent are indistinguishable).
"""

from __future__ import annotations

import pytest
from sklearn.base import clone
from sklearn.datasets import make_classification

from honestml import AutoML, ConfigError, EnsembleConfig, HPOConfig
from honestml.composition import presets

pytestmark = pytest.mark.unit

_ALL_NONE = dict.fromkeys(presets.PRESET_SURFACE)


def _data():
    return make_classification(
        n_samples=60, n_features=6, n_informative=4, n_redundant=0, random_state=0
    )


# --- resolve_preset (pure, no fits) -------------------------------------------------------


def test_builtin_presets_are_valid() -> None:
    """FR-DLV-1: every built-in profile normalizes into facade forms without errors."""
    for name, entries in presets.PRESETS.items():
        _, block = presets.resolve_preset(name, _ALL_NONE)
        assert block["name"] == name
        assert set(block["applied"]) == set(entries)
    fast, _ = presets.resolve_preset("fast", _ALL_NONE)
    assert fast["cv"] == 3
    best, _ = presets.resolve_preset("best", _ALL_NONE)
    assert isinstance(best["hpo"], HPOConfig)
    assert isinstance(best["ensemble"], EnsembleConfig)


def test_unknown_preset_lists_known() -> None:
    with pytest.raises(ConfigError, match=r"unknown preset.*balanced.*best.*fast"):
        presets.resolve_preset("speedrun", _ALL_NONE)


def test_out_of_surface_key_rejected() -> None:
    """ADR-0074 §1: honesty/domain parameters are not presettable — typos do not pass."""
    for key in ("task", "significance", "finalize", "random_state", "tracker"):
        with pytest.raises(ConfigError, match="outside the presettable surface"):
            presets.resolve_preset({key: "x"}, _ALL_NONE)


def test_non_str_non_mapping_rejected() -> None:
    with pytest.raises(ConfigError, match="preset must be"):
        presets.resolve_preset(42, _ALL_NONE)


def test_explicit_value_wins_over_preset() -> None:
    """FR-DLV-2: the preset fills ONLY parameters left as None."""
    current = {**_ALL_NONE, "cv": 4}
    eff, block = presets.resolve_preset("fast", current)
    assert eff["cv"] == 4
    assert block == {"name": "fast", "applied": []}


def test_custom_mapping_preset_applies() -> None:
    """FR-DLV-1 (Mapping form) + escape hatch: 'best without HPO' is a profile copy."""
    custom = {key: value for key, value in presets.PRESETS["best"].items() if key != "hpo"}
    eff, block = presets.resolve_preset(custom, _ALL_NONE)
    assert eff["hpo"] is None
    assert isinstance(eff["ensemble"], EnsembleConfig)
    assert block == {"name": None, "applied": ["ensemble"]}


def test_explicit_none_value_is_a_noop_not_applied() -> None:
    """DM-D2: a None VALUE in a custom Mapping fills nothing — provenance stays truthful."""
    eff, block = presets.resolve_preset({"hpo": None, "cv": 3}, _ALL_NONE)
    assert eff["hpo"] is None and eff["cv"] == 3
    assert block == {"name": None, "applied": ["cv"]}


def test_invalid_config_value_is_config_error() -> None:
    with pytest.raises(ConfigError, match="invalid preset value for 'hpo'"):
        presets.resolve_preset({"hpo": {"bogus": 1}}, _ALL_NONE)


def test_models_list_normalized_to_tuple() -> None:
    eff, _ = presets.resolve_preset({"models": ["baseline", "linear"]}, _ALL_NONE)
    assert eff["models"] == ("baseline", "linear")


def test_new_profile_is_a_dict_entry(monkeypatch) -> None:
    """OCP (FR-DLV-1): adding a profile touches only the data registry."""
    monkeypatch.setitem(presets.PRESETS, "tiny", {"cv": 2})
    eff, block = presets.resolve_preset("tiny", _ALL_NONE)
    assert eff["cv"] == 2 and block["name"] == "tiny"


# --- facade integration (real fits) -------------------------------------------------------


def test_preset_applied_through_fit_and_reported() -> None:
    """FR-DLV-2: the effective config and the additive report block reflect the preset."""
    X, y = _data()
    model = AutoML(task="binary", models=("baseline", "linear"), random_state=0, preset="fast")
    model.fit(X, y)
    assert model.run_report_["preset"] == {"name": "fast", "applied": ["cv"]}
    assert model.run_report_["config"]["cv"]["n_splits"] == 3


def test_explicit_cv_wins_through_fit() -> None:
    X, y = _data()
    model = AutoML(
        task="binary", models=("baseline", "linear"), random_state=0, preset="fast", cv=4
    )
    model.fit(X, y)
    assert model.run_report_["preset"] == {"name": "fast", "applied": []}
    assert model.run_report_["config"]["cv"]["n_splits"] == 4


def test_no_preset_reports_none() -> None:
    X, y = _data()
    model = AutoML(task="binary", models=("baseline",), random_state=0).fit(X, y)
    assert model.run_report_["preset"] is None


def test_preset_fingerprint_equals_explicit_equivalent() -> None:
    """NFR-DLV-2: a preset is input sugar — the fingerprint carries resolved parameters."""
    X, y = _data()
    via_preset = AutoML(
        task="binary", models=("baseline", "linear"), random_state=0, preset="fast"
    ).fit(X, y)
    explicit = AutoML(task="binary", models=("baseline", "linear"), random_state=0, cv=3).fit(X, y)
    assert via_preset.run_report_["run_fingerprint"] == explicit.run_report_["run_fingerprint"]


def test_preset_survives_sklearn_clone() -> None:
    model = AutoML(preset="best")
    assert clone(model).preset == "best"
    assert AutoML(preset={"cv": 3}).get_params()["preset"] == {"cv": 3}
