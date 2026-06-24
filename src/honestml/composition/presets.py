"""Built-in facade presets (ADR-0074) — data, not branching (P14).

A preset is a partial, plain-dict (JSON-primitive, YAML-ready) set of facade
parameters covering EXACTLY the None-default surface. It fills only the parameters
the user left as ``None`` (None == "not set" is the exact semantics here — every
surface parameter defaults to None), so an explicit value always wins.
Honesty-controlling parameters (``significance``/``finalize``/``run_mode``) are NOT
presettable by construction (ADR-0074 §1): a preset cannot silently downgrade the
honest-selection contract. The explicit opt-out escape hatch is a custom Mapping
(a copy of a profile without the key).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ValidationError

from honestml.core import (
    BudgetConfig,
    ConfigError,
    CVConfig,
    EnsembleConfig,
    FeatureSelectionConfig,
    FEConfig,
    HPOConfig,
    get_logger,
)

logger = get_logger("composition.presets")

PRESET_SURFACE = (
    "cv",
    "models",
    "budget",
    "hpo",
    "ensemble",
    "feature_selection",
    "feature_engineering",
)

# a new profile is a new entry here (+ CHANGELOG: preset contents are public contract)
PRESETS: dict[str, dict[str, Any]] = {
    # quick prototype: fewer folds; the significance band stays on (not presettable)
    "fast": {"cv": 3},
    # defaults + a blend shipped only when significantly better (ADR-0063 gate)
    "balanced": {"ensemble": {}},
    # maximum quality; HPO needs the optuna extra (honest MissingDependencyError)
    "best": {"hpo": {}, "ensemble": {}},
}

# config-valent keys: a Mapping value is normalized via the pydantic model ({} -> defaults);
# the facade's scalar forms (cv=3, budget=600.0) and ready instances pass through untouched
_CONFIG_FORMS: dict[str, type[BaseModel]] = {
    "cv": CVConfig,
    "budget": BudgetConfig,
    "hpo": HPOConfig,
    "ensemble": EnsembleConfig,
    "feature_selection": FeatureSelectionConfig,
    "feature_engineering": FEConfig,
}


def resolve_preset(
    preset: str | Mapping[str, Any] | None, current: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Effective surface values + the additive run-report ``preset`` block (ADR-0074 §2/§3).

    ``current`` maps surface parameter -> constructor-verbatim value. Returns
    ``(eff, block)``: ``eff`` covers the full surface; ``block`` is
    ``{"name": str | None, "applied": [...]}`` (``name=None`` for a custom Mapping)
    or ``None`` when no preset was requested.
    """
    eff = dict(current)
    if preset is None:
        return eff, None
    if isinstance(preset, str):
        entries = PRESETS.get(preset)
        if entries is None:
            raise ConfigError(f"unknown preset {preset!r}; known presets: {sorted(PRESETS)}")
        name: str | None = preset
    elif isinstance(preset, Mapping):
        name, entries = None, dict(preset)  # snapshot: a shared Mapping is read once per fit
    else:
        raise ConfigError(
            f"preset must be a preset name, a Mapping or None, got {type(preset).__name__}"
        )
    unknown = sorted(set(entries) - set(PRESET_SURFACE))
    if unknown:
        raise ConfigError(
            f"preset keys {unknown} are outside the presettable surface "
            f"{list(PRESET_SURFACE)} (honesty/domain/infra parameters "
            "are not presettable)"
        )
    # an explicit None VALUE in a custom Mapping is a no-op (the escape hatch is omitting
    # the key) — it must not show up in `applied`, the provenance stays truthful (DM-D2)
    applied = [
        key for key in PRESET_SURFACE if entries.get(key) is not None and current[key] is None
    ]
    for key in applied:
        eff[key] = _normalize(key, entries[key])
    logger.info("preset %s applied: %s", name or "<custom>", applied)
    return eff, {"name": name, "applied": applied}


def _normalize(key: str, value: Any) -> Any:
    """Plain-dict preset value -> the facade form (ADR-0074 §1); scalars pass through."""
    if key == "models" and isinstance(value, list):
        return tuple(value)
    config_cls = _CONFIG_FORMS.get(key)
    if config_cls is not None and isinstance(value, Mapping):
        try:
            return config_cls.model_validate(dict(value))
        except ValidationError as exc:
            raise ConfigError(f"invalid preset value for {key!r}: {exc}") from exc
    return value
