"""M3a (ADR-0019): ``Capabilities.probabilistic`` is an additive static tag."""

from __future__ import annotations

import pytest

from honestml.core import Capabilities, ModelSpec

pytestmark = pytest.mark.unit


def test_capabilities_probabilistic_default_false() -> None:
    caps = Capabilities(tasks=("regression",))
    assert caps.probabilistic is False


def test_capabilities_probabilistic_explicit() -> None:
    caps = Capabilities(tasks=("binary",), probabilistic=True)
    assert caps.probabilistic is True


def test_legacy_capabilities_without_probabilistic_loads() -> None:
    """A pre-M3a serialized Capabilities (no ``probabilistic``) deserializes (NFR-3)."""
    legacy = '{"tasks": ["binary"], "handles_cat": false, "handles_missing": false}'
    caps = Capabilities.model_validate_json(legacy)
    assert caps.probabilistic is False
    assert caps.tasks == ("binary",)


def test_model_spec_carries_probabilistic() -> None:
    spec = ModelSpec(
        name="linear", capabilities=Capabilities(tasks=("binary",), probabilistic=True)
    )
    assert spec.capabilities.probabilistic is True
    assert spec.supports("binary")
