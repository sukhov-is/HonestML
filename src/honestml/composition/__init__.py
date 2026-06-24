"""Composition root: wire dependencies, expose the public facade (M2).

Lazy barrel (PEP 562, ADR-0066 §1): importing ``load_artifact`` (the standalone serving entry) does not pull
the training facade/build (registry → all adapters). Names resolve to their submodule on first access; the
public surface (``__all__``) is unchanged.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

_SUBMODULES = {
    "ARTIFACT_VERSION": ".artifact",
    "FittedModel": ".artifact",
    "load_artifact": ".artifact",
    "save_artifact": ".artifact",
    "Components": ".build",
    "build_default_components": ".build",
    "AutoML": ".facade",
    "export_onnx": ".onnx_bundle",
    "save_run_report": ".run_report",
    "render_report": ".run_report",
}

__all__ = [
    "AutoML",
    "build_default_components",
    "Components",
    "save_artifact",
    "load_artifact",
    "FittedModel",
    "ARTIFACT_VERSION",
    "export_onnx",
    "save_run_report",
    "render_report",
]


def __getattr__(name: str) -> object:
    sub = _SUBMODULES.get(name)
    if sub is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(sub, __name__), name)
    globals()[name] = value  # cache: subsequent access skips __getattr__
    return value


def __dir__() -> list[str]:
    return sorted(__all__)


if (
    TYPE_CHECKING
):  # static type-checkers / IDEs see the real names without triggering the eager import
    from .artifact import ARTIFACT_VERSION, FittedModel, load_artifact, save_artifact
    from .build import Components, build_default_components
    from .facade import AutoML
    from .onnx_bundle import export_onnx
    from .run_report import render_report, save_run_report
