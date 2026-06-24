"""The ``ModelSerializer`` port — pluggable model-body (de)serialization.

``save_artifact``/``load_artifact`` dispatch the model BODY through this port by the
manifest's ``model_type``: a new format is a new adapter plus a registry
entry, not an edit of the artifact core (OCP). The serializer owns only the body file(s);
manifest, calibrator, classes and integrity stay with the orchestrator (SRP).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from .estimator import Estimator


@dataclass(frozen=True)
class ModelFiles:
    """Basenames written by a serializer + the runtime extra needed to load them back.

    ``files[0]`` is the primary body recorded as ``model_file`` in the manifest; every name
    lands in ``checksums.files``. ``required_extra`` self-describes the runtime so
    a load without it raises ``MissingDependencyError`` before deserialization.
    """

    files: tuple[str, ...]
    required_extra: str | None = None


@runtime_checkable
class ModelSerializer(Protocol):
    """One serialization format: keyed by ``model_type``, matched via ``can_serialize``."""

    model_type: str

    def can_serialize(self, estimator: Estimator) -> bool: ...

    def save(self, estimator: Estimator, directory: Path) -> ModelFiles: ...

    def load(self, directory: Path, manifest: Mapping[str, Any]) -> Estimator: ...
