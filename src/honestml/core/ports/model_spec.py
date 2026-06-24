"""``ModelSpec`` and ``Capabilities``.

Capabilities are declared (auto-sklearn ``get_properties`` style) so the registry
can auto-skip components that do not fit the task. ``handles_missing`` states
whether raw NaN reach the model or must be imputed beforehand — identically
on train and inference. The ``search_space`` is a backend-neutral declaration the
tuner consumes; here it is just carried data.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from ..task import TaskKind


class Capabilities(BaseModel):
    """What a model can do — used by the registry to skip unsuitable components.

    ``probabilistic`` is a *static* tag: it lets the registry filter
    ``predict_proba``-needing metrics without materializing the (possibly heavy)
    adapter. Additive with a ``False`` default — older ``Capabilities`` load unchanged.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    tasks: tuple[TaskKind, ...]
    probabilistic: bool = False
    handles_cat: bool = False
    handles_missing: bool = False
    # the model early-stops on a held-out es tail (ADR-0080); composition carves the tail only when a
    # supports_early_stopping model is in the zoo. Additive False default.
    supports_early_stopping: bool = False
    needs_scaling: bool = False
    gpu: bool = False
    max_rows: int | None = None
    max_cols: int | None = None


class ModelSpec(BaseModel):
    """A registered model: name, capabilities and a declarative search space."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    capabilities: Capabilities
    search_space: dict[str, Any] = {}

    def supports(self, kind: TaskKind) -> bool:
        return kind in self.capabilities.tasks
