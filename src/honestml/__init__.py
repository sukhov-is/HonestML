"""AutoML — general tabular AutoML library.

The top-level import pulls only the pure domain ``core`` (numpy/pydantic; no frame/ML library, no I/O). The
facade and the model artifact resolve lazily on first attribute access (PEP 562), so
``import honestml`` followed by ``load_artifact(...).predict(...)`` never executes the training stack
(optuna/shap/sklearn cluster·ensemble/tuner/ensembler/significance). The public surface (``__all__``) is
unchanged.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from .core import (
    ArtifactIntegrityError,
    AutoMLError,
    BudgetConfig,
    BudgetExhaustedError,
    Candidate,
    ColumnRole,
    ConfigError,
    CVConfig,
    Dataset,
    EnsembleConfig,
    FeatureSchema,
    FeatureSelectionConfig,
    FeatureSelectionError,
    FEConfig,
    HPOConfig,
    MissingDependencyError,
    NotFittedError,
    RunConfig,
    RunContext,
    SchemaValidationError,
    SelectionPolicy,
    Task,
    TrackerConfig,
    get_logger,
    select_best,
)

__version__ = "1.0.0"

# lazy: facade + artifact (resolved via the composition barrel, itself lazy) — keeps `import honestml` slim
_SUBMODULES = {
    "AutoML": ".composition",
    "save_artifact": ".composition",
    "load_artifact": ".composition",
    "FittedModel": ".composition",
    "save_run_report": ".composition",
    "render_report": ".composition",
    "export_onnx": ".composition",
}

__all__ = [
    "__version__",
    # exceptions
    "AutoMLError",
    "ConfigError",
    "MissingDependencyError",
    "SchemaValidationError",
    "ArtifactIntegrityError",
    "NotFittedError",
    "BudgetExhaustedError",
    "FeatureSelectionError",
    # config / context / logging
    "RunConfig",
    "CVConfig",
    "BudgetConfig",
    "FEConfig",
    "FeatureSelectionConfig",
    "HPOConfig",
    "EnsembleConfig",
    "TrackerConfig",
    "RunContext",
    "get_logger",
    # domain data core
    "Task",
    "FeatureSchema",
    "ColumnRole",
    "Dataset",
    # selection (ports live in honestml.core / honestml.core.ports)
    "SelectionPolicy",
    "Candidate",
    "select_best",
    # facade + artifact (M2) — lazy
    "AutoML",
    "save_artifact",
    "load_artifact",
    "FittedModel",
    # run report (M5b JSON + M9-2 md/html rendering) — lazy; matplotlib only inside html
    "save_run_report",
    "render_report",
    # onnx export bundle (M8b-2) — lazy; onnx tooling is imported only when called
    "export_onnx",
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
):  # static type-checkers / IDEs see the facade/artifact names without the eager import
    from .composition import (
        AutoML,
        FittedModel,
        export_onnx,
        load_artifact,
        render_report,
        save_artifact,
        save_run_report,
    )
