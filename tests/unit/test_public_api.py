"""M0-1/M0-5: public surface is pinned and the top-level import stays lightweight."""

from __future__ import annotations

import subprocess
import sys

import pytest

import honestml

pytestmark = pytest.mark.unit

EXPECTED_PUBLIC = {
    "__version__",
    "AutoMLError",
    "ConfigError",
    "MissingDependencyError",
    "SchemaValidationError",
    "ArtifactIntegrityError",
    "NotFittedError",
    "BudgetExhaustedError",
    "FeatureSelectionError",
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
    "Task",
    "FeatureSchema",
    "ColumnRole",
    "Dataset",
    "SelectionPolicy",
    "Candidate",
    "select_best",
    "AutoML",
    "save_artifact",
    "load_artifact",
    "FittedModel",
    "save_run_report",
    "render_report",
    "export_onnx",
}


def test_public_surface_is_pinned() -> None:
    assert set(honestml.__all__) == EXPECTED_PUBLIC


def test_version_present() -> None:
    assert honestml.__version__ == "1.0.0"


def test_feature_selection_config_exported() -> None:
    """M6b FR-FS-1: the FS public config is importable from the top level (like FEConfig)."""
    assert "FeatureSelectionConfig" in honestml.__all__
    assert honestml.FeatureSelectionConfig is not None


def test_import_does_not_pull_shap() -> None:
    """M6b NFR-FS-5: the default FS catalog is dependency-free; `import honestml` never pulls shap."""
    code = "import sys, honestml; print('shap' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def test_import_adapters_does_not_pull_shap() -> None:
    """M6d NFR-FSH-4: importing the adapters package (incl. ShapRanker class) must not pull shap either;
    the heavy library is lazy inside ShapRanker.__init__/rank, so the class import stays light."""
    code = "import sys, honestml.adapters; print('shap' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def test_top_level_import_is_lightweight() -> None:
    """`import honestml` must not pull heavy ML libraries (ADR-0001)."""
    # psutil is intentionally NOT checked here: joblib (a transitive sklearn dependency) imports it
    # when present, independently of honestml. honestml's own lazy psutil import (ADR-0039 §3) is covered
    # by test_run_budget.py::test_memory_limit_requires_psutil (needed only when a limit is set).
    code = (
        "import sys, honestml; "
        "heavy = [m for m in ('catboost','lightgbm','xgboost','optuna','mlflow','matplotlib') "
        "if m in sys.modules]; "
        "print(','.join(heavy))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "", f"heavy modules pulled: {out.stdout!r}"
