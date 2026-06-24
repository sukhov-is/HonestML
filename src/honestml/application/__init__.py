"""Use-cases: orchestrate the domain through ports. Never name a concrete adapter.

Lazy barrel (PEP 562, ADR-0066 §1): public names resolve to their submodule on first attribute access. The
use-case layer is already adapter-free (the ``usecases-independent-of-adapters`` contract), so importing
``design_matrix`` pulls only pure submodules; laziness keeps the inference cone minimal and the public
surface (``__all__``) unchanged.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

_SUBMODULES = {
    "calibrate_winner": ".calibration",
    "crossfit_calibrate": ".calibration",
    "viable_blocks": ".calibration",
    "EnsembleOutcome": ".ensemble",
    "ensemble_selection": ".ensemble",
    "refit_members": ".ensemble",
    "crossfit_encode": ".feature_encoding",
    "crossfit_encode_expanding": ".feature_encoding",
    "apply_cutoff": ".feature_selection",
    "select_features": ".feature_selection",
    "FINGERPRINT_VERSION": ".run_report",
    "RUN_MANIFEST_VERSION": ".run_report",
    "build_run_report": ".run_report",
    "collect_lib_versions": ".run_report",
    "compute_run_fingerprint": ".run_report",
    "dataset_signature": ".run_report",
    "BudgetReport": ".slice",
    "EstimatorFactory": ".slice",
    "FailedCandidate": ".slice",
    "FeatureSelectionBundle": ".slice",
    "FeatureSelectionReport": ".slice",
    "LeaderboardEntry": ".slice",
    "SliceResult": ".slice",
    "align_proba": ".projection",
    "design_matrix": ".slice",
    "project_for_metric": ".projection",
    "refit_best": ".slice",
    "resolve_positive": ".slice",
    "run_slice": ".slice",
    "MakeFactory": ".tuning",
    "tune_estimators": ".tuning",
}

__all__ = [
    "run_slice",
    "refit_best",
    "tune_estimators",
    "MakeFactory",
    "ensemble_selection",
    "refit_members",
    "EnsembleOutcome",
    "build_run_report",
    "RUN_MANIFEST_VERSION",
    "FINGERPRINT_VERSION",
    "compute_run_fingerprint",
    "dataset_signature",
    "collect_lib_versions",
    "crossfit_calibrate",
    "crossfit_encode",
    "crossfit_encode_expanding",
    "select_features",
    "apply_cutoff",
    "calibrate_winner",
    "viable_blocks",
    "project_for_metric",
    "align_proba",
    "design_matrix",
    "resolve_positive",
    "SliceResult",
    "BudgetReport",
    "FeatureSelectionReport",
    "FailedCandidate",
    "FeatureSelectionBundle",
    "LeaderboardEntry",
    "EstimatorFactory",
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
    from .calibration import calibrate_winner, crossfit_calibrate, viable_blocks
    from .ensemble import EnsembleOutcome, ensemble_selection, refit_members
    from .feature_encoding import crossfit_encode, crossfit_encode_expanding
    from .feature_selection import apply_cutoff, select_features
    from .projection import align_proba, project_for_metric
    from .run_report import (
        FINGERPRINT_VERSION,
        RUN_MANIFEST_VERSION,
        build_run_report,
        collect_lib_versions,
        compute_run_fingerprint,
        dataset_signature,
    )
    from .slice import (
        BudgetReport,
        EstimatorFactory,
        FailedCandidate,
        FeatureSelectionBundle,
        FeatureSelectionReport,
        LeaderboardEntry,
        SliceResult,
        design_matrix,
        refit_best,
        resolve_positive,
        run_slice,
    )
    from .tuning import MakeFactory, tune_estimators
