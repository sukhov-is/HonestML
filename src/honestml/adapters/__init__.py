"""Adapters: implement domain ports over concrete libraries.

Lazy barrel (PEP 562, ADR-0066 §1): public names resolve to their submodule on first attribute access, so
``from honestml.adapters import Reader`` does not eager-import the training stack (rankers→sklearn cluster/
ensemble, tuner, ensembler search, significance, splitters, budget, cache, feature-selectors). This keeps the
standalone inference cone slim (NFR-SRV-1) while the public surface (``__all__``) is unchanged.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

_SUBMODULES = {
    "IsotonicCalibrator": ".calibration",
    "SigmoidCalibrator": ".calibration",
    "resolve_calibrator": ".calibration",
    "CACHE_VERSION": ".candidate_cache",
    "JoblibCandidateCache": ".candidate_cache",
    "BlendedEstimator": ".ensembling",
    "CaruanaEnsembler": ".ensembling",
    "WeightedEnsembler": ".ensembling",
    "BaselineClassifier": ".estimators",
    "BaselineRegressor": ".estimators",
    "LinearClassifier": ".estimators",
    "LinearRegressor": ".estimators",
    "ImportanceRanker": ".feature_rankers",
    "NullImportanceRanker": ".feature_rankers",
    "RandomProbeRanker": ".feature_rankers",
    "ShapRanker": ".feature_rankers",
    "make_ranker_fit_predict": ".feature_rankers",
    "SequentialSelector": ".feature_selectors",
    "load_table": ".loader",
    "Accuracy": ".metrics",
    "Brier": ".metrics",
    "Ece": ".metrics",
    "LogLoss": ".metrics",
    "Mae": ".metrics",
    "PrAuc": ".metrics",
    "Rmse": ".metrics",
    "RocAuc": ".metrics",
    "resolve_metric": ".metrics",
    "PolarsDataset": ".polars_dataset",
    "Reader": ".reader",
    "TypingDecision": ".reader",
    "RunBudget": ".run_budget",
    "BootstrapSignificanceTest": ".significance",
    "GroupKFoldSplitter": ".splitters",
    "HoldoutSplitter": ".splitters",
    "KFoldSplitter": ".splitters",
    "PeriodTimeSeriesSplitter": ".splitters",
    "StratifiedGroupKFoldSplitter": ".splitters",
    "StratifiedKFoldSplitter": ".splitters",
    "TimeSeriesSplitter": ".splitters",
    "outer_holdout_carve": ".splitters",
    "MlflowTracker": ".tracking",
    "OptunaTuner": ".tuning",
}

__all__ = [
    "PolarsDataset",
    "Reader",
    "TypingDecision",
    "load_table",
    "BaselineClassifier",
    "BaselineRegressor",
    "LinearClassifier",
    "LinearRegressor",
    "ImportanceRanker",
    "RandomProbeRanker",
    "NullImportanceRanker",
    "ShapRanker",
    "SequentialSelector",
    "make_ranker_fit_predict",
    "RocAuc",
    "PrAuc",
    "Accuracy",
    "LogLoss",
    "Brier",
    "Ece",
    "Rmse",
    "Mae",
    "resolve_metric",
    "SigmoidCalibrator",
    "IsotonicCalibrator",
    "resolve_calibrator",
    "BootstrapSignificanceTest",
    "MlflowTracker",
    "RunBudget",
    "JoblibCandidateCache",
    "CACHE_VERSION",
    "HoldoutSplitter",
    "StratifiedKFoldSplitter",
    "KFoldSplitter",
    "GroupKFoldSplitter",
    "StratifiedGroupKFoldSplitter",
    "TimeSeriesSplitter",
    "PeriodTimeSeriesSplitter",
    "outer_holdout_carve",
    "OptunaTuner",
    "CaruanaEnsembler",
    "WeightedEnsembler",
    "BlendedEstimator",
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
    from .calibration import IsotonicCalibrator, SigmoidCalibrator, resolve_calibrator
    from .candidate_cache import CACHE_VERSION, JoblibCandidateCache
    from .ensembling import BlendedEstimator, CaruanaEnsembler, WeightedEnsembler
    from .estimators import (
        BaselineClassifier,
        BaselineRegressor,
        LinearClassifier,
        LinearRegressor,
    )
    from .feature_rankers import (
        ImportanceRanker,
        NullImportanceRanker,
        RandomProbeRanker,
        ShapRanker,
        make_ranker_fit_predict,
    )
    from .feature_selectors import SequentialSelector
    from .loader import load_table
    from .metrics import Accuracy, Brier, Ece, LogLoss, Mae, PrAuc, Rmse, RocAuc, resolve_metric
    from .polars_dataset import PolarsDataset
    from .reader import Reader, TypingDecision
    from .run_budget import RunBudget
    from .significance import BootstrapSignificanceTest
    from .splitters import (
        GroupKFoldSplitter,
        HoldoutSplitter,
        KFoldSplitter,
        PeriodTimeSeriesSplitter,
        StratifiedGroupKFoldSplitter,
        StratifiedKFoldSplitter,
        TimeSeriesSplitter,
        outer_holdout_carve,
    )
    from .tracking import MlflowTracker
    from .tuning import OptunaTuner
