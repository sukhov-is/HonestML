"""Domain ports (Protocols) — extension points implemented by adapters."""

from __future__ import annotations

from .budget import Budget
from .cache import CandidateCache
from .calibration import Calibrator, CalibratorFactory
from .ensembler import Ensembler, EnsembleRecipe
from .estimator import (
    Estimator,
    ProbabilisticEstimator,
    SupportsEarlyStopping,
    SupportsFeatureImportance,
    SupportsNativeCategorical,
    SupportsNativeModel,
    SupportsShap,
)
from .feature_ranker import FeatureRanker
from .feature_subset_selector import FeatureSubsetSelector
from .metric import Metric, MetricNeeds
from .model_serializer import ModelFiles, ModelSerializer
from .model_spec import Capabilities, ModelSpec
from .significance import NoSignificanceTest, SignificanceTest
from .splitter import (
    CVSplitter,
    Fold,
    GroupAwareSplitter,
    ReportsSplitMeta,
    TimeOrderedSplitter,
    validate_fold,
)
from .tracker import ExperimentTracker
from .tuner import (
    CategoricalParam,
    FloatParam,
    IntParam,
    ParamSpec,
    SearchSpace,
    TuneOutcome,
    Tuner,
    parse_search_space,
)

__all__ = [
    "Metric",
    "MetricNeeds",
    "Calibrator",
    "CalibratorFactory",
    "CVSplitter",
    "TimeOrderedSplitter",
    "GroupAwareSplitter",
    "ReportsSplitMeta",
    "Fold",
    "validate_fold",
    "Estimator",
    "ProbabilisticEstimator",
    "SupportsEarlyStopping",
    "SupportsFeatureImportance",
    "SupportsNativeCategorical",
    "SupportsNativeModel",
    "SupportsShap",
    "FeatureRanker",
    "FeatureSubsetSelector",
    "ModelSerializer",
    "ModelFiles",
    "ModelSpec",
    "Capabilities",
    "Budget",
    "CandidateCache",
    "SignificanceTest",
    "NoSignificanceTest",
    "ExperimentTracker",
    "Ensembler",
    "EnsembleRecipe",
    "Tuner",
    "TuneOutcome",
    "ParamSpec",
    "SearchSpace",
    "parse_search_space",
    "IntParam",
    "FloatParam",
    "CategoricalParam",
]
