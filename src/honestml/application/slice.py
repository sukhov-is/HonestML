"""The ``run_slice`` use-case: cross-validated OOF leaderboard (ADR-0010).

Orchestrates the binary vertical slice over the domain ports only (no adapters,
``import-linter`` ``usecases-independent-of-adapters``): split → per-model CV →
out-of-fold predictions → ``Metric`` → ``equivalence_band`` (the honest significance band,
ADR-0026). The domain stays a Humble Object — all I/O-free, synchronously testable on fake
ports (NFR-3). The final model is refit on the full training data by :func:`refit_best`.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from copy import deepcopy
from dataclasses import dataclass, field, replace

import numpy as np
from pydantic import BaseModel, ConfigDict

from honestml.core import (
    Budget,
    BudgetExhaustedError,
    CalibratorFactory,
    Candidate,
    CandidateCache,
    ConfigError,
    Dataset,
    Estimator,
    FeatureRanker,
    FeatureSubsetSelector,
    FitFailedError,
    Fold,
    GroupAwareSplitter,
    Metric,
    NoSignificanceTest,
    ProbabilisticEstimator,
    ReportsSplitMeta,
    RunConfig,
    RunContext,
    SchemaValidationError,
    SelectionPolicy,
    SignificanceTest,
    SupportsEarlyStopping,
    SupportsNativeCategorical,
    Task,
    TimeOrderedSplitter,
    TuneOutcome,
    equivalence_band,
    get_logger,
    rank,
    resolve_positive,
    validate_fold,
)
from honestml.core.config import (
    FeatureSelectionConfig,
    FEConfig,
    SearchConfig,
    SelectionMode,
    WeightingMode,
)
from honestml.core.ports.cache import StageCache
from honestml.core.ports.estimator import (
    SupportsFitContext,
    SupportsIterationBudget,
    SupportsIterationPlan,
    SupportsThreadLimit,
)
from honestml.core.ports.splitter import CVSplitter
from honestml.core.schema import categorical_positions, native_routing, te_output_name
from honestml.core.task import TaskKind

from .calibration import crossfit_calibrate, viable_blocks
from .feature_compare import compare_features, no_selection_gate
from .feature_encoding import crossfit_encode, crossfit_encode_expanding
from .feature_selection import (
    BoundedFeatureRanker,
    _degenerate_counts,
    bounded_fit_predict,
    select_features,
    structure_labels,
)
from .oof_scorer import FitPredict, OOFSubsetCache
from .projection import _PROBA_NEEDS, align_proba, project_for_metric
from .search import probe_models, profile_fs_cost, scout_feature_recipes

EstimatorFactory = Callable[[], Estimator]


class LeaderboardEntry(BaseModel):
    """One ranked leaderboard row — public surface (facade ``leaderboard_``, artifact).

    ``protected_namespaces=()`` is intentional: it exposes the SemVer-stable
    ``model_id`` field name (ADR-0010 §8) without pydantic's ``model_`` warning.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, protected_namespaces=())

    model_id: str
    score: float
    metric: str
    n_features: int
    train_time: float
    rank: int


@dataclass(frozen=True)
class FailedCandidate:
    """A candidate excluded from the leaderboard after its model raised (ADR-0022)."""

    id: str
    reason: str


@dataclass(frozen=True)
class BudgetReport:
    """Budget-degradation observability (ADR-0032 §4 / ADR-0039 §5).

    ``skipped`` are candidates not started once the budget was exhausted, ``exhausted`` whether the
    run degraded, ``exhausted_by`` the axis that hit the limit ("time"/"trials"/"memory"/None, read
    from the Budget port's ``exhausted_reason``). Defaults are a within-budget run.
    """

    skipped: tuple[str, ...] = ()
    exhausted: bool = False
    exhausted_by: str | None = None


@dataclass(frozen=True)
class FeatureSelectionReport:
    """Feature-selection observability — present on the result only when selection produced a subset.

    ``selected_features`` is the kept names in ``schema.features`` order (the facade attaches it to the
    schema for refit/artifact/holdout). The remaining fields are the M6c+ compare record (None on the
    M6b single-ranker path): the no-selection gate verdict (ADR-0063), the winning strategy and
    per-strategy arbitration record (ADR-0048/0049), the nested/significance winner rule + band
    (ADR-0052/0053), the structure-aware null diagnostics (ADR-0050/0055) and the per-fold re-selection
    stability (ADR-0054).
    """

    selected_features: tuple[str, ...]
    selection_gate: str | None = None
    selected_strategy: str | None = None
    per_strategy: tuple[tuple[str, int, float], ...] | None = None
    winner_rule: str | None = None
    band_members: tuple[str, ...] | None = None
    per_strategy_std: tuple[tuple[str, float], ...] | None = None
    null_block_stats: dict[str, float | str] | None = None
    arbitration_effective: str | None = None
    fold_subset_jaccard: float | None = None
    per_strategy_mean_features: tuple[tuple[str, float], ...] | None = None
    # in-sequential band of the winning wrapper selector (ADR-0086 §1), distinct from the
    # strategy-arbitration band (winner_rule/band_members). None unless the winner is sequential w/ band.
    seq_band: dict[str, object] | None = None
    # cascade refinement stage sizes of the winner (ADR-0100); None when refinement did not run.
    refine: dict[str, object] | None = None
    subset_cache: dict[str, int] | None = None


_PREPARED_STAGE = "prepared_features_v1"


@dataclass(frozen=True)
class _PreparedFeatures:
    schema_features: tuple[str, ...]
    model_ids: tuple[str, ...]
    report: FeatureSelectionReport | None
    search_report: dict[str, object] | None
    wide_candidate: Candidate | None


def search_completed(search_report: dict[str, object] | None) -> bool:
    """Whether search finished without transient failures, skipped work or budget fallback.

    Deterministic structural fallbacks, including unavailable inner validation, are completed
    decisions. Interrupted probes and failed candidate/control computations remain retryable.
    """
    if search_report is None:
        return True
    if any(
        search_report.get(key)
        for key in (
            "model_failures",
            "confirmation_failures",
            "recipe_failures",
            "skipped_models",
            "skipped_recipes",
            "stop_reason",
        )
    ):
        return False
    return search_report.get("model_reason") not in (
        "incomplete_probe_budget",
        "incomplete_confirmation_budget",
        "confirmation_failed",
    ) and search_report.get("recipe_reason") not in (
        "probe_budget",
        "incomplete_recipe_budget",
        "control_failed",
        "prefilter_failed",
    )


@dataclass
class SliceResult:
    """Outcome of :func:`run_slice`: leaderboard, winner id, OOF candidates, failures.

    The honesty-band fields (ADR-0026 §6) carry the equivalence band's outcome: who is
    statistically indistinguishable from the absolute anchor (``band_member_ids``), whether the
    band is anchor-sensitive (``band_unstable``), its size (``band_width``), and whether the winner
    was chosen by the Occam tie-break rather than being the anchor (``winner_by_tiebreak``). They
    default to the lone-anchor band so a run without a real test is unchanged. Budget degradation and
    feature-selection observability are grouped in nested reports (``budget``/``feature_selection``).
    """

    leaderboard: list[LeaderboardEntry]
    best_model_id: str
    candidates: list[Candidate]
    failed: list[FailedCandidate] = field(default_factory=list)
    band_member_ids: tuple[str, ...] = ()
    band_unstable: bool = False
    band_width: int = 1
    winner_by_tiebreak: bool = False
    # refinement-based selection observability (ADR-0031 §6): the mode actually used after any
    # fallback, and whether the leaderboard score is the raw or the cross-fitted calibrated loss.
    selection_mode: SelectionMode = "raw"
    score_space: str = "raw_oof"
    # honest-regime holdout (ADR-0029 §3): the winner's unbiased score on the once-touched outer
    # holdout. Set by composition (the carve+score is orchestrated there); a plain dev run leaves None.
    holdout_score: float | None = None
    # budget degradation report (ADR-0032 §4); defaults to a within-budget run.
    budget: BudgetReport = field(default_factory=BudgetReport)
    # stage-cache observability (ADR-0036 §3 / ADR-0037 §3): candidate ids reused from cache (skip-on-
    # hit) vs freshly computed. Empty when cache is off; the primary hit/miss channel for the run-report.
    reused: tuple[str, ...] = ()
    computed: tuple[str, ...] = ()
    # CV fold id per OOF row (-1 where uncovered), for the calibration cross-fit gate (ADR-0030 §3)
    # and refinement blocks (ADR-0031 §3); built only when proba is captured.
    oof_fold_index: np.ndarray | None = None
    # feature-selection report (ADR-0044/0045/...): None when selection did not produce a subset.
    feature_selection: FeatureSelectionReport | None = None
    # native-categorical routing verdict (ADR-0095): per routed CATEGORICAL column,
    # "native"/"high_cardinality". Populated by run_slice ONLY when the cardinality gate demoted >=1
    # column (None = nothing demoted / gate off), so the opt-out path's report is bit-identical;
    # build_run_report surfaces it. Demotion is never silent (FR-5).
    native_routing: dict[str, str] | None = None
    # period CV split diagnostics (ADR-0096 §4): {period, n_periods, n_folds, n_dropped_empty} for a
    # timeseries_period run (None otherwise), surfaced in the run-report `cv` block for a truthful manifest.
    cv_split: dict[str, object] | None = None
    # HPO outcomes when tuning ran inside the slice (ADR-0102, post-FS objective); the facade folds the
    # tuned factories into its estimator mapping for refit/ensemble and builds the hpo report from this.
    hpo: dict[str, TuneOutcome] | None = None
    search: dict[str, object] | None = None


class _CandidateFailed(Exception):
    """Internal signal: one candidate's model raised; isolate it (ADR-0022 §1)."""

    def __init__(self, name: str, reason: object) -> None:
        self.name = name
        self.reason = str(reason)
        super().__init__(f"candidate {name!r} failed: {self.reason}")


def project_by_name(feature_names: Sequence[str], selected: Sequence[str]) -> list[int]:
    """Column indices selecting ``selected`` in ``feature_names`` (schema) order (ADR-0102 §1, FR-FS-7).

    The single post-FS projection rule, shared by :func:`design_matrix` and ``tune_estimators``: keep the
    columns whose name is selected, ordered by their position in ``feature_names`` (NOT the subset's tuple
    order), so the HPO objective, the CV matrix and the shipped refit stay column-aligned regardless of how
    the subset was stored. Leakage-critical — both call sites must project identically, so the rule lives once.
    """
    selected_set = set(selected)
    return [i for i, f in enumerate(feature_names) if f in selected_set]


def design_matrix(dataset: Dataset) -> np.ndarray:
    """Model input: numeric block ⊕ categorical codes, in ``schema.features`` order.

    Also the single ADR-0013 §F9 ≥1-feature boundary guard (reused by the facade,
    ``refit_best`` and the artifact). When the schema carries a feature-selection subset
    (``selected_features``, ADR-0045 §2), the full matrix is projected to it by name — the one
    choke-point that keeps train==inference without touching the predict path; a selected feature
    absent from the matrix fails loud (FR-FS-4).
    """
    numeric = dataset.to_numpy()
    codes = dataset.categorical_codes()
    if numeric.shape[1] == 0 and codes.shape[1] == 0:
        raise SchemaValidationError("dataset has no model features")
    full: np.ndarray = np.hstack([numeric, codes.astype(np.float64, copy=False)])
    selected = dataset.schema.selected_features
    if selected is None:
        return full
    features = dataset.schema.features
    missing = set(selected) - set(features)
    if missing:
        raise SchemaValidationError(
            f"selected feature {sorted(missing)!r} absent from the design matrix"
        )
    return full[:, project_by_name(features, selected)]


def _wants_oof(significance_test: SignificanceTest | None) -> bool:
    """True if a real significance test will consume the OOF predictions (M4)."""
    return significance_test is not None and not isinstance(significance_test, NoSignificanceTest)


def _fold_index(n: int, folds: Sequence[Fold]) -> np.ndarray:
    """Per-row CV fold id, ``-1`` where no fold covers the row.

    The single source for the time-series band block index and the cross-fit OOF index (TE/calibration/
    refinement), so the fold-to-row map cannot diverge between leakage-sensitive consumers (ADR-0041 §1).
    """
    idx = np.full(n, -1, dtype=np.int64)
    for fold_id, fold in enumerate(folds):
        idx[fold.test_idx] = fold_id
    return idx


def _score_weighted(
    metric: Metric,
    y: np.ndarray,
    pred: np.ndarray,
    mask: np.ndarray,
    block_index: np.ndarray | None,
    sample_weight: np.ndarray | None,
    weighting: WeightingMode,
) -> float:
    """Leaderboard score over the valid OOF rows: pooled (one metric) or macro-by-period (ADR-0098 §2).

    ``pooled`` is the unchanged single-``metric.score`` path (NFR-5 byte-identical). ``period`` scores the
    metric per block (CV fold/period) and averages over blocks with a FINITE metric — a block whose metric
    is undefined (single-class roc_auc) is dropped (R-6); all blocks invalid -> ``nan`` (as an empty mask).
    ``pred`` is the metric-ready OOF, ``mask`` marks its valid rows.
    """
    if weighting == "pooled" or block_index is None:
        sw = sample_weight[mask] if sample_weight is not None else None
        return float(metric.score(y[mask], pred[mask], sw))
    scores = _period_block_scores(metric, y, pred, mask, block_index, sample_weight)
    return float(np.mean(scores)) if scores else float("nan")


def _period_block_scores(
    metric: Metric,
    y: np.ndarray,
    pred: np.ndarray,
    mask: np.ndarray,
    block_index: np.ndarray,
    sample_weight: np.ndarray | None,
) -> list[float]:
    """Finite per-block metric scores (uncovered id ``-1`` and undefined-metric blocks dropped, R-6).

    The valid-block set is fixed by ``y``+metric (e.g. single-class roc_auc), not by the candidate, so the
    per-candidate scoring set equals the common set across candidates (F7); the band enforces the same at
    the pairwise comparison level via its common mask.
    """
    scores: list[float] = []
    for b in np.unique(block_index):
        if b < 0:  # uncovered rows carry id -1 and never form a real block
            continue
        bm = mask & (block_index == b)
        if not bm.any():
            continue
        sw = sample_weight[bm] if sample_weight is not None else None
        # a block where the metric is undefined (e.g. single-class roc_auc) is dropped (R-6): newer
        # sklearn returns nan (caught by the isfinite guard), older raises ValueError -> both handled.
        try:
            s = float(metric.score(y[bm], pred[bm], sw))
        except ValueError:
            continue
        if np.isfinite(s):
            scores.append(s)
    return scores


def _augment_oof_te(
    x_full: np.ndarray,
    dataset: Dataset,
    y: np.ndarray,
    positive: object,
    oof_fold_index: np.ndarray,
    smoothing: float,
    feature_names: list[str],
    *,
    time_ordered: bool = False,
) -> np.ndarray:
    """Overwrite the full-train TE columns of ``x_full`` with out-of-fold values (ADR-0041 §1).

    The Reader materialized ``{col}_te`` as the full-train smoothed mean; for an honest leaderboard the
    evaluation matrix must instead carry the cross-fitted OOF encoding (a row never sees its own fold's
    target). Computed once and shared by every candidate (ADR-0040 §2); returns a copy so the dataset's
    full-train columns (used by ``refit_best``/inference) are untouched. Source codes are read back from
    ``x_full`` (the categorical block follows the numeric block), so ``design_matrix``'s encode is reused
    — no second materialization (NFR-FE-5). ``reserve_from`` = per-column ``null_code`` keeps null/unknown
    rows at ``global_mean``, matching the full-train spec (ADR-0041 §2). ``time_ordered`` routes to the
    expanding-window encoder (each fold from strictly earlier folds, no look-ahead, ADR-0082) for a
    time-series split; otherwise the plain leave-one-fold-out cross-fit (ADR-0041 §1).
    """
    schema = dataset.schema
    spec = schema.target_encoding
    if spec is None or not spec.encodings:
        return x_full
    te_cols = list(spec.encodings)
    n_numeric = len(schema.numeric)
    categorical = schema.categorical
    src_idx = [n_numeric + categorical.index(c) for c in te_cols]  # code column in x_full
    codes_te = np.ascontiguousarray(x_full[:, src_idx].astype(np.int64))
    reserve_from = np.array([schema.categories[c].null_code for c in te_cols], dtype=np.int64)
    y_te = (y == positive).astype(np.float64)
    encode = crossfit_encode_expanding if time_ordered else crossfit_encode
    oof = encode(codes_te, y_te, oof_fold_index, smoothing=smoothing, reserve_from=reserve_from)
    out = x_full.copy()
    for j, col in enumerate(te_cols):
        out[:, feature_names.index(te_output_name(col))] = oof[:, j]
    return out


@dataclass(frozen=True)
class TuningBundle:
    """The HPO injectables passed to :func:`run_slice` as one unit (ADR-0102).

    ``tune(dataset, selected_features)`` runs the inner-CV search AFTER the FS block, so the
    objective sees the post-FS width (supersedes ADR-0062 §2a); ``tuned_factories`` maps the
    outcomes to estimator-factory updates (replace, or ``{name}__tuned`` append) exactly like the
    facade's legacy write-back. Both are closures composed in the facade — injection, not import:
    ``tuning.py`` imports from this module, so calling it here directly would be a cycle.
    """

    tune: Callable[[Dataset, tuple[str, ...] | None], dict[str, TuneOutcome]]
    tuned_factories: Callable[[dict[str, TuneOutcome]], dict[str, EstimatorFactory]]
    select_models: Callable[[tuple[str, ...]], None] | None = None
    profile_cost: (
        Callable[[Dataset, tuple[str, ...], Callable[[], bool]], dict[str, dict[str, object]]]
        | None
    ) = None
    keep_baseline: bool = False


@dataclass(frozen=True)
class FeatureSelectionBundle:
    """The feature-selection injectables passed to :func:`run_slice` as one unit (ADR-0044/0046).

    Collapses the six loose ``feature_*`` parameters whose all-or-nothing invariant used to ride a
    runtime assert: composition wires either the single-ranker spine (``config`` + ``ranker``) or the
    M6c compare (``config`` + ``strategies`` + ``carve`` + ``fit_predict`` [+ ``arbitration_splitter``]).
    """

    config: FeatureSelectionConfig
    ranker: FeatureRanker | None = None
    strategies: Sequence[tuple[str, FeatureRanker | FeatureSubsetSelector]] | None = None
    carve: Callable[[Dataset, float, int], tuple[np.ndarray, np.ndarray]] | None = None
    fit_predict: (
        Callable[
            [np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, int],
            tuple[np.ndarray | None, np.ndarray, np.ndarray | None],
        ]
        | None
    ) = None
    arbitration_splitter: CVSplitter | None = None
    prefilter: FeatureRanker | None = None
    probe_fit_predict: FitPredict | None = None
    probe_strategies: Sequence[tuple[str, FeatureRanker | FeatureSubsetSelector]] | None = None
    execution_resources: Mapping[str, int] | None = None


def run_slice(
    dataset: Dataset,
    task: Task,
    *,
    estimators: Mapping[str, EstimatorFactory],
    splitter: CVSplitter,
    metric: Metric,
    policy: SelectionPolicy,
    significance_test: SignificanceTest | None = None,
    calibrator_factory: CalibratorFactory | None = None,
    selection: SelectionMode = "raw",
    refinement_min_oof: int = 2000,
    weighting: WeightingMode = "pooled",
    capture_proba: bool = False,
    fe: FEConfig | None = None,
    features: FeatureSelectionBundle | None = None,
    tuning: TuningBundle | None = None,
    search: SearchConfig | None = None,
    completion_refit_rows: tuple[int, ...] = (),
    budget: Budget | None = None,
    cache: CandidateCache | None = None,
    stage_cache: StageCache | None = None,
    ctx: RunContext | None = None,
) -> SliceResult:
    """Run CV selection; search limits actual fits recorded through the shared context.

    Custom FS callbacks must implement ``SupportsFitContext`` and record each underlying fit
    to participate in actual-fit budgets. Opaque callbacks have no inferred backend work count.
    """
    if search is not None and ctx is None:
        ctx = RunContext(run_config=RunConfig(seed=0, search=search))
    if search is not None and features is not None and ctx is not None:
        resources = [
            features.ranker,
            features.prefilter,
            features.fit_predict,
            features.probe_fit_predict,
            *(strategy for _, strategy in features.strategies or ()),
            *(strategy for _, strategy in features.probe_strategies or ()),
        ]
        for resource in resources:
            if isinstance(resource, SupportsFitContext):
                resource.set_run_context(ctx)
    logger = ctx.logger if ctx is not None else get_logger()
    schema = dataset.schema
    if not estimators:
        raise ConfigError("run_slice requires at least one estimator")
    # unpack the FS bundle into the leakage-critical injectables the body wires (ADR-0044); None == off
    feature_selection = features.config if features is not None else None
    execution_resources = (features.execution_resources or {}) if features is not None else {}
    feature_ranker = features.ranker if features is not None else None
    feature_strategies = features.strategies if features is not None else None
    feature_carve = features.carve if features is not None else None
    feature_fit_predict = features.fit_predict if features is not None else None
    feature_arbitration_splitter = features.arbitration_splitter if features is not None else None

    y = dataset.target()
    if y is None:
        raise SchemaValidationError("run_slice requires a target column")
    classes = np.unique(y)
    if task.is_classification and classes.size < 2:
        raise SchemaValidationError("classification requires at least 2 classes in y")
    # binary keeps the positive-column path; multiclass reindexes to `classes` (ADR-0021)
    positive = resolve_positive(task, classes) if task.kind == "binary" else None

    x_full = design_matrix(dataset)  # built once; also the §F9 no-features guard
    feature_names = list(schema.features)
    folds = list(splitter.split(dataset))
    if not folds:
        raise SchemaValidationError("splitter produced no folds")
    # a group-aware splitter (ADR-0023) guarantees no group spans fit/test — assert it here;
    # a shuffling scheme over a group column is warned at composition, not failed here.
    groups = dataset.groups() if isinstance(splitter, GroupAwareSplitter) else None
    time_ordered = isinstance(splitter, TimeOrderedSplitter)
    times = dataset.time() if time_ordered else None
    for fold in folds:
        validate_fold(fold, groups=groups, time_ordered=time_ordered, times=times)
    # period-CV split diagnostics for the truthful manifest (ADR-0096 §4); None unless the splitter
    # reports them (only PeriodTimeSeriesSplitter does, after split has run above).
    cv_split = splitter.split_meta() if isinstance(splitter, ReportsSplitMeta) else None

    # fold-block bootstrap for time-series significance (ADR-0026 §2): i.i.d. row resampling
    # understates variance under autocorrelation, so the band resamples whole CV test folds.
    block_index = _fold_index(y.shape[0], folds) if time_ordered else None

    sample_weight = dataset.sample_weight()
    n_features = len(feature_names)
    # proba is produced only when the metric needs it; an active band additionally captures the
    # metric-ready class/value OOF for non-proba metrics (ADR-0026 §3), not extra proba.
    need_proba = metric.needs in _PROBA_NEEDS
    # period weighting scores the leaderboard per block too, so it needs the metric-ready OOF even when
    # no significance test would otherwise capture it (ADR-0098 §2).
    capture_oof = _wants_oof(significance_test) or weighting == "period"

    # fold id per OOF row (-1 where uncovered): the cross-fit blocks shared by the OOF target-encoding
    # augmentation (ADR-0041 §1), the calibration gate (ADR-0030 §3) and refinement (ADR-0031 §3).
    # SEPARATE from the band's TS bootstrap block_index above. Built BEFORE the candidate loop because
    # the TE augmentation must rewrite x_full once, ahead of any candidate (ADR-0040 §2).
    te_on = fe is not None and fe.target_encoding and schema.target_encoding is not None
    oof_fold_index: np.ndarray | None = None
    if capture_proba or selection == "refinement" or te_on:
        oof_fold_index = _fold_index(y.shape[0], folds)
    # OOF target encoding for EVALUATION only: replace the full-train TE columns of x_full with
    # out-of-fold values, so the leaderboard score carries no target bleed (ADR-0041 §1/§3). refit and
    # inference keep the full-train TE spec on the boundary (Reader), not this augmentation. Under a
    # time-ordered split the expanding-window encoder is used (each fold from strictly earlier folds, no
    # look-ahead, ADR-0082); an IID split uses the plain leave-one-fold-out cross-fit (ADR-0041 §1).
    if te_on:
        assert fe is not None and oof_fold_index is not None and positive is not None
        x_full = _augment_oof_te(
            x_full,
            dataset,
            y,
            positive,
            oof_fold_index,
            fe.te_smoothing,
            feature_names,
            time_ordered=time_ordered,
        )

    cached_preparation = stage_cache.get_stage(_PREPARED_STAGE) if stage_cache is not None else None
    prepared = (
        cached_preparation
        if isinstance(cached_preparation, _PreparedFeatures)
        and cached_preparation.schema_features == tuple(feature_names)
        and all(name in estimators for name in cached_preparation.model_ids)
        else None
    )
    schema_features = tuple(feature_names)
    if prepared is not None:
        estimators = {name: estimators[name] for name in prepared.model_ids}
        if search is not None and tuning is not None and tuning.select_models is not None:
            tuning.select_models(prepared.model_ids)
        feature_selection = None
        feature_ranker = None
        feature_strategies = None

    probe_active = False
    probe_fit_count = 0
    fs_fit_count = 0

    def before_fit(stage: str) -> None:
        nonlocal probe_fit_count, fs_fit_count
        memory_left = budget.memory_left() if budget is not None else None
        if memory_left is not None and memory_left <= 0:
            raise BudgetExhaustedError("memory", completed=0, skipped=1, failed=0)
        if search is not None and probe_active and stage in ("fs", "scouting"):
            if (
                budget is not None and budget.exhausted
            ) or time.perf_counter() - probe_started >= probe_seconds:
                raise BudgetExhaustedError(
                    budget.exhausted_reason
                    if budget is not None and budget.exhausted_reason
                    else "time",
                    completed=probe_fit_count,
                    skipped=1,
                    failed=0,
                )
            if probe_fit_count >= search.max_probe_fits:
                raise BudgetExhaustedError("trials", completed=probe_fit_count, skipped=1, failed=0)
            probe_fit_count += 1
        if search is not None and stage == "fs" and not probe_active:
            if fs_fit_count >= search.max_fs_fits:
                raise BudgetExhaustedError("trials", completed=fs_fit_count, skipped=1, failed=0)
            fs_fit_count += 1
        if stage in ("fs", "hpo", "cv", "scouting") and budget is not None and budget.exhausted:
            raise BudgetExhaustedError(
                budget.exhausted_reason or "time", completed=1, skipped=1, failed=0
            )

    if search is not None and ctx is not None:
        ctx.before_fit = before_fit

    search_report = deepcopy(prepared.search_report) if prepared is not None else None
    wide_candidate = prepared.wide_candidate if prepared is not None else None
    if prepared is not None and search_report is not None:
        search_report["preparation_reused"] = True
    if search is not None and prepared is None:
        original_estimators = estimators
        routing_before_fs = native_routing(schema, task.native_cat_max_unique)
        native_names = [name for name, route in routing_before_fs.items() if route == "native"]
        encoded_names = (
            {te_output_name(name) for name in schema.target_encoding.encodings}
            if schema.target_encoding is not None
            else set()
        )
        probe_columns = tuple(
            i for i, name in enumerate(feature_names) if name not in encoded_names
        )

        full_iterations: dict[str, int | None] = {}
        full_training_rows: dict[str, int] = {}

        def evaluate_probe(
            name: str,
            probe_folds: Sequence[Fold],
            iterations: int,
            indices: tuple[int, ...] | None = None,
        ) -> Candidate:
            columns = (
                list(probe_columns)
                if indices is None
                else [i for i in indices if i in probe_columns]
            )
            names = [feature_names[i] for i in columns]

            def factory() -> Estimator:
                est = original_estimators[name]()
                full_training_rows[name] = sum(
                    len(f.fit_idx)
                    + (0 if isinstance(est, SupportsEarlyStopping) else len(f.es_idx))
                    for f in folds
                )
                full_iterations[name] = (
                    est.iteration_limit(early_stopping=any(f.es_idx.size for f in folds))
                    if isinstance(est, SupportsIterationPlan)
                    else None
                )
                if isinstance(est, SupportsIterationBudget):
                    cap = (
                        min(
                            iterations,
                            est.iteration_limit(
                                early_stopping=any(f.es_idx.size for f in probe_folds)
                            ),
                        )
                        if isinstance(est, SupportsIterationPlan)
                        else iterations
                    )
                    est.set_refit_iterations(cap)
                if isinstance(est, SupportsThreadLimit):
                    est.set_threads(search.threads)
                return est

            try:
                return _run_candidate(
                    name,
                    factory,
                    x_full=x_full if len(columns) == n_features else x_full[:, columns],
                    y=y,
                    feature_names=names,
                    categorical_indices=categorical_positions(names, native_names),
                    kind=task.kind,
                    positive=positive,
                    global_classes=classes,
                    metric=metric,
                    folds=list(probe_folds),
                    sample_weight=sample_weight,
                    n_features=len(columns),
                    need_proba=need_proba,
                    capture_oof=True,
                    capture_proba=capture_proba,
                    block_index=block_index,
                    weighting=weighting,
                    logger=logger,
                    ctx=ctx,
                    stage="scouting",
                )
            except _CandidateFailed as exc:
                if isinstance(exc.__cause__, BudgetExhaustedError):
                    raise exc.__cause__
                raise FitFailedError([(name, exc.reason)]) from exc

        probe_started = time.perf_counter()
        probe_seconds = (
            max(0.0, budget.time_left()) * search.probe_fraction
            if budget is not None
            else float("inf")
        )

        def can_probe() -> bool:
            return (
                (budget is None or not budget.exhausted)
                and probe_fit_count < search.max_probe_fits
                and time.perf_counter() - probe_started < probe_seconds
            )

        def profile_completion(names: tuple[str, ...]) -> dict[str, dict[str, object]]:
            disabled: dict[str, object] = {
                "status": "disabled",
                "estimated_s": 0.0,
                "planned_fit_count": 0,
            }
            fs_cost = disabled
            if feature_selection is not None:
                fixed_name = (feature_selection.compare or (feature_selection.strategy,))[0]
                strategy = feature_ranker
                if feature_strategies is not None and len(feature_strategies) == 1:
                    item = feature_strategies[0][1]
                    strategy = item if isinstance(item, FeatureRanker) else None
                effective_fs = feature_selection.model_copy(
                    update={
                        "strategy": fixed_name,
                        "n_runs": execution_resources.get("n_runs", feature_selection.n_runs),
                        "n_probes": execution_resources.get("n_probes", feature_selection.n_probes),
                    }
                )
                fs_groups = structure_labels(
                    groups,
                    times,
                    effective_fs.null_block_size,
                    mode=effective_fs.null_block_mode,
                    window=effective_fs.null_block_window,
                )
                fs_cost = profile_fs_cost(
                    strategy,
                    x_full,
                    y,
                    folds,
                    categorical=np.array([name in schema.categorical for name in feature_names]),
                    feature_names=feature_names,
                    config=effective_fs,
                    search=search,
                    fit_predict=feature_fit_predict,
                    metric=metric,
                    task=task,
                    seed=effective_fs.random_state if effective_fs.random_state is not None else 0,
                    sample_weight=sample_weight,
                    groups=fs_groups,
                    significance_test=significance_test,
                    policy=policy,
                    ctx=ctx,
                    can_start=can_probe,
                )
            hpo_costs = (
                tuning.profile_cost(dataset, names, can_probe)
                if tuning is not None and tuning.profile_cost is not None
                else {}
            )
            result: dict[str, dict[str, object]] = {}
            for name in names:
                hpo_cost = hpo_costs.get(
                    name,
                    disabled
                    if tuning is None
                    else {
                        "status": "unavailable",
                        "estimated_s": None,
                        "reason": "missing_hpo_cost_profiler",
                    },
                )
                has_fs = feature_selection is not None
                has_hpo = hpo_cost["status"] != "disabled"
                additional_cv = int(has_fs or has_hpo) + int(
                    has_hpo and tuning is not None and tuning.keep_baseline
                )
                result[name] = {
                    "fs": fs_cost,
                    "hpo": hpo_cost,
                    "additional_cv_count": additional_cv,
                    "assumptions": ["full_width_post_search_cv", "confirmation_based_refit_cost"],
                    "hpo_timeout_is_a_limit": True,
                }
            return result

        probe_active = True
        try:
            with ctx.timed_stage("run", "model_scouting") if ctx is not None else nullcontext():
                probe = probe_models(
                    tuple(estimators),
                    folds,
                    y=y,
                    groups=groups,
                    task=task,
                    metric=metric,
                    config=search,
                    seed=ctx.run_config.seed if ctx is not None else 0,
                    evaluate=evaluate_probe,
                    can_start=can_probe,
                    significance_test=significance_test,
                    sample_weight=sample_weight,
                    block_index=block_index,
                    times=times,
                    full_iterations=full_iterations,
                    full_training_rows=full_training_rows,
                    completion_refit_rows=completion_refit_rows,
                    profile_completion=profile_completion
                    if feature_selection is not None or tuning is not None
                    else None,
                )
        finally:
            probe_active = False
        probe_seconds -= time.perf_counter() - probe_started
        estimators = {probe.winner: original_estimators[probe.winner]}
        if tuning is not None and tuning.select_models is not None:
            tuning.select_models((probe.winner,))
        search_report = {
            "selected_model": probe.winner,
            "model_reason": probe.reason,
            "model_probes": [
                {"model": c.id, "score": c.score, "rows": sum(len(f.test_idx) for f in probe.folds)}
                for c in probe.candidates
            ],
            "model_failures": list(probe.failures),
            "confirmation_probes": [
                {
                    "model": c.id,
                    "score": c.score,
                    "elapsed_s": c.train_time,
                    "iterations": c.refit_iterations,
                }
                for c in probe.confirmations
            ],
            "confirmation_failures": list(probe.confirmation_failures),
            "probe_diagnostics": probe.diagnostics,
            "probe_issues": list(probe.issues),
            "rank_changed": probe.rank_changed,
            "estimated_cv_refit_s": probe.cost_estimates,
            "completion_cost_forecast": probe.completion_costs,
            "completion_cost_basis": "conditional_fs_recipe_and_central_hpo_parameters_full_trials",
            "completion_cost_scope": "remaining_profiled_procedures_cv_and_refit; excludes total_run_wall",
            "initial_estimated_cv_refit_s": probe.initial_cost_estimates,
            "cost_model": probe.cost_model,
            "completion_refit_rows": list(completion_refit_rows),
            "cost_estimate_basis": "probe_elapsed_scaled_by_actual_cv_training_and_planned_refit_rows; native_ceiling_extrapolated",
            "cost_estimate_excludes": ["fs", "hpo", "feature_width_changes", "total_run_wall"],
            "model_margin": search.model_margin,
            "threads": search.threads,
            "fs_execution": {
                "rows_per_fit": search.max_rows,
                "fit_limit": search.max_fs_fits,
                "requested": feature_selection is not None,
                "n_runs": execution_resources.get("n_runs"),
                "n_probes": execution_resources.get("n_probes"),
                "ranker_iterations": execution_resources.get("ranker_iterations"),
                "proxy_resource_source": "composition"
                if execution_resources
                else "native_component",
                "evaluation": "post_search_dev",
                "approximate": feature_selection is not None,
            },
            "skipped_models": list(probe.skipped),
            "probe_folds": len(probe.folds),
            "evaluation": "post_search_dev",
            "quality_acceptance": "pending",
            "probe_target_encoding": "raw_features_only",
        }
        # a complete wide DEV estimate is reserved before optional FS/HPO.
        if budget is not None and budget.exhausted:
            raise BudgetExhaustedError(
                budget.exhausted_reason or "time", completed=0, skipped=1, failed=0
            )
        with ctx.timed_stage("run", "wide_control") if ctx is not None else nullcontext():
            try:
                wide_candidate = _run_candidate(
                    probe.winner,
                    estimators[probe.winner],
                    x_full=x_full,
                    y=y,
                    feature_names=feature_names,
                    categorical_indices=categorical_positions(feature_names, native_names),
                    kind=task.kind,
                    positive=positive,
                    global_classes=classes,
                    metric=metric,
                    folds=folds,
                    sample_weight=sample_weight,
                    n_features=n_features,
                    need_proba=need_proba,
                    capture_oof=capture_oof,
                    capture_proba=capture_proba,
                    block_index=block_index,
                    weighting=weighting,
                    logger=logger,
                    ctx=ctx,
                    stage="wide_control",
                )
            except _CandidateFailed as exc:
                raise FitFailedError([(probe.winner, exc.reason)]) from exc
        probe_started = time.perf_counter()

        if (
            feature_selection is not None
            and feature_strategies is not None
            and len(feature_strategies) > 1
        ):
            assert (
                features is not None
                and features.prefilter is not None
                and feature_fit_predict is not None
            )
            cat_mask = np.array([name in schema.categorical for name in feature_names])
            probe_active = True
            try:
                with ctx.timed_stage("run", "fs_scouting") if ctx is not None else nullcontext():
                    fs_probe = scout_feature_recipes(
                        features.probe_strategies or feature_strategies,
                        x_full[:, list(probe_columns)]
                        if len(probe_columns) != n_features
                        else x_full,
                        y,
                        probe.folds,
                        categorical=cat_mask[list(probe_columns)],
                        fs_config=feature_selection,
                        search_config=search,
                        metric=metric,
                        task=task,
                        fit_predict=features.probe_fit_predict or feature_fit_predict,
                        prefilter=features.prefilter,
                        seed=ctx.run_config.seed if ctx is not None else 0,
                        evaluate=lambda indices, probe_folds: evaluate_probe(
                            probe.winner,
                            probe_folds,
                            search.model_iterations,
                            tuple(probe_columns[i] for i in indices),
                        ),
                        sample_weight=sample_weight,
                        groups=groups if groups is not None else times,
                        can_start=can_probe,
                        ctx=ctx,
                    )
            finally:
                probe_active = False
            search_report.update(
                {
                    "selected_recipe": fs_probe.winner,
                    "recipe_reason": fs_probe.reason,
                    "recipe_probes": [
                        {"recipe": c.id, "score": c.score, "features": c.n_features}
                        for c in fs_probe.candidates
                    ],
                    "recipe_failures": list(fs_probe.failures),
                    "skipped_recipes": list(fs_probe.skipped),
                }
            )
            if fs_probe.winner is None:
                feature_selection = None
                feature_strategies = None
                feature_ranker = None
            else:
                feature_strategies = tuple(
                    (name, strategy)
                    for name, strategy in feature_strategies
                    if name == fs_probe.winner
                )
                feature_selection = feature_selection.model_copy(
                    update={"compare": None, "strategy": fs_probe.winner}
                )
        search_report["probe_fit_count"] = probe_fit_count if ctx is not None else None
        search_report["probe_fit_limit"] = search.max_probe_fits
        if budget is not None and budget.exhausted:
            feature_selection = None
            feature_strategies = None
            feature_ranker = None
            tuning = None
            search_report["stop_reason"] = "completion_reserve"
            search_report["use_original_factory"] = True

    if (
        search is not None
        and feature_selection is not None
        and task.is_classification
        and np.unique(y).size > search.max_rows
    ):
        feature_selection = None
        feature_strategies = None
        feature_ranker = None
        if search_report is not None:
            search_report["fs_execution_reason"] = "infeasible_training_classes"

    if search is not None and feature_selection is not None:
        if feature_strategies is not None:
            feature_strategies = tuple(
                (
                    name,
                    BoundedFeatureRanker(strategy, task=task, max_rows=search.max_rows)
                    if isinstance(strategy, FeatureRanker)
                    else strategy,
                )
                for name, strategy in feature_strategies
            )
        if feature_ranker is not None:
            feature_ranker = BoundedFeatureRanker(
                feature_ranker, task=task, max_rows=search.max_rows
            )
        if feature_fit_predict is not None:
            feature_fit_predict = bounded_fit_predict(
                feature_fit_predict, task=task, max_rows=search.max_rows
            )
        if feature_selection.refine:
            feature_selection = feature_selection.model_copy(
                update={
                    "refine_max_features": min(
                        feature_selection.refine_max_features or search.max_features,
                        search.max_features,
                    ),
                    "refine_drop_frac": max(0.5, feature_selection.refine_drop_frac),
                }
            )

    # feature selection (ADR-0044): keep ONE subset shared by all candidates and refit, computed here
    # over the final FE-augmented set; the eval matrix/feature_names are then projected and the subset
    # travels to refit/inference via the schema. M6c compare/sequential (ADR-0046/0048) goes through
    # compare_features (carve + per-strategy select + arbitrate); the M6b single-ranker path is unchanged.
    selected_features: tuple[str, ...] | None = None
    fs_idx: tuple[int, ...] | None = None
    selection_gate: str | None = None
    selected_strategy: str | None = None
    per_strategy: tuple[tuple[str, int, float], ...] | None = None
    winner_rule: str | None = None
    band_members: tuple[str, ...] | None = None
    per_strategy_std: tuple[tuple[str, float], ...] | None = None
    arbitration_effective: str | None = None
    fold_subset_jaccard: float | None = None
    per_strategy_mean_features: tuple[tuple[str, float], ...] | None = None
    seq_band: dict[str, object] | None = None
    fs_refine: dict[str, object] | None = None
    # per-row structure label for structure-aware null_importance (M6d, ADR-0050): reuse the group/time
    # arrays already derived above; None (i.i.d. scheme) keeps the M6c uniform permutation.
    feature_groups = (
        structure_labels(
            groups,
            times,
            feature_selection.null_block_size,
            mode=feature_selection.null_block_mode,
            window=feature_selection.null_block_window,
        )
        if feature_selection is not None
        else None
    )
    null_block_stats: dict[str, float | str] | None = None
    if feature_groups is not None:
        block_ids, counts = np.unique(feature_groups, return_counts=True)
        degenerate = _degenerate_counts(
            feature_groups, y
        )  # vectorized O(n), shared with per-fold (ADR-0059 §2)
        null_block_stats = {
            "n_blocks": float(block_ids.size),
            "mean_block_size": float(counts.mean()),
            "degenerate_blocks": float(degenerate),
        }
        # M6e (ADR-0055 §4): surface the binning mode/parameter; under time_window the degenerate WARNING
        # below becomes more load-bearing (a narrow window fragments blocks on irregular series).
        if feature_selection is not None and groups is None and times is not None:
            null_block_stats["block_mode"] = feature_selection.null_block_mode
            if (
                feature_selection.null_block_mode == "time_window"
                and feature_selection.null_block_window
            ):
                null_block_stats["block_window"] = float(feature_selection.null_block_window)
        if degenerate > block_ids.size // 2:
            logger.warning(
                "structure-aware null: %d/%d blocks have a constant target -> weak null signal "
                "(common for group classification); consider a coarser scheme",
                degenerate,
                block_ids.size,
            )
    memory_headroom = budget.memory_left() if budget is not None else None
    cache_bytes = 32 * 1024 * 1024
    if memory_headroom is not None:
        cache_bytes = min(cache_bytes, max(0, int(memory_headroom * 1024 * 1024)))
    subset_cache = OOFSubsetCache(max_bytes=cache_bytes)
    try:
        with ctx.timed_stage("run", "fs") if ctx is not None else nullcontext():
            if feature_selection is not None and feature_strategies is not None:
                # composition wires the three M6c components as a bundle (build.py); narrow for the call
                assert feature_carve is not None and feature_fit_predict is not None
                categorical_mask = np.zeros(n_features, dtype=bool)
                categorical_mask[len(schema.numeric) :] = True
                fs_seed = (
                    feature_selection.random_state
                    if feature_selection.random_state is not None
                    else 0
                )
                outcome = compare_features(
                    dataset,
                    x_full,
                    y,
                    task=task,
                    metric=metric,
                    strategies=feature_strategies,
                    subset_cache=subset_cache,
                    config=feature_selection,
                    splitter=splitter,
                    carve=feature_carve,
                    fit_predict=feature_fit_predict,
                    categorical=categorical_mask,
                    feature_names=feature_names,
                    sample_weight=sample_weight,
                    random_state=fs_seed,
                    groups=feature_groups,
                    arbitration_splitter=feature_arbitration_splitter,
                    significance_test=significance_test,
                    policy=policy,
                    run_context=ctx,
                )
                fs_idx = outcome.winner_idx
                selected_strategy = outcome.winner
                per_strategy = outcome.per_strategy
                winner_rule = outcome.winner_rule
                band_members = outcome.band_members or None
                per_strategy_std = outcome.per_strategy_std or None
                arbitration_effective = outcome.arbitration_effective
                fold_subset_jaccard = outcome.fold_subset_jaccard
                per_strategy_mean_features = outcome.per_strategy_mean_features or None
                seq_band = outcome.seq_band
                fs_refine = outcome.refine
                # M6f (ADR-0059 §1a): merge the winner's per-fold block-fragmentation aggregate into the full-DEV
                # null_block_stats (built above) so the honesty metric reflects the smaller per-fold trains too.
                if outcome.per_fold_block_stats is not None:
                    null_block_stats = {**(null_block_stats or {}), **outcome.per_fold_block_stats}
            elif feature_selection is not None and feature_ranker is not None:
                categorical_mask = np.zeros(n_features, dtype=bool)
                categorical_mask[len(schema.numeric) :] = True
                fs_idx = tuple(
                    select_features(
                        x_full,
                        y,
                        folds,
                        ranker=feature_ranker,
                        categorical=categorical_mask,
                        config=feature_selection,
                        sample_weight=sample_weight,
                        groups=feature_groups,
                        ctx=ctx,
                    )
                )

            # no-selection honest gate (finding #10): an FS subset must not be SIGNIFICANTLY worse than the full
            # feature set, mirroring the ensemble's choose_better gate (ADR-0063). Covers BOTH the M6c compare and
            # the M6b single-ranker paths; on a "no_selection_better" verdict we ship all features (never silent).
            if fs_idx is not None:
                if (
                    search is None
                    and feature_selection is not None
                    and feature_fit_predict is not None
                    and significance_test is not None
                    and len(fs_idx) < n_features
                ):
                    fs_gate_seed = (
                        feature_selection.random_state
                        if feature_selection.random_state is not None
                        else 0
                    )
                    keep, selection_gate = no_selection_gate(
                        x_full,
                        y,
                        fs_idx,
                        folds,
                        fit_predict=feature_fit_predict,
                        metric=metric,
                        task=task,
                        sample_weight=sample_weight,
                        significance_test=significance_test,
                        policy=policy,
                        random_state=fs_gate_seed,
                        block_index=block_index,
                        refine_tol=feature_selection.refine_tol,
                        subset_cache=subset_cache,
                        run_context=ctx,
                    )
                    if not keep:
                        logger.warning(
                            "feature selection (%d of %d features) is not significantly better than "
                            "no-selection and risks regressing; shipping all features (finding #10)",
                            len(fs_idx),
                            n_features,
                        )
                        fs_idx = tuple(range(n_features))
                selected_features = tuple(feature_names[i] for i in fs_idx)
                x_full = x_full[:, list(fs_idx)]
                feature_names = [feature_names[i] for i in fs_idx]
                n_features = len(feature_names)

    except BudgetExhaustedError:
        if wide_candidate is None:
            raise
        fs_idx = None
        selected_features = None
        tuning = None
        if search_report is not None:
            search_report["stop_reason"] = (
                "fs_fit_limit"
                if search is not None and fs_fit_count >= search.max_fs_fits
                else "completion_reserve"
            )
            search_report["use_original_factory"] = True

    if search_report is not None:
        search_report["fs_fit_count"] = fs_fit_count if ctx is not None else None
    feature_report = (
        FeatureSelectionReport(
            selected_features=selected_features,
            selection_gate=selection_gate,
            selected_strategy=selected_strategy,
            per_strategy=per_strategy,
            winner_rule=winner_rule,
            band_members=band_members,
            per_strategy_std=per_strategy_std,
            null_block_stats=null_block_stats,
            arbitration_effective=arbitration_effective,
            fold_subset_jaccard=fold_subset_jaccard,
            per_strategy_mean_features=per_strategy_mean_features,
            seq_band=seq_band,
            refine=fs_refine,
            subset_cache={
                "hits": subset_cache.hits,
                "misses": subset_cache.misses,
                "retained_bytes": subset_cache.retained_bytes,
                "peak_bytes": subset_cache.peak_bytes,
            },
        )
        if selected_features is not None
        else None
    )
    if prepared is not None:
        feature_report = prepared.report
        if feature_report is not None:
            selected_features = feature_report.selected_features
            fs_idx = tuple(project_by_name(feature_names, selected_features))
            x_full = x_full[:, list(fs_idx)]
            feature_names = [feature_names[i] for i in fs_idx]
            n_features = len(feature_names)
    elif (
        stage_cache is not None
        and (search is not None or features is not None)
        and search_completed(search_report)
        and (budget is None or not budget.exhausted)
    ):
        stage_cache.put_stage(
            _PREPARED_STAGE,
            _PreparedFeatures(
                schema_features,
                tuple(estimators),
                feature_report,
                deepcopy(search_report),
                wide_candidate,
            ),
        )

    # HPO stage (ADR-0102, supersedes ADR-0062 §2a): tune INSIDE the slice, after the FS projection,
    # so the inner objective sees the post-FS width; FS off passes None -> the legacy full-DEV
    # objective, statement-for-statement. Tuned factories replace (or `__tuned`-augment) the
    # candidates before the loop below, exactly like the facade's legacy write-back.
    hpo_outcomes: dict[str, TuneOutcome] | None = None
    if tuning is not None and not (search is not None and budget is not None and budget.exhausted):
        if search_report is not None:
            search_report["hpo_on_subset"] = selected_features is not None
        hpo_outcomes = tuning.tune(dataset, selected_features)
        estimators = {**estimators, **tuning.tuned_factories(hpo_outcomes)}

    if not search_completed(search_report) or (
        hpo_outcomes is not None and any(not outcome.completed for outcome in hpo_outcomes.values())
    ):
        cache = None

    # native-categorical routing (ADR-0087/0088/0092, FR-1/FR-2/FR-3): the cardinality-GATED verdict over
    # the frozen schema, computed ONCE here; positions of the natively-routed CATEGORICAL columns in the
    # FINAL (post-FS) feature_names. The same gate (native_routing) backs FeatureSchema.categorical_indices
    # (refit_best) and tune_estimators, so the routing indices cannot drift across CV/refit/HPO (R-3/R-6).
    # High-card columns are excluded here and ride the existing ordinal-codes path.
    cap = task.native_cat_max_unique
    routing = native_routing(schema, cap)
    categorical_indices = categorical_positions(
        feature_names, [c for c, r in routing.items() if r == "native"]
    )
    # routing verdict over the categoricals that actually reach the model (post-FS), surfaced in the
    # run-report; a demotion is never silent (ADR-0095, FR-5). None when the gate demoted nothing.
    routed = set(feature_names)
    verdict: dict[str, str] = {c: r for c, r in routing.items() if c in routed}
    demoted = [c for c, r in verdict.items() if r != "native"]
    native_routing_verdict = verdict if demoted else None
    if demoted:
        logger.warning(
            "native categorical gate demoted %d high-cardinality column(s) to ordinal codes: %s "
            "(native_cat_max_unique=%s)",
            len(demoted),
            demoted,
            cap,
        )

    candidates: list[Candidate] = []
    failed: list[FailedCandidate] = []
    skipped: list[str] = []
    reused: list[str] = []
    computed: list[str] = []
    budget_exhausted = (
        search_report is not None and search_report.get("stop_reason") == "completion_reserve"
    )
    # the exhausted axis, captured at the MOMENT of exhaustion (ADR-0039 §5): truthful and robust to a
    # later non-monotonic RSS read; first capture wins. None on a within-budget run.
    exhausted_by: str | None = (
        budget.exhausted_reason if budget_exhausted and budget is not None else None
    )
    for name, factory in estimators.items():
        if (
            wide_candidate is not None
            and selected_features is None
            and not any(outcome.successful_trials for outcome in (hpo_outcomes or {}).values())
        ):
            candidates.append(wide_candidate)
            computed.append(wide_candidate.id)
            break
        # cooperative per-candidate gate (ADR-0032 §1): once exhausted, skip the rest (continue, not
        # break, so skipped_by_budget is complete); a failed candidate does NOT consume a trial.
        if budget is not None and budget.exhausted:
            skipped.append(name)
            budget_exhausted = True
            if exhausted_by is None:
                exhausted_by = budget.exhausted_reason
            continue
        # stage-cache skip-on-hit (ADR-0036 §3): a cached candidate (same fingerprint + id) is reused
        # without retraining; its restored OOF feeds band/calibration/refinement identically (FR-RC-2).
        cand = cache.get(name) if cache is not None else None
        if cand is not None:
            reused.append(name)
        else:
            try:
                with ctx.timed_stage(name, "cv") if ctx is not None else nullcontext():
                    cand = _run_candidate(
                        name,
                        factory,
                        x_full=x_full,
                        y=y,
                        feature_names=feature_names,
                        categorical_indices=categorical_indices,
                        kind=task.kind,
                        positive=positive,
                        global_classes=classes,
                        metric=metric,
                        folds=folds,
                        sample_weight=sample_weight,
                        n_features=n_features,
                        need_proba=need_proba,
                        capture_oof=capture_oof,
                        capture_proba=capture_proba,
                        block_index=block_index,
                        weighting=weighting,
                        logger=logger,
                        ctx=ctx,
                    )
            except BudgetExhaustedError:
                skipped.append(name)
                budget_exhausted = True
                exhausted_by = budget.exhausted_reason if budget is not None else "time"
                continue
            except _CandidateFailed as exc:
                logger.warning("candidate %r failed and was skipped: %s", name, exc.reason)
                failed.append(FailedCandidate(id=name, reason=exc.reason))
                continue
            # durable on completion (atomic) -> resume after a crash recomputes only the remainder
            # (FR-RC-3); a failed candidate is NOT cached (it carries no OOF; retry stays honest).
            if cache is not None:
                cache.put(name, cand)
            computed.append(name)
        candidates.append(cand)
        if budget is not None:
            # one consume = one completed trial; a cache-hit consumes alike (trials/none determinism,
            # ADR-0037 §2) — under time/none consume is a no-op so a cached train_time is not billed.
            budget.consume(cand.train_time)

    if not candidates and wide_candidate is not None:
        candidates.append(wide_candidate)
        computed.append(wide_candidate.id)
        selected_features = None
        if search_report is not None:
            search_report["stop_reason"] = "completion_reserve"
            search_report["use_original_factory"] = True
    if wide_candidate is not None and selected_features is not None and candidates:
        selected_candidate = rank(candidates, policy)[0]
        control = replace(wide_candidate, id="__wide_control")
        selected = replace(selected_candidate, id="__selected")
        with ctx.timed_stage("run", "final_control") if ctx is not None else nullcontext():
            control_band = equivalence_band(
                [control, selected],
                policy,
                significance_test,
                y if significance_test is not None else None,
                block_index=block_index,
                sample_weight=sample_weight,
            )
        if control_band.winner == "__wide_control":
            candidates = [wide_candidate]
            selected_features = None
            if search_report is not None:
                search_report["final_control"] = "wide_control"
                search_report["use_original_factory"] = True
        elif search_report is not None:
            search_report["final_control"] = "selected_subset"

    if not candidates:
        # 0 completed: a budget that skipped candidates is budget-degraded -> BudgetExhaustedError;
        # otherwise every candidate failed on its own -> FitFailedError (M3 behavior; ADR-0032 §3).
        if budget_exhausted and skipped:
            # the exhausted axis captured at skip time (ADR-0039 §4, fix B1) — NOT BudgetConfig.mode, so a
            # memory-only run (mode="none") reports "memory", not "none". "budget" is an unreachable last
            # resort (a non-monotonic probe read at the skip would have to yield no axis).
            raise BudgetExhaustedError(
                exhausted_by or "budget",
                completed=0,
                skipped=len(skipped),
                failed=len(failed),
            )
        raise FitFailedError([(f.id, f.reason) for f in failed])

    if wide_candidate is not None and selected_features is None:
        full_verdict: dict[str, str] = {
            name: route
            for name, route in native_routing(schema, task.native_cat_max_unique).items()
        }
        native_routing_verdict = (
            full_verdict if any(route != "native" for route in full_verdict.values()) else None
        )

    # refinement-based selection (ADR-0031): rank by cross-fitted calibrated proper-loss. Only for
    # a proper-proba metric, classification, non-time-series, with enough OOF and >1 candidate;
    # all-or-nothing -> any non-viable candidate falls the whole run back to the raw selection.
    with ctx.timed_stage("run", "statistics") if ctx is not None else nullcontext():
        selection_mode: SelectionMode = "raw"
        score_space = "raw_oof"
        if selection == "refinement":
            refined = _maybe_refine(
                candidates,
                task=task,
                metric=metric,
                calib_blocks=oof_fold_index,
                y=y,
                positive=positive,
                classes=classes,
                sample_weight=sample_weight,
                calibrator_factory=calibrator_factory,
                refinement_min_oof=refinement_min_oof,
                time_ordered=time_ordered,
                logger=logger,
            )
            if refined is not None:
                candidates = refined
                selection_mode = "refinement"
                score_space = "calibrated_oof"

        band = equivalence_band(
            candidates,
            policy,
            significance_test,
            y if significance_test is not None else None,
            block_index=block_index,
            sample_weight=sample_weight,
        )
        ordered = rank(candidates, policy)
    if weighting == "period" and block_index is not None:
        # surface HOW the score was computed and on how many periods (G7, ADR-0098 §4) via the cv block;
        # n_periods_used is candidate-independent (finiteness is y+metric-determined), so the anchor's
        # valid-block count is the shared one. NOT in LeaderboardEntry (extra='forbid') or the artifact.
        anchor = ordered[0]
        assert (
            anchor.oof_pred is not None and anchor.oof_mask is not None
        )  # capture_oof forced (period)
        n_used = len(
            _period_block_scores(
                metric, y, anchor.oof_pred, anchor.oof_mask, block_index, sample_weight
            )
        )
        cv_split = {**(cv_split or {}), "weighting": weighting, "n_periods_used": n_used}
    leaderboard = [
        LeaderboardEntry(
            model_id=c.id,
            score=c.score,
            metric=metric.name,
            n_features=c.n_features,
            train_time=c.train_time,
            rank=i + 1,
        )
        for i, c in enumerate(ordered)
    ]
    return SliceResult(
        leaderboard=leaderboard,
        search=search_report,
        best_model_id=band.winner,
        candidates=candidates,
        failed=failed,
        band_member_ids=band.member_ids,
        band_unstable=band.unstable,
        band_width=band.width,
        winner_by_tiebreak=band.winner_by_tiebreak,
        selection_mode=selection_mode,
        score_space=score_space,
        budget=BudgetReport(
            skipped=tuple(skipped), exhausted=budget_exhausted, exhausted_by=exhausted_by
        ),
        reused=tuple(reused),
        computed=tuple(computed),
        oof_fold_index=oof_fold_index,
        feature_selection=feature_report if selected_features is not None else None,
        native_routing=native_routing_verdict,
        cv_split=cv_split,
        hpo=hpo_outcomes,
    )


def _run_candidate(
    name: str,
    factory: EstimatorFactory,
    *,
    x_full: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    categorical_indices: list[int],
    kind: TaskKind,
    positive: object | None,
    global_classes: np.ndarray,
    metric: Metric,
    folds: list[Fold],
    sample_weight: np.ndarray | None,
    n_features: int,
    need_proba: bool,
    capture_oof: bool,
    capture_proba: bool,
    block_index: np.ndarray | None,
    weighting: WeightingMode,
    logger: logging.Logger,
    ctx: RunContext | None = None,
    stage: str = "cv",
    trial: int | None = None,
) -> Candidate:
    n = y.shape[0]
    is_classification = kind in ("binary", "multiclass")
    multiclass = kind == "multiclass"
    # proba is produced when the metric needs it OR a downstream calibrator wants it (ADR-0030 §1)
    want_proba = (need_proba or capture_proba) and is_classification
    # only the multiclass proba path needs an (n, K) buffer; skip it when no proba is produced
    oof_proba = (
        np.full((n, global_classes.size), np.nan)
        if multiclass and want_proba
        else np.full(n, np.nan)
    )
    oof_class = np.empty(n, dtype=y.dtype)
    mask = np.zeros(n, dtype=bool)
    needs_proba_metric = metric.needs in _PROBA_NEEDS
    produced_proba = False

    t0 = time.perf_counter()
    iteration_counts: list[int] = []
    for fold_id, fold in enumerate(folds):
        test_idx = fold.test_idx
        if needs_proba_metric and np.unique(y[test_idx]).size < 2:
            logger.warning("fold skipped: single-class test split (model=%s)", name)
            continue
        train_idx = (
            fold.fit_idx
            if fold.es_idx.size == 0 or stage == "scouting"
            else np.concatenate([fold.fit_idx, fold.es_idx])
        )
        x_test = x_full[test_idx]
        sw_train = sample_weight[train_idx] if sample_weight is not None else None
        # Narrow isolation: only the external model calls are guarded (ADR-0022 §2);
        # a failure of any fold fails the whole candidate (ADR-0022 §1, fair OOF coverage).
        try:
            est = factory()
            est.feature_names = feature_names
            # native categorical routing (ADR-0088): hand the cat-column positions to a native-capable
            # model (injected like feature_names); others are untouched and stay on the codes path (FR-1).
            if isinstance(est, SupportsNativeCategorical):
                est.categorical_indices = categorical_indices
            # early stopping (ADR-0080): an ES-capable model holds the carved es tail out as validation
            # and trains on fit only; everything else merges fit ∪ es and trains on the union (unchanged).
            fit_rows = (
                fold.fit_idx.size
                if fold.es_idx.size and isinstance(est, SupportsEarlyStopping)
                else train_idx.size
            )
            fit_scope: AbstractContextManager[dict[str, int | None]] = (
                ctx.timed_fit(
                    stage,
                    model_id=name,
                    rows=int(fit_rows),
                    columns=n_features,
                    fold=fold_id,
                    trial=trial,
                )
                if ctx is not None
                else nullcontext(dict[str, int | None]())
            )
            with fit_scope as resources:
                if fold.es_idx.size > 0 and isinstance(est, SupportsEarlyStopping):
                    sw_fit = sample_weight[fold.fit_idx] if sample_weight is not None else None
                    est.fit(
                        x_full[fold.fit_idx],
                        y[fold.fit_idx],
                        X_val=x_full[fold.es_idx],
                        y_val=y[fold.es_idx],
                        sample_weight=sw_fit,
                    )
                else:
                    est.fit(x_full[train_idx], y[train_idx], sample_weight=sw_train)
                if isinstance(est, SupportsIterationBudget):
                    count = est.fitted_iterations
                    if count is not None:
                        iteration_counts.append(count)
                        resources["iterations"] = count
                    resources["tree_budget"] = est.iteration_budget
            raw_pred = est.predict(x_test)
            raw_proba = (
                est.predict_proba(x_test)
                if want_proba and isinstance(est, ProbabilisticEstimator)
                else None
            )
        except BudgetExhaustedError:
            raise
        except Exception as exc:
            raise _CandidateFailed(name, exc) from exc
        # Our code, outside the isolation: a bug here surfaces, it is not masked as "failed".
        oof_class[test_idx] = raw_pred
        if raw_proba is not None and isinstance(est, ProbabilisticEstimator):
            if multiclass:
                oof_proba[test_idx] = align_proba(raw_proba, est.classes_, global_classes)
            else:
                pos_idx = int(np.where(est.classes_ == positive)[0][0])
                oof_proba[test_idx] = raw_proba[:, pos_idx]
            produced_proba = True
        mask[test_idx] = True

    if not mask.any():
        raise _CandidateFailed(name, "produced no valid OOF predictions")
    y_valid = y[mask]
    if needs_proba_metric and np.unique(y_valid).size < 2:
        raise SchemaValidationError("OOF target has a single class; cannot score a proba metric")

    # metric-ready OOF (proba for proba-metrics, else class/value), projected once over the full array;
    # the same array the band aligns on. pooled scores it over `mask`; period macro-averages per block.
    metric_ready = project_for_metric(
        metric, proba=oof_proba if produced_proba else None, pred=oof_class, kind=kind
    )
    score = _score_weighted(metric, y, metric_ready, mask, block_index, sample_weight, weighting)

    # metric-ready OOF the band aligns on (ADR-0026 §3): proba for proba-metrics, else the
    # predicted class/value when a real test will consume it; validity is `mask`, never np.isnan.
    if needs_proba_metric:
        captured = oof_proba if produced_proba else None
    elif capture_oof:
        captured = oof_class
    else:
        captured = None
    # the raw proba channel for the calibrator (ADR-0030 §1 / ADR-0031 §3), kept separate from the
    # metric-ready oof_pred; valid rows are `mask`, the rest NaN.
    proba_channel = oof_proba if produced_proba else None
    return Candidate(
        id=name,
        score=score,
        n_features=n_features,
        train_time=round(time.perf_counter() - t0, 4),
        oof_pred=captured,
        oof_mask=mask if captured is not None else None,
        oof_proba=proba_channel,
        refit_iterations=max(1, int(np.median(iteration_counts))) if iteration_counts else None,
    )


def _maybe_refine(
    candidates: list[Candidate],
    *,
    task: Task,
    metric: Metric,
    calib_blocks: np.ndarray | None,
    y: np.ndarray,
    positive: object | None,
    classes: np.ndarray,
    sample_weight: np.ndarray | None,
    calibrator_factory: CalibratorFactory | None,
    refinement_min_oof: int,
    time_ordered: bool,
    logger: logging.Logger,
) -> list[Candidate] | None:
    """Replace candidates' score/oof_pred with cross-fitted calibrated values, or None (ADR-0031).

    Returns ``None`` (the whole run falls back to raw selection) on any gate miss: a non-proper /
    regression metric (no-op, §2), time-series (§3, disabled in M4), a single candidate (§3), too
    few OOF rows (§4b), or a candidate whose per-block calibration is not viable (§4a). ``calib_blocks``
    is the CV fold id per OOF row — SEPARATE from the band's bootstrap block_index (never fed to the
    band), so the non-TS band scheme is unchanged (fix B1).
    """
    if not (metric.proper_proba and task.is_classification):
        return None  # no-op by the proper_proba gate (ranking/argmax/regression) — ADR-0031 §2
    if time_ordered:
        logger.warning("refinement selection is disabled for time-series CV; using raw")
        return None
    if calibrator_factory is None or calib_blocks is None or len(candidates) < 2:
        return None  # 1 candidate -> nothing to choose, keep raw score (ADR-0031 §3)
    if y.shape[0] < refinement_min_oof:
        logger.warning(
            "refinement selection needs >= %d OOF rows (have %d); using raw",
            refinement_min_oof,
            y.shape[0],
        )
        return None
    refined: list[Candidate] = []
    for c in candidates:
        out = _refine_candidate(
            c,
            kind=task.kind,
            metric=metric,
            y=y,
            positive=positive,
            classes=classes,
            calib_blocks=calib_blocks,
            factory=calibrator_factory,
            sample_weight=sample_weight,
        )
        if out is None:
            logger.warning(
                "refinement selection unavailable (candidate %r calibration not viable); using raw",
                c.id,
            )
            return None
        refined.append(out)
    return refined


def _refine_candidate(
    candidate: Candidate,
    *,
    kind: TaskKind,
    metric: Metric,
    y: np.ndarray,
    positive: object | None,
    classes: np.ndarray,
    calib_blocks: np.ndarray,
    factory: CalibratorFactory,
    sample_weight: np.ndarray | None,
) -> Candidate | None:
    """One candidate's cross-fitted calibrated score/oof, or None if its blocks are not viable."""
    if candidate.oof_proba is None or candidate.oof_mask is None:
        return None
    mask = candidate.oof_mask
    proba_m = candidate.oof_proba[mask]
    blocks_m = calib_blocks[mask]
    y_m = y[mask]
    y_code = (
        np.searchsorted(classes, y_m)
        if kind == "multiclass"
        else (y_m == positive).astype(np.int64)
    )
    if not viable_blocks(
        blocks_m, y_code, n_classes=classes.size if kind == "multiclass" else None
    ):
        return None
    sw_m = sample_weight[mask] if sample_weight is not None else None
    cal_m = crossfit_calibrate(proba_m, y_code, blocks_m, factory, sample_weight=sw_m)
    score = metric.score(y_m, cal_m, sw_m)
    cal_full = np.full_like(candidate.oof_proba, np.nan)
    cal_full[mask] = cal_m
    # the band ranks on the calibrated oof_pred; oof_proba stays RAW so a production calibrator
    # (ADR-0030, if also enabled) fits the raw winner OOF, not an already-calibrated one.
    return replace(candidate, score=score, oof_pred=cal_full)


def refit_best(
    dataset: Dataset,
    task: Task,
    *,
    factory: EstimatorFactory,
    ctx: RunContext | None = None,
    iterations: int | None = None,
    model_id: str = "winner",
) -> Estimator:
    """Refit the winning model on the full training data (es tail included)."""
    y = dataset.target()
    if y is None:
        raise SchemaValidationError("refit_best requires a target column")
    est = factory()
    # the shipped model trains on the selected subset when selection ran (ADR-0045 §2); design_matrix
    # already projects to it (in schema.features order), so feature_names must match that order.
    schema = dataset.schema
    selected = schema.selected_features
    if selected is None:
        est.feature_names = list(schema.features)
    else:
        kept = set(selected)
        est.feature_names = [f for f in schema.features if f in kept]
    # native categorical routing (ADR-0088/0092, FR-2/FR-4): the schema (with selected_features) computes
    # the same cardinality-gated projected indices the CV path used (cap threaded from the task), so the
    # shipped model trains native-consistently with the leaderboard — and the manifest n_cat is post-gate.
    if isinstance(est, SupportsNativeCategorical):
        est.categorical_indices = schema.categorical_indices(task.native_cat_max_unique)
    if iterations is not None and isinstance(est, SupportsIterationBudget):
        est.set_refit_iterations(iterations)
    matrix = design_matrix(dataset)
    fit_scope: AbstractContextManager[dict[str, int | None]] = (
        ctx.timed_fit("refit", model_id=model_id, rows=dataset.n_rows, columns=matrix.shape[1])
        if ctx is not None
        else nullcontext(dict[str, int | None]())
    )
    with fit_scope as resources:
        est.fit(matrix, y, sample_weight=dataset.sample_weight())
        if isinstance(est, SupportsIterationBudget):
            resources["iterations"] = est.fitted_iterations
    return est
