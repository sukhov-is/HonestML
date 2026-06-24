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
    RunContext,
    SchemaValidationError,
    SelectionPolicy,
    SignificanceTest,
    SupportsEarlyStopping,
    SupportsNativeCategorical,
    Task,
    TimeOrderedSplitter,
    equivalence_band,
    get_logger,
    rank,
    resolve_positive,
    validate_fold,
)
from honestml.core.config import FeatureSelectionConfig, FEConfig, SelectionMode, WeightingMode
from honestml.core.ports.splitter import CVSplitter
from honestml.core.schema import categorical_positions, native_routing, te_output_name
from honestml.core.task import TaskKind

from .calibration import crossfit_calibrate, viable_blocks
from .feature_compare import compare_features, no_selection_gate
from .feature_encoding import crossfit_encode, crossfit_encode_expanding
from .feature_selection import _degenerate_counts, select_features, structure_labels
from .projection import _PROBA_NEEDS, align_proba, project_for_metric

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


class _CandidateFailed(Exception):
    """Internal signal: one candidate's model raised; isolate it (ADR-0022 §1)."""

    def __init__(self, name: str, reason: object) -> None:
        self.name = name
        self.reason = str(reason)
        super().__init__(f"candidate {name!r} failed: {self.reason}")


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
    full = np.hstack([numeric, codes.astype(np.float64, copy=False)])
    selected = dataset.schema.selected_features
    if selected is None:
        return full
    features = dataset.schema.features
    selected_set = set(selected)
    missing = selected_set - set(features)
    if missing:
        raise SchemaValidationError(
            f"selected feature {sorted(missing)!r} absent from the design matrix"
        )
    # project in schema.features order (not the subset's tuple order) so columns stay aligned with
    # refit's feature_names regardless of how the subset was stored (R-FS-COLDRIFT, FR-FS-7).
    keep = [i for i, f in enumerate(features) if f in selected_set]
    return full[:, keep]


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
    budget: Budget | None = None,
    cache: CandidateCache | None = None,
    ctx: RunContext | None = None,
) -> SliceResult:
    """Run the binary slice and return its OOF leaderboard."""
    logger = ctx.logger if ctx is not None else get_logger()
    schema = dataset.schema
    if not estimators:
        raise ConfigError("run_slice requires at least one estimator")
    # unpack the FS bundle into the leakage-critical injectables the body wires (ADR-0044); None == off
    feature_selection = features.config if features is not None else None
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
    if feature_selection is not None and feature_strategies is not None:
        # composition wires the three M6c components as a bundle (build.py); narrow for the call
        assert feature_carve is not None and feature_fit_predict is not None
        categorical_mask = np.zeros(n_features, dtype=bool)
        categorical_mask[len(schema.numeric) :] = True
        fs_seed = (
            feature_selection.random_state if feature_selection.random_state is not None else 0
        )
        outcome = compare_features(
            dataset,
            x_full,
            y,
            task=task,
            metric=metric,
            strategies=feature_strategies,
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
            )
        )

    # no-selection honest gate (finding #10): an FS subset must not be SIGNIFICANTLY worse than the full
    # feature set, mirroring the ensemble's choose_better gate (ADR-0063). Covers BOTH the M6c compare and
    # the M6b single-ranker paths; on a "no_selection_better" verdict we ship all features (never silent).
    if fs_idx is not None:
        if (
            feature_selection is not None
            and feature_fit_predict is not None
            and significance_test is not None
            and len(fs_idx) < n_features
        ):
            fs_gate_seed = (
                feature_selection.random_state if feature_selection.random_state is not None else 0
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
    budget_exhausted = False
    # the exhausted axis, captured at the MOMENT of exhaustion (ADR-0039 §5): truthful and robust to a
    # later non-monotonic RSS read; first capture wins. None on a within-budget run.
    exhausted_by: str | None = None
    for name, factory in estimators.items():
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
                )
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

    # refinement-based selection (ADR-0031): rank by cross-fitted calibrated proper-loss. Only for
    # a proper-proba metric, classification, non-time-series, with enough OOF and >1 candidate;
    # all-or-nothing -> any non-viable candidate falls the whole run back to the raw selection.
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
        feature_selection=(
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
            )
            if selected_features is not None
            else None
        ),
        native_routing=native_routing_verdict,
        cv_split=cv_split,
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
    for fold in folds:
        test_idx = fold.test_idx
        if needs_proba_metric and np.unique(y[test_idx]).size < 2:
            logger.warning("fold skipped: single-class test split (model=%s)", name)
            continue
        train_idx = (
            fold.fit_idx if fold.es_idx.size == 0 else np.concatenate([fold.fit_idx, fold.es_idx])
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
            raw_pred = est.predict(x_test)
            raw_proba = (
                est.predict_proba(x_test)
                if want_proba and isinstance(est, ProbabilisticEstimator)
                else None
            )
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
    est.fit(design_matrix(dataset), y, sample_weight=dataset.sample_weight())
    return est
