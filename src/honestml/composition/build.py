"""Composition root: assemble default components for a task (ADR-0009 / ADR-0016).

The single place that names concrete adapters. The use-case stays adapter-blind;
this factory picks the metric, splitter, estimators and a selection policy whose
direction is synced to the metric (ADR-0009 §F10). Pairs whose metric needs
probabilities but whose model is not probabilistic — or whose model does not
support the task — are dropped (ADR-0009 §F3); if none remain it fails fast with
``ConfigError`` **before** CV is resolved (ADR-0016 §2).

CV selection is honest (ADR-0016): the ``CVConfig`` scheme drives the splitter,
``"auto"`` resolves to ``Task.default_cv_scheme``, and an unimplemented scheme or
``purge``/``embargo`` fails fast with ``UnsupportedSchemeError`` instead of silently
falling back to a shuffling split. A shuffling scheme over datetime columns is
warned about (look-ahead risk). The entry-point registry / boosting zoo are M3;
time-series CV with purge/embargo is M4.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, NamedTuple, cast

import numpy as np

from honestml.adapters import (
    BootstrapSignificanceTest,
    GroupKFoldSplitter,
    HoldoutSplitter,
    KFoldSplitter,
    PeriodTimeSeriesSplitter,
    StratifiedGroupKFoldSplitter,
    StratifiedKFoldSplitter,
    TimeSeriesSplitter,
    resolve_calibrator,
    resolve_metric,
)
from honestml.application import EstimatorFactory, MakeFactory
from honestml.application.feature_compare import FitPredict, SelectionCarve, Strategy
from honestml.application.feature_selection import estimate_fs_refits
from honestml.core import (
    CalibratorFactory,
    ConfigError,
    CVConfig,
    CVSplitter,
    Dataset,
    Ensembler,
    Estimator,
    FeatureRanker,
    Metric,
    MissingDependencyError,
    NoSignificanceTest,
    SelectionPolicy,
    SignificanceTest,
    SupportsNativeCategorical,
    Task,
    Tuner,
    UnsupportedSchemeError,
    get_logger,
    parse_search_space,
    resolve_positive,
)
from honestml.core.config import (
    CalibrateMethod,
    EnsembleConfig,
    FeatureSelectionConfig,
    HPOConfig,
    SelectionMode,
    SignificanceMode,
    WeightingMode,
)

from .registry import _BUILTIN_DIST, ComponentRegistry, model_registry

_PROBA_NEEDS = ("proba", "threshold")
_CLASS_NEEDS = ("proba", "threshold", "class")
_SHUFFLING_SCHEMES = ("stratified", "holdout", "kfold")
_KFOLD_FAMILY = ("stratified", "kfold", "group")
# value-ordered schemes that accept purge/embargo gaps (ADR-0027 / ADR-0096)
_TIME_SCHEMES = ("timeseries", "timeseries_period")
# i.i.d. early-stopping validation fraction carved from each fold's train rows (ADR-0080); only
# active when a supports_early_stopping model is in the zoo, so non-boosting runs are unchanged.
_ES_FRACTION = 0.1
# SPIKE-0002: the band decision is stable (flip-rate 0%) for n_boot from 200 to 10000, and
# n_boot>=1000 is sufficient; 2000 was statistical headroom. The band recomputes the metric per
# resample through the Metric port (no metric-specific fast path, to keep core clean), so the cost
# is n_boot sklearn calls per pair — a constant per fit. We pick 1000 (meets the §7 floor
# n_boot*alpha>=50 at the default alpha=0.05) to bound that constant; pass an explicit test for more.
_DEFAULT_N_BOOT = 1000
logger = get_logger("composition.build")


class Components(NamedTuple):
    """Default wiring handed to ``run_slice`` (``cv`` is the resolved scheme)."""

    estimators: dict[str, EstimatorFactory]
    splitter: CVSplitter
    metric: Metric
    policy: SelectionPolicy
    cv: CVConfig
    significance: SignificanceTest
    # period weighting of the leaderboard score (ADR-0098): threaded into run_slice; "pooled" by default.
    weighting: WeightingMode
    # calibration (ADR-0030/0031): the production calibrate method, the refinement-selection mode
    # and its (low-DOF) calibrator factory, and the OOF row floor for refinement.
    calibrate: CalibrateMethod
    selection: SelectionMode
    refinement_calibrator: CalibratorFactory | None
    refinement_min_oof: int
    # feature selection (ADR-0043): the resolved config (with seed filled) + its ranker, or None when off.
    # M6b single-ranker path uses `feature_ranker`; the M6c compare/sequential path (ADR-0046/0048) uses
    # `feature_strategies` + an injected scheme-aware carve + estimator-agnostic fit_predict (mutually
    # exclusive with feature_ranker). All None when FS is off.
    feature_selection: FeatureSelectionConfig | None
    feature_ranker: FeatureRanker | None
    feature_strategies: tuple[tuple[str, Strategy], ...] | None = None
    feature_carve: SelectionCarve | None = None
    feature_fit_predict: FitPredict | None = None
    # M6d nested arbitration (ADR-0052): a scheme-aware K-fold splitter on DEV (n_splits=arbitration_n_splits);
    # None unless arbitration="nested" with a compare list (timeseries -> expanding-window).
    feature_arbitration_splitter: CVSplitter | None = None
    # M7a HPO (ADR-0061/0062): the resolved Tuner, a tuned-factory builder, the inner-CV splitter and the
    # tunable name->search_space map. All None when hpo is off; the facade runs the tuning stage on DEV.
    tuner: Tuner | None = None
    make_factory: MakeFactory | None = None
    inner_splitter: CVSplitter | None = None
    tunable: dict[str, dict[str, Any]] | None = None
    # M7b ensembling (ADR-0063/0064): the resolved Ensembler and the metric the blend is scored on
    # (EnsembleConfig.metric or the run metric). Both None when ensemble is off.
    ensembler: Ensembler | None = None
    ensemble_metric: Metric | None = None
    # early stopping (ADR-0080): True when a boosting early-stops on a carved es tail this run; the
    # facade records it in the manifest. False only for non-boosting runs -- every scheme (i.i.d./group/
    # timeseries) carves an es tail when a boosting model is in the zoo.
    early_stopping: bool = False


def build_default_components(
    task: Task,
    *,
    random_state: int,
    metric: str | Metric | None = None,
    cv: int | CVConfig | None = None,
    models: tuple[str, ...] | None = None,
    has_datetime: bool = False,
    has_group: bool = False,
    has_time: bool = False,
    has_missing: bool = False,
    classes: np.ndarray | None = None,
    significance: SignificanceMode = "bootstrap",
    feature_selection: FeatureSelectionConfig | None = None,
    hpo: HPOConfig | None = None,
    ensemble: EnsembleConfig | None = None,
) -> Components:
    """Build the default (estimators, splitter, metric, policy, cv) for *task*.

    ``classes`` (the global class order, ``np.unique(y)`` for classification) is passed by the
    facade so the metric carries ``labels`` for multiclass scoring (ADR-0021 §4). ``has_group``
    drives group-CV routing; ``has_time`` drives time-series routing and the look-ahead warning
    (ADR-0023/0027). ``has_missing`` (NaN in the numeric block) drops models that require
    imputed input instead of letting them crash mid-fit.
    """
    resolved = _resolve_metric(task, metric, classes)

    # Estimators from the registry; capability filter first: an unsupported task must
    # fail on the estimator ("no estimator supports …"), not on CV resolution (ADR-0016 §2).
    estimators = _select_estimators(task, resolved, models, random_state, has_missing=has_missing)

    # early stopping (ADR-0080): carve an es tail only when a boosting (supports_early_stopping) is in
    # the zoo, so non-boosting runs are byte-identical. Every scheme carves it — i.i.d. and group
    # subsample the fold's train (group-disjointly), timeseries takes its end tail.
    es_enabled = _any_early_stopping(estimators)
    splitter, cv_config = _resolve_splitter(
        _normalize_cv(cv),
        task,
        random_state,
        has_datetime,
        has_group,
        has_time,
        es_fraction=_ES_FRACTION if es_enabled else 0.0,
    )
    early_stopping = es_enabled
    policy = SelectionPolicy(greater_is_better=resolved.greater_is_better)
    # honest selection is the default (ADR-0026 §4): the band is built on the selection metric,
    # seeded from random_state for reproducibility. The public "off" toggle (ADR-0034) returns the
    # inert NoSignificanceTest -> a pure argmax with no band membership and no forced OOF capture.
    significance_test: SignificanceTest = (
        NoSignificanceTest()
        if significance == "off"
        # aggregate mirrors weighting: 'period' macro-averages per-block Δ (ADR-0098 §3); only time-ordered
        # schemes reach 'period' (gated above), so a block_index is always present when it is set.
        else BootstrapSignificanceTest(
            resolved, seed=random_state, n_boot=_DEFAULT_N_BOOT, aggregate=cv_config.weighting
        )
    )
    # calibration is classification-only (ADR-0030); refinement uses a low-DOF sigmoid by default
    # (ADR-0031 §5), n_calib=None keeps 'auto' deterministic at sigmoid for the selection path.
    if task.kind == "regression" and cv_config.calibrate != "off":
        raise ConfigError("probability calibration is classification-only; set calibrate='off'")
    refinement_calibrator: CalibratorFactory | None = None
    if cv_config.selection == "refinement":
        method = cv_config.calibrate if cv_config.calibrate != "off" else "sigmoid"
        refinement_calibrator = resolve_calibrator(method)
    # FS routing (ADR-0046 §2): the M6c compare/sequential path needs the second port + carve + scorer;
    # a single ranker strategy (importance/random_probe/null_importance) stays the M6b path (feature_ranker).
    fs_ranker: FeatureRanker | None = None
    fs_strategies: tuple[tuple[str, Strategy], ...] | None = None
    fs_carve: SelectionCarve | None = None
    fs_fit_predict: FitPredict | None = None
    fs_arb_splitter: CVSplitter | None = None
    if feature_selection is not None:
        # the estimator-agnostic ranker-model: the M6c arbitration scorer AND the M6b/M6c no-selection
        # gate (finding #10) both score subsets with it, so it is wired whenever FS runs (not only compare).
        fs_fit_predict = _make_fit_predict(task)
        if feature_selection.compare is not None or feature_selection.strategy == "sequential":
            fs_strategies = _resolve_strategies(
                task, feature_selection, full_descent=significance != "off"
            )
            fs_carve = _make_selection_carve(task, cv_config)
            if feature_selection.arbitration in ("nested", "nested_per_fold"):
                fs_arb_splitter = _resolve_splitter(
                    cv_config.model_copy(
                        update={"n_splits": feature_selection.arbitration_n_splits}
                    ),
                    task,
                    random_state,
                    has_datetime,
                    has_group,
                    has_time,
                )[0]
        else:
            fs_ranker = _resolve_feature_ranker(task, feature_selection)
            if feature_selection.arbitration in ("nested", "nested_per_fold"):
                # arbitration only acts when comparing >= 2 strategies; a single strategy ignores it
                logger.warning(
                    "arbitration=%r has no effect without a `compare` list of >= 2 strategies",
                    feature_selection.arbitration,
                )
        # per-fold re-selection (ADR-0054) on timeseries is only boundary-leak-safe if the arbitration splitter
        # purges; with purge=0 and no label_time the inner re-selection trains on rows adjacent to outer-test.
        if (
            feature_selection.arbitration in ("nested", "nested_per_fold")
            and has_time
            and cv_config.purge == 0
        ):
            logger.warning(
                "arbitration=%r on a timeseries scheme with purge=0 does not purge the inner/outer boundary; "
                "set cv purge>0 (or declare label_time) for leak-safe per-fold arbitration",
                feature_selection.arbitration,
            )
        # dead-config (ADR-0055 §1): a window only acts under null_block_mode='time_window' (WARNING, not error)
        if (
            feature_selection.null_block_mode == "rank"
            and feature_selection.null_block_window is not None
        ):
            logger.warning(
                "null_block_window is set but null_block_mode='rank' ignores it; "
                "set null_block_mode='time_window' to bin by Δt windows"
            )
        # dead-config (ADR-0056 §1): background only acts under shap_perturbation='interventional'
        if (
            feature_selection.shap_perturbation == "tree_path_dependent"
            and feature_selection.shap_background_samples is not None
        ):
            logger.warning(
                "shap_background_samples is set but shap_perturbation='tree_path_dependent' uses no background; "
                "set shap_perturbation='interventional' to use it"
            )
        # dead-config (ADR-0060 §3): kmeans background only acts under shap_perturbation='interventional'
        if (
            feature_selection.shap_perturbation == "tree_path_dependent"
            and feature_selection.shap_background == "kmeans"
        ):
            logger.warning(
                "shap_background='kmeans' is set but shap_perturbation='tree_path_dependent' uses no background; "
                "set shap_perturbation='interventional' to use it"
            )
        _warn_fs_cost(feature_selection)

    # M7a HPO wiring (ADR-0061/0062): resolve the Tuner (extras-gated), the inner-CV splitter and the
    # tunable name->search_space map; the facade runs the tuning stage on DEV. All None when hpo is off.
    tuner: Tuner | None = None
    make_factory: MakeFactory | None = None
    inner_splitter: CVSplitter | None = None
    tunable: dict[str, dict[str, Any]] | None = None
    if hpo is not None:
        tuner = _resolve_tuner(hpo)
        inner_splitter = _resolve_splitter(
            cv_config.model_copy(update={"n_splits": hpo.inner_cv}),
            task,
            random_state,
            has_datetime,
            has_group,
            has_time,
        )[0]
        make_factory = _make_tuned_factory_builder(model_registry(), task, random_state)
        tunable = _resolve_tunable(estimators, hpo)

    # M7b ensembling wiring (ADR-0063/0064): resolve the Ensembler adapter + the blend metric; the facade
    # runs the ensemble stage after run_slice. Both None when ensemble is off.
    ensembler: Ensembler | None = None
    ensemble_metric: Metric | None = None
    if ensemble is not None:
        ensembler = _resolve_ensembler(ensemble)
        ensemble_metric = (
            _resolve_metric(task, ensemble.metric, classes)
            if ensemble.metric is not None
            else resolved
        )

    return Components(
        estimators=estimators,
        splitter=splitter,
        metric=resolved,
        policy=policy,
        cv=cv_config,
        significance=significance_test,
        weighting=cv_config.weighting,
        calibrate=cv_config.calibrate,
        selection=cv_config.selection,
        refinement_calibrator=refinement_calibrator,
        refinement_min_oof=cv_config.refinement_min_oof,
        feature_selection=feature_selection,
        feature_ranker=fs_ranker,
        feature_strategies=fs_strategies,
        feature_carve=fs_carve,
        feature_fit_predict=fs_fit_predict,
        feature_arbitration_splitter=fs_arb_splitter,
        tuner=tuner,
        make_factory=make_factory,
        inner_splitter=inner_splitter,
        tunable=tunable,
        ensembler=ensembler,
        ensemble_metric=ensemble_metric,
        early_stopping=early_stopping,
    )


def _resolve_tuner(hpo: HPOConfig) -> Tuner:
    """Resolve the HPO backend, extras-gated via ``find_spec`` (ADR-0061 §3 / ADR-0062)."""
    from importlib.util import find_spec

    if hpo.backend == "optuna":
        if find_spec("optuna") is None:
            raise MissingDependencyError("optuna")
        from honestml.adapters import OptunaTuner

        return OptunaTuner()
    raise ConfigError(f"unknown hpo backend {hpo.backend!r}")  # pragma: no cover (Literal-guarded)


def _resolve_ensembler(ensemble: EnsembleConfig) -> Ensembler:
    """Resolve the ensembler adapter from the method literal (ADR-0063 §3); no extra needed (scipy rides
    in with sklearn, Caruana is pure numpy)."""
    from honestml.adapters import CaruanaEnsembler, WeightedEnsembler

    if ensemble.method == "caruana":
        return CaruanaEnsembler(size=ensemble.size, n_bags=ensemble.n_bags)
    if ensemble.method == "weighted":
        return WeightedEnsembler()
    raise ConfigError(
        f"unknown ensemble method {ensemble.method!r}"
    )  # pragma: no cover (Literal-guarded)


def _make_tuned_factory_builder(
    registry: ComponentRegistry, task: Task, random_state: int
) -> MakeFactory:
    """A 2-arg builder ``(name, params) -> EstimatorFactory`` that validates params then closes a factory.

    Tuned param keys must be a subset of the component's declared ``search_space`` (composition-time
    validation, ADR-0061 §4); a stray key fails loud with ``ConfigError`` rather than being silently
    dropped by the native ``**kwargs`` ctor.
    """
    by_name = registry.by_name()

    def make(name: str, params: Mapping[str, Any]) -> EstimatorFactory:
        space = by_name[name].spec.search_space
        unknown = set(params) - set(space)
        if unknown:
            raise ConfigError(f"tuned params {sorted(unknown)} are not in {name!r} search_space")
        kw = dict(params)
        return lambda: registry.build(name, task=task, random_state=random_state, **kw)

    return make


def _resolve_tunable(
    estimators: Mapping[str, EstimatorFactory], hpo: HPOConfig
) -> dict[str, dict[str, Any]]:
    """The tunable ``name -> search_space`` map: selected estimators with a non-empty space (ADR-0062 §1).

    ``hpo.models`` narrows the set; a requested model that is not a selected estimator, or has an empty
    space, is dropped with a WARNING (it cannot be tuned) rather than failing the run.
    """
    by_name = model_registry().by_name()
    names = hpo.models if hpo.models is not None else tuple(estimators)
    out: dict[str, dict[str, Any]] = {}
    for name in names:
        if name not in estimators:
            logger.warning("hpo.models lists %r which is not a selected estimator; skipping", name)
            continue
        space = by_name[name].spec.search_space
        if parse_search_space(space):  # non-empty + valid
            out[name] = space
        elif hpo.models is not None:
            logger.warning("model %r has no search_space; HPO cannot tune it (skipped)", name)
    return out


def _warn_fs_cost(fs: FeatureSelectionConfig) -> None:
    """Log a cost WARNING when an expensive FS strategy is enabled (NFR-FSH-2/NFR-FSC-3/NFR-FSE-2)."""
    names = set(fs.compare) if fs.compare is not None else {fs.strategy}
    if "null_importance" in names:
        logger.warning(
            "strategy 'null_importance' refits the ranker-model n_folds x (1 + n_runs=%d) times",
            fs.n_runs,
        )
    if "shap" in names:
        logger.warning(
            "strategy 'shap' runs a TreeExplainer per fold (cost ~ importance + SHAP pass)"
        )
    if "sequential" in names:
        logger.warning(
            "strategy 'sequential' is a wrapper: O(n_features^2) score_subset evaluations"
        )
    if fs.arbitration == "nested_per_fold":
        # per-fold re-selection multiplies the SELECTION cost by the outer fold count (ADR-0054 §5); with
        # null_importance this is the most expensive FS mode (hours on large data) -> opt-in, loud at build.
        logger.warning(
            "arbitration='nested_per_fold' re-selects features inside every one of %d outer folds (x K inner) "
            "-> SELECTION cost x %d; prefer importance/shap, use null_importance per-fold only on small data",
            fs.arbitration_n_splits,
            fs.arbitration_n_splits,
        )


# M6f auto-resolve thresholds (ADR-0057 §1/§2); conservative, M9-tunable (R-AUTOCLAIM, not optimality claims)
_AUTO_N_SMALL = 2000  # n_rows below this -> nested_per_fold affordable; below _AUTO_N_MED -> nested; else holdout
_AUTO_N_MED = 20000
_AUTO_CV_IRREG = (
    0.25  # Δt coefficient-of-variation above which a series is "irregular" -> time_window blocks
)


def _resolve_block_mode(
    times: np.ndarray | None, block_size: int, explicit_window: float | None
) -> tuple[str, float | None]:
    """Auto null_block_mode (ADR-0057 §2): time_window on an irregular series, else rank.

    Returns ``(mode, window_to_set)``; ``window_to_set`` is None when no write-back is needed (rank, or
    time_window with a user-set window). Degenerate Δt (empty / duplicate timestamps -> median≤0) falls back
    to rank so a derived window of 0 never violates ``null_block_window: gt=0`` on write-back.
    """
    if times is None:
        return "rank", None
    dt = np.diff(np.sort(np.asarray(times, dtype=np.float64)))
    if dt.size == 0:
        return "rank", None
    mean, median = float(dt.mean()), float(np.median(dt))
    if (
        mean <= 0 or median <= 0
    ):  # duplicate timestamps -> derived window would be 0 (ADR-0057 §2 guard)
        return "rank", None
    if float(dt.std()) / mean <= _AUTO_CV_IRREG:
        return "rank", None  # regular series -> equal-count rank blocks cover it
    return "time_window", (None if explicit_window is not None else median * block_size)


def _budget_downgrade(
    fs: FeatureSelectionConfig,
    arb: str,
    *,
    budget: int,
    n_strategies: int,
    n_features: int,
    inner_n_splits: int,
) -> str:
    """Lowest-honesty-loss arbitration whose estimated refits fit the budget; fail loud at the floor (ADR-0058)."""
    ladder = ("nested_per_fold", "nested", "holdout")
    start = ladder.index(arb) if arb in ladder else len(ladder) - 1
    for cand in ladder[start:]:
        cost = estimate_fs_refits(
            fs.model_copy(update={"arbitration": cand}),
            n_strategies=n_strategies,
            n_features=n_features,
            inner_n_splits=inner_n_splits,
        )
        if cost <= budget:
            return cand
    floor = estimate_fs_refits(
        fs.model_copy(update={"arbitration": "holdout"}),
        n_strategies=n_strategies,
        n_features=n_features,
        inner_n_splits=inner_n_splits,
    )
    raise ConfigError(
        f"cost_budget_refits={budget} is below the holdout floor (~{floor} ranker-refits); raise the budget "
        f"or cheapen the ranker (lower n_runs / avoid sequential)"
    )


def resolve_fs_defaults(
    fs: FeatureSelectionConfig,
    *,
    n_rows: int,
    n_features: int,
    inner_n_splits: int,
    times: np.ndarray | None,
    scheme: str,
    purge: int,
    purge_delta: float | None = None,
) -> tuple[FeatureSelectionConfig, dict[str, str]]:
    """Resolve data-shape ``"auto"`` sentinels + apply the hard cost-budget (ADR-0057/0058).

    Pure composition step (called from ``facade.fit`` post-read). Returns ``(effective_fs, resolve_record)``.
    Never touches an explicit non-auto value (NFR-FSF-6) except a hard cost-budget downgrade (loud). The
    record carries resolve provenance (auto/cost) for the run-report ``fs_resolution`` block (ADR-0058 §4).
    """
    updates: dict[str, object] = {}
    record: dict[str, str] = {}
    n_strategies = len(fs.compare) if fs.compare is not None else 1

    if fs.arbitration == "auto":  # C1: most-honest affordable locus by data shape (ADR-0057 §1)
        if n_strategies < 2:
            arb = "holdout"  # arbitration has nothing to resolve for a single strategy (incl. sequential)
        elif scheme in _TIME_SCHEMES and purge == 0 and purge_delta is None:
            # anti-leakage: don't enable leak-prone per-fold on an unpurged boundary. A Δt purge
            # (purge_delta) also separates the inner/outer boundary, so it counts as purged (ADR-0097).
            arb = "holdout"
        elif n_rows < _AUTO_N_SMALL:
            arb = "nested_per_fold"
        elif n_rows < _AUTO_N_MED:
            arb = "nested"
        else:
            arb = "holdout"
        updates["arbitration"] = arb
        record |= {"arbitration_requested": "auto", "arbitration_resolved_from": "auto"}

    if (
        fs.null_block_mode == "auto"
    ):  # C1: time_window on an irregular series, else rank (ADR-0057 §2)
        mode, window = _resolve_block_mode(times, fs.null_block_size, fs.null_block_window)
        updates["null_block_mode"] = mode
        if window is not None:
            updates["null_block_window"] = window
        record |= {"block_mode_requested": "auto", "block_mode_resolved_from": "auto"}

    if (
        fs.cost_budget_refits is not None
    ):  # C2: hard ceiling -> downgrade arbitration / fail loud (ADR-0058)
        arb_now = str(updates.get("arbitration", fs.arbitration))
        chosen = _budget_downgrade(
            fs,
            arb_now,
            budget=fs.cost_budget_refits,
            n_strategies=n_strategies,
            n_features=n_features,
            inner_n_splits=inner_n_splits,
        )
        if chosen != arb_now:
            logger.warning(
                "cost_budget_refits=%d: arbitration %r exceeds budget -> downgraded to %r",
                fs.cost_budget_refits,
                arb_now,
                chosen,
            )
            updates["arbitration"] = chosen
            record["arbitration_requested"] = record.get("arbitration_requested", fs.arbitration)
            record["arbitration_resolved_from"] = "cost_budget"

    return (fs.model_copy(update=updates) if updates else fs), record


def _make_strategy(
    task: Task, fs: FeatureSelectionConfig, name: str, *, full_descent: bool = False
) -> Strategy:
    from honestml.adapters import (
        ImportanceRanker,
        NullImportanceRanker,
        RandomProbeRanker,
        SequentialSelector,
        ShapRanker,
    )

    if name == "sequential":
        # full_descent under an active band (ADR-0084): explore to the floor so the band sees the
        # whole path; off -> legacy patience early-stop (FR-2 back-compat).
        return SequentialSelector(
            min_features=fs.seq_min_features, patience=fs.seq_patience, full_descent=full_descent
        )
    if name == "null_importance":
        return NullImportanceRanker(task, n_runs=fs.n_runs, null_percentile=fs.null_percentile)
    if name == "random_probe":
        return RandomProbeRanker(task, n_probes=fs.n_probes)
    if name == "shap":
        return ShapRanker(
            task,
            max_samples=fs.shap_max_samples,
            perturbation=fs.shap_perturbation,
            background_samples=fs.shap_background_samples,
            shap_background=fs.shap_background,
        )
    return ImportanceRanker(task)


def _resolve_strategies(
    task: Task, fs: FeatureSelectionConfig, *, full_descent: bool = False
) -> tuple[tuple[str, Strategy], ...]:
    """Resolve the compare/sequential strategy list (ADR-0046 §2); one adapter per name, by port.

    null_importance is supported on every scheme (M6d, ADR-0050): i.i.d. schemes permute uniformly,
    group/timeseries permute within structure — the spine threads the per-row label, so no resolve guard.
    ``full_descent`` (ADR-0084) is applied to the ``sequential`` wrapper only (set when the band is active).
    """
    names = fs.compare if fs.compare is not None else (fs.strategy,)
    resolved: list[tuple[str, Strategy]] = [
        (name, _make_strategy(task, fs, name, full_descent=full_descent)) for name in names
    ]
    return tuple(resolved)


def _make_selection_carve(task: Task, cv: CVConfig) -> SelectionCarve:
    """Bind the scheme-aware selection-holdout carve (reuses ``outer_holdout_carve`` on DEV; ADR-0048 §1).

    Injected into the application arbiter so it imports no adapter (NFR-FSC-2). Inherits the CV scheme
    and ``purge`` (timeseries future-window), like the outer holdout.
    """
    from honestml.adapters import outer_holdout_carve

    def carve(
        dataset: Dataset, fraction: float, random_state: int
    ) -> tuple[np.ndarray, np.ndarray]:
        return outer_holdout_carve(
            dataset,
            scheme=cv.scheme,
            fraction=fraction,
            stratify=task.is_classification,
            random_state=random_state,
            purge=cv.purge,
            purge_delta=cv.purge_delta,
            period=cv.period,
            period_size=cv.period_size,
        )

    return carve


def _make_fit_predict(task: Task) -> FitPredict:
    from honestml.adapters import make_ranker_fit_predict

    return make_ranker_fit_predict(task)


def _resolve_feature_ranker(task: Task, fs: FeatureSelectionConfig | None) -> FeatureRanker | None:
    """Resolve the default feature ranker for the single-strategy FS path (ADR-0043 §3); None when off.

    A separate cheap ranker-model (estimator-agnostic subset, ADR-0043 §4). The ``strategy`` Literal is
    validated at config construction; ``null_importance`` works on every scheme now (M6d, ADR-0050 —
    structure-aware permutation via the spine-threaded label), so no resolve-time scheme guard.
    """
    if fs is None:
        return None
    # the single-strategy path is rankers only (sequential routes to the M6c compare path); reuse the
    # shared instantiation so the M6b and M6c paths can never drift in how a strategy is built.
    return cast(FeatureRanker, _make_strategy(task, fs, fs.strategy))


def _normalize_cv(cv: int | CVConfig | None) -> CVConfig:
    """Coerce the facade ``cv`` parameter to a ``CVConfig`` (``int`` = fold count)."""
    if cv is None:
        return CVConfig()
    if isinstance(cv, CVConfig):
        return cv
    if cv < 2:
        raise ConfigError(f"cv must be >= 2 (got {cv}); omit it for the default of 5")
    return CVConfig(scheme="auto", n_splits=cv)


def _resolve_splitter(
    cfg: CVConfig,
    task: Task,
    random_state: int,
    has_datetime: bool,
    has_group: bool,
    has_time: bool,
    es_fraction: float = 0.0,
) -> tuple[CVSplitter, CVConfig]:
    """Map a ``CVConfig`` to a concrete splitter; fail fast on unimplemented schemes.

    Returns the splitter and the *resolved* config (``auto`` replaced by the concrete
    scheme) so the run manifest records what actually ran (ADR-0016 §2). ``es_fraction`` (>0 only
    for the main run splitter when a boosting is in the zoo, ADR-0080) carves an early-stopping
    tail from each fold's train — i.i.d. by a stratified subsample, group by holding out whole groups.
    """
    scheme = task.default_cv_scheme if cfg.scheme == "auto" else cfg.scheme
    if (
        cfg.purge > 0
        or cfg.embargo > 0
        or cfg.purge_delta is not None
        or cfg.embargo_delta is not None
    ) and scheme not in _TIME_SCHEMES:
        raise ConfigError(
            "cv purge/embargo (rows/periods) and purge_delta/embargo_delta (Δt) require a time-series "
            "scheme ('timeseries' or 'timeseries_period'); set them to 0/None otherwise"
        )
    # period knobs are valid only under the period scheme; a stray one would be silently dead (FR-9)
    if (
        cfg.period is not None or cfg.period_size is not None or cfg.step_periods is not None
    ) and scheme != "timeseries_period":
        raise ConfigError("cv period/period_size/step_periods require scheme='timeseries_period'")
    # rolling lookback caps are scheme-specific units: periods vs rows (ADR-0099 §1); a mismatch is dead
    if cfg.max_train_periods is not None and scheme != "timeseries_period":
        raise ConfigError(
            "cv max_train_periods requires scheme='timeseries_period' (rolling lookback in periods)"
        )
    if cfg.max_train_size is not None and scheme != "timeseries":
        raise ConfigError(
            "cv max_train_size requires scheme='timeseries' (rolling lookback in rows)"
        )
    # period weighting macro-averages over periods/folds, so it needs a time-ordered scheme (block index)
    if cfg.weighting == "period" and scheme not in _TIME_SCHEMES:
        raise ConfigError(
            "cv weighting='period' requires a time-ordered scheme ('timeseries' or 'timeseries_period'); "
            "it macro-averages the score over periods/folds"
        )
    if scheme in _KFOLD_FAMILY and cfg.n_splits < 2:
        raise ConfigError(f"{scheme} cv requires n_splits >= 2 (got {cfg.n_splits})")

    splitter: CVSplitter
    if scheme == "stratified":
        splitter = StratifiedKFoldSplitter(
            n_splits=cfg.n_splits, shuffle=True, random_state=random_state, es_fraction=es_fraction
        )
    elif scheme == "kfold":
        splitter = KFoldSplitter(
            n_splits=cfg.n_splits, shuffle=True, random_state=random_state, es_fraction=es_fraction
        )
    elif scheme == "group":
        if not has_group:
            raise ConfigError("cv scheme 'group' requires a group column (none in the schema)")
        splitter = (
            StratifiedGroupKFoldSplitter(
                n_splits=cfg.n_splits,
                shuffle=True,
                random_state=random_state,
                es_fraction=es_fraction,
            )
            if task.is_classification
            else GroupKFoldSplitter(
                n_splits=cfg.n_splits, random_state=random_state, es_fraction=es_fraction
            )
        )
    elif scheme == "holdout":
        # regression cannot be stratified on a continuous target (ADR-0020 §4 regression path)
        splitter = HoldoutSplitter(
            shuffle=True,
            stratify=task.is_classification,
            random_state=random_state,
            es_fraction=es_fraction,
        )
    elif scheme == "timeseries":
        if not has_time:
            raise ConfigError("cv scheme 'timeseries' requires a time column (pass time= to fit)")
        splitter = TimeSeriesSplitter(
            n_splits=cfg.n_splits,
            n_test=cfg.n_test,
            n_es=cfg.n_es,
            purge=cfg.purge,
            embargo=cfg.embargo,
            purge_delta=cfg.purge_delta,
            embargo_delta=cfg.embargo_delta,
            max_train_size=cfg.max_train_size,
        )
    elif scheme == "timeseries_period":
        if not has_time:
            raise ConfigError(
                "cv scheme 'timeseries_period' requires a time column (pass time= to fit)"
            )
        if cfg.period is None:
            raise ConfigError(
                "cv scheme 'timeseries_period' requires a period unit (set cv.period to "
                "'month'/'week'/'day'/'delta')"
            )
        splitter = PeriodTimeSeriesSplitter(
            period=cfg.period,
            n_splits=cfg.n_splits,
            n_test=cfg.n_test,
            n_es=cfg.n_es,
            purge=cfg.purge,
            embargo=cfg.embargo,
            period_size=cfg.period_size,
            step_periods=cfg.step_periods,
            purge_delta=cfg.purge_delta,
            embargo_delta=cfg.embargo_delta,
            max_train_periods=cfg.max_train_periods,
        )
    else:  # any future scheme not yet implemented
        raise UnsupportedSchemeError(
            f"cv scheme {scheme!r} is not available yet; "
            "use 'stratified', 'kfold', 'group', 'holdout', 'timeseries', 'timeseries_period' or 'auto'"
        )

    # a time axis (datetime feature OR declared TIME role) under a shuffling scheme risks look-ahead
    if (has_datetime or has_time) and scheme in _SHUFFLING_SCHEMES:
        logger.warning(
            "cv scheme=%s shuffles rows but the data has a time axis; this risks look-ahead "
            "leakage. Use scheme='timeseries' with a declared time= column for time-safe CV.",
            scheme,
        )
    if has_group and scheme != "group":
        logger.warning(
            "a group column is present but cv scheme=%s is not group-aware; rows of the same "
            "group may span train and test (leakage). Use scheme='group' for group-safe CV.",
            scheme,
        )

    resolved = cfg if cfg.scheme != "auto" else cfg.model_copy(update={"scheme": scheme})
    return splitter, resolved


def _resolve_metric(task: Task, metric: str | Metric | None, classes: np.ndarray | None) -> Metric:
    """Resolve and validate the metric, carrying the global ``classes`` and binary ``positive`` (ADR-0021 §4).

    ``positive`` (``Task.positive_label``-aware) orients the binary proba metrics on ``P(positive)`` —
    without it they read the 1-D score as ``P(greatest label)`` and invert when ``positive`` is not
    the greatest label (F111).
    """
    positive = (
        resolve_positive(task, classes) if task.kind == "binary" and classes is not None else None
    )
    if metric is None:
        resolved = resolve_metric(task.target_metric, classes=classes, positive=positive)
    elif isinstance(metric, str):
        resolved = resolve_metric(metric, classes=classes, positive=positive)
    else:
        resolved = metric
    _validate_task_metric(task, resolved)
    return resolved


def _validate_task_metric(task: Task, metric: Metric) -> None:
    """Fail fast on an incompatible task/metric pairing (ADR-0021 §4), before CV/fit."""
    if task.is_classification and metric.needs == "value":
        raise ConfigError(
            f"value metric {metric.name!r} cannot score a {task.kind} classification task"
        )
    if task.kind == "regression" and metric.needs in _CLASS_NEEDS:
        raise ConfigError(
            f"metric {metric.name!r} (needs {metric.needs!r}) cannot score a regression task"
        )
    # keyed on task.kind (not classes.size) so the early guard fires even when classes is None
    if metric.name == "pr_auc" and task.kind == "multiclass":
        raise ConfigError("pr_auc is not supported for multiclass")


def _select_estimators(
    task: Task,
    metric: Metric,
    models: tuple[str, ...] | None,
    random_state: int,
    *,
    has_missing: bool = False,
) -> dict[str, EstimatorFactory]:
    """Resolve the registry, validate ``models``, filter by *static* capabilities.

    The filter reads ``descriptor.spec.capabilities`` — no ``factory()`` materialization, so
    an unselected heavy adapter is never imported (ADR-0019 §2). Default selection
    (``models=None``) also drops components whose extra is not installed (``is_available``,
    ADR-0020 §5); an explicitly requested but uninstalled model fails with
    ``MissingDependencyError``. With ``has_missing`` the data carries NaN: models declaring
    ``handles_missing=False`` are skipped with a WARNING instead of crashing mid-fit.
    """
    registry = model_registry()
    available = registry.by_name()

    if models is not None:
        unknown = [m for m in models if m not in available]
        if unknown:
            raise ConfigError(f"unknown models {unknown}; available: {sorted(available)}")
        missing = [m for m in models if not registry.is_available(m)]
        if missing:
            raise MissingDependencyError(missing[0])
        selected = [n for n in available if n in models]
    else:
        selected = [n for n in available if registry.is_available(n)]

    keep: dict[str, EstimatorFactory] = {}
    nan_skipped: list[str] = []
    for name in selected:
        caps = available[name].spec.capabilities
        if task.kind not in caps.tasks:
            continue
        if metric.needs in _PROBA_NEEDS and not caps.probabilistic:
            continue
        if has_missing and not caps.handles_missing:
            nan_skipped.append(name)
            continue
        keep[name] = _estimator_factory(registry, name, task, random_state)

    if nan_skipped:
        logger.warning(
            "data contains NaN in numeric features; skipping models that require imputed input: %s",
            nan_skipped,
        )
    if not keep:
        if nan_skipped:
            raise ConfigError(
                f"no estimator left for task {task.kind!r}: the data contains NaN and "
                f"{nan_skipped} require imputed input — impute the data or add a NaN-capable "
                'model (e.g. pip install "honestml[boosting]")'
            )
        raise ConfigError(f"no estimator supports task {task.kind!r} with metric {metric.name!r}")
    # plugin contract guard (FR-2): a third-party model declaring handles_cat=True must implement the
    # SupportsNativeCategorical marker to be routed natively; otherwise warn and let it train on codes
    # (the marker drives routing, not the static flag). Built-ins are aligned by construction — skip them
    # so the check never eagerly imports a heavy boosting adapter (laziness, ADR-0019 §2).
    for name, factory in keep.items():
        descriptor = available[name]
        if descriptor.dist == _BUILTIN_DIST or not descriptor.spec.capabilities.handles_cat:
            continue
        if not isinstance(factory(), SupportsNativeCategorical):
            logger.warning(
                "model %r declares handles_cat=True but its estimator does not implement "
                "SupportsNativeCategorical; its categorical features fall back to numeric codes",
                name,
            )
    return keep


def _any_early_stopping(estimators: Mapping[str, EstimatorFactory]) -> bool:
    """Whether any selected model declares ``supports_early_stopping`` (ADR-0080), read from the
    registry descriptors without materializing a factory (so no heavy boosting import here)."""
    by_name = model_registry().by_name()
    return any(
        by_name[n].spec.capabilities.supports_early_stopping for n in estimators if n in by_name
    )


def _estimator_factory(
    registry: ComponentRegistry, name: str, task: Task, random_state: int
) -> EstimatorFactory:
    """A zero-arg factory that lazily materializes ``name`` from the registry."""

    def make() -> Estimator:
        return registry.build(name, task=task, random_state=random_state)

    return make
