"""The ``tune_estimators`` use-case: honest inner-CV HPO over the Tuner port (ADR-0062).

For each tunable model type, builds an inner-CV objective on DEV and lets the injected :class:`Tuner`
search it; the tuned factory then competes in the outer honest selection (``run_slice``) unchanged
(ADR-0062 §2/§3). The objective REUSES ``run_slice``'s per-fold engine (:func:`_run_candidate`) plus a
SEPARATE out-of-fold target-encoding step (:func:`_augment_oof_te` on an INNER fold index) — because
``_run_candidate`` alone does no TE and the full-train TE would leak the target into the search
(ADR-0062 §2, R2 fix). With ``selected_features`` set (the post-FS subset, ADR-0102 superseding
ADR-0062 §2a) the objective is projected to that width AFTER the TE step — the TE positional lookups
assume the full matrix; ``None`` keeps the legacy full-DEV-width objective statement-for-statement.
``sample_weight`` weights inner fit AND inner score, matching the weighted leaderboard. Budget is
cooperative with graceful degradation (best-so-far / baseline, §5).
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import numpy as np

from honestml.core import (
    Budget,
    Dataset,
    Estimator,
    FEConfig,
    Fold,
    Metric,
    RunContext,
    SearchConfig,
    SelectionPolicy,
    Task,
    TuneOutcome,
    Tuner,
    get_logger,
    parse_search_space,
    resolve_positive,
)
from honestml.core.exceptions import AutoMLError, BudgetExhaustedError
from honestml.core.ports.estimator import (
    SupportsEarlyStopping,
    SupportsIterationBudget,
    SupportsIterationPlan,
    SupportsThreadLimit,
)
from honestml.core.ports.splitter import CVSplitter, TimeOrderedSplitter
from honestml.core.ports.tuner import (
    CategoricalParam,
    FloatParam,
    IntParam,
    ReportsTuningTrial,
    SupportsTuningCache,
)
from honestml.core.schema import categorical_positions, native_routable

from .projection import _PROBA_NEEDS
from .search import bounded_probe_folds
from .slice import (
    EstimatorFactory,
    _augment_oof_te,
    _CandidateFailed,
    _fold_index,
    _run_candidate,
    design_matrix,
    project_by_name,
)

# name -> tuned EstimatorFactory builder (closes over registry/task/seed in composition, ADR-0062 §2)
MakeFactory = Callable[[str, Mapping[str, Any]], EstimatorFactory]


def _timeout(hpo_timeout_s: float | None, budget: Budget | None, n_remaining: int) -> float | None:
    """The per-model wall-clock cap (ADR-0062 §5): the tighter of the HPO timeout and a FAIR SHARE of
    the run budget's time left (``time_left / n_remaining``) so the first model cannot starve the rest."""
    cap = hpo_timeout_s
    if budget is not None:
        left = budget.time_left()
        if left != float("inf"):
            share = left / n_remaining
            cap = share if cap is None else min(cap, share)
    return cap


@dataclass(frozen=True)
class _ObjectiveData:
    y: np.ndarray
    classes: np.ndarray
    positive: Any
    x_eval: np.ndarray
    feature_names: list[str]
    categorical_indices: list[int]
    inner_folds: list[Fold]


def _prepare_objective(
    ds_dev: Dataset,
    task: Task,
    inner_splitter: CVSplitter,
    fe: FEConfig | None,
    selected_features: tuple[str, ...] | None,
) -> _ObjectiveData:
    y = ds_dev.target()
    if y is None:
        raise ValueError("tune_estimators requires a target column")
    schema = ds_dev.schema
    classes = np.unique(y)
    positive = resolve_positive(task, classes) if task.kind == "binary" else None

    # full DEV feature space; the post-FS projection (ADR-0102) happens AFTER the TE step below
    x_full = design_matrix(ds_dev)
    feature_names = list(schema.features)
    # native-categorical routing (ADR-0088/0092, FR-1/FR-2): the cardinality-GATED CATEGORICAL-column
    # positions are taken over the full feature list via the same gate run_slice/refit_best use (cap
    # from the task), keeping CV/refit/HPO routing identical (R-3); recomputed after any projection.
    categorical_indices = categorical_positions(
        feature_names, native_routable(schema, task.native_cat_max_unique)
    )
    inner_folds = list(inner_splitter.split(ds_dev))

    # OOF target-encoding for the inner objective (ADR-0062 §2): a SEPARATE step keyed on the INNER
    # fold index — `_run_candidate` does no TE, and the full-train TE columns would leak the target.
    te_on = fe is not None and fe.target_encoding and schema.target_encoding is not None
    x_eval = x_full
    if te_on:
        assert fe is not None and positive is not None
        # a time-ordered inner CV uses the expanding-window encoder (each fold from strictly earlier inner
        # folds, no look-ahead, ADR-0082); an IID inner CV uses the plain cross-fit (ADR-0041 §1).
        x_eval = _augment_oof_te(
            x_full,
            ds_dev,
            y,
            positive,
            _fold_index(y.shape[0], inner_folds),
            fe.te_smoothing,
            feature_names,
            time_ordered=isinstance(inner_splitter, TimeOrderedSplitter),
        )

    # post-FS projection (ADR-0102): strictly AFTER the TE step (its positional lookups assume the
    # full-width matrix) and by NAME in schema.features order — the same rule design_matrix applies —
    # with the routing indices recomputed by the same gate, so CV/refit/HPO routing stays identical (R-3).
    if selected_features is not None:
        keep = project_by_name(feature_names, selected_features)
        x_eval = x_eval[:, keep]
        feature_names = [feature_names[i] for i in keep]
        categorical_indices = categorical_positions(
            feature_names, native_routable(schema, task.native_cat_max_unique)
        )

    return _ObjectiveData(
        y, classes, positive, x_eval, feature_names, categorical_indices, inner_folds
    )


def tune_estimators(
    ds_dev: Dataset,
    task: Task,
    *,
    tunable: Mapping[str, dict[str, Any]],
    make_factory: MakeFactory,
    tuner: Tuner,
    metric: Metric,
    policy: SelectionPolicy,
    inner_splitter: CVSplitter,
    n_trials: int,
    timeout_s: float | None,
    random_state: int,
    fe: FEConfig | None = None,
    sample_weight: np.ndarray | None = None,
    budget: Budget | None = None,
    ctx: RunContext | None = None,
    selected_features: tuple[str, ...] | None = None,
) -> dict[str, TuneOutcome]:
    """Tune each model type on an inner-CV of DEV; return ``name -> TuneOutcome`` (ADR-0062 §2)."""
    logger = ctx.logger if ctx is not None else get_logger()
    prepared = _prepare_objective(ds_dev, task, inner_splitter, fe, selected_features)
    y, classes, positive = prepared.y, prepared.classes, prepared.positive
    x_eval, feature_names = prepared.x_eval, prepared.feature_names
    categorical_indices, inner_folds = prepared.categorical_indices, prepared.inner_folds
    n_features = len(feature_names)

    need_proba = metric.needs in _PROBA_NEEDS
    worst = float("-inf") if policy.greater_is_better else float("inf")
    # tunable models with a non-empty (valid) space; the empty ones keep their baseline (ADR-0062 §1)
    to_tune = [(name, sp) for name, raw in tunable.items() if (sp := parse_search_space(raw))]
    outcomes: dict[str, TuneOutcome] = {}

    for i, (name, space) in enumerate(to_tune):
        # HPO is gated by the run budget on the TIME axis only (ADR-0062 §5): a tighter (time/memory)
        # exhaustion skips the rest, keeping their baseline. HPO does NOT consume() a candidate trial —
        # that is the run_slice candidate-loop axis (ADR-0062 §6); its wall-clock time is billed by the
        # shared time-mode clock automatically (RunBudget.time_left is clock-derived, not consume-driven).
        if budget is not None and budget.exhausted:
            logger.warning("HPO: budget exhausted before tuning %r; keeping the baseline", name)
            continue

        def score(params: Mapping[str, Any], _name: str = name) -> float:
            factory = make_factory(_name, params)
            try:
                cand = _run_candidate(
                    _name,
                    factory,
                    x_full=x_eval,
                    y=y,
                    feature_names=feature_names,
                    categorical_indices=categorical_indices,
                    kind=task.kind,
                    positive=positive,
                    global_classes=classes,
                    metric=metric,
                    folds=inner_folds,
                    sample_weight=sample_weight,
                    n_features=n_features,
                    need_proba=need_proba,
                    capture_oof=False,
                    capture_proba=False,
                    # the HPO inner objective stays pooled (ADR-0098 is the outer leaderboard/band only);
                    # keeps the tuning score byte-identical to before the feature (NFR-5).
                    block_index=None,
                    weighting="pooled",
                    logger=logger,
                    ctx=ctx,
                    stage="hpo",
                    trial=tuner.current_trial_number
                    if isinstance(tuner, ReportsTuningTrial)
                    else None,
                )
            except _CandidateFailed:
                return worst  # an invalid hyper-combo: steer the search away, do not crash
            return cand.score

        if isinstance(tuner, SupportsTuningCache):
            tuner.set_search_context(name, tuple(feature_names))
        outcomes[name] = tuner.tune(
            space,
            score,
            max_trials=n_trials,
            timeout_s=_timeout(timeout_s, budget, len(to_tune) - i),
            greater_is_better=policy.greater_is_better,
            random_state=random_state,
        )
    return outcomes


def _central_parameters(raw: Mapping[str, Any]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for name, spec in parse_search_space(raw).items():
        if isinstance(spec, IntParam):
            params[name] = spec.low + ((spec.high - spec.low) // spec.step // 2) * spec.step
        elif isinstance(spec, FloatParam):
            params[name] = (
                math.exp((math.log(spec.low) + math.log(spec.high)) / 2)
                if spec.log
                else spec.low / 2 + spec.high / 2
            )
        elif isinstance(spec, CategoricalParam):
            params[name] = spec.choices[len(spec.choices) // 2]
        else:
            raise ValueError("unsupported parameter specification")
    json.dumps(params, allow_nan=False)
    return params


def _cost_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode("utf-8")
    ).hexdigest()


def _partition_profile(folds: list[Fold]) -> list[dict[str, object]]:
    return [
        {
            label: {
                "rows": len(indices),
                "sha256": hashlib.sha256(np.asarray(indices, dtype=np.int64).tobytes()).hexdigest(),
            }
            for label, indices in (
                ("fit", fold.fit_idx),
                ("es", fold.es_idx),
                ("test", fold.test_idx),
            )
        }
        for fold in folds
    ]


def _scale_tuning_time(
    elapsed: float, rows: int, iterations: int, cap: int, target_rows: int, target_cap: int
) -> float:
    scale = max(1.0, target_cap / iterations) if iterations >= cap else 1.0
    return elapsed * target_rows / rows * scale


def profile_tuning_cost(
    ds_dev: Dataset,
    task: Task,
    *,
    tunable: Mapping[str, dict[str, Any]],
    make_factory: MakeFactory,
    metric: Metric,
    policy: SelectionPolicy,
    inner_splitter: CVSplitter,
    n_trials: int,
    timeout_s: float | None,
    random_state: int,
    search: SearchConfig,
    fe: FEConfig | None = None,
    sample_weight: np.ndarray | None = None,
    ctx: RunContext | None = None,
    can_start: Callable[[], bool] | None = None,
) -> dict[str, dict[str, object]]:
    """Measure a conditional wide-feature HPO scenario without invoking the tuner.

    Central search-space parameters are held fixed across two bounded DEV resources.
    The projection assumes all requested trials use those parameters; it is neither
    an expected duration over the search space nor a wall-time budget guarantee.
    """
    if not tunable:
        return {}
    result: dict[str, dict[str, object]] = {}
    parameters: dict[str, dict[str, Any]] = {}
    for name, raw in tunable.items():
        entry: dict[str, object] = {
            "status": "unknown",
            "estimated_s": None,
            "planned_fit_count": None,
            "planned_trials": n_trials,
            "timeout_limit_s": timeout_s,
            "assumptions": [
                "all_requested_trials_complete",
                "central_parameters_for_every_trial",
                "wide_features_unchanged",
                "linear_training_row_scaling",
                "native_ceiling_scaling_only_when_probe_reaches_ceiling",
                "resource_dependent_native_defaults_may_change",
            ],
            "includes": ["inner_fit", "inner_predict", "inner_score"],
            "excludes": [
                "inner_preparation_and_target_encoding",
                "tuner_and_checkpoint_overhead",
                "fs",
                "feature_width_changes",
                "parameter_search_variation",
                "selection_cv",
                "refit",
                "total_run_wall",
            ],
            "profiles": [],
            "validation": None,
        }
        result[name] = entry
        try:
            params = _central_parameters(raw)
        except (AutoMLError, ValueError, TypeError) as exc:
            entry["reason"] = f"unsupported_parameters:{type(exc).__name__}"
            continue
        if not params or n_trials <= 0:
            entry.update(
                status="disabled",
                estimated_s=0.0,
                planned_fit_count=0,
                reason="no_tunable_parameters" if not params else "no_trials",
            )
            continue
        parameters[name] = params
    if not parameters:
        return result
    try:
        prepared = _prepare_objective(ds_dev, task, inner_splitter, fe, None)
    except (AutoMLError, ValueError, TypeError, RuntimeError, MemoryError) as exc:
        for name in parameters:
            result[name]["reason"] = f"preparation_failed:{type(exc).__name__}"
        return result
    for name, params in parameters.items():
        entry = result[name]
        entry["planned_fit_count"] = n_trials * len(prepared.inner_folds)
        entry["features_sha256"] = _cost_hash(prepared.feature_names)
        profiles: list[dict[str, object]] = []
        entry["profiles"] = profiles
        try:
            base_factory = make_factory(name, params)
            model = base_factory()
            if not isinstance(model, SupportsIterationBudget):
                entry["reason"] = "unsupported_iteration_protocol"
                continue
            if not isinstance(model, SupportsIterationPlan):
                entry["reason"] = "unsupported_iteration_protocol"
                continue
            if not isinstance(model, SupportsThreadLimit):
                entry["reason"] = "unsupported_thread_protocol"
                continue
            uses_es = isinstance(model, SupportsEarlyStopping)
            full_cap = model.iteration_limit(
                early_stopping=uses_es and any(f.es_idx.size for f in prepared.inner_folds)
            )
            planned_rows = (
                sum(
                    len(f.fit_idx) + (0 if uses_es else len(f.es_idx)) for f in prepared.inner_folds
                )
                * n_trials
            )
            entry["planned_training_rows"] = planned_rows
            entry["full_iteration_cap"] = full_cap
            if not prepared.inner_folds or full_cap < 1:
                entry["reason"] = "infeasible_inner_resource"
                continue
            measurements: list[tuple[float, int, int, int]] = []
            with ctx.timed_stage("run", "hpo_cost_probe") if ctx is not None else nullcontext():
                for label, rows, fold_count, iterations in (
                    ("initial", search.max_rows, search.max_folds, search.model_iterations),
                    (
                        "confirmation",
                        search.confirmation_rows,
                        search.confirmation_folds,
                        search.confirmation_iterations,
                    ),
                ):
                    if can_start is not None and not can_start():
                        raise BudgetExhaustedError(
                            "trials", completed=len(measurements), skipped=1, failed=0
                        )
                    bounded = bounded_probe_folds(
                        prepared.inner_folds,
                        y=prepared.y,
                        task=task,
                        config=search.model_copy(
                            update={"max_rows": rows, "max_folds": fold_count}
                        ),
                        seed=random_state,
                    )
                    if not bounded:
                        raise ValueError("infeasible probe partitions")
                    folds = [
                        f
                        if uses_es
                        else Fold(
                            np.concatenate((f.fit_idx, f.es_idx)),
                            np.empty(0, dtype=np.int64),
                            f.test_idx,
                        )
                        for f in bounded
                    ]
                    cap = min(
                        iterations,
                        model.iteration_limit(
                            early_stopping=uses_es and any(f.es_idx.size for f in folds)
                        ),
                    )
                    if cap < 1:
                        raise ValueError("invalid native iteration ceiling")
                    partitions = _partition_profile(folds)
                    partition_hash = _cost_hash(partitions)
                    if profiles and (
                        profiles[-1]["partitions_sha256"] == partition_hash
                        and profiles[-1]["iteration_cap"] == cap
                    ):
                        entry["reason"] = "identical_probe_resources"
                        break

                    def factory() -> Estimator:
                        estimator = base_factory()
                        if not isinstance(estimator, SupportsIterationBudget):
                            raise ValueError("factory changed iteration capability")
                        if not isinstance(estimator, SupportsThreadLimit):
                            raise ValueError("factory changed thread capability")
                        estimator.set_refit_iterations(cap)
                        estimator.set_threads(search.threads)
                        return estimator

                    candidate = _run_candidate(
                        f"{name}__cost_hpo",
                        factory,
                        x_full=prepared.x_eval,
                        y=prepared.y,
                        feature_names=prepared.feature_names,
                        categorical_indices=prepared.categorical_indices,
                        kind=task.kind,
                        positive=prepared.positive,
                        global_classes=prepared.classes,
                        metric=metric,
                        folds=folds,
                        sample_weight=sample_weight,
                        n_features=len(prepared.feature_names),
                        need_proba=metric.needs in _PROBA_NEEDS,
                        capture_oof=False,
                        capture_proba=False,
                        block_index=None,
                        weighting="pooled",
                        logger=ctx.logger if ctx is not None else get_logger(),
                        ctx=ctx,
                        stage="scouting",
                    )
                    count = candidate.refit_iterations
                    if (
                        count is None
                        or count < 1
                        or not np.isfinite(candidate.score)
                        or not np.isfinite(candidate.train_time)
                        or candidate.train_time <= 0
                    ):
                        raise ValueError("invalid completed profile measurement")
                    train_rows = sum(len(f.fit_idx) for f in folds)
                    profile: dict[str, object] = {
                        "level": label,
                        "model_id": candidate.id,
                        "params": dict(params),
                        "params_sha256": _cost_hash(params),
                        "features_sha256": _cost_hash(prepared.feature_names),
                        "feature_count": len(prepared.feature_names),
                        "partitions": partitions,
                        "partitions_sha256": partition_hash,
                        "training_rows": train_rows,
                        "fit_count": len(folds),
                        "iteration_cap": cap,
                        "native_iterations": count,
                        "native_iterations_aggregation": "median_of_completed_folds",
                        "elapsed_s": candidate.train_time,
                        "score": candidate.score,
                        "greater_is_better": policy.greater_is_better,
                        "threads": search.threads,
                    }
                    profiles.append(profile)
                    measurements.append((candidate.train_time, train_rows, count, cap))
            if len(measurements) != 2:
                continue
            low, high = measurements
            predicted = _scale_tuning_time(*low, high[1], high[3])
            estimated = _scale_tuning_time(*high, planned_rows, full_cap)
            if not np.isfinite(predicted) or not np.isfinite(estimated):
                raise ValueError("non-finite extrapolation")
            error = high[0] - predicted
            entry["validation"] = {
                "predicted_s": predicted,
                "actual_s": high[0],
                "error_s": error,
                "relative_error": error / predicted if predicted > 0 else None,
            }
            entry["estimated_s"] = estimated
            entry["status"] = "conditional"
        except BudgetExhaustedError:
            entry["reason"] = "probe_budget"
        except (
            AutoMLError,
            _CandidateFailed,
            ValueError,
            TypeError,
            RuntimeError,
            MemoryError,
        ) as exc:
            entry["reason"] = f"profile_failed:{type(exc).__name__}"
    return result
