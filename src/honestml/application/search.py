"""Resource-bounded DEV probes with independent recipe confirmation rows."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass, field, replace
from typing import cast

import numpy as np

from honestml.core import FeatureRanker, FeatureSelectionConfig, Fold, Metric, RunContext, Task
from honestml.core.config import SearchConfig
from honestml.core.exceptions import AutoMLError, BudgetExhaustedError, FitFailedError
from honestml.core.ports.significance import SignificanceTest
from honestml.core.selection_policy import Candidate, SelectionPolicy, rank

from .feature_compare import Strategy, _select_one, _strategy_seed
from .feature_selection import (
    BoundedFeatureRanker,
    bounded_fit_predict,
    estimate_fs_refits,
    sample_training_rows,
    select_features,
)
from .oof_scorer import FitPredict, OOFSubsetCache


@dataclass(frozen=True)
class ModelProbeOutcome:
    winner: str
    folds: tuple[Fold, ...]
    candidates: tuple[Candidate, ...]
    failures: tuple[tuple[str, str], ...]
    reason: str
    skipped: tuple[str, ...] = ()
    confirmations: tuple[Candidate, ...] = ()
    confirmation_failures: tuple[tuple[str, str], ...] = ()
    diagnostics: dict[str, list[dict[str, object]]] = field(default_factory=dict)
    issues: tuple[str, ...] = ()
    rank_changed: bool = False
    cost_estimates: dict[str, float] = field(default_factory=dict)
    initial_cost_estimates: dict[str, float] = field(default_factory=dict)
    cost_model: dict[str, dict[str, object]] = field(default_factory=dict)
    completion_costs: dict[str, dict[str, object]] = field(default_factory=dict)


@dataclass(frozen=True)
class FeatureProbeOutcome:
    winner: str | None
    subset: tuple[int, ...]
    candidates: tuple[Candidate, ...]
    failures: tuple[tuple[str, str], ...]
    reason: str
    skipped: tuple[str, ...] = ()


def _sample_partition(
    indices: np.ndarray,
    y: np.ndarray,
    limit: int,
    *,
    classification: bool,
    rng: np.random.Generator,
) -> np.ndarray | None:
    if len(indices) <= limit:
        return indices.copy()
    if limit < 1 or (classification and np.unique(y[indices]).size > limit):
        return None
    selected = sample_training_rows(
        y[indices],
        max_rows=limit,
        task=Task(kind="multiclass" if classification else "regression"),
        random_state=int(rng.integers(0, 2**31)),
    )
    return indices[selected]


def bounded_probe_folds(
    folds: Sequence[Fold],
    *,
    y: np.ndarray,
    task: Task,
    config: SearchConfig,
    seed: int,
) -> tuple[Fold, ...]:
    """Subsample within original partitions; never move rows across fit/ES/test boundaries.

    Subsets inherit group disjointness, temporal ordering and purge/embargo exclusions. Each
    probe keeps every class present in each original partition and requires all DEV classes in
    fit and test. An infeasible cap yields no probes. The aggregate partition budget across all
    retained folds is at most ``max_rows``; overlap across different folds is counted repeatedly.
    """
    count = min(len(folds), config.max_folds)
    if not count:
        return ()
    rng = np.random.default_rng(seed)
    classes = np.unique(y) if task.is_classification else None
    bounded: list[Fold] = []
    per_fold = config.max_rows // count
    for position in np.linspace(0, len(folds) - 1, count, dtype=int):
        fold = folds[position]
        parts = (fold.fit_idx, fold.es_idx, fold.test_idx)
        fractions = (0.6, 0.2, 0.2) if fold.es_idx.size else (0.8, 0.0, 0.2)
        caps = [int(per_fold * fraction) for fraction in fractions]
        caps[0] += per_fold - sum(caps)
        sampled = [
            _sample_partition(part, y, cap, classification=task.is_classification, rng=rng)
            for part, cap in zip(parts, caps, strict=True)
        ]
        if any(part is None for part in sampled):
            continue
        fit, es, test = sampled
        assert fit is not None and es is not None and test is not None
        if not fit.size or not test.size:
            continue
        if classes is not None and any(
            not np.array_equal(np.unique(y[part]), classes) for part in (fit, test)
        ):
            continue
        bounded.append(Fold(fit, es, test))
    return tuple(bounded)


def _partition_diagnostics(
    indices: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray | None,
    times: np.ndarray | None,
    task: Task,
) -> dict[str, object]:
    classes, counts = (
        np.unique(y[indices], return_counts=True) if task.is_classification else ([], [])
    )
    return {
        "rows": len(indices),
        "class_counts": dict(zip((str(value) for value in classes), map(int, counts), strict=True)),
        "groups": int(np.unique(groups[indices]).size) if groups is not None else None,
        "time_values": int(np.unique(times[indices]).size) if times is not None else None,
    }


def _probe_diagnostics(
    folds: Sequence[Fold],
    y: np.ndarray,
    groups: np.ndarray | None,
    times: np.ndarray | None,
    task: Task,
    config: SearchConfig,
) -> tuple[list[dict[str, object]], tuple[str, ...]]:
    diagnostics: list[dict[str, object]] = []
    issues: set[str] = set()
    for fold in folds:
        record: dict[str, object] = {}
        for name, indices in (("fit", fold.fit_idx), ("es", fold.es_idx), ("test", fold.test_idx)):
            record[name] = _partition_diagnostics(indices, y, groups, times, task)
            if name == "es" and 0 < len(indices) < config.min_es_rows:
                issues.add("small_es")
            if task.is_classification and len(indices):
                _, counts = np.unique(y[indices], return_counts=True)
                if counts.min() < config.min_class_count or len(counts) != len(np.unique(y)):
                    issues.add(f"few_{name}_class_rows")
        diagnostics.append(record)
    if times is not None and len(folds) < 2:
        issues.add("single_time_window")
    return diagnostics, tuple(sorted(issues))


def _completion_cost(
    candidate: Candidate,
    probe_folds: Sequence[Fold],
    completion_rows: int,
    iteration_cap: int,
    full_cap: int | None,
) -> float:
    probe_rows = sum(len(f.fit_idx) for f in probe_folds)
    iterations = candidate.refit_iterations
    iteration_scale = (
        max(1.0, full_cap / iterations)
        if full_cap is not None and iterations is not None and iterations >= iteration_cap
        else 1.0
    )
    return candidate.train_time * completion_rows / max(1, probe_rows) * iteration_scale


def _cost_winner(
    candidates: Sequence[Candidate],
    costs: dict[str, float],
    *,
    y: np.ndarray,
    metric: Metric,
    margin: float,
    test: SignificanceTest | None,
    block_index: np.ndarray | None,
    sample_weight: np.ndarray | None,
) -> str:
    anchor = rank(candidates, SelectionPolicy(greater_is_better=metric.greater_is_better))[0]
    sign = 1.0 if metric.greater_is_better else -1.0
    eligible = [anchor]
    for candidate in candidates:
        if candidate.id == anchor.id or sign * (anchor.score - candidate.score) > margin:
            continue
        if (
            test is None
            or candidate.oof_pred is None
            or anchor.oof_pred is None
            or candidate.oof_mask is None
            or anchor.oof_mask is None
        ):
            continue
        mask = candidate.oof_mask & anchor.oof_mask
        if not mask.any():
            continue
        if test.noninferior(
            candidate.oof_pred[mask],
            anchor.oof_pred[mask],
            y[mask],
            alpha=0.05,
            margin=margin,
            block_index=block_index[mask] if block_index is not None else None,
            sample_weight=sample_weight[mask] if sample_weight is not None else None,
        ):
            eligible.append(candidate)
    return min(eligible, key=lambda candidate: (costs[candidate.id], candidate.id)).id


def probe_models(
    names: Sequence[str],
    folds: Sequence[Fold],
    *,
    y: np.ndarray,
    groups: np.ndarray | None,
    task: Task,
    metric: Metric,
    config: SearchConfig,
    seed: int,
    evaluate: Callable[[str, Sequence[Fold], int], Candidate],
    can_start: Callable[[], bool] | None = None,
    significance_test: SignificanceTest | None = None,
    sample_weight: np.ndarray | None = None,
    block_index: np.ndarray | None = None,
    times: np.ndarray | None = None,
    full_iterations: Mapping[str, int | None] | None = None,
    full_training_rows: Mapping[str, int] | None = None,
    completion_refit_rows: tuple[int, ...] = (),
    profile_completion: Callable[[tuple[str, ...]], Mapping[str, Mapping[str, object]]]
    | None = None,
) -> ModelProbeOutcome:
    """Confirm at most two families on richer DEV probes before full-resource selection."""
    bounded = bounded_probe_folds(folds, y=y, task=task, config=config, seed=seed)
    initial_diagnostics, initial_issues = _probe_diagnostics(
        bounded, y, groups, times, task, config
    )
    diagnostics = {"initial": initial_diagnostics}
    if len(names) == 1:
        return ModelProbeOutcome(names[0], bounded, (), (), "single_model", diagnostics=diagnostics)
    if not bounded:
        return ModelProbeOutcome(
            names[0], (), (), (), "infeasible_probe_folds", diagnostics=diagnostics
        )

    def run_round(
        pool: Sequence[str],
        parts: Sequence[Fold],
        iterations: int,
    ) -> tuple[list[Candidate], list[tuple[str, str]], bool]:
        candidates: list[Candidate] = []
        failures: list[tuple[str, str]] = []
        for name in pool:
            if can_start is not None and not can_start():
                return candidates, failures, False
            try:
                candidate = evaluate(name, parts, iterations)
                if not np.isfinite(candidate.score):
                    raise ValueError("non-finite probe score")
                candidates.append(replace(candidate, id=name))
            except BudgetExhaustedError:
                return candidates, failures, False
            except (AutoMLError, ValueError, RuntimeError, MemoryError) as exc:
                failures.append((name, str(exc)))
        return candidates, failures, True

    def native_cap(name: str, requested: int) -> int:
        cap = full_iterations.get(name) if full_iterations is not None else None
        return min(cap, requested) if cap is not None else requested

    def completion_rows(name: str) -> int:
        training_rows = (
            full_training_rows[name]
            if full_training_rows is not None
            else sum(len(f.fit_idx) for f in folds)
        )
        return training_rows + sum(completion_refit_rows)

    def estimate(
        pool: Sequence[Candidate], parts: Sequence[Fold], requested: int
    ) -> dict[str, float]:
        return {
            c.id: _completion_cost(
                c,
                parts,
                completion_rows(c.id),
                native_cap(c.id, requested),
                full_iterations.get(c.id) if full_iterations is not None else None,
            )
            for c in pool
        }

    candidates, failures, complete = run_round(names, bounded, config.model_iterations)
    initial_costs = estimate(candidates, bounded, config.model_iterations)
    failed_names = {name for name, _ in failures}
    if len(failed_names) == len(names):
        raise FitFailedError(failures)
    attempted = failed_names | {candidate.id for candidate in candidates}
    skipped = tuple(name for name in names if name not in attempted)
    if not complete:
        return ModelProbeOutcome(
            next(name for name in names if name not in failed_names),
            bounded,
            tuple(candidates),
            tuple(failures),
            "incomplete_probe_budget",
            skipped,
            diagnostics=diagnostics,
            initial_cost_estimates=initial_costs,
            issues=initial_issues,
        )
    ordered = rank(candidates, SelectionPolicy(greater_is_better=metric.greater_is_better))
    first_winner = ordered[0].id
    if len(ordered) == 1:
        return ModelProbeOutcome(
            first_winner,
            bounded,
            tuple(candidates),
            tuple(failures),
            "probe_score_with_failures",
            diagnostics=diagnostics,
            initial_cost_estimates=initial_costs,
            issues=initial_issues,
        )
    confirmation_config = config.model_copy(
        update={
            "max_rows": config.confirmation_rows,
            "max_folds": config.confirmation_folds,
        }
    )
    confirmation = bounded_probe_folds(folds, y=y, task=task, config=confirmation_config, seed=seed)
    confirmation_diagnostics, issues = _probe_diagnostics(
        confirmation, y, groups, times, task, config
    )
    diagnostics["confirmation"] = confirmation_diagnostics
    if not confirmation:
        return ModelProbeOutcome(
            first_winner,
            bounded,
            tuple(candidates),
            tuple(failures),
            "infeasible_confirmation",
            diagnostics=diagnostics,
            initial_cost_estimates=initial_costs,
            issues=issues,
        )
    confirmed, confirm_failures, confirmed_complete = run_round(
        [c.id for c in ordered[:2]], confirmation, config.confirmation_iterations
    )
    costs = estimate(confirmed, confirmation, config.confirmation_iterations)
    initial_by_name = {c.id: c for c in candidates}
    cost_model: dict[str, dict[str, object]] = {}
    for candidate in confirmed:
        first = initial_by_name[candidate.id]
        predicted = _completion_cost(
            first,
            bounded,
            sum(len(f.fit_idx) for f in confirmation),
            native_cap(candidate.id, config.model_iterations),
            native_cap(candidate.id, config.confirmation_iterations)
            if full_iterations is not None and full_iterations.get(candidate.id) is not None
            else None,
        )
        error = candidate.train_time - predicted
        cost_model[candidate.id] = {
            "cv_training_rows": completion_rows(candidate.id) - sum(completion_refit_rows),
            "initial_training_rows": sum(len(f.fit_idx) for f in bounded),
            "confirmation_training_rows": sum(len(f.fit_idx) for f in confirmation),
            "initial_iteration_cap": native_cap(candidate.id, config.model_iterations)
            if full_iterations is not None and full_iterations.get(candidate.id) is not None
            else None,
            "confirmation_iteration_cap": native_cap(candidate.id, config.confirmation_iterations)
            if full_iterations is not None and full_iterations.get(candidate.id) is not None
            else None,
            "full_iteration_cap": full_iterations.get(candidate.id)
            if full_iterations is not None
            else None,
            "initial_elapsed_s": first.train_time,
            "predicted_confirmation_s": predicted,
            "actual_confirmation_s": candidate.train_time,
            "confirmation_error_s": error,
            "confirmation_relative_error": error / predicted if predicted > 0 else None,
        }
    if not confirmed_complete or confirm_failures:
        return ModelProbeOutcome(
            first_winner,
            bounded,
            tuple(candidates),
            tuple(failures),
            "incomplete_confirmation_budget" if not confirmed_complete else "confirmation_failed",
            confirmations=tuple(confirmed),
            confirmation_failures=tuple(confirm_failures),
            diagnostics=diagnostics,
            initial_cost_estimates=initial_costs,
            cost_estimates=costs,
            cost_model=cost_model,
            issues=issues,
        )
    completion_costs: dict[str, dict[str, object]] = {}
    selection_costs = costs
    incomplete_cost = False
    if profile_completion is not None:
        profiles = profile_completion(tuple(candidate.id for candidate in confirmed))
        selection_costs = {}
        for candidate in confirmed:
            profile = dict(profiles.get(candidate.id, {}))
            fs = cast(Mapping[str, object], profile.get("fs", {}))
            hpo = cast(Mapping[str, object], profile.get("hpo", {}))
            fs_s = cast(float | None, fs.get("estimated_s"))
            hpo_s = cast(float | None, hpo.get("estimated_s"))
            count = cast(int, profile.get("additional_cv_count", 0))
            cv_rows = completion_rows(candidate.id) - sum(completion_refit_rows)
            extra_cv = costs[candidate.id] * cv_rows / max(1, completion_rows(candidate.id)) * count
            total = (
                costs[candidate.id] + extra_cv + fs_s + hpo_s
                if fs_s is not None and hpo_s is not None
                else None
            )
            completion_costs[candidate.id] = {
                **profile,
                "status": "conditional" if total is not None else "unavailable",
                "estimated_s": total,
                "cv_refit_s": costs[candidate.id],
                "additional_cv_s": extra_cv,
                "additional_cv_count": count,
                "excludes": [
                    "already_spent_probes",
                    "global_statistics",
                    "hpo_preparation_and_sampler",
                    "ensemble",
                    "io",
                    "total_run_wall",
                ],
            }
            if total is None:
                incomplete_cost = True
            else:
                selection_costs[candidate.id] = total
    anchor = rank(confirmed, SelectionPolicy(greater_is_better=metric.greater_is_better))[0]
    blocks = block_index
    if blocks is None and groups is not None:
        _, blocks = np.unique(groups, return_inverse=True)
    winner = (
        anchor.id
        if issues or incomplete_cost
        else _cost_winner(
            confirmed,
            selection_costs,
            y=y,
            metric=metric,
            margin=config.model_margin,
            test=significance_test,
            block_index=blocks,
            sample_weight=sample_weight,
        )
    )
    reason = (
        "insufficient_confirmation"
        if issues
        else "incomplete_completion_cost"
        if incomplete_cost
        else "confirmed_cost"
        if winner != anchor.id
        else "confirmation_score"
    )
    return ModelProbeOutcome(
        winner,
        bounded,
        tuple(candidates),
        tuple(failures),
        reason,
        confirmations=tuple(confirmed),
        diagnostics=diagnostics,
        issues=issues,
        rank_changed=anchor.id != first_winner,
        cost_estimates=costs,
        initial_cost_estimates=initial_costs,
        cost_model=cost_model,
        completion_costs=completion_costs,
    )


def profile_fs_cost(
    strategy: Strategy | None,
    x: np.ndarray,
    y: np.ndarray,
    folds: Sequence[Fold],
    *,
    categorical: np.ndarray,
    feature_names: Sequence[str],
    config: FeatureSelectionConfig,
    search: SearchConfig,
    fit_predict: FitPredict | None,
    metric: Metric,
    task: Task,
    seed: int,
    sample_weight: np.ndarray | None = None,
    groups: np.ndarray | None = None,
    significance_test: SignificanceTest | None = None,
    policy: SelectionPolicy | None = None,
    ctx: RunContext | None = None,
    can_start: Callable[[], bool] | None = None,
) -> dict[str, object]:
    """Profile one fixed ranker recipe on two training caps inside an original DEV fold.

    The forecast extrapolates the observed procedure to the remaining original folds. It is
    conditional on similar refinement paths and costs; the fit ceiling is reported separately.
    No subset or score from these profiles participates in feature or model-quality selection.
    """
    unavailable: dict[str, object] = {"status": "unavailable", "estimated_s": None}
    if config.compare is not None and len(config.compare) > 1:
        return {**unavailable, "reason": "recipe_not_fixed"}
    if (
        not isinstance(strategy, FeatureRanker)
        or ctx is None
        or (config.refine and fit_predict is None)
        or not folds
    ):
        return {**unavailable, "reason": "missing_ranker_profile_components"}
    cfg = config.model_copy(
        update={
            "compare": None,
            "refine_max_features": min(
                config.refine_max_features or search.max_features, search.max_features
            ),
            "refine_drop_frac": max(0.5, config.refine_drop_frac),
        }
    )
    fit_upper = min(
        search.max_fs_fits,
        estimate_fs_refits(
            cfg,
            n_strategies=1,
            n_features=x.shape[1],
            inner_n_splits=len(folds),
        ),
    )
    fold = folds[0]
    train = np.concatenate((fold.fit_idx, fold.es_idx))
    if not train.size or not fold.test_idx.size:
        return {**unavailable, "reason": "empty_profile_partition"}
    high_cap = min(search.max_rows, len(train))
    low_cap = max(1, high_cap // 2)
    n_classes = len(np.unique(y[train])) if task.is_classification else 1
    if low_cap < n_classes or high_cap == low_cap:
        return {**unavailable, "reason": "infeasible_profile_resources"}
    profiles: list[dict[str, object]] = []
    try:
        for cap in (low_cap, high_cap):
            if can_start is not None and not can_start():
                return {**unavailable, "reason": "probe_budget", "profiles": profiles}
            before = len(ctx.cost_report()["work"])
            started = time.perf_counter()
            with ctx.timed_stage("run", "fs_cost_probe"):
                bounded_ranker = BoundedFeatureRanker(strategy, task=task, max_rows=cap)
                if not cfg.refine:
                    subset = select_features(
                        x,
                        y,
                        [fold],
                        ranker=bounded_ranker,
                        categorical=categorical,
                        config=cfg.model_copy(update={"random_state": seed}),
                        sample_weight=sample_weight,
                        groups=groups,
                        ctx=ctx,
                    )
                else:
                    assert fit_predict is not None
                    subset, _, _ = _select_one(
                        bounded_ranker,
                        x,
                        y,
                        (fold,),
                        categorical=categorical,
                        config=cfg,
                        seed=seed,
                        sample_weight=sample_weight,
                        fit_predict=bounded_fit_predict(fit_predict, task=task, max_rows=cap),
                        metric=metric,
                        task=task,
                        global_classes=np.unique(y) if task.is_classification else None,
                        groups=groups,
                        significance_test=significance_test,
                        policy=policy,
                        subset_cache=OOFSubsetCache(),
                        run_context=ctx,
                    )
            elapsed = time.perf_counter() - started
            work = ctx.cost_report()["work"][before:]
            if not work or any(w["status"] != "completed" for w in work):
                return {
                    **unavailable,
                    "reason": "unmeasured_or_failed_backend_work",
                    "profiles": profiles,
                }
            profiles.append(
                {
                    "train_cap": cap,
                    "elapsed_s": elapsed,
                    "test_rows": len(fold.test_idx),
                    "feature_count": x.shape[1],
                    "selected_feature_count": len(subset),
                    "fit_count": len(work),
                    "work": work,
                }
            )
    except BudgetExhaustedError:
        return {**unavailable, "reason": "probe_budget", "profiles": profiles}
    except (AutoMLError, ValueError, RuntimeError, MemoryError) as exc:
        return {**unavailable, "reason": "profile_failed", "error": str(exc), "profiles": profiles}
    low_s, high_s = (cast(float, record["elapsed_s"]) for record in profiles)
    predicted_high = low_s * high_cap / low_cap
    scale = sum(min(search.max_rows, len(f.fit_idx) + len(f.es_idx)) for f in folds) / high_cap
    estimated = high_s * scale

    def signature(record: Mapping[str, object]) -> list[tuple[object, object]]:
        work = cast(Sequence[Mapping[str, object]], record["work"])
        return [(w["columns"], w["tree_budget"]) for w in work]

    paths_match = signature(profiles[0]) == signature(profiles[1])
    return {
        "status": "conditional",
        "estimated_s": estimated,
        "recipe": strategy.name,
        "planned_fit_count": None,
        "observed_path_projected_fit_count": cast(int, profiles[-1]["fit_count"]) * len(folds),
        "fit_count_upper_bound": fit_upper,
        "profiles": profiles,
        "configuration_sha256": hashlib.sha256(cfg.model_dump_json().encode()).hexdigest(),
        "features_sha256": hashlib.sha256(json.dumps(list(feature_names)).encode()).hexdigest(),
        "partition_sha256": {
            name: hashlib.sha256(np.asarray(idx, dtype="<i8").tobytes()).hexdigest()
            for name, idx in (("fit", fold.fit_idx), ("es", fold.es_idx), ("test", fold.test_idx))
        },
        "seed": seed,
        "threads": search.threads,
        "validation": {
            "predicted_s": predicted_high,
            "actual_s": high_s,
            "error_s": high_s - predicted_high,
            "relative_error": (high_s - predicted_high) / predicted_high
            if predicted_high > 0
            else None,
            "same_observed_fit_path": paths_match,
        },
        "assumptions": [
            "first_fold_represents_remaining_folds",
            "row_proportional_stage_time",
            "refinement_path_and_cache_misses_may_change",
            "all_folds_complete",
        ],
        "includes": ["ranker", "refinement", "subset_scoring_and_local_statistics"],
        "excludes": ["wide_cv", "post_selection_cv", "hpo", "refit", "recipe_comparison"],
        "timeout_is_not_an_estimate": True,
    }


def scout_feature_recipes(
    strategies: Sequence[tuple[str, Strategy]],
    x: np.ndarray,
    y: np.ndarray,
    probe_folds: Sequence[Fold],
    *,
    categorical: np.ndarray,
    fs_config: FeatureSelectionConfig,
    search_config: SearchConfig,
    metric: Metric,
    task: Task,
    fit_predict: FitPredict,
    prefilter: FeatureRanker,
    seed: int,
    evaluate: Callable[[tuple[int, ...], Sequence[Fold]], Candidate],
    sample_weight: np.ndarray | None = None,
    groups: np.ndarray | None = None,
    can_start: Callable[[], bool] | None = None,
    ctx: RunContext | None = None,
) -> FeatureProbeOutcome:
    """Compare bounded recipes, confirming them with the chosen family and wide control.

    Selection sees only the first probe fold's fit/ES rows. Its fit rows train the prefilter and
    rankers; ES is the inner validation for short refinement. The outer test labels are available
    only to ``evaluate``. Missing inner validation conservatively keeps all features. The chosen
    recipe name identifies the original procedure to execute on the permitted full resource.
    """
    full = tuple(range(x.shape[1]))
    if not probe_folds:
        return FeatureProbeOutcome(None, full, (), (), "infeasible_probe_folds")
    outer = probe_folds[0]
    if not outer.es_idx.size:
        return FeatureProbeOutcome(None, full, (), (), "no_inner_validation")
    if can_start is not None and not can_start():
        return FeatureProbeOutcome(None, full, (), (), "probe_budget")
    evaluation_folds = (outer,)
    try:
        with ctx.fit_attribution(recipe="no_selection") if ctx is not None else nullcontext():
            control = replace(evaluate(full, evaluation_folds), id="no_selection")
        if not np.isfinite(control.score):
            raise ValueError("non-finite wide control score")
    except BudgetExhaustedError:
        return FeatureProbeOutcome(
            None, full, (), (), "probe_budget", tuple(name for name, _ in strategies)
        )
    except (AutoMLError, ValueError, RuntimeError, MemoryError) as exc:
        return FeatureProbeOutcome(None, full, (), (("no_selection", str(exc)),), "control_failed")
    candidates = [control]
    if can_start is not None and not can_start():
        return FeatureProbeOutcome(
            None, full, tuple(candidates), (), "probe_budget", tuple(name for name, _ in strategies)
        )
    failures: list[tuple[str, str]] = []
    rows = np.concatenate((outer.fit_idx, outer.es_idx))
    x_train, y_train = x[rows], y[rows]
    sw = sample_weight[rows] if sample_weight is not None else None
    grp = groups[rows] if groups is not None else None
    n_fit = len(outer.fit_idx)
    inner = [Fold(np.arange(n_fit), np.empty(0, dtype=int), np.arange(n_fit, len(rows)))]
    if task.is_classification and not np.array_equal(
        np.unique(y_train[:n_fit]), np.unique(y_train[n_fit:])
    ):
        return FeatureProbeOutcome(None, full, tuple(candidates), (), "infeasible_inner_classes")
    try:
        if len(full) > search_config.max_features:
            with (
                ctx.fit_attribution(recipe="prefilter", fold=0)
                if ctx is not None
                else nullcontext()
            ):
                importance = prefilter.rank(
                    x_train[:n_fit],
                    y_train[:n_fit],
                    categorical=categorical,
                    random_state=seed,
                    sample_weight=sw[:n_fit] if sw is not None else None,
                    groups=grp[:n_fit] if grp is not None else None,
                )
            if importance.shape != (len(full),) or not np.all(np.isfinite(importance)):
                raise ValueError("invalid prefilter importance")
            preselected = tuple(
                sorted(
                    int(i)
                    for i in np.argsort(-importance, kind="stable")[: search_config.max_features]
                )
            )
        else:
            preselected = full
    except BudgetExhaustedError:
        return FeatureProbeOutcome(
            None, full, (), (), "probe_budget", tuple(name for name, _ in strategies)
        )
    except (AutoMLError, ValueError, RuntimeError, MemoryError) as exc:
        return FeatureProbeOutcome(
            None, full, tuple(candidates), (("prefilter", str(exc)),), "prefilter_failed"
        )
    cfg = fs_config.model_copy(
        update={
            "refine_max_features": search_config.max_features,
            "refine_drop_frac": max(0.5, fs_config.refine_drop_frac),
        }
    )
    oof_fits = 0

    def bounded_fit_predict(
        x_fit: np.ndarray,
        y_fit: np.ndarray,
        x_test: np.ndarray,
        weights: np.ndarray | None,
        random_state: int,
    ) -> tuple[np.ndarray | None, np.ndarray, np.ndarray | None]:
        nonlocal oof_fits
        if oof_fits >= search_config.max_probe_fits:
            raise BudgetExhaustedError("trials", completed=oof_fits, skipped=1, failed=0)
        oof_fits += 1
        return fit_predict(x_fit, y_fit, x_test, weights, random_state)

    cache = OOFSubsetCache()
    subsets: dict[str, tuple[int, ...]] = {"no_selection": full}
    complete = True
    evaluated: dict[tuple[int, ...], Candidate] = {full: control}
    for name, strategy in strategies:
        if can_start is not None and not can_start():
            complete = False
            break
        try:
            local, _, _ = _select_one(
                strategy,
                x_train[:, list(preselected)],
                y_train,
                inner,
                categorical=categorical[list(preselected)],
                config=cfg,
                seed=_strategy_seed(name, seed),
                sample_weight=sw,
                fit_predict=bounded_fit_predict,
                metric=metric,
                task=task,
                global_classes=np.unique(y_train) if task.is_classification else None,
                groups=grp,
                subset_cache=cache,
                run_context=ctx,
            )
            subset = tuple(preselected[i] for i in local)
            if can_start is not None and not can_start():
                complete = False
                break
            if subset not in evaluated:
                with ctx.fit_attribution(recipe=name) if ctx is not None else nullcontext():
                    evaluated[subset] = evaluate(subset, evaluation_folds)
            candidate = replace(evaluated[subset], id=name, n_features=len(subset))
            if not np.isfinite(candidate.score):
                raise ValueError("non-finite recipe probe score")
            candidates.append(candidate)
            subsets[name] = subset
        except BudgetExhaustedError:
            complete = False
            break
        except (AutoMLError, ValueError, RuntimeError, MemoryError) as exc:
            failures.append((name, str(exc)))
    if not complete:
        attempted = {name for name, _ in failures} | {candidate.id for candidate in candidates}
        return FeatureProbeOutcome(
            None,
            full,
            tuple(candidates),
            tuple(failures),
            "incomplete_recipe_budget",
            tuple(name for name, _ in strategies if name not in attempted),
        )
    sign = -1.0 if metric.greater_is_better else 1.0
    winner = min(
        candidates,
        key=lambda candidate: (sign * candidate.score, len(subsets[candidate.id]), candidate.id),
    )
    selected = winner.id != "no_selection"
    return FeatureProbeOutcome(
        winner.id if selected else None,
        subsets[winner.id],
        tuple(candidates),
        tuple(failures),
        "recipe_score" if selected else "wide_control",
    )
