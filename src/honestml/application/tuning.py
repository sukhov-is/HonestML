"""The ``tune_estimators`` use-case: honest inner-CV HPO over the Tuner port (ADR-0062).

For each tunable model type, builds an inner-CV objective on DEV and lets the injected :class:`Tuner`
search it; the tuned factory then competes in the outer honest selection (``run_slice``) unchanged
(ADR-0062 §2/§3). The objective REUSES ``run_slice``'s per-fold engine (:func:`_run_candidate`) plus a
SEPARATE out-of-fold target-encoding step (:func:`_augment_oof_te` on an INNER fold index) — because
``_run_candidate`` alone does no TE and the full-train TE would leak the target into the search
(ADR-0062 §2, R2 fix). It does NOT touch the feature-selection block: the inner objective sees the full
DEV feature width (ADR-0062 §2a). ``sample_weight`` weights inner fit AND inner score, matching the
weighted leaderboard. Budget is cooperative with graceful degradation (best-so-far / baseline, §5).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from honestml.core import (
    Budget,
    Dataset,
    FEConfig,
    Metric,
    RunContext,
    SelectionPolicy,
    Task,
    TuneOutcome,
    Tuner,
    get_logger,
    parse_search_space,
    resolve_positive,
)
from honestml.core.ports.splitter import CVSplitter, TimeOrderedSplitter
from honestml.core.schema import categorical_positions, native_routable

from .projection import _PROBA_NEEDS
from .slice import (
    EstimatorFactory,
    _augment_oof_te,
    _CandidateFailed,
    _fold_index,
    _run_candidate,
    design_matrix,
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
) -> dict[str, TuneOutcome]:
    """Tune each model type on an inner-CV of DEV; return ``name -> TuneOutcome`` (ADR-0062 §2)."""
    logger = ctx.logger if ctx is not None else get_logger()
    y = ds_dev.target()
    if y is None:
        raise ValueError("tune_estimators requires a target column")
    schema = ds_dev.schema
    classes = np.unique(y)
    positive = resolve_positive(task, classes) if task.kind == "binary" else None

    # full DEV feature space (NO FS projection — the inner objective sees full width, ADR-0062 §2a)
    x_full = design_matrix(ds_dev)
    feature_names = list(schema.features)
    n_features = len(feature_names)
    # native-categorical routing (ADR-0088/0092, FR-1/FR-2): the inner objective sees full width (no FS),
    # so the cardinality-GATED CATEGORICAL-column positions are taken over the full feature list via the
    # same gate run_slice/refit_best use (cap from the task), keeping CV/refit/HPO routing identical (R-3).
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
                )
            except _CandidateFailed:
                return worst  # an invalid hyper-combo: steer the search away, do not crash
            return cand.score

        outcomes[name] = tuner.tune(
            space,
            score,
            max_trials=n_trials,
            timeout_s=_timeout(timeout_s, budget, len(to_tune) - i),
            greater_is_better=policy.greater_is_better,
            random_state=random_state,
        )
    return outcomes
