"""Honest multi-strategy feature-selection compare (ADR-0046/0048) — pure application logic.

Each strategy picks a subset on ``dev_fit`` (DEV minus a carved selection-holdout); the arbiter scores
each subset on the **independent** ``sel_holdout`` with an estimator-agnostic ranker-model and keeps the
single winner. The external holdout (ADR-0029) is never touched here, so it stays an unbiased final
check (SPIKE-M6c-1). The leakage-critical model fit/predict is an injected ``fit_predict`` adapter; the
scheme-aware carve is an injected callable — this module imports no adapter (Humble Object, NFR-FSC-2).
A single-strategy ``compare`` skips the carve and selects on full DEV (tantamount to M6b).
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from honestml.core import (
    Dataset,
    FeatureRanker,
    FeatureSelectionConfig,
    FeatureSelectionError,
    FeatureSubsetSelector,
    Fold,
    Metric,
    Task,
    get_logger,
)
from honestml.core.ports.significance import NoSignificanceTest, SignificanceTest
from honestml.core.selection_policy import (
    BandResult,
    Candidate,
    SelectionPolicy,
    equivalence_band,
)

from .feature_selection import _degenerate_counts, estimate_fs_refits, select_features
from .oof_scorer import (
    FitPredict,
    _fold_proba,
    _positive_of,
    _score_and_band_vector,
    make_oof_scorer,
    make_oof_vector_scorer,
)
from .projection import _PROBA_NEEDS, _scorer_setup

logger = get_logger("application.feature_compare")

# (dataset, selection_holdout fraction, random_state) -> (dev_fit row idx, sel_holdout row idx)
SelectionCarve = Callable[[Dataset, float, int], tuple[np.ndarray, np.ndarray]]
Strategy = FeatureRanker | FeatureSubsetSelector


@dataclass(frozen=True)
class CompareOutcome:
    """The chosen subset plus the per-strategy arbitration record (ADR-0048/0049/0052/0053)."""

    winner: str
    winner_idx: tuple[int, ...]
    winner_subset: tuple[str, ...]
    per_strategy: tuple[tuple[str, int, float], ...]  # (name, n_selected, arb_score)
    # M6d nested/significance observability (ADR-0052/0053); defaults keep the holdout/M6c shape
    winner_rule: str = "argmax_holdout"  # argmax_holdout | argmax_band_empty | band_tiebreak
    band_members: tuple[str, ...] = ()
    per_strategy_std: tuple[
        tuple[str, float], ...
    ] = ()  # (name, arb_score_std over K nested folds)
    # M6e per-fold re-selection observability (ADR-0054 §6/§Проверки)
    arbitration_effective: str = "holdout"  # holdout | nested | nested_per_fold | holdout_degraded_c5_{outer,inner} | per_fold_partial_c5_inner
    fold_subset_jaccard: float | None = (
        None  # winner procedure stability (mean pairwise Jaccard over outer folds)
    )
    per_strategy_mean_features: tuple[
        tuple[str, float], ...
    ] = ()  # (name, raw mean per-fold subset size)
    # M6f (ADR-0059): winner's per-fold block-fragmentation aggregate, merged into null_block_stats by run_slice
    per_fold_block_stats: dict[str, float] | None = None
    # in-sequential band observability (ADR-0086 §1): the wrapper-selector band of the WINNER strategy,
    # SEPARATE from the strategy-arbitration band (winner_rule/band_members). None unless the winner is a
    # wrapper selector with significance on. Keys: width, winner_by_tiebreak, members, rule.
    seq_band: dict[str, object] | None = None


def _strategy_seed(name: str, random_state: int) -> int:
    """Stable per-strategy seed (ADR-0049 §1) — isolates strategies, reproducible across runs.

    ``blake2b`` (not Python ``hash()``, which is per-process randomized): same ``(name, seed)`` always
    maps to the same value; different names map to different values, so strategy A's randomness cannot
    bleed into strategy B (FR-FSC-7).
    """
    digest = hashlib.blake2b(f"{name}:{random_state}".encode(), digest_size=4).digest()
    return int.from_bytes(digest, "big")


def _strategy_fold_seed(name: str, random_state: int, fold_id: int) -> int:
    """Per-outer-fold stable seed (ADR-0054 §4): blake2b like :func:`_strategy_seed`, **not** Python ``hash()``.

    Folds ``fold_id`` into the digest so each outer fold's null-permutation draws are independent (honest
    per-fold dispersion), while staying a fixed function of the run seed — ``hash()`` is ``PYTHONHASHSEED``-salted
    and would differ across processes, breaking the determinism guarantee (NFR-FSE-3).
    """
    digest = hashlib.blake2b(f"{name}:{random_state}:{fold_id}".encode(), digest_size=4).digest()
    return int.from_bytes(digest, "big")


def no_selection_gate(
    x_full: np.ndarray,
    y: np.ndarray,
    selected_idx: Sequence[int],
    folds: Sequence[Fold],
    *,
    fit_predict: FitPredict,
    metric: Metric,
    task: Task,
    sample_weight: np.ndarray | None,
    significance_test: SignificanceTest,
    policy: SelectionPolicy,
    random_state: int,
    block_index: np.ndarray | None = None,
) -> tuple[bool, str]:
    """Honest gate of a feature subset against the no-selection baseline (finding #10, ADR-0063 §5).

    Scores the selected subset AND the full feature set over the same CV folds with the cheap
    estimator-agnostic ranker-model, then runs the M4 equivalence band over the two: the subset ships
    only when it is statistically indistinguishable from (or better than) no-selection — never when it
    is *significantly* worse, the same choose_better semantics the ensemble gate uses. Returns
    ``(keep_selection, reason)``; the decision is never silent. The ranker chose the subset on full DEV,
    so the subset carries mild optimism here, but the full set is scored identically — the RELATIVE
    verdict still catches a subset aggressive enough to regress (the backlog's "cheap full-feature
    control"). ``block_index`` feeds the time-series block bootstrap, exactly like the main band.
    """
    full_idx = tuple(range(x_full.shape[1]))
    if len(selected_idx) >= len(full_idx):
        return True, "all_features_selected"
    scorer = make_oof_vector_scorer(
        x_full,
        y,
        folds,
        fit_predict=fit_predict,
        metric=metric,
        task=task,
        random_state=random_state,
        sample_weight=sample_weight,
        global_classes=np.unique(y) if task.is_classification else None,
    )
    # the scorer returns a higher-is-better (sign-flipped) score; multiply back to the metric's own
    # orientation so the band ranks with the metric-oriented policy, exactly like the main leaderboard band.
    sign = 1.0 if metric.greater_is_better else -1.0
    sel_flip, sel_oof, sel_mask = scorer(list(selected_idx))
    full_flip, full_oof, full_mask = scorer(full_idx)
    candidates = [
        Candidate(
            id="selected",
            score=sign * sel_flip,
            n_features=len(selected_idx),
            oof_pred=sel_oof,
            oof_mask=sel_mask,
        ),
        Candidate(
            id="no_selection",
            score=sign * full_flip,
            n_features=len(full_idx),
            oof_pred=full_oof,
            oof_mask=full_mask,
        ),
    ]
    band = equivalence_band(
        candidates,
        policy,
        significance_test,
        y,
        block_index=block_index,
        sample_weight=sample_weight,
    )
    keep = "selected" in band.member_ids
    return keep, ("selection_kept" if keep else "no_selection_better")


def _arbitrate_score(
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    x_hold: np.ndarray,
    y_hold: np.ndarray,
    indices: Sequence[int],
    *,
    fit_predict: FitPredict,
    metric: Metric,
    task: Task,
    random_state: int,
    sw_fit: np.ndarray | None,
    sw_hold: np.ndarray | None,
    global_classes: np.ndarray | None,
) -> float:
    """Score one subset on the independent selection-holdout (ADR-0048 §1 п.3).

    ``global_classes`` is the whole-DEV class order (not ``y_fit``'s) so multiclass proba aligns even
    when ``dev_fit`` is missing a class present on the holdout.
    """
    idx = list(indices)
    positive = _positive_of(task, global_classes)
    proba, pred, cls = fit_predict(x_fit[:, idx], y_fit, x_hold[:, idx], sw_fit, random_state)
    if (
        metric.needs in _PROBA_NEEDS
        and proba is not None
        and cls is not None
        and global_classes is not None
    ):
        proj = _fold_proba(
            proba,
            cls,
            multiclass=task.kind == "multiclass",
            global_classes=global_classes,
            positive=positive,
        )
    else:
        proj = pred
    sign = 1.0 if metric.greater_is_better else -1.0
    return sign * metric.score(y_hold, proj, sw_hold)


def _band_over_trajectory(
    trajectory: Sequence[tuple[int, ...]],
    x: np.ndarray,
    y: np.ndarray,
    folds: Sequence[Fold],
    *,
    fit_predict: FitPredict,
    metric: Metric,
    task: Task,
    random_state: int,
    sample_weight: np.ndarray | None,
    global_classes: np.ndarray | None,
    groups: np.ndarray | None,
    significance_test: SignificanceTest | None,
    policy: SelectionPolicy | None,
) -> tuple[tuple[int, ...], BandResult | None]:
    """Pick the final subset from a backward trajectory by significance band + Occam (ADR-0083/0085).

    Each visited subset is scored on the SAME selection folds (Same-OOF, ADR-0085 §1) via
    :func:`make_oof_vector_scorer`; the metric-ready OOF + mask feed :func:`equivalence_band`. The winner is
    the most compact subset statistically indistinguishable from the best; with ``NoSignificanceTest`` the
    band collapses to the absolute argmax = the current ``best_keep`` (FR-2). ``id`` encodes the size so the
    off-path anchor on score ties is the LARGEST subset, matching ``best_keep`` (ADR-0085 §3); the scorer's
    higher-is-better output is flipped back to the metric's native orientation for the metric-oriented policy.

    The anchor is the argmax on the SAME OOF the band tests on, so the band is **narrower** than a fully
    independent one — strictly more conservative than argmax (toward keeping MORE features), never an
    over-prune; the residual is documented in ADR-0085 §5 (independent-OOF is a Day-2 improvement). Returns
    ``BandResult = None`` on the off-path (``None``/``NoSignificanceTest``): the winner is still the absolute
    argmax, but there is no band observability (FR-2 / ADR-0086).
    """
    scorer = make_oof_vector_scorer(
        x,
        y,
        folds,
        fit_predict=fit_predict,
        metric=metric,
        task=task,
        random_state=random_state,
        sample_weight=sample_weight,
        global_classes=global_classes,
    )
    sign = 1.0 if metric.greater_is_better else -1.0
    n_full = x.shape[1]
    pol = (
        policy
        if policy is not None
        else SelectionPolicy(greater_is_better=metric.greater_is_better)
    )
    candidates: list[Candidate] = []
    subset_by_id: dict[str, tuple[int, ...]] = {}
    for subset in trajectory:
        flip, oof_vec, mask = scorer(subset)
        cid = f"k{n_full - len(subset):05d}"
        subset_by_id[cid] = subset
        candidates.append(
            Candidate(
                id=cid, score=sign * flip, n_features=len(subset), oof_pred=oof_vec, oof_mask=mask
            )
        )
    band = equivalence_band(
        candidates, pol, significance_test, y, block_index=groups, sample_weight=sample_weight
    )
    # off-path (None / NoSignificanceTest) reproduces pure argmax with NO band observability (FR-2/ADR-0086):
    # the winner is still the absolute argmax = current best_keep, but seq_band stays absent.
    real = significance_test is not None and not isinstance(significance_test, NoSignificanceTest)
    return subset_by_id[band.winner], (band if real else None)


def _select_one(
    strategy: Strategy,
    x: np.ndarray,
    y: np.ndarray,
    folds: Sequence[Fold],
    *,
    categorical: np.ndarray,
    config: FeatureSelectionConfig,
    seed: int,
    sample_weight: np.ndarray | None,
    fit_predict: FitPredict,
    metric: Metric,
    task: Task,
    global_classes: np.ndarray | None,
    groups: np.ndarray | None = None,
    significance_test: SignificanceTest | None = None,
    policy: SelectionPolicy | None = None,
) -> tuple[tuple[int, ...], BandResult | None]:
    """Run one strategy, dispatching by port; a wrapper selector gets the significance band (ADR-0083).

    ranker spine -> ``(subset, None)``; wrapper selector -> greedy trajectory scored Same-OOF, band + Occam
    pick the final subset, returning its :class:`BandResult` for observability. ``groups`` (M6d) is the
    per-row structure label (``block_index`` for the band; structure-aware rankers); already sliced to
    ``x``'s rows by the caller (ADR-0050 §3).
    """
    if isinstance(strategy, FeatureSubsetSelector):
        score_subset = make_oof_scorer(
            x,
            y,
            folds,
            fit_predict=fit_predict,
            metric=metric,
            task=task,
            random_state=seed,
            sample_weight=sample_weight,
            global_classes=global_classes,
        )
        trajectory = strategy.select(
            x,
            y,
            folds,
            categorical=categorical,
            score_subset=score_subset,
            random_state=seed,
            sample_weight=sample_weight,
        )
        return _band_over_trajectory(
            trajectory,
            x,
            y,
            folds,
            fit_predict=fit_predict,
            metric=metric,
            task=task,
            random_state=seed,
            sample_weight=sample_weight,
            global_classes=global_classes,
            groups=groups,
            significance_test=significance_test,
            policy=policy,
        )
    seeded = config.model_copy(update={"random_state": seed})
    return (
        select_features(
            x,
            y,
            list(folds),
            ranker=strategy,
            categorical=categorical,
            config=seeded,
            sample_weight=sample_weight,
            groups=groups,
        ),
        None,
    )


def _nested_winner(
    subsets: Sequence[tuple[str, tuple[int, ...]]],
    x_full: np.ndarray,
    y: np.ndarray,
    arb_folds: Sequence[Fold],
    *,
    metric: Metric,
    task: Task,
    fit_predict: FitPredict,
    sample_weight: np.ndarray | None,
    random_state: int,
    global_classes: np.ndarray | None,
    groups: np.ndarray | None,
    significance_test: SignificanceTest,
    policy: SelectionPolicy,
) -> tuple[
    str,
    tuple[int, ...],
    list[tuple[str, int, float]],
    list[tuple[str, float]],
    str,
    tuple[str, ...],
]:
    """Arbitrate subsets by pooled-OOF over K DEV folds + significance band (ADR-0052/0053).

    Each subset is refit on every arbitration fold's train part and scored on the pooled OOF (the
    selection-time scores are never reused -> no wrapper home-advantage, ADR-0052 §3). The winner is the
    most compact subset statistically indistinguishable from the best (Occam, ADR-0053); with the inert
    ``NoSignificanceTest`` the band collapses to a plain argmax (back-compat).
    """
    scorer = make_oof_vector_scorer(
        x_full,
        y,
        arb_folds,
        fit_predict=fit_predict,
        metric=metric,
        task=task,
        random_state=random_state,
        sample_weight=sample_weight,
        global_classes=global_classes,
    )
    sign = 1.0 if metric.greater_is_better else -1.0
    candidates: list[Candidate] = []
    per_strategy: list[tuple[str, int, float]] = []
    per_std: list[tuple[str, float]] = []
    by_name: dict[str, tuple[int, ...]] = {}
    for name, idx in subsets:
        score, oof_vec, mask = scorer(idx)
        # per-fold dispersion from the metric-ready OOF (no refit): score each arb fold's slice; a fold the
        # metric cannot score (e.g. a single-class proba slice) is skipped — dispersion is a diagnostic.
        fold_scores: list[float] = []
        for f in arb_folds:
            if not f.test_idx.size:
                continue
            sw_f = sample_weight[f.test_idx] if sample_weight is not None else None
            try:
                fold_scores.append(sign * metric.score(y[f.test_idx], oof_vec[f.test_idx], sw_f))
            except ValueError:
                continue
        per_strategy.append((name, len(idx), score))
        per_std.append((name, float(np.std(fold_scores)) if fold_scores else 0.0))
        candidates.append(
            Candidate(id=name, score=score, n_features=len(idx), oof_pred=oof_vec, oof_mask=mask)
        )
        by_name[name] = idx
    result = equivalence_band(
        candidates, policy, significance_test, y, block_index=groups, sample_weight=sample_weight
    )
    winner_rule = "band_tiebreak" if result.winner_by_tiebreak else "argmax_band_empty"
    return (
        result.winner,
        by_name[result.winner],
        per_strategy,
        per_std,
        winner_rule,
        result.member_ids,
    )


def _mean_jaccard(subsets: Sequence[tuple[int, ...]]) -> float | None:
    """Mean pairwise Jaccard of per-fold subsets — procedure stability diagnostic (ADR-0054 §Проверки).

    Low values mean the re-selection picked very different features across outer folds (unstable procedure).
    ``None`` when fewer than two folds were scored (no pair to compare).
    """
    if len(subsets) < 2:
        return None
    sets = [set(s) for s in subsets]
    vals = [
        (len(sets[i] & sets[j]) / len(sets[i] | sets[j]) if (sets[i] | sets[j]) else 1.0)
        for i in range(len(sets))
        for j in range(i + 1, len(sets))
    ]
    return float(np.mean(vals)) if vals else None


def _score_procedure(
    strategy: Strategy,
    name: str,
    dataset: Dataset,
    x_full: np.ndarray,
    y: np.ndarray,
    arb_folds: Sequence[Fold],
    inner_splitter: object,
    *,
    categorical: np.ndarray,
    config: FeatureSelectionConfig,
    metric: Metric,
    task: Task,
    fit_predict: FitPredict,
    sample_weight: np.ndarray | None,
    random_state: int,
    global_classes: np.ndarray | None,
    groups: np.ndarray | None,
    significance_test: SignificanceTest | None = None,
    policy: SelectionPolicy | None = None,
) -> (
    tuple[float, np.ndarray, np.ndarray, list[tuple[int, ...]], int, dict[str, float] | None] | None
):
    """Honest per-fold re-selection (ADR-0054 §2): re-select the subset INSIDE each outer fold's train.

    For each outer (arbitration) fold: build inner folds over the outer-train rows only
    (``inner_splitter.split(dataset.take(tr))``), re-select the subset on that train with ``_select_one``,
    refit ``fit_predict`` on the selected columns and predict the outer-test rows; pool the OOF. Returns
    ``(score, metric-ready OOF, mask, per-fold subsets, n_degraded_folds, per-fold block stats)`` or ``None`` when no outer fold
    survived (whole-arbitration must fall back to holdout). ``groups``/``sample_weight`` are sliced to the
    train rows BEFORE selection so the ranker never sees outer-test (anti-leakage, NFR-FSE-1); per-outer-fold
    blake2b seed keeps null draws independent yet reproducible (NFR-FSE-3). Inner C5 (ADR-0054 §6): an outer
    fold whose train has a class with fewer than the inner ``n_splits`` rows is dropped from the pooled OOF
    (fold-local), not a whole-arbitration failure.
    """
    n = y.shape[0]
    multiclass = task.kind == "multiclass"
    classes, positive, need_proba, sign = _scorer_setup(
        task, metric, y, global_classes=global_classes
    )
    inner_n_splits = int(getattr(inner_splitter, "n_splits", 2))
    oof_proba = (
        np.full((n, classes.size), np.nan)
        if multiclass and classes is not None
        else np.full(n, np.nan)
    )
    oof_pred = np.empty(n, dtype=y.dtype)
    mask = np.zeros(n, dtype=bool)
    produced_proba = False
    subsets: list[tuple[int, ...]] = []
    degraded = 0
    fold_deg_frac: list[
        float
    ] = []  # per-fold degenerate-block fraction (ADR-0059 §1: fold-train, not full-DEV)
    fold_n_blocks: list[float] = []
    for fold_id, outer in enumerate(arb_folds):
        test_idx = outer.test_idx
        if not test_idx.size:
            continue
        tr = (
            outer.fit_idx
            if outer.es_idx.size == 0
            else np.concatenate([outer.fit_idx, outer.es_idx])
        )
        if (
            task.is_classification
            and int(np.unique(y[tr], return_counts=True)[1].min()) < inner_n_splits
        ):
            degraded += 1  # inner-C5 fold-local: drop this fold from the pooled OOF (ADR-0054 §6)
            continue
        sw_tr = sample_weight[tr] if sample_weight is not None else None
        grp_tr = groups[tr] if groups is not None else None
        if (
            grp_tr is not None
        ):  # per-fold honesty (ADR-0059 §1): blocks fragment more on the smaller fold-train
            n_blk = int(np.unique(grp_tr).size)
            fold_deg_frac.append(_degenerate_counts(grp_tr, y[tr]) / n_blk)
            fold_n_blocks.append(float(n_blk))
        seed_f = _strategy_fold_seed(name, random_state, fold_id)
        inner = list(inner_splitter.split(dataset.take(tr)))  # type: ignore[attr-defined]
        idx_f, _ = _select_one(
            strategy,
            x_full[tr],
            y[tr],
            inner,
            categorical=categorical,
            config=config,
            seed=seed_f,
            sample_weight=sw_tr,
            fit_predict=fit_predict,
            metric=metric,
            task=task,
            global_classes=global_classes,
            groups=grp_tr,
            significance_test=significance_test,
            policy=policy,
        )
        # per-fold band runs (ADR-0083 §3a) but its BandResult is discarded: seq_band is reported only for
        # the winner's FINAL selection on full DEV (ADR-0086 §1), not for intermediate per-fold re-selections.
        cols = list(idx_f)
        proba, pred, cls = fit_predict(
            x_full[tr][:, cols], y[tr], x_full[test_idx][:, cols], sw_tr, random_state
        )
        oof_pred[test_idx] = pred
        if need_proba and proba is not None and cls is not None and classes is not None:
            oof_proba[test_idx] = _fold_proba(
                proba, cls, multiclass=multiclass, global_classes=classes, positive=positive
            )
            produced_proba = True
        mask[test_idx] = True
        subsets.append(idx_f)
    if (
        not mask.any()
    ):  # every outer fold inner-C5-degraded -> per-fold infeasible (caller falls back to holdout)
        return None
    score, oof_vec, _ = _score_and_band_vector(
        y,
        oof_pred,
        oof_proba if produced_proba else None,
        mask,
        metric=metric,
        task=task,
        sample_weight=sample_weight,
        sign=sign,
    )
    pf_block = (
        {
            "per_fold_degenerate_mean": float(np.mean(fold_deg_frac)),
            "per_fold_degenerate_max": float(np.max(fold_deg_frac)),
            "per_fold_n_blocks_mean": float(np.mean(fold_n_blocks)),
        }
        if fold_deg_frac
        else None
    )
    return score, oof_vec, mask, subsets, degraded, pf_block


def _per_fold_winner(
    strategies: Sequence[tuple[str, Strategy]],
    dataset: Dataset,
    x_full: np.ndarray,
    y: np.ndarray,
    arb_folds: Sequence[Fold],
    inner_splitter: object,
    *,
    categorical: np.ndarray,
    config: FeatureSelectionConfig,
    metric: Metric,
    task: Task,
    fit_predict: FitPredict,
    sample_weight: np.ndarray | None,
    random_state: int,
    global_classes: np.ndarray | None,
    groups: np.ndarray | None,
    significance_test: SignificanceTest,
    policy: SelectionPolicy,
) -> (
    tuple[
        str,
        list[tuple[str, int, float]],
        list[tuple[str, float]],
        str,
        tuple[str, ...],
        float | None,
        int,
        dict[str, float] | None,
    ]
    | None
):
    """Arbitrate strategies by their per-fold re-selection PROCEDURE (ADR-0054).

    Each strategy is scored by :func:`_score_procedure` (re-selection inside every outer fold); the Occam key
    is the int ``round(mean per-fold subset size)`` (raw float kept for observability), so the band compares
    the compactness of the same object it scored. ``None`` if any strategy is per-fold-infeasible (caller
    degrades the whole arbitration to holdout). The shipped subset is re-derived on full DEV by the caller.
    """
    candidates: list[Candidate] = []
    per_strategy: list[tuple[str, int, float]] = []
    per_mean: list[tuple[str, float]] = []
    subsets_by_name: dict[str, list[tuple[int, ...]]] = {}
    block_by_name: dict[str, dict[str, float] | None] = {}
    total_degraded = 0
    for name, strat in strategies:
        result = _score_procedure(
            strat,
            name,
            dataset,
            x_full,
            y,
            arb_folds,
            inner_splitter,
            categorical=categorical,
            config=config,
            metric=metric,
            task=task,
            fit_predict=fit_predict,
            sample_weight=sample_weight,
            random_state=random_state,
            global_classes=global_classes,
            groups=groups,
            significance_test=significance_test,
            policy=policy,
        )
        if result is None:
            return None
        score, oof_vec, mask, subsets, degraded, pf_block = result
        total_degraded += degraded
        block_by_name[name] = pf_block
        mean_size = float(np.mean([len(s) for s in subsets]))
        n_key = max(1, round(mean_size))  # int Occam key (ADR-0054 §2); raw mean in per_mean
        per_strategy.append((name, n_key, score))
        per_mean.append((name, mean_size))
        subsets_by_name[name] = subsets
        candidates.append(
            Candidate(id=name, score=score, n_features=n_key, oof_pred=oof_vec, oof_mask=mask)
        )
    band = equivalence_band(
        candidates, policy, significance_test, y, block_index=groups, sample_weight=sample_weight
    )
    winner_rule = "band_tiebreak" if band.winner_by_tiebreak else "argmax_band_empty"
    jaccard = _mean_jaccard(subsets_by_name[band.winner])
    return (
        band.winner,
        per_strategy,
        per_mean,
        winner_rule,
        band.member_ids,
        jaccard,
        total_degraded,
        block_by_name[band.winner],
    )


@dataclass(frozen=True)
class _CompareCtx:
    """The shared compare inputs, built once by :func:`compare_features` and threaded to each mode."""

    dataset: Dataset
    x_full: np.ndarray
    y: np.ndarray
    task: Task
    metric: Metric
    config: FeatureSelectionConfig
    splitter: object
    carve: SelectionCarve
    fit_predict: FitPredict
    categorical: np.ndarray
    feature_names: Sequence[str]
    sample_weight: np.ndarray | None
    random_state: int
    groups: np.ndarray | None
    global_classes: np.ndarray | None
    # band wiring (ADR-0083 §3): threaded UNCONDITIONALLY (incl. single-strategy) so sequential's
    # in-trajectory band activates everywhere; off -> NoSignificanceTest -> argmax (FR-2).
    significance_test: SignificanceTest | None = None
    policy: SelectionPolicy | None = None

    def select(
        self,
        name: str,
        strat: Strategy,
        x: np.ndarray,
        ys: np.ndarray,
        folds: Sequence[Fold],
        sw: np.ndarray | None,
        seed: int,
        grp: np.ndarray | None,
    ) -> tuple[tuple[int, ...], BandResult | None]:
        """Run one strategy -> ``(subset, BandResult|None)``; map non-:class:`FeatureSelectionError` to fail-fast (ADR-0048 §4)."""
        try:
            return _select_one(
                strat,
                x,
                ys,
                folds,
                categorical=self.categorical,
                config=self.config,
                seed=seed,
                sample_weight=sw,
                fit_predict=self.fit_predict,
                metric=self.metric,
                task=self.task,
                global_classes=self.global_classes,
                groups=grp,
                significance_test=self.significance_test,
                policy=self.policy,
            )
        except FeatureSelectionError:
            raise
        except Exception as exc:  # fail-fast: no silent strategy drop (ADR-0048 §4 п.4)
            raise FeatureSelectionError(name, exc) from exc


def _seq_band_dict(band: BandResult | None) -> dict[str, object] | None:
    """Format a wrapper-selector :class:`BandResult` for ``CompareOutcome.seq_band`` (ADR-0086 §1)."""
    if band is None:
        return None
    return {
        "width": band.width,
        "winner_by_tiebreak": band.winner_by_tiebreak,
        "members": list(band.member_ids),
        "rule": "band_tiebreak" if band.winner_by_tiebreak else "argmax",
    }


def _outcome(
    ctx: _CompareCtx,
    winner: str,
    winner_idx: tuple[int, ...],
    per_strategy: Sequence[tuple[str, int, float]],
    *,
    winner_rule: str = "argmax_holdout",
    band_members: tuple[str, ...] = (),
    per_strategy_std: tuple[tuple[str, float], ...] = (),
    arbitration_effective: str = "holdout",
    fold_subset_jaccard: float | None = None,
    per_strategy_mean_features: tuple[tuple[str, float], ...] = (),
    per_fold_block_stats: dict[str, float] | None = None,
    seq_band: BandResult | None = None,
) -> CompareOutcome:
    """The single :class:`CompareOutcome` constructor — resolves ``winner_subset`` once for every mode."""
    return CompareOutcome(
        winner=winner,
        winner_idx=winner_idx,
        winner_subset=tuple(ctx.feature_names[i] for i in winner_idx),
        per_strategy=tuple(per_strategy),
        winner_rule=winner_rule,
        band_members=band_members,
        per_strategy_std=per_strategy_std,
        arbitration_effective=arbitration_effective,
        fold_subset_jaccard=fold_subset_jaccard,
        per_strategy_mean_features=per_strategy_mean_features,
        per_fold_block_stats=per_fold_block_stats,
        seq_band=_seq_band_dict(seq_band),
    )


def _compare_single(
    ctx: _CompareCtx,
    strategies: Sequence[tuple[str, Strategy]],
    *,
    arbitration_effective: str = "holdout",
) -> CompareOutcome:
    """Select the first strategy on full DEV — no carve/arbitration (single-strategy or degraded path)."""
    name, strat = strategies[0]
    folds = list(ctx.splitter.split(ctx.dataset))  # type: ignore[attr-defined]
    idx, band = ctx.select(
        name, strat, ctx.x_full, ctx.y, folds, ctx.sample_weight, ctx.random_state, ctx.groups
    )
    return _outcome(
        ctx,
        name,
        idx,
        ((name, len(idx), float("nan")),),
        arbitration_effective=arbitration_effective,
        seq_band=band,
    )


def _compare_per_fold(
    ctx: _CompareCtx,
    strategies: Sequence[tuple[str, Strategy]],
    arbitration_splitter: object,
    significance_test: SignificanceTest,
    policy: SelectionPolicy,
) -> CompareOutcome | None:
    """Per-fold re-selection arbitration (ADR-0054); ``None`` when every outer fold is inner-C5-degraded.

    Re-selects the subset INSIDE each outer fold (cost ~ N*K_outer SELECTIONS, not just scoring); the winner
    is the strategy whose procedure generalizes best, its subset then re-derived on full DEV.
    """
    inner_n = int(getattr(ctx.splitter, "n_splits", 5))  # type: ignore[attr-defined]
    # canonical cost (ADR-0058 §1): the single numeric source, same as the cost-budget gate
    n_refits = estimate_fs_refits(
        ctx.config,
        n_strategies=len(strategies),
        n_features=ctx.x_full.shape[1],
        inner_n_splits=inner_n,
    )
    logger.warning(
        "nested_per_fold arbitration: re-selects per outer fold -> ~%d ranker-model fits (%d strategies "
        "x %d outer x %d inner); selection_holdout is ignored",
        n_refits,
        len(strategies),
        ctx.config.arbitration_n_splits,
        inner_n,
    )
    arb_folds = list(arbitration_splitter.split(ctx.dataset))  # type: ignore[attr-defined]
    pf = _per_fold_winner(
        strategies,
        ctx.dataset,
        ctx.x_full,
        ctx.y,
        arb_folds,
        ctx.splitter,
        categorical=ctx.categorical,
        config=ctx.config,
        metric=ctx.metric,
        task=ctx.task,
        fit_predict=ctx.fit_predict,
        sample_weight=ctx.sample_weight,
        random_state=ctx.random_state,
        global_classes=ctx.global_classes,
        groups=ctx.groups,
        significance_test=significance_test,
        policy=policy,
    )
    if pf is None:
        return None
    winner, pf_per_strategy, pf_per_mean, pf_rule, pf_band, jaccard, degraded, pf_block = pf
    sel_folds = list(ctx.splitter.split(ctx.dataset))  # type: ignore[attr-defined]
    winner_idx, winner_band = ctx.select(
        winner,
        dict(strategies)[winner],
        ctx.x_full,
        ctx.y,
        sel_folds,
        ctx.sample_weight,
        _strategy_seed(winner, ctx.random_state),
        ctx.groups,
    )
    return _outcome(
        ctx,
        winner,
        winner_idx,
        pf_per_strategy,
        winner_rule=pf_rule,
        band_members=pf_band,
        seq_band=winner_band,
        arbitration_effective="per_fold_partial_c5_inner" if degraded else "nested_per_fold",
        fold_subset_jaccard=jaccard,
        per_strategy_mean_features=tuple(pf_per_mean),
        per_fold_block_stats=pf_block,
    )


def _compare_nested(
    ctx: _CompareCtx,
    strategies: Sequence[tuple[str, Strategy]],
    arbitration_splitter: object,
    significance_test: SignificanceTest,
    policy: SelectionPolicy,
) -> CompareOutcome:
    """Nested-CV arbitration (ADR-0052): refit each fixed subset over K DEV folds + significance band.

    No carve; strategies select on full DEV, the arbiter refits each FIXED subset over K independent DEV
    folds and scores the pooled OOF; significance picks the most compact among the indistinguishable.
    """
    logger.warning(
        "nested arbitration: ~%d ranker-model fits (%d strategies x %d folds); selection_holdout is "
        "ignored in nested mode",
        len(strategies) * ctx.config.arbitration_n_splits,
        len(strategies),
        ctx.config.arbitration_n_splits,
    )
    sel_folds = list(ctx.splitter.split(ctx.dataset))  # type: ignore[attr-defined]
    selected = {
        name: ctx.select(
            name,
            strat,
            ctx.x_full,
            ctx.y,
            sel_folds,
            ctx.sample_weight,
            _strategy_seed(name, ctx.random_state),
            ctx.groups,
        )
        for name, strat in strategies
    }
    subsets = [(name, sub) for name, (sub, _band) in selected.items()]
    seq_bands = {name: band for name, (_sub, band) in selected.items()}
    arb_folds = list(arbitration_splitter.split(ctx.dataset))  # type: ignore[attr-defined]
    n_winner, n_winner_idx, n_per_strategy, n_per_std, n_rule, n_band = _nested_winner(
        subsets,
        ctx.x_full,
        ctx.y,
        arb_folds,
        metric=ctx.metric,
        task=ctx.task,
        fit_predict=ctx.fit_predict,
        sample_weight=ctx.sample_weight,
        random_state=ctx.random_state,
        global_classes=ctx.global_classes,
        groups=ctx.groups,
        significance_test=significance_test,
        policy=policy,
    )
    return _outcome(
        ctx,
        n_winner,
        n_winner_idx,
        n_per_strategy,
        winner_rule=n_rule,
        band_members=n_band,
        per_strategy_std=tuple(n_per_std),
        arbitration_effective="nested",
        seq_band=seq_bands.get(n_winner),
    )


def _compare_holdout(
    ctx: _CompareCtx,
    strategies: Sequence[tuple[str, Strategy]],
    *,
    arbitration_effective: str,
) -> CompareOutcome:
    """Holdout arbitration (ADR-0048): carve an independent ``sel_holdout``, score each subset on it, argmax."""
    dev_fit_idx, sel_idx = ctx.carve(ctx.dataset, ctx.config.selection_holdout, ctx.random_state)
    if sel_idx.size == 0 or dev_fit_idx.size == 0:
        # an empty carve (tiny DEV / extreme selection_holdout) makes holdout arbitration infeasible:
        # degrade to selecting the first strategy on full DEV (like the nested->holdout C5 degradation)
        # instead of letting a raw sklearn ValueError surface — the decision is never silent.
        logger.warning(
            "selection-holdout carve produced an empty dev_fit/sel_holdout (%d/%d rows); degrading "
            "to single-strategy selection on full DEV",
            dev_fit_idx.size,
            sel_idx.size,
        )
        return _compare_single(
            ctx, strategies, arbitration_effective="holdout_degraded_empty_carve"
        )
    if sel_idx.size < 300:
        logger.warning(
            "selection-holdout is small (%d rows): compare arbitration may be noisy; "
            "consider a smaller selection_holdout or fewer strategies",
            sel_idx.size,
        )
    dev_fit_ds = ctx.dataset.take(dev_fit_idx)
    dev_folds = list(ctx.splitter.split(dev_fit_ds))  # type: ignore[attr-defined]
    x_devfit, y_devfit = ctx.x_full[dev_fit_idx], ctx.y[dev_fit_idx]
    x_sel, y_sel = ctx.x_full[sel_idx], ctx.y[sel_idx]
    sw_devfit = ctx.sample_weight[dev_fit_idx] if ctx.sample_weight is not None else None
    sw_sel = ctx.sample_weight[sel_idx] if ctx.sample_weight is not None else None
    groups_devfit = ctx.groups[dev_fit_idx] if ctx.groups is not None else None

    # per-strategy hashed seed isolates strategies (A's randomness can't bleed into B; FR-FSC-7)
    selected = {
        name: ctx.select(
            name,
            strat,
            x_devfit,
            y_devfit,
            dev_folds,
            sw_devfit,
            _strategy_seed(name, ctx.random_state),
            groups_devfit,
        )
        for name, strat in strategies
    }
    seq_bands = {name: band for name, (_sub, band) in selected.items()}

    per_strategy: list[tuple[str, int, float]] = []
    best_name: str = ""
    best_idx: tuple[int, ...] = ()
    best_score = -np.inf
    for name, (idx, _band) in selected.items():
        arb = _arbitrate_score(
            x_devfit,
            y_devfit,
            x_sel,
            y_sel,
            idx,
            fit_predict=ctx.fit_predict,
            metric=ctx.metric,
            task=ctx.task,
            random_state=ctx.random_state,
            sw_fit=sw_devfit,
            sw_hold=sw_sel,
            global_classes=ctx.global_classes,
        )
        per_strategy.append((name, len(idx), arb))
        if arb > best_score:  # strict > -> ties keep the first strategy in order (ADR-0048 §1 п.3)
            best_name, best_idx, best_score = name, idx, arb
    return _outcome(
        ctx,
        best_name,
        best_idx,
        per_strategy,
        arbitration_effective=arbitration_effective,
        seq_band=seq_bands.get(best_name),
    )


def compare_features(
    dataset: Dataset,
    x_full: np.ndarray,
    y: np.ndarray,
    *,
    task: Task,
    metric: Metric,
    strategies: Sequence[tuple[str, Strategy]],
    config: FeatureSelectionConfig,
    splitter: object,
    carve: SelectionCarve,
    fit_predict: FitPredict,
    categorical: np.ndarray,
    feature_names: Sequence[str],
    sample_weight: np.ndarray | None,
    random_state: int,
    groups: np.ndarray | None = None,
    arbitration_splitter: object | None = None,
    significance_test: SignificanceTest | None = None,
    policy: SelectionPolicy | None = None,
) -> CompareOutcome:
    """Pick one subset by honest compare (ADR-0046 §3 / ADR-0048) — dispatch the arbitration mode.

    ``len(strategies) == 1`` short-circuits to :func:`_compare_single` (no carve, select on full DEV,
    tantamount to M6b). Otherwise the mode is dispatched: ``nested_per_fold`` re-selects inside each outer
    fold (:func:`_compare_per_fold`), ``nested`` refits fixed subsets over K folds (:func:`_compare_nested`),
    and the default carves an independent ``sel_holdout`` (:func:`_compare_holdout`); an infeasible nested
    mode (C5 rare class / inner-degraded) degrades to holdout. ``groups`` (M6d) is the per-row structure
    label for structure-aware rankers (ADR-0050 §3).
    """
    ctx = _CompareCtx(
        dataset=dataset,
        x_full=x_full,
        y=y,
        task=task,
        metric=metric,
        config=config,
        splitter=splitter,
        carve=carve,
        fit_predict=fit_predict,
        categorical=categorical,
        feature_names=feature_names,
        sample_weight=sample_weight,
        random_state=random_state,
        groups=groups,
        # whole-DEV class order so multiclass proba aligns even when a sub-split misses a class
        global_classes=np.unique(y) if task.is_classification else None,
        # band wiring threaded UNCONDITIONALLY so single-strategy sequential's band activates too (ADR-0083 §3)
        significance_test=significance_test,
        policy=policy,
    )
    if len(strategies) == 1:
        return _compare_single(ctx, strategies)

    nested = (
        config.arbitration in ("nested", "nested_per_fold") and arbitration_splitter is not None
    )
    per_fold = config.arbitration == "nested_per_fold" and arbitration_splitter is not None
    arbitration_effective = "holdout"
    if nested and task.is_classification:
        # C5 OUTER (ADR-0052 §2): a globally rare class (< K) cannot be stratified into K arbitration folds
        # -> degrade the WHOLE arbitration to holdout rather than letting StratifiedKFold raise a raw ValueError.
        min_count = int(np.unique(y, return_counts=True)[1].min())
        if min_count < config.arbitration_n_splits:
            logger.warning(
                "nested arbitration needs >= arbitration_n_splits (%d) rows per class; the rarest class "
                "has %d -> falling back to holdout arbitration",
                config.arbitration_n_splits,
                min_count,
            )
            nested = per_fold = False
            arbitration_effective = "holdout_degraded_c5_outer"

    if nested:
        # composition wires policy/significance_test UNCONDITIONALLY (ADR-0083 §3b, also feeds the
        # single-strategy band); this assert only narrows the optional for the nested arb-splitter dispatch.
        assert (
            policy is not None
            and significance_test is not None
            and arbitration_splitter is not None
        )
        if per_fold:
            out = _compare_per_fold(
                ctx, strategies, arbitration_splitter, significance_test, policy
            )
            if out is not None:
                return out
            # every outer fold inner-C5-degraded -> per-fold infeasible; degrade to holdout arbitration
            logger.warning(
                "nested_per_fold: no outer fold had enough rows per class for inner selection -> "
                "falling back to holdout arbitration"
            )
            arbitration_effective = "holdout_degraded_c5_inner"
        else:
            return _compare_nested(ctx, strategies, arbitration_splitter, significance_test, policy)

    return _compare_holdout(ctx, strategies, arbitration_effective=arbitration_effective)
