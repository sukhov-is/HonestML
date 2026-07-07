"""OOF anti-leakage feature-selection spine (ADR-0044) — pure numpy, no adapters.

Mirror of :func:`crossfit_encode`: rank features on each fold's training part (``fit ⊕ es``, never the
test rows), L1-normalize per fold so no fold's magnitude dominates, average, then cut to a subset
(ADR-0044 §1/§3). The :class:`FeatureRanker` port supplies the per-fold scores; this module owns the
leakage-critical loop, the scale-invariant aggregation and the cutoff+floor — one place, every strategy
(Humble Object, NFR-FS-2). Synchronously testable on a fake ranker without any model training.
"""

from __future__ import annotations

import math

import numpy as np

from honestml.core import FeatureRanker, Fold, get_logger
from honestml.core.config import FeatureSelectionConfig

logger = get_logger("application.feature_selection")


def structure_labels(
    groups: np.ndarray | None,
    times: np.ndarray | None,
    block_size: int,
    *,
    mode: str = "rank",
    window: float | None = None,
) -> np.ndarray | None:
    """Per-row structure label for structure-aware ``null_importance`` (M6d ADR-0050; M6e ADR-0055).

    ``group`` scheme -> the group array itself (``mode``/``window`` ignored). ``timeseries`` ->
    ``mode="rank"`` (M6d): equal-COUNT blocks of ``block_size`` rows by **time rank**
    (``argsort(argsort(t))``, valid at ~regular frequency); ``mode="time_window"`` (M6e ADR-0055): equal-Δt
    windows of width ``window`` over the **raw times**, densified to ``0..k-1`` so empty windows of an
    irregular series leave no gaps in the label space. Otherwise ``None`` (i.i.d. -> uniform permutation).
    The one source of the structure block, shared by the null permutation and the significance block_index
    (ADR-0053 §3), so the block semantics can never diverge. ``window`` is guaranteed non-None under
    ``time_window`` by the config validator (ADR-0055 §2); no defensive re-check here.
    """
    if groups is not None:
        return groups
    if times is not None:
        if mode == "time_window":
            assert (
                window is not None
            )  # config validator guarantees this under time_window (ADR-0055 §2)
            t = np.asarray(times, dtype=np.float64)
            raw = ((t - float(t.min())) / window).astype(np.int64)
            return np.unique(raw, return_inverse=True)[1].astype(np.int64)
        rank = np.argsort(np.argsort(times, kind="stable"), kind="stable")
        return (rank // block_size).astype(np.int64)
    return None


def _degenerate_counts(block_labels: np.ndarray, y: np.ndarray) -> int:
    """Count structure blocks whose target is constant (< 2 distinct values) — O(n), vectorized (ADR-0059 §2).

    Replaces the O(n_blocks·n) masked loop (one ``y[labels==b]`` pass per block): densify both axes, count
    distinct ``(block, target)`` pairs, tally blocks holding a single target value. A 1-row block (common
    under ``time_window`` densify) is constant -> degenerate, matching the reference exactly.
    """
    _, y_codes = np.unique(y, return_inverse=True)
    block_ids, block_codes = np.unique(block_labels, return_inverse=True)
    n_classes = int(y_codes.max()) + 1
    pairs = np.unique(block_codes.astype(np.int64) * n_classes + y_codes)
    distinct_per_block = np.bincount(pairs // n_classes, minlength=block_ids.size)
    return int(np.count_nonzero(distinct_per_block < 2))


def _refine_steps(fs: FeatureSelectionConfig, n_features: int) -> int:
    """Deterministic upper bound on the refinement trajectory length (ADR-0100).

    Mirrors :func:`refine_trajectory` step-for-step from the worst-case stage-1 survivor count
    (``top_k``/``top_frac`` bound it; ``auto`` has no a-priori bound below ``n``). Each trajectory
    point costs one OOF evaluation = ``inner_n_splits`` ranker-model fits, so the count adds to the
    per-strategy ``base``. Pure loop arithmetic, no RNG (NFR-FSF-1).
    """
    if not fs.refine:
        return 0
    if fs.cutoff == "top_k":
        k0 = min(fs.top_k or n_features, n_features)
    elif fs.cutoff == "top_frac":
        k0 = max(1, math.ceil(fs.top_frac * n_features))
    else:  # auto
        k0 = n_features
    # apply_cutoff raises the survivor set to the min_features floor, so the trajectory starts from that
    # many features (F127: keeps the estimate a true upper bound when min_features > the cutoff size)
    k0 = max(k0, min(max(1, fs.min_features), n_features))
    floor = max(1, fs.seq_min_features)
    steps = 1  # trajectory[0]: the uncapped survivor set
    cur = k0
    if fs.refine_max_features is not None:
        cap = max(
            fs.refine_max_features, floor
        )  # F126: the cap never truncates below the descent floor
        if cur > cap:
            cur = cap
            steps += 1
    while cur > floor:
        cur -= min(max(1, math.ceil(cur * fs.refine_drop_frac)), cur - floor)
        steps += 1
    return steps


def _strategy_base(name: str, fs: FeatureSelectionConfig, n_features: int) -> int:
    """Per-strategy ranker-fit count for the cost estimator (ADR-0058 §1, upper bound)."""
    if name == "sequential":
        return n_features * n_features  # O(n²) score_subset upper bound (no runtime reference)
    base = 1 + fs.n_runs if name == "null_importance" else 1
    return base + _refine_steps(fs, n_features)


def estimate_fs_refits(
    fs: FeatureSelectionConfig, *, n_strategies: int, n_features: int, inner_n_splits: int
) -> int:
    """Deterministic upper-bound on selection ranker-refits for the cost budget (ADR-0058 §1).

    Canonical SELECTION cost = ``n_strategies × base × mult`` — numerically identical to the runtime per_fold
    WARNING (``n_strat × arbitration_n_splits × cv.n_splits × (1+n_runs)``). ``base`` is the per-strategy
    ranker-fit count (**max** over compared strategies, upper bound): ``null_importance`` -> ``1+n_runs``;
    ``sequential`` -> ``n_features²`` (O(n²) score_subset upper bound, no runtime reference); else ``1``.
    Ranker strategies with ``refine`` add the trajectory-length bound (:func:`_refine_steps`, ADR-0100 —
    one OOF evaluation per visited subset). ``mult`` is the arbitration factor: ``holdout``/``nested`` ->
    ``inner_n_splits``; ``nested_per_fold`` -> ``arbitration_n_splits × inner_n_splits``. Pure arithmetic,
    no RNG (NFR-FSF-1). ``inner_n_splits`` is the main selection splitter's ``cv.n_splits`` (not a config
    field).
    """
    names = set(fs.compare) if fs.compare is not None else {fs.strategy}
    base = max(_strategy_base(n, fs, n_features) for n in names)
    mult = (
        fs.arbitration_n_splits * inner_n_splits
        if fs.arbitration == "nested_per_fold"
        else inner_n_splits
    )
    return n_strategies * base * mult


def _normalize_fold(imp: np.ndarray) -> np.ndarray:
    """Scale-normalize one fold's scores before aggregation (ADR-0044 §1, fix A4).

    Non-negative importances -> L1 share (``imp / imp.sum()``) so a fold with a larger raw magnitude
    does not dominate the average; an all-zero vector -> zeros (no division by zero). Signed scores
    (``random_probe`` margins) are already fold-relative and pass through unchanged.
    """
    if np.any(imp < 0):
        return imp
    total = float(imp.sum())
    if total <= 0.0:
        return np.zeros_like(imp)
    return imp / total


def aggregate_scores(
    x_full: np.ndarray,
    y: np.ndarray,
    folds: list[Fold],
    *,
    ranker: FeatureRanker,
    categorical: np.ndarray,
    config: FeatureSelectionConfig,
    sample_weight: np.ndarray | None = None,
    groups: np.ndarray | None = None,
) -> np.ndarray:
    """OOF feature ranking on the evaluation folds -> one aggregate score vector (ADR-0044 §1).

    For each fold the ranker scores features on the train part (``fit ⊕ es``, never ``test``); per-fold
    scores are normalized and averaged. The cascade (ADR-0100) ranks ONCE and reuses this aggregate for
    both the cutoff and the refinement drop order. ``folds`` is the SAME list ``run_slice`` evaluates on,
    so selection and evaluation share folds (R-FS-FOLD-ALIGN). ``categorical`` is the per-column mask of
    ``x_full``. ``groups`` (M6d) is the per-row structure label, sliced to each fold's train rows for
    structure-aware rankers; the sliced labels stay aligned with the train rows, so the ranker still
    never sees test rows (ADR-0050).
    """
    n_features = x_full.shape[1]
    random_state = config.random_state if config.random_state is not None else 0
    scores = np.zeros(n_features, dtype=np.float64)
    k = 0
    for fold in folds:
        train_idx = (
            fold.fit_idx if fold.es_idx.size == 0 else np.concatenate([fold.fit_idx, fold.es_idx])
        )
        sw = sample_weight[train_idx] if sample_weight is not None else None
        imp = np.asarray(
            ranker.rank(
                x_full[train_idx],
                y[train_idx],
                categorical=categorical,
                random_state=random_state,
                sample_weight=sw,
                groups=groups[train_idx] if groups is not None else None,
            ),
            dtype=np.float64,
        )
        if imp.shape != (n_features,) or not bool(np.all(np.isfinite(imp))):
            raise ValueError(
                f"ranker {ranker.name!r} returned an invalid score vector: shape {imp.shape} "
                f"(expected ({n_features},)), all-finite={bool(np.all(np.isfinite(imp)))}"
            )
        scores += _normalize_fold(imp)
        k += 1
    return scores / max(k, 1)


def select_features(
    x_full: np.ndarray,
    y: np.ndarray,
    folds: list[Fold],
    *,
    ranker: FeatureRanker,
    categorical: np.ndarray,
    config: FeatureSelectionConfig,
    sample_weight: np.ndarray | None = None,
    groups: np.ndarray | None = None,
) -> tuple[int, ...]:
    """Single-cut selection: :func:`aggregate_scores` + :func:`apply_cutoff` (ADR-0044 §1).

    The legacy M6b path (``refine=False``); the cascade path composes the same pieces with
    :func:`refine_trajectory` in the application compare (ADR-0100).
    """
    agg = aggregate_scores(
        x_full,
        y,
        folds,
        ranker=ranker,
        categorical=categorical,
        config=config,
        sample_weight=sample_weight,
        groups=groups,
    )
    return apply_cutoff(agg, config, ranker.auto_threshold(x_full.shape[1]))


def refine_trajectory(
    subset: tuple[int, ...],
    agg: np.ndarray,
    *,
    max_features: int | None,
    drop_frac: float,
    min_features: int,
) -> tuple[tuple[int, ...], ...]:
    """Deterministic backward-descent trajectory over the stage-1 survivors (ADR-0100).

    Pure arithmetic over the aggregate score vector — zero ``score_subset`` calls; the honest
    Same-OOF scoring of every visited subset happens in ``_band_over_trajectory``. The drop order
    is descending ``agg`` with :func:`apply_cutoff`'s stable tie semantics. ``trajectory[0]`` is
    the UNCAPPED survivor set, so the band's anchor can veto a harmful ``max_features`` truncation
    (the capped top-k set is the next point when it applies); each step then drops the
    ``max(1, ceil(len*drop_frac))`` weakest members, clamped to the ``>= max(1, min_features)``
    floor. Sizes strictly decrease; every subset is emitted sorted by column position (FR-FS-7).
    """
    floor = max(1, min_features)
    members = set(subset)
    order = [int(i) for i in np.argsort(-agg, kind="stable") if int(i) in members]
    trajectory: list[tuple[int, ...]] = [tuple(sorted(order))]
    current = order
    if max_features is not None:
        cap = max(max_features, floor)  # F126: the cap never truncates below the descent floor
        if len(current) > cap:
            current = current[:cap]
            trajectory.append(tuple(sorted(current)))
    while len(current) > floor:
        drop = min(max(1, math.ceil(len(current) * drop_frac)), len(current) - floor)
        current = current[:-drop]
        trajectory.append(tuple(sorted(current)))
    return tuple(trajectory)


def mass_floor(agg: np.ndarray, refine_min_mass: float) -> int:
    """Smallest feature count covering ``refine_min_mass`` of the stage-1 importance mass (ADR-0104).

    Insurance against proxy-blindness: the cascade descent never prunes below the ``k`` strongest
    features whose cumulative (non-negative) ``agg`` mass reaches ``refine_min_mass``. Zero extra fits
    (``agg`` already computed). Signed rankers give negative margins that carry no signal mass, so the
    mass is over ``clip(agg, 0)``; ``0`` when disabled (``<= 0``) or there is no positive mass. A small
    tolerance at the crossing keeps ``searchsorted`` platform-stable (deterministic by construction).
    """
    if refine_min_mass <= 0.0:
        return 0
    pos = np.clip(agg, 0.0, None)
    total = float(pos.sum())
    if total == 0.0:
        return 0
    frac = np.sort(pos)[::-1].cumsum() / total
    return int(np.searchsorted(frac, refine_min_mass - 1e-12) + 1)


def apply_cutoff(
    agg: np.ndarray, config: FeatureSelectionConfig, auto_threshold: float
) -> tuple[int, ...]:
    """Turn an aggregate score vector into a kept-column index subset (ADR-0044 §3).

    Policies: ``top_k`` (k strongest), ``top_frac`` (strongest ``ceil(frac*n)``), ``auto``
    (``> auto_threshold``). A ``>= max(1, min_features)`` floor guarantees ``design_matrix`` never
    loses all features (§F9); truncation, floor and a sub-baseline keep log a WARNING. Indices are
    returned sorted by column position (relative ``schema.features`` order preserved, FR-FS-7).
    """
    n = agg.shape[0]
    order = np.argsort(-agg, kind="stable")
    if config.cutoff == "top_k":
        keep = order[: min(config.top_k or n, n)]
    elif config.cutoff == "top_frac":
        keep = order[: max(1, int(np.ceil(config.top_frac * n)))]
    else:  # auto
        keep = np.flatnonzero(agg > auto_threshold)
    floor = max(1, config.min_features)
    if keep.size < floor:
        keep = order[: min(floor, n)]
        logger.warning("feature selection floored to %d feature(s): cutoff left too few", keep.size)
    keep = np.sort(keep)
    if keep.size < n:
        logger.warning("feature selection kept %d of %d features", keep.size, n)
    if keep.size and bool(np.any(agg[keep] < 0.0)):
        logger.warning(
            "feature selection kept feature(s) scoring below the random-probe baseline; "
            "consider cutoff='auto'"
        )
    return tuple(int(i) for i in keep)
