"""OOF anti-leakage feature-selection spine (ADR-0044) — pure numpy, no adapters.

Mirror of :func:`crossfit_encode`: rank features on each fold's training part (``fit ⊕ es``, never the
test rows), L1-normalize per fold so no fold's magnitude dominates, average, then cut to a subset
(ADR-0044 §1/§3). The :class:`FeatureRanker` port supplies the per-fold scores; this module owns the
leakage-critical loop, the scale-invariant aggregation and the cutoff+floor — one place, every strategy
(Humble Object, NFR-FS-2). Synchronously testable on a fake ranker without any model training.
"""

from __future__ import annotations

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


def _strategy_base(name: str, n_runs: int, n_features: int) -> int:
    """Per-strategy ranker-fit count for the cost estimator (ADR-0058 §1, upper bound)."""
    if name == "null_importance":
        return 1 + n_runs
    if name == "sequential":
        return n_features * n_features  # O(n²) score_subset upper bound (no runtime reference)
    return 1


def estimate_fs_refits(
    fs: FeatureSelectionConfig, *, n_strategies: int, n_features: int, inner_n_splits: int
) -> int:
    """Deterministic upper-bound on selection ranker-refits for the cost budget (ADR-0058 §1).

    Canonical SELECTION cost = ``n_strategies × base × mult`` — numerically identical to the runtime per_fold
    WARNING (``n_strat × arbitration_n_splits × cv.n_splits × (1+n_runs)``). ``base`` is the per-strategy
    ranker-fit count (**max** over compared strategies, upper bound): ``null_importance`` -> ``1+n_runs``;
    ``sequential`` -> ``n_features²`` (O(n²) score_subset upper bound, no runtime reference); else ``1``.
    ``mult`` is the arbitration factor: ``holdout``/``nested`` -> ``inner_n_splits``; ``nested_per_fold`` ->
    ``arbitration_n_splits × inner_n_splits``. Pure arithmetic, no RNG (NFR-FSF-1). ``inner_n_splits`` is the
    main selection splitter's ``cv.n_splits`` (not a config field).
    """
    names = set(fs.compare) if fs.compare is not None else {fs.strategy}
    base = max(_strategy_base(n, fs.n_runs, n_features) for n in names)
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
    """OOF feature ranking on the evaluation folds -> one kept-column index subset (ADR-0044 §1).

    For each fold the ranker scores features on the train part (``fit ⊕ es``, never ``test``); per-fold
    scores are normalized and averaged; :func:`apply_cutoff` turns the aggregate into a subset with a
    ``>= 1`` floor. ``folds`` is the SAME list ``run_slice`` evaluates on, so selection and evaluation
    share folds (R-FS-FOLD-ALIGN). ``categorical`` is the per-column mask of ``x_full``. ``groups`` (M6d)
    is the per-row structure label, sliced to each fold's train rows for structure-aware rankers; the
    sliced labels stay aligned with the train rows, so the ranker still never sees test rows (ADR-0050).
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
    agg = scores / max(k, 1)
    return apply_cutoff(agg, config, ranker.auto_threshold(n_features))


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
