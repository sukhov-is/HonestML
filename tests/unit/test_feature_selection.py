"""M6b: the feature-selection spine (ADR-0044) + design_matrix projection (ADR-0045).

Humble Object: select_features/apply_cutoff are exercised on fake rankers and plain arrays — no model
training, no polars (NFR-FS-2).
"""

from __future__ import annotations

import numpy as np
import pytest

from honestml.application import apply_cutoff, design_matrix, select_features
from honestml.application.feature_selection import (
    _degenerate_counts,
    _normalize_fold,
    aggregate_scores,
    estimate_fs_refits,
    mass_floor,
    refine_trajectory,
    structure_labels,
)
from honestml.core import ColumnRole, FeatureSchema, Fold
from honestml.core.config import FeatureSelectionConfig

pytestmark = pytest.mark.unit


def _folds(n: int, k: int = 3) -> list[Fold]:
    """k contiguous test blocks over n rows; train = the complement (es empty)."""
    blocks = np.array_split(np.arange(n), k)
    folds = []
    for b in blocks:
        test = np.asarray(b, dtype=int)
        train = np.setdiff1d(np.arange(n), test)
        folds.append(Fold(fit_idx=train, es_idx=np.empty(0, dtype=int), test_idx=test))
    return folds


class _SpyRanker:
    """Records the row-ids (column 0 of x) and weights it is called with; importance = column mean."""

    name = "spy"

    def __init__(self) -> None:
        self.seen_rows: list[np.ndarray] = []
        self.seen_weights: list[np.ndarray | None] = []

    def rank(self, x, y, *, categorical, random_state, sample_weight=None, groups=None):
        if x.shape[0] == 0:
            raise ValueError("empty training matrix")
        self.seen_rows.append(x[:, 0].astype(int).copy())
        self.seen_weights.append(None if sample_weight is None else sample_weight.copy())
        return np.abs(x).mean(axis=0)

    def auto_threshold(self, n_features):
        return 1.0 / n_features


class _FixedRanker:
    """Returns a fixed score vector scaled per call — to probe scale-invariant aggregation."""

    name = "fixed"

    def __init__(self, base: np.ndarray, scales: list[float]) -> None:
        self.base = base
        self.scales = scales
        self.i = 0

    def rank(self, x, y, *, categorical, random_state, sample_weight=None, groups=None):
        s = self.scales[self.i % len(self.scales)]
        self.i += 1
        return self.base * s

    def auto_threshold(self, n_features):
        return 1.0 / n_features


_DEFAULT = FeatureSelectionConfig()


# --- spine: anti-leakage + aggregation (ADR-0044) ---


def test_ranker_sees_only_fit_part() -> None:
    # FR-FS-3 / NFR-FS-1: the ranker is called once per fold with exactly the train rows (no test_idx)
    n = 12
    x = np.column_stack([np.arange(n, dtype=float), np.random.default_rng(0).random((n, 2))])
    y = (np.arange(n) % 2).astype(float)
    folds = _folds(n)
    spy = _SpyRanker()
    select_features(x, y, folds, ranker=spy, categorical=np.zeros(3, dtype=bool), config=_DEFAULT)
    assert len(spy.seen_rows) == len(folds)
    for fold, rows in zip(folds, spy.seen_rows, strict=True):
        assert set(rows.tolist()) == set(fold.fit_idx.tolist())
        assert not (set(rows.tolist()) & set(fold.test_idx.tolist()))


def test_target_permutation_in_test_fold_leaves_subset_unchanged() -> None:
    # NFR-FS-1 property: permuting y inside each fold's test part cannot change the subset (the ranker
    # only ever sees train-part targets). Proven without model training.
    n = 12
    rng = np.random.default_rng(1)
    x = np.column_stack([np.arange(n, dtype=float), rng.random((n, 3))])
    y = (np.arange(n) % 2).astype(float)
    folds = _folds(n)
    cfg = FeatureSelectionConfig(cutoff="top_k", top_k=2)
    base = select_features(
        x, y, folds, ranker=_SpyRanker(), categorical=np.zeros(4, dtype=bool), config=cfg
    )
    y2 = y.copy()
    for fold in folds:
        y2[fold.test_idx] = rng.permutation(y2[fold.test_idx])
    perturbed = select_features(
        x, y2, folds, ranker=_SpyRanker(), categorical=np.zeros(4, dtype=bool), config=cfg
    )
    assert base == perturbed


def test_aggregate_scale_invariant() -> None:
    # fix A4: a fold scaled by a constant must not dominate the subset (per-fold L1 normalization)
    base = np.array([0.1, 0.5, 0.4])
    folds = _folds(9)
    x = np.zeros((9, 3))
    y = np.zeros(9)
    cfg = FeatureSelectionConfig(cutoff="top_k", top_k=2)
    unscaled = select_features(
        x, y, folds, ranker=_FixedRanker(base, [1.0]), categorical=np.zeros(3, bool), config=cfg
    )
    scaled = select_features(
        x,
        y,
        folds,
        ranker=_FixedRanker(base, [1.0, 100.0, 0.01]),
        categorical=np.zeros(3, bool),
        config=cfg,
    )
    assert unscaled == scaled == (1, 2)  # the two largest base components


def test_sample_weight_passed_to_ranker() -> None:
    n = 9
    x = np.column_stack([np.arange(n, dtype=float), np.ones((n, 1))])
    y = np.zeros(n)
    sw = np.linspace(0.1, 1.0, n)
    folds = _folds(n)
    spy = _SpyRanker()
    select_features(
        x, y, folds, ranker=spy, categorical=np.zeros(2, bool), config=_DEFAULT, sample_weight=sw
    )
    for fold, w in zip(folds, spy.seen_weights, strict=True):
        assert w is not None and np.allclose(w, sw[fold.fit_idx])


def test_invalid_score_vector_raises() -> None:
    class _BadRanker:
        name = "bad"

        def rank(self, x, y, *, categorical, random_state, sample_weight=None, groups=None):
            return np.array([np.nan, 1.0, 2.0])

        def auto_threshold(self, n_features):
            return 0.0

    with pytest.raises(ValueError, match="invalid score vector"):
        select_features(
            np.ones((6, 3)),
            np.zeros(6),
            _folds(6),
            ranker=_BadRanker(),
            categorical=np.zeros(3, bool),
            config=_DEFAULT,
        )


# --- structure_labels: block-by-time-window (M6e, ADR-0055) ---


def test_structure_labels_rank_mode_matches_m6d() -> None:
    # default mode="rank": equal-COUNT blocks by time rank (unchanged from M6d)
    times = np.array([10.0, 40.0, 20.0, 30.0, 50.0, 5.0])
    labels = structure_labels(None, times, block_size=2)
    # ranks: 5->1->? ; sorted times [5,10,20,30,40,50] -> rank of each row, //2
    assert labels is not None and labels.tolist() == [0, 2, 1, 1, 2, 0]


def test_structure_labels_time_window_equal_duration_on_irregular() -> None:
    # mode="time_window": equal-Δt windows over RAW times; irregular spacing -> blocks of equal DURATION
    # (unequal cardinality). Dense early region + sparse gap.
    times = np.array([0.0, 1.0, 2.0, 3.0, 100.0, 101.0])  # 4 rows in [0,3], 2 rows near 100
    labels = structure_labels(None, times, block_size=2, mode="time_window", window=10.0)
    assert labels is not None
    # rows 0..3 share one window; rows 4,5 share a later window; labels densified to 0..k-1 (no gaps)
    assert labels.tolist() == [0, 0, 0, 0, 1, 1]
    assert set(np.unique(labels).tolist()) == {0, 1}  # contiguous ids, empty windows dropped


def test_structure_labels_group_ignores_mode_and_window() -> None:
    groups = np.array([7, 7, 3, 3, 3])
    out = structure_labels(groups, np.arange(5.0), block_size=2, mode="time_window", window=1.0)
    assert out is not None and out.tolist() == groups.tolist()  # group scheme: block == group


# --- M6f: vectorized degenerate-count + cost estimator (ADR-0059 §2 / ADR-0058 §1) ---


def _ref_degenerate(labels: np.ndarray, y: np.ndarray) -> int:
    # the O(n_blocks·n) reference from M6e slice.py:394 — masks the full y per block
    return sum(1 for b in np.unique(labels) if np.unique(y[labels == b]).size < 2)


def test_degenerate_count_vectorized_equals_reference() -> None:
    rng = np.random.default_rng(0)
    tw = structure_labels(
        None, np.array([0.0, 1.0, 2.0, 3.0, 100.0, 101.0]), 2, mode="time_window", window=0.5
    )
    cases = [
        (
            np.array([0, 0, 1, 1, 2, 2]),
            np.array([0, 0, 1, 0, 1, 1]),
        ),  # block 0 constant -> degenerate
        (np.repeat(np.arange(5), 2), rng.integers(0, 2, 10)),  # rank-like equal-count blocks
        (
            tw,
            np.array([1, 0, 1, 0, 1, 0]),
        ),  # time_window densify -> all 1-row blocks (each degenerate)
        (np.array([7, 7, 3, 3, 3]), np.array([1, 1, 0, 0, 0])),  # group-like, all constant
    ]
    for labels, y in cases:
        assert labels is not None
        assert _degenerate_counts(labels, y) == _ref_degenerate(labels, y)
    assert (
        _degenerate_counts(tw, np.array([1, 0, 1, 0, 1, 0])) == 6
    )  # 6 one-row blocks all degenerate


# --- cascade refinement trajectory (ADR-0100) ---


def test_refine_trajectory_shape_and_order() -> None:
    # agg descending: 9,8,...,0 -> survivors {0..9}; cap 6; drop 25% per step; floor 2
    agg = np.arange(10, dtype=float)  # column 9 strongest
    subset = tuple(range(10))
    traj = refine_trajectory(subset, agg, max_features=6, drop_frac=0.25, min_features=2)
    assert traj[0] == subset  # uncapped survivor set anchors the band
    assert traj[1] == (4, 5, 6, 7, 8, 9)  # capped top-6 by agg
    sizes = [len(t) for t in traj]
    assert sizes == sorted(sizes, reverse=True) and len(set(sizes)) == len(sizes)  # strictly dec
    assert sizes[-1] == 2  # floor reached exactly (drop clamped, never overshoots)
    assert all(t == tuple(sorted(t)) for t in traj)  # column-position order (FR-FS-7)
    # each step drops the weakest tail of the current subset
    assert traj[2] == (6, 7, 8, 9)  # ceil(6*0.25)=2 dropped: columns 4,5 (weakest)
    # deterministic
    assert traj == refine_trajectory(subset, agg, max_features=6, drop_frac=0.25, min_features=2)


def test_refine_trajectory_no_cap_and_tiny_subset() -> None:
    agg = np.array([0.4, 0.1, 0.3, 0.2])
    traj = refine_trajectory((0, 1, 2, 3), agg, max_features=None, drop_frac=0.5, min_features=1)
    assert traj[0] == (0, 1, 2, 3)
    assert traj[1] == (0, 2)  # ceil(4*0.5)=2 weakest dropped (cols 1,3)
    assert traj[-1] == (0,)
    # subset at the floor -> single-point trajectory (the caller skips refinement anyway)
    assert refine_trajectory((2,), agg, max_features=None, drop_frac=0.5, min_features=1) == ((2,),)


def test_refine_trajectory_tie_order_matches_apply_cutoff() -> None:
    # equal scores: stable argsort ties toward the LOWER column index being stronger, exactly
    # like apply_cutoff's order (parity keeps the cascade cut and descent consistent)
    agg = np.array([0.5, 0.5, 0.5, 0.5])
    traj = refine_trajectory((0, 1, 2, 3), agg, max_features=2, drop_frac=0.5, min_features=1)
    assert traj[1] == (0, 1)  # top-2 under stable ties


def test_select_features_equals_aggregate_plus_cutoff() -> None:
    # refactor safety: select_features (M6b path) == apply_cutoff(aggregate_scores(...)) byte-for-byte
    x = np.column_stack([np.arange(30, dtype=float), np.ones(30), -np.arange(30, dtype=float)])
    y = (np.arange(30) % 2).astype(int)
    folds = _folds(30)
    ranker = _FixedRanker(np.array([0.7, 0.1, 0.2]), scales=[1.0])
    cfg = FeatureSelectionConfig(cutoff="top_k", top_k=2, refine=False, random_state=0)
    direct = select_features(
        x, y, folds, ranker=ranker, categorical=np.zeros(3, dtype=bool), config=cfg
    )
    agg = aggregate_scores(
        x, y, folds, ranker=ranker, categorical=np.zeros(3, dtype=bool), config=cfg
    )
    assert direct == apply_cutoff(agg, cfg, ranker.auto_threshold(3))


def test_estimate_fs_refits_matches_compare_formula() -> None:
    # per_fold == runtime compare formula (feature_compare:617-622): n_strat × K_outer × inner × per_fit;
    # refine=False reproduces the pre-cascade numbers exactly (ADR-0100 back-compat)
    fs = FeatureSelectionConfig(
        compare=("importance", "null_importance"),
        arbitration="nested_per_fold",
        arbitration_n_splits=4,
        n_runs=10,
        refine=False,
    )
    per_fit = 1 + fs.n_runs
    assert (
        estimate_fs_refits(fs, n_strategies=2, n_features=8, inner_n_splits=5)
        == 2 * 4 * 5 * per_fit
    )
    # holdout: no outer factor -> n_strat × inner × per_fit
    fs_h = FeatureSelectionConfig(
        compare=("importance", "null_importance"), n_runs=10, refine=False
    )
    assert estimate_fs_refits(fs_h, n_strategies=2, n_features=8, inner_n_splits=5) == 2 * 5 * (
        1 + 10
    )
    # sequential -> n_features² upper bound (no runtime reference; excluded from byte-identity, ADR-0058 §1);
    # the wrapper is unaffected by the refine default (its own descent)
    fs_s = FeatureSelectionConfig(strategy="sequential")
    assert (
        estimate_fs_refits(fs_s, n_strategies=1, n_features=6, inner_n_splits=5) == 1 * (6 * 6) * 5
    )


def test_estimate_fs_refits_adds_refine_trajectory_bound() -> None:
    # ADR-0100: a ranker strategy with refine adds the deterministic trajectory-length bound to base.
    # cutoff='auto' has no a-priori bound below n -> k0=n; verify against refine_trajectory's actual
    # length from a full survivor set (the estimator mirrors it step-for-step).
    n = 12
    fs = FeatureSelectionConfig(
        strategy="importance", cutoff="auto", refine_max_features=8, refine_drop_frac=0.25
    )
    traj = refine_trajectory(
        tuple(range(n)), np.arange(n, dtype=float), max_features=8, drop_frac=0.25, min_features=1
    )
    expected_base = 1 + len(traj)  # 1 ranker fit + one OOF eval per visited subset
    assert (
        estimate_fs_refits(fs, n_strategies=1, n_features=n, inner_n_splits=3) == expected_base * 3
    )
    # top_k bounds the survivor count below the cap -> no cap point in the bound
    fs_k = FeatureSelectionConfig(strategy="importance", cutoff="top_k", top_k=4)
    traj_k = refine_trajectory(
        tuple(range(4)), np.arange(4, dtype=float), max_features=200, drop_frac=0.05, min_features=1
    )
    assert estimate_fs_refits(fs_k, n_strategies=1, n_features=n, inner_n_splits=3) == (
        (1 + len(traj_k)) * 3
    )


def test_refine_trajectory_cap_clamped_to_floor() -> None:
    # F126: refine_max_features below the seq_min_features floor must not truncate below the floor -> the
    # cap point is clamped to the floor, so no shipped subset drops under the documented minimum.
    agg = np.arange(10, dtype=float)
    traj = refine_trajectory(tuple(range(10)), agg, max_features=2, drop_frac=0.25, min_features=5)
    assert all(len(t) >= 5 for t in traj)  # never below the floor despite max_features=2 < 5
    assert traj[0] == tuple(range(10))  # uncapped survivor anchor
    assert traj[-1] == (5, 6, 7, 8, 9)  # capped to the floor (5), not to max_features (2)


def test_mass_floor_profiles_and_signed() -> None:
    # ADR-0104/FR-3/NFR-5: dominant importance -> low floor; spread -> high; signed clamped; off/no-mass -> 0
    assert (
        mass_floor(np.array([0.99, 0.005, 0.003, 0.002]), 0.95) == 1
    )  # one feature covers the mass
    assert mass_floor(np.full(20, 1.0), 0.95) == 19  # uniform: need 19/20 to reach 0.95
    assert (
        mass_floor(np.array([0.8, -0.5, 0.2]), 0.9) == 2
    )  # negative clamped to 0, not counted as mass
    assert mass_floor(np.array([0.99, 0.01]), 0.0) == 0  # refine_min_mass=0 -> disabled
    assert mass_floor(np.array([-1.0, -2.0]), 0.99) == 0  # no positive mass -> not active


def test_mass_floor_adaptive_by_importance_concentration() -> None:
    # Empirical acceptance (ADR-0104, Major-2): the floor ADAPTS to importance concentration — spread
    # importance (adult-like TE encodings) -> HIGH floor (blocks the -2.2pp over-prune to 5 features);
    # a truly dominant feature -> floor 1 (a compact size stays reachable when the data supports it).
    # The real adult/titanic showcase re-run is the owner's pre-release validation (not cached locally).
    rng = np.random.default_rng(0)
    spread = rng.uniform(0.5, 1.0, size=30)  # ~uniform importance over 30 features
    assert mass_floor(spread, 0.99) > 5  # the harmful 5-feature cut is floored out
    dominant = np.concatenate([[100.0], np.full(20, 1e-4)])  # one feature carries ~all the mass
    assert mass_floor(dominant, 0.99) == 1


def test_mass_floor_keeps_cost_estimate_upper_bound() -> None:
    # NFR-3: mass_floor only RAISES the runtime descent floor -> shortens the trajectory; the a-priori
    # estimate (which ignores the data-dependent mass_floor) stays a valid upper bound.
    n = 40
    agg = np.concatenate([np.full(10, 1.0), np.zeros(n - 10)])  # 10 features carry all the mass
    fs = FeatureSelectionConfig(
        strategy="importance",
        cutoff="top_frac",
        top_frac=1.0,
        refine_drop_frac=0.25,
        refine_min_mass=0.99,
    )
    floor = max(1, fs.min_features, fs.seq_min_features, mass_floor(agg, fs.refine_min_mass))  # ~10
    actual = refine_trajectory(
        tuple(range(n)),
        agg,
        max_features=fs.refine_max_features,
        drop_frac=0.25,
        min_features=floor,
    )
    est = estimate_fs_refits(fs, n_strategies=1, n_features=n, inner_n_splits=3)
    assert est >= (1 + len(actual)) * 3  # a-priori estimate >= actual mass-floored trajectory cost


def test_estimate_fs_refits_upper_bound_when_min_features_exceeds_cutoff() -> None:
    # F127: apply_cutoff raises a small top_k survivor set to the min_features floor, so _refine_steps must
    # start the trajectory from that floor -> the cost estimate stays a true upper bound (was: understated).
    n = 20
    fs = FeatureSelectionConfig(
        strategy="importance", cutoff="top_k", top_k=2, min_features=8, refine_drop_frac=0.25
    )
    # the shipped survivor set is min_features=8 (floored), so the bound mirrors a trajectory from 8
    traj = refine_trajectory(
        tuple(range(8)), np.arange(8, dtype=float), max_features=200, drop_frac=0.25, min_features=1
    )
    assert (
        estimate_fs_refits(fs, n_strategies=1, n_features=n, inner_n_splits=3)
        == (1 + len(traj)) * 3
    )


# --- _normalize_fold (ADR-0044 §1) ---


def test_normalize_fold_l1_and_zero_sum_and_signed() -> None:
    assert np.allclose(_normalize_fold(np.array([1.0, 3.0])), [0.25, 0.75])
    assert np.allclose(_normalize_fold(np.zeros(3)), np.zeros(3))  # no division by zero
    signed = np.array([-1.0, 2.0])
    assert np.allclose(_normalize_fold(signed), signed)  # margins pass through unchanged


# --- apply_cutoff (ADR-0044 §3) ---


def test_apply_cutoff_top_k_top_frac_auto() -> None:
    agg = np.array([0.4, 0.1, 0.3, 0.2])
    assert apply_cutoff(agg, FeatureSelectionConfig(cutoff="top_k", top_k=2), 0.25) == (0, 2)
    assert apply_cutoff(agg, FeatureSelectionConfig(cutoff="top_frac", top_frac=0.5), 0.25) == (
        0,
        2,
    )
    # auto keeps strictly-above-threshold (0.25): {0.4, 0.3}
    assert apply_cutoff(agg, FeatureSelectionConfig(cutoff="auto"), 0.25) == (0, 2)


def test_apply_cutoff_top_k_clamped_to_n_features() -> None:
    agg = np.array([0.6, 0.4])
    assert apply_cutoff(agg, FeatureSelectionConfig(cutoff="top_k", top_k=5), 0.5) == (0, 1)


def test_apply_cutoff_empty_result_floored_to_one() -> None:
    # auto threshold above everything -> 0 kept -> floor to the single strongest (FR-FS-5, §F9)
    agg = np.array([0.1, 0.2, 0.05])
    keep = apply_cutoff(agg, FeatureSelectionConfig(cutoff="auto"), 1.0)
    assert keep == (1,)


def test_apply_cutoff_preserves_feature_order() -> None:
    agg = np.array([0.2, 0.9, 0.5, 0.7])
    # top_k=3 keeps cols {1,3,2} but returns them in column order
    assert apply_cutoff(agg, FeatureSelectionConfig(cutoff="top_k", top_k=3), 0.25) == (1, 2, 3)


# --- design_matrix projection (ADR-0045 §2) ---


class _MatDataset:
    def __init__(self, numeric: np.ndarray, codes: np.ndarray, schema: FeatureSchema) -> None:
        self._n, self._c, self._s = numeric, codes, schema

    @property
    def schema(self):
        return self._s

    def to_numpy(self):
        return self._n

    def categorical_codes(self):
        return self._c


def _mat_schema(selected=None) -> FeatureSchema:
    schema = FeatureSchema(
        roles={"n1": ColumnRole.NUMERIC, "n2": ColumnRole.NUMERIC, "c1": ColumnRole.CATEGORICAL}
    )
    return schema if selected is None else schema.with_selected_features(selected)


def test_design_matrix_projects_in_features_order() -> None:
    num = np.array([[1.0, 2.0], [3.0, 4.0]])
    codes = np.array([[5], [6]], dtype=np.int64)
    ds = _MatDataset(num, codes, _mat_schema(selected=("c1", "n1")))  # given out of order
    out = design_matrix(ds)
    # projection preserves schema.features order (n1, n2, c1) -> kept {n1, c1} => columns [n1, c1]
    assert np.allclose(out, np.array([[1.0, 5.0], [3.0, 6.0]]))


def test_design_matrix_no_selection_returns_full() -> None:
    num = np.array([[1.0, 2.0]])
    codes = np.array([[5]], dtype=np.int64)
    out = design_matrix(_MatDataset(num, codes, _mat_schema()))
    assert out.shape == (1, 3)


def test_design_matrix_missing_selected_feature_raises() -> None:
    from honestml.core import SchemaValidationError

    ds = _MatDataset(
        np.array([[1.0, 2.0]]), np.array([[5]], dtype=np.int64), _mat_schema(selected=("ghost",))
    )
    with pytest.raises(SchemaValidationError, match="absent"):
        design_matrix(ds)
