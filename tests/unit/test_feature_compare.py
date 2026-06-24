"""M6c FR-FSC-4: honest-compare driver (arbitration, fail-fast, N=1) on fakes — no model training."""

from __future__ import annotations

import numpy as np
import pytest

from honestml.application.feature_compare import (
    _strategy_seed,
    compare_features,
    no_selection_gate,
)
from honestml.core import FeatureSelectionConfig, FeatureSelectionError, Fold

pytestmark = pytest.mark.unit


class _FixedSelector:
    """A FeatureSubsetSelector returning a preset subset (ignores score_subset)."""

    def __init__(self, name: str, subset: tuple[int, ...]) -> None:
        self.name = name
        self._subset = subset

    def select(self, x, y, folds, *, categorical, score_subset, random_state, sample_weight=None):
        return (self._subset,)  # single-point trajectory (ADR-0083): band collapses to this subset


class _RaisingSelector:
    name = "boom"

    def select(self, x, y, folds, *, categorical, score_subset, random_state, sample_weight=None):
        raise RuntimeError("pathological data")


class _FakeMetric:
    name = "size"
    greater_is_better = True
    needs = "value"
    optimum = float("inf")
    average = None
    proper_proba = False

    def score(self, y_true, y_pred, sample_weight=None):
        return float(np.mean(y_pred))


class _FakeTask:
    kind = "regression"
    is_classification = False


def _fit_predict(x_tr, y_tr, x_te, sample_weight, random_state):
    # estimator-agnostic stand-in: prediction == subset width -> a wider subset scores higher
    return None, np.full(x_te.shape[0], float(x_te.shape[1])), None


class _FakeDataset:
    def __init__(self, n: int) -> None:
        self._n = n

    def take(self, indices):
        return _FakeDataset(len(indices))


class _FakeSplitter:
    def split(self, dataset):
        n = dataset._n
        half = max(1, n // 2)
        return [Fold(np.arange(half), np.array([], dtype=int), np.arange(half, n))]


def _carve(dataset, fraction, random_state):
    n = dataset._n
    cut = int(round(n * (1.0 - fraction)))
    return np.arange(cut), np.arange(cut, n)


def _run(strategies, *, n=40, n_features=4):
    # config.compare just needs to be a valid non-None tuple; the driver iterates the passed
    # `strategies` list (resolved adapters), not config.compare (FSStrategy names live in config only).
    rng = np.random.RandomState(0)
    x = rng.random((n, n_features))
    y = rng.random(n)
    cfg = FeatureSelectionConfig(compare=("importance", "random_probe"), selection_holdout=0.3)
    return compare_features(
        _FakeDataset(n),
        x,
        y,
        task=_FakeTask(),
        metric=_FakeMetric(),
        strategies=strategies,
        config=cfg,
        splitter=_FakeSplitter(),
        carve=_carve,
        fit_predict=_fit_predict,
        categorical=np.zeros(n_features, dtype=bool),
        feature_names=[f"f{i}" for i in range(n_features)],
        sample_weight=None,
        random_state=42,
    )


def test_arbiter_picks_widest_subset_winner() -> None:
    # under _fit_predict, a wider subset scores higher -> the size-3 subset wins
    out = _run([("a", _FixedSelector("a", (0, 1))), ("b", _FixedSelector("b", (0, 1, 2)))])
    assert out.winner == "b"
    assert out.winner_idx == (0, 1, 2)
    assert out.winner_subset == ("f0", "f1", "f2")
    assert {n for n, _, _ in out.per_strategy} == {"a", "b"}


def test_winner_tracks_subset_not_strategy_order() -> None:
    # swap which strategy carries the wider subset: the winner follows the subset, not the order
    out = _run([("a", _FixedSelector("a", (0, 1, 2))), ("b", _FixedSelector("b", (0, 1)))])
    assert out.winner == "a"


def test_compare_is_deterministic() -> None:
    s = [("a", _FixedSelector("a", (0, 1))), ("b", _FixedSelector("b", (0, 1, 2)))]
    assert _run(s).per_strategy == _run(s).per_strategy


def test_single_strategy_skips_carve() -> None:
    calls = []

    def _spy_carve(dataset, fraction, random_state):
        calls.append(1)
        return _carve(dataset, fraction, random_state)

    rng = np.random.RandomState(0)
    x, y = rng.random((20, 3)), rng.random(20)
    out = compare_features(
        _FakeDataset(20),
        x,
        y,
        task=_FakeTask(),
        metric=_FakeMetric(),
        strategies=[("solo", _FixedSelector("solo", (0, 2)))],
        config=FeatureSelectionConfig(compare=("importance",)),
        splitter=_FakeSplitter(),
        carve=_spy_carve,
        fit_predict=_fit_predict,
        categorical=np.zeros(3, dtype=bool),
        feature_names=["f0", "f1", "f2"],
        sample_weight=None,
        random_state=1,
    )
    assert out.winner == "solo" and out.winner_idx == (0, 2)
    assert calls == []  # N=1 -> no carve (ADR-0046 §3)


def test_strategy_failure_fails_fast() -> None:
    with pytest.raises(FeatureSelectionError, match="boom"):
        _run([("a", _FixedSelector("a", (0, 1))), ("boom", _RaisingSelector())])


def test_strategy_seed_is_stable_and_isolated() -> None:
    # same (name, seed) -> same value across calls; different names -> different seeds (FR-FSC-7)
    assert _strategy_seed("importance", 42) == _strategy_seed("importance", 42)
    assert _strategy_seed("importance", 42) != _strategy_seed("sequential", 42)


# --- M6d nested-CV arbitration + significance-winner (ADR-0052/0053, FR-FSH-6/8) ---

from honestml.core import SelectionPolicy  # noqa: E402


class _FakeKSplitter:
    """Two arbitration folds that partition the rows (DEV K-fold stand-in)."""

    def split(self, dataset):
        n = dataset._n
        half = max(1, n // 2)
        return [
            Fold(np.arange(half, n), np.array([], dtype=int), np.arange(half)),
            Fold(np.arange(half), np.array([], dtype=int), np.arange(half, n)),
        ]


class _AllEquivalent:
    """SignificanceTest declaring every candidate indistinguishable from the anchor (band = all)."""

    def equivalent(self, a, b, y_true, *, alpha=0.05, block_index=None, sample_weight=None):
        return True


class _NoneEquivalent:
    """Inert test (like NoSignificanceTest): nothing is equivalent -> band collapses to argmax."""

    def equivalent(self, a, b, y_true, *, alpha=0.05, block_index=None, sample_weight=None):
        return False


def _run_nested(strategies, sig_test, *, n=40, n_features=4):
    rng = np.random.RandomState(0)
    x, y = rng.random((n, n_features)), rng.random(n)
    cfg = FeatureSelectionConfig(compare=("importance", "random_probe"), arbitration="nested")
    return compare_features(
        _FakeDataset(n),
        x,
        y,
        task=_FakeTask(),
        metric=_FakeMetric(),
        strategies=strategies,
        config=cfg,
        splitter=_FakeSplitter(),
        carve=_carve,
        fit_predict=_fit_predict,
        categorical=np.zeros(n_features, dtype=bool),
        feature_names=[f"f{i}" for i in range(n_features)],
        sample_weight=None,
        random_state=42,
        arbitration_splitter=_FakeKSplitter(),
        significance_test=sig_test,
        policy=SelectionPolicy(greater_is_better=True),
    )


def test_nested_argmax_when_band_empty() -> None:
    # inert significance -> nested winner is the plain argmax (wider subset scores higher under _fit_predict)
    out = _run_nested(
        [("a", _FixedSelector("a", (0, 1))), ("b", _FixedSelector("b", (0, 1, 2)))],
        _NoneEquivalent(),
    )
    assert out.winner == "b" and out.winner_idx == (0, 1, 2)
    assert out.winner_rule == "argmax_band_empty"
    assert dict(out.per_strategy_std).keys() == {"a", "b"}


def test_nested_significance_picks_compact() -> None:
    # all strategies indistinguishable -> Occam tie-break keeps the SMALLEST subset (FR-FSH-8 differentiator)
    out = _run_nested(
        [("wide", _FixedSelector("wide", (0, 1, 2))), ("compact", _FixedSelector("compact", (0,)))],
        _AllEquivalent(),
    )
    assert out.winner == "compact" and out.winner_idx == (0,)
    assert out.winner_rule == "band_tiebreak"
    assert set(out.band_members) == {"wide", "compact"}


def test_nested_winner_deterministic() -> None:
    s = [("a", _FixedSelector("a", (0, 1))), ("b", _FixedSelector("b", (0, 1, 2)))]
    assert _run_nested(s, _NoneEquivalent()).winner == _run_nested(s, _NoneEquivalent()).winner


def test_nested_warns_fit_estimate_and_dead_selection_holdout(caplog) -> None:
    # NFR-FSH-2 + ADR-0052 §1: nested logs the N*K fit estimate and that selection_holdout is ignored
    with caplog.at_level("WARNING"):
        _run_nested(
            [("a", _FixedSelector("a", (0, 1))), ("b", _FixedSelector("b", (0, 1, 2)))],
            _NoneEquivalent(),
        )
    msgs = " ".join(r.message for r in caplog.records)
    assert "nested arbitration" in msgs and "selection_holdout is ignored" in msgs


class _FakeClsTask:
    kind = "binary"
    is_classification = True
    positive_label = None


def test_nested_degrades_to_holdout_on_rare_class(caplog) -> None:
    # C5 (ADR-0052 §2): a class rarer than arbitration_n_splits cannot stratify K folds -> holdout fallback
    # (a graceful WARNING, not a raw StratifiedKFold ValueError).
    rng = np.random.RandomState(0)
    n, nf = 40, 4
    x, y = rng.random((n, nf)), np.zeros(n, dtype=int)
    y[:2] = 1  # rarest class has 2 rows < default arbitration_n_splits=5
    with caplog.at_level("WARNING"):
        out = compare_features(
            _FakeDataset(n),
            x,
            y,
            task=_FakeClsTask(),
            metric=_FakeMetric(),
            strategies=[("a", _FixedSelector("a", (0, 1))), ("b", _FixedSelector("b", (0, 1, 2)))],
            config=FeatureSelectionConfig(
                compare=("importance", "random_probe"), arbitration="nested"
            ),
            splitter=_FakeSplitter(),
            carve=_carve,
            fit_predict=_fit_predict,
            categorical=np.zeros(nf, dtype=bool),
            feature_names=[f"f{i}" for i in range(nf)],
            sample_weight=None,
            random_state=42,
            arbitration_splitter=_FakeKSplitter(),
            significance_test=_NoneEquivalent(),
            policy=SelectionPolicy(greater_is_better=True),
        )
    assert out.winner in {"a", "b"}  # ran via holdout fallback, no crash
    assert any("falling back to holdout" in r.message for r in caplog.records)


# --- M6e per-fold re-selection (ADR-0054, FR-FSE-1..4, NFR-FSE-1/3) ---


class _SpyPerFoldSelector:
    """Records the global row-ids (col 0 of x) it is re-selected on per outer fold; returns a fixed subset."""

    def __init__(self, name: str, subset: tuple[int, ...]) -> None:
        self.name = name
        self._subset = subset
        self.seen_rows: list[np.ndarray] = []

    def select(self, x, y, folds, *, categorical, score_subset, random_state, sample_weight=None):
        self.seen_rows.append(x[:, 0].astype(int).copy())
        return (self._subset,)  # single-point trajectory (ADR-0083)


def _run_per_fold(strategies, sig_test, *, n=40, n_features=4, y=None, groups=None):
    rng = np.random.RandomState(0)
    # col 0 = global row id so a spy can prove which rows the per-fold re-selection saw
    x = np.column_stack([np.arange(n, dtype=float), rng.random((n, n_features - 1))])
    if y is None:
        y = rng.random(n)
    cfg = FeatureSelectionConfig(
        compare=("importance", "random_probe"), arbitration="nested_per_fold"
    )
    return compare_features(
        _FakeDataset(n),
        x,
        y,
        task=_FakeTask(),
        metric=_FakeMetric(),
        strategies=strategies,
        config=cfg,
        splitter=_FakeSplitter(),
        carve=_carve,
        fit_predict=_fit_predict,
        categorical=np.zeros(n_features, dtype=bool),
        feature_names=[f"f{i}" for i in range(n_features)],
        sample_weight=None,
        random_state=42,
        groups=groups,
        arbitration_splitter=_FakeKSplitter(),
        significance_test=sig_test,
        policy=SelectionPolicy(greater_is_better=True),
    )


def test_per_fold_reselects_only_on_outer_train() -> None:
    # FR-FSE-2 / NFR-FSE-1: the ranker is re-invoked once per outer fold on EXACTLY that fold's train rows;
    # outer-test rows never reach selection (anti-leakage). _FakeKSplitter has 2 outer folds (each tr = a half).
    spy = _SpyPerFoldSelector("a", (0, 1))
    _run_per_fold([("a", spy), ("b", _FixedSelector("b", (0, 1, 2)))], _NoneEquivalent())
    assert len(spy.seen_rows) == 2  # one re-selection per outer fold
    halves = (
        {*range(20, 40)},
        {*range(0, 20)},
    )  # the two outer-train id sets (fit_idx of each K-fold)
    for rows in spy.seen_rows:
        assert set(rows.tolist()) in halves
        # the complementary half (this fold's outer-test) is never seen by the ranker
        other = halves[1] if set(rows.tolist()) == halves[0] else halves[0]
        assert not (set(rows.tolist()) & other)


def test_per_fold_ships_full_dev_subset_and_marks_effective() -> None:
    # FR-FSE-3: winner subset is re-derived on full DEV; arbitration_effective records the honest procedure ran
    out = _run_per_fold(
        [("a", _FixedSelector("a", (0, 1))), ("b", _FixedSelector("b", (0, 1, 2)))],
        _NoneEquivalent(),
    )
    assert out.winner == "b" and out.winner_idx == (0, 1, 2)  # full-DEV selection of the winner
    assert out.arbitration_effective == "nested_per_fold"
    assert dict(out.per_strategy_mean_features).keys() == {
        "a",
        "b",
    }  # raw mean per-fold sizes surfaced


def test_per_fold_significance_picks_compact() -> None:
    # FR-FSE-3 + ADR-0053: indistinguishable strategies -> Occam keeps the smaller (round(mean per-fold size))
    out = _run_per_fold(
        [("wide", _FixedSelector("wide", (0, 1, 2))), ("compact", _FixedSelector("compact", (0,)))],
        _AllEquivalent(),
    )
    assert out.winner == "compact" and out.winner_idx == (0,)
    assert out.winner_rule == "band_tiebreak"


def test_per_fold_deterministic() -> None:
    s = [("a", _FixedSelector("a", (0, 1))), ("b", _FixedSelector("b", (0, 1, 2)))]
    assert _run_per_fold(s, _NoneEquivalent()).winner == _run_per_fold(s, _NoneEquivalent()).winner


def test_per_fold_warns_cost_estimate(caplog) -> None:
    # NFR-FSE-2: per-fold logs a projected fit estimate distinguishing it from M6d nested
    with caplog.at_level("WARNING"):
        _run_per_fold(
            [("a", _FixedSelector("a", (0, 1))), ("b", _FixedSelector("b", (0, 1, 2)))],
            _NoneEquivalent(),
        )
    msgs = " ".join(r.message for r in caplog.records)
    assert "nested_per_fold arbitration" in msgs and "re-selects per outer fold" in msgs


def test_per_fold_inner_c5_is_fold_local(caplog) -> None:
    # ADR-0054 §6 (fix R2): a class rare on ONE outer-train (but >= K globally) degrades only THAT fold, not
    # the whole arbitration; the procedure still runs per-fold (no whole fallback to holdout).
    rng = np.random.RandomState(0)
    n, nf = 40, 4
    x = np.column_stack([np.arange(n, dtype=float), rng.random((n, nf - 1))])
    y = np.zeros(n, dtype=int)
    y[[0, 1, 2, 3, 4, 20]] = (
        1  # 6 globally (>= K=5 -> outer ok); only 1 in rows 20..39 -> one inner fold degrades
    )
    with caplog.at_level("WARNING"):
        out = compare_features(
            _FakeDataset(n),
            x,
            y,
            task=_FakeClsTask(),
            metric=_FakeMetric(),
            strategies=[("a", _FixedSelector("a", (0, 1))), ("b", _FixedSelector("b", (0, 1, 2)))],
            config=FeatureSelectionConfig(
                compare=("importance", "random_probe"), arbitration="nested_per_fold"
            ),
            splitter=_FakeSplitter(),
            carve=_carve,
            fit_predict=_fit_predict,
            categorical=np.zeros(nf, dtype=bool),
            feature_names=[f"f{i}" for i in range(nf)],
            sample_weight=None,
            random_state=42,
            arbitration_splitter=_FakeKSplitter(),
            significance_test=_NoneEquivalent(),
            policy=SelectionPolicy(greater_is_better=True),
        )
    assert (
        out.arbitration_effective == "per_fold_partial_c5_inner"
    )  # fold-local, not whole-arbitration fallback
    assert out.winner in {"a", "b"}  # still ran per-fold, no crash


class _CorrRanker:
    """A FeatureRanker scoring |corr(col, y)| on the train rows it sees — so the selected subset RESPONDS to
    the fold's data (unlike a fixed-subset selector). Proves per-fold re-selection is not a no-op (FR-FSE-2)."""

    name = "corr"

    def rank(self, x, y, *, categorical, random_state, sample_weight=None, groups=None):
        ys = y - y.mean()
        out = np.zeros(x.shape[1])
        for j in range(x.shape[1]):
            xs = x[:, j] - x[:, j].mean()
            d = float(np.sqrt((xs**2).sum() * (ys**2).sum()))
            out[j] = abs(float((xs * ys).sum()) / d) if d > 0 else 0.0
        return out

    def auto_threshold(self, n_features):
        return 1.0 / n_features


def test_per_fold_reselection_responds_to_fold_data() -> None:
    # FR-FSE-2 core: re-selection actually depends on each outer fold's data -> the two outer folds, whose
    # train halves carry DIFFERENT signal columns, select DIFFERENT subsets. A fixed-subset (M6d) procedure
    # could never produce this. (_FakeSplitter inner fold trains on the first half of each outer-train.)
    from honestml.application.feature_compare import _score_procedure

    rng = np.random.RandomState(0)
    n, nf = 40, 4
    x = rng.random((n, nf))
    y = np.empty(n)
    y[:20] = x[:20, 1] * 10.0  # first half: signal in col 1
    y[20:] = x[20:, 3] * 10.0  # second half: signal in col 3
    cfg = FeatureSelectionConfig(
        compare=("importance", "random_probe"),
        arbitration="nested_per_fold",
        cutoff="top_k",
        top_k=1,
    )
    arb_folds = _FakeKSplitter().split(
        _FakeDataset(n)
    )  # fold0 tr=rows20-39 (col3), fold1 tr=rows0-19 (col1)
    res = _score_procedure(
        _CorrRanker(),
        "corr",
        _FakeDataset(n),
        x,
        y,
        arb_folds,
        _FakeSplitter(),
        categorical=np.zeros(nf, dtype=bool),
        config=cfg,
        metric=_FakeMetric(),
        task=_FakeTask(),
        fit_predict=_fit_predict,
        sample_weight=None,
        random_state=0,
        global_classes=None,
        groups=None,
    )
    subsets = res[3]
    assert (
        len(subsets) == 2 and subsets[0] != subsets[1]
    )  # data-driven re-selection differs per fold


class _PurgedKSplitter:
    """One outer fold with a PURGE GAP: row 19 sits between train 0..18 and test 20..39 (excluded from both),
    mimicking TimeSeriesSplitter(purge>0). Lets us assert the gap row never reaches per-fold re-selection."""

    def split(self, dataset):
        n = dataset._n
        return [Fold(np.arange(19), np.array([], dtype=int), np.arange(20, n))]


def test_per_fold_purged_boundary_row_absent_from_reselection_tr() -> None:
    # FR-FSE-4 (assert on per-fold tr, not splitter output -> not vacuous): a purged outer splitter keeps the
    # boundary row out of the tr handed to _select_one, so re-selection never trains on a leaked-label row.
    n, nf = 40, 4
    rng = np.random.RandomState(0)
    x = np.column_stack(
        [np.arange(n, dtype=float), rng.random((n, nf - 1))]
    )  # col0 = global id for the spy
    spy = _SpyPerFoldSelector("a", (0, 1))
    compare_features(
        _FakeDataset(n),
        x,
        rng.random(n),
        task=_FakeTask(),
        metric=_FakeMetric(),
        strategies=[("a", spy), ("b", _FixedSelector("b", (0, 1, 2)))],
        config=FeatureSelectionConfig(
            compare=("importance", "random_probe"), arbitration="nested_per_fold"
        ),
        splitter=_FakeSplitter(),
        carve=_carve,
        fit_predict=_fit_predict,
        categorical=np.zeros(nf, dtype=bool),
        feature_names=[f"f{i}" for i in range(nf)],
        sample_weight=None,
        random_state=42,
        arbitration_splitter=_PurgedKSplitter(),
        significance_test=_NoneEquivalent(),
        policy=SelectionPolicy(greater_is_better=True),
    )
    seen = set(spy.seen_rows[0].tolist())
    assert seen == set(range(19))  # tr is exactly the purged train part (rows 0..18)
    assert 19 not in seen  # the purged boundary row never reaches re-selection
    assert not (seen & set(range(20, 40)))  # nor any outer-test row


class _FakeInner5:
    """Inner splitter advertising n_splits=5 (so the inner-C5 gate requires >= 5 rows per class on outer-train)."""

    n_splits = 5

    def split(self, dataset):
        n = dataset._n
        half = max(1, n // 2)
        return [Fold(np.arange(half), np.array([], dtype=int), np.arange(half, n))]


def test_per_fold_all_folds_inner_degrade_falls_back_to_holdout(caplog) -> None:
    # ADR-0054 §6 (impl-review fix): if NO outer fold survives inner-C5, per-fold is infeasible -> the whole
    # arbitration degrades to holdout, reported distinctly as 'holdout_degraded_c5_inner'.
    rng = np.random.RandomState(0)
    n, nf = 40, 4
    x = np.column_stack([np.arange(n, dtype=float), rng.random((n, nf - 1))])
    y = np.zeros(n, dtype=int)
    # class 1 has 6 rows globally (>= K_outer=5 -> outer ok), but only 3 in each outer-train half (< inner 5)
    y[[0, 1, 2, 20, 21, 22]] = 1
    with caplog.at_level("WARNING"):
        out = compare_features(
            _FakeDataset(n),
            x,
            y,
            task=_FakeClsTask(),
            metric=_FakeMetric(),
            strategies=[("a", _FixedSelector("a", (0, 1))), ("b", _FixedSelector("b", (0, 1, 2)))],
            config=FeatureSelectionConfig(
                compare=("importance", "random_probe"), arbitration="nested_per_fold"
            ),
            splitter=_FakeInner5(),
            carve=_carve,
            fit_predict=_fit_predict,
            categorical=np.zeros(nf, dtype=bool),
            feature_names=[f"f{i}" for i in range(nf)],
            sample_weight=None,
            random_state=42,
            arbitration_splitter=_FakeKSplitter(),
            significance_test=_NoneEquivalent(),
            policy=SelectionPolicy(greater_is_better=True),
        )
    assert out.arbitration_effective == "holdout_degraded_c5_inner"
    assert any("no outer fold had enough rows per class" in r.message for r in caplog.records)


# --- M6f: per-fold degenerate-block aggregation (ADR-0059) ---


def test_per_fold_degenerate_aggregated_across_folds() -> None:
    # ADR-0059 §1: with structural blocks, the winner's per-fold degenerate fraction is aggregated over outer
    # folds and returned via CompareOutcome (merged into null_block_stats by run_slice).
    n = 40
    groups = (np.arange(n) // 8).astype(np.int64)  # 5 contiguous blocks of 8 rows
    y = (groups % 2).astype(float)  # each block has a constant target -> every block degenerate
    out = _run_per_fold(
        [("a", _FixedSelector("a", (0, 1))), ("b", _FixedSelector("b", (0, 1, 2)))],
        _NoneEquivalent(),
        n=n,
        y=y,
        groups=groups,
    )
    pf = out.per_fold_block_stats
    assert pf is not None
    assert (
        pf["per_fold_degenerate_mean"] == 1.0 and pf["per_fold_degenerate_max"] == 1.0
    )  # all blocks constant
    assert pf["per_fold_n_blocks_mean"] > 0


def test_per_fold_keys_absent_outside_per_fold() -> None:
    # no structural groups (iid) -> per-fold path runs but emits NO per-fold block keys (no false honesty)
    out_pf = _run_per_fold(
        [("a", _FixedSelector("a", (0, 1))), ("b", _FixedSelector("b", (0, 1, 2)))],
        _NoneEquivalent(),
    )
    assert out_pf.per_fold_block_stats is None
    # a plain holdout compare never sets it either (default None)
    out_holdout = _run([("a", _FixedSelector("a", (0, 1))), ("b", _FixedSelector("b", (0, 1, 2)))])
    assert out_holdout.per_fold_block_stats is None


# --- no-selection honest gate (finding #10, ADR-0063 semantics) ---


class _Rmse:
    name = "rmse"
    greater_is_better = False
    needs = "value"
    optimum = 0.0
    average = None
    proper_proba = False

    def score(self, y_true, y_pred, sample_weight=None):  # noqa: ANN001
        err = np.asarray(y_true, float) - np.asarray(y_pred, float)
        return float(np.sqrt(np.average(err**2, weights=sample_weight)))


def _ols_fit_predict(x_tr, y_tr, x_te, sample_weight, random_state):  # noqa: ANN001
    coef, *_ = np.linalg.lstsq(np.column_stack([x_tr, np.ones(len(x_tr))]), y_tr, rcond=None)
    return None, np.column_stack([x_te, np.ones(len(x_te))]) @ coef, None


def _kfolds(n: int, k: int = 4) -> list[Fold]:
    idx = np.arange(n)
    return [
        Fold(np.setdiff1d(idx, idx[i::k]), np.array([], dtype=int), idx[i::k]) for i in range(k)
    ]


def _gate(selected_idx, sig, *, n=80):
    # the only signal lives in column 0; columns 1..3 are pure noise
    rng = np.random.RandomState(0)
    x = rng.normal(size=(n, 4))
    y = 3.0 * x[:, 0] + 0.01 * rng.normal(size=n)
    return no_selection_gate(
        x,
        y,
        selected_idx,
        _kfolds(n),
        fit_predict=_ols_fit_predict,
        metric=_Rmse(),
        task=_FakeTask(),
        sample_weight=None,
        significance_test=sig,
        policy=SelectionPolicy(greater_is_better=False),
        random_state=0,
    )


def test_gate_drops_subset_significantly_worse_than_full() -> None:
    # a subset that EXCLUDES the signal column is much worse than no-selection -> drop it, ship all features
    keep, reason = _gate((1, 2, 3), _NoneEquivalent())
    assert keep is False and reason == "no_selection_better"


def test_gate_keeps_subset_equivalent_to_full() -> None:
    # statistically indistinguishable from no-selection -> ship the compact subset (Occam, not a regression)
    keep, reason = _gate((0,), _AllEquivalent())
    assert keep is True and reason == "selection_kept"


def test_gate_noop_when_nothing_dropped() -> None:
    # selecting all features is not a selection -> no gate, no scoring
    keep, reason = _gate((0, 1, 2, 3), _NoneEquivalent())
    assert keep is True and reason == "all_features_selected"


# --- in-sequential significance band over the trajectory (ADR-0083..0086, FR-1/2/5/6) ---

from honestml.adapters import SequentialSelector  # noqa: E402
from honestml.core.ports.significance import NoSignificanceTest  # noqa: E402


def _run_seq(sig_test, policy, *, n=40, n_features=4, full_descent=True):
    # single-strategy sequential on full DEV; _fit_predict makes a WIDER subset score higher,
    # so the greedy trajectory descends full -> floor with strictly decreasing scores.
    rng = np.random.RandomState(0)
    x, y = rng.random((n, n_features)), rng.random(n)
    seq = SequentialSelector(min_features=1, patience=2, full_descent=full_descent)
    return compare_features(
        _FakeDataset(n),
        x,
        y,
        task=_FakeTask(),
        metric=_FakeMetric(),
        strategies=[("sequential", seq)],
        config=FeatureSelectionConfig(strategy="sequential"),
        splitter=_FakeSplitter(),
        carve=_carve,
        fit_predict=_fit_predict,
        categorical=np.zeros(n_features, dtype=bool),
        feature_names=[f"f{i}" for i in range(n_features)],
        sample_weight=None,
        random_state=42,
        significance_test=sig_test,
        policy=policy,
    )


def test_sequential_band_picks_smallest_when_equivalent() -> None:
    # FR-1: every trajectory subset indistinguishable from the peak -> Occam keeps the FLOOR (1 feature)
    out = _run_seq(_AllEquivalent(), SelectionPolicy(greater_is_better=True))
    assert len(out.winner_idx) == 1
    assert out.seq_band is not None and out.seq_band["rule"] == "band_tiebreak"
    assert out.seq_band["winner_by_tiebreak"] is True and out.seq_band["width"] == 4


def test_sequential_argmax_when_significance_off() -> None:
    # FR-2: NoSignificanceTest -> band collapses to the absolute argmax = the full set; seq_band ABSENT
    out = _run_seq(NoSignificanceTest(), SelectionPolicy(greater_is_better=True))
    assert out.winner_idx == (0, 1, 2, 3)
    assert out.seq_band is None


def test_sequential_band_argmax_anchor_matches_off_path() -> None:
    # the band's absolute anchor (largest max-score subset) equals the off-path winner -> consistent argmax
    on = _run_seq(
        _NoneEquivalent(), SelectionPolicy(greater_is_better=True)
    )  # band empty -> argmax
    off = _run_seq(NoSignificanceTest(), SelectionPolicy(greater_is_better=True))
    assert on.winner_idx == off.winner_idx == (0, 1, 2, 3)


def test_sequential_band_is_deterministic() -> None:
    # NFR-1: identical inputs -> identical winner and band
    pol = SelectionPolicy(greater_is_better=True)
    a, b = _run_seq(_AllEquivalent(), pol), _run_seq(_AllEquivalent(), pol)
    assert a.winner_idx == b.winner_idx and a.seq_band == b.seq_band


def test_sequential_band_in_compare_mode_is_independent_of_arbitration() -> None:
    # FR-6: sequential inside compare -> its in-trajectory band is reported in seq_band, SEPARATE from the
    # strategy-arbitration band (band_members). The wrapper's subset is band-selected before arbitration.
    rng = np.random.RandomState(0)
    x, y = rng.random((40, 4)), rng.random(40)
    out = compare_features(
        _FakeDataset(40),
        x,
        y,
        task=_FakeTask(),
        metric=_FakeMetric(),
        strategies=[
            ("sequential", SequentialSelector(min_features=1, full_descent=True)),
            ("fixed", _FixedSelector("fixed", (0, 1))),
        ],
        config=FeatureSelectionConfig(compare=("sequential", "importance"), arbitration="nested"),
        splitter=_FakeSplitter(),
        carve=_carve,
        fit_predict=_fit_predict,
        categorical=np.zeros(4, dtype=bool),
        feature_names=[f"f{i}" for i in range(4)],
        sample_weight=None,
        random_state=42,
        arbitration_splitter=_FakeKSplitter(),
        significance_test=_AllEquivalent(),
        policy=SelectionPolicy(greater_is_better=True),
    )
    # sequential's band drops it to the floor {0}; arbitration (all-equivalent) then Occam-picks the
    # most compact strategy subset -> sequential wins with its floor subset.
    assert out.winner == "sequential" and out.winner_idx == (0,)
    # the two band channels coexist independently: seq_band (size-keyed, in-trajectory) vs band_members
    # (strategy-keyed, arbitration). They are SEPARATE fields with disjoint member spaces.
    assert out.seq_band is not None and out.seq_band["rule"] == "band_tiebreak"
    assert set(out.band_members) == {"sequential", "fixed"}
    assert all(m.startswith("k") for m in out.seq_band["members"])  # size-keys, not strategy names
