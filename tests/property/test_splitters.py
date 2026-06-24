"""M2-3: CV splitters yield valid, reproducible folds (ADR-0013).

Each fold is the anti-leakage mechanism (R-6): it must pass ``validate_fold``.
For k-fold the test indices must partition all rows; stratification must keep
both classes present per fold. Reproducibility (NFR-4) is checked on equal seeds.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from honestml.adapters import (
    GroupKFoldSplitter,
    HoldoutSplitter,
    KFoldSplitter,
    PeriodTimeSeriesSplitter,
    PolarsDataset,
    Reader,
    StratifiedGroupKFoldSplitter,
    StratifiedKFoldSplitter,
    TimeSeriesSplitter,
)
from honestml.adapters.splitters import _period_key
from honestml.core import (
    ColumnRole,
    FeatureSchema,
    GroupAwareSplitter,
    ReportsSplitMeta,
    SchemaValidationError,
    Task,
    TimeOrderedSplitter,
    validate_fold,
)

pytestmark = pytest.mark.property


def test_splitters_satisfy_capability_role_interfaces() -> None:
    # run_slice routes via isinstance against the marker role-interfaces (ADR-0027 §2 / ADR-0082): a real
    # adapter that fails to satisfy its marker would SILENTLY disable the time/group leakage guards, so pin
    # the conformance here (presence == capability; plain i.i.d. splitters must NOT match).
    assert isinstance(TimeSeriesSplitter(n_splits=2, n_test=1), TimeOrderedSplitter)
    assert not isinstance(KFoldSplitter(n_splits=5, random_state=0), TimeOrderedSplitter)
    for make in (GroupKFoldSplitter, StratifiedGroupKFoldSplitter):
        assert isinstance(make(n_splits=4, random_state=0), GroupAwareSplitter)
    assert not isinstance(StratifiedKFoldSplitter(n_splits=5, random_state=0), GroupAwareSplitter)


def _dataset(n: int = 100, seed: int = 0):
    rng = np.random.default_rng(seed)
    frame = pl.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    y = rng.integers(0, 2, size=n)
    # guard both classes present
    y[0], y[1] = 0, 1
    return Reader(Task(kind="binary")).read(frame, y)


def _group_dataset(n: int = 90, n_groups: int = 15, seed: int = 0):
    """PolarsDataset with a GROUP-role column (the Reader does not assign GROUP itself)."""
    rng = np.random.default_rng(seed)
    g = rng.integers(0, n_groups, size=n)
    y = rng.integers(0, 2, size=n)
    y[0], y[1] = 0, 1
    frame = pl.DataFrame(
        {"a": rng.normal(size=n), "b": rng.normal(size=n), "g": g, "__target__": y}
    )
    schema = FeatureSchema(
        roles={
            "a": ColumnRole.NUMERIC,
            "b": ColumnRole.NUMERIC,
            "g": ColumnRole.GROUP,
            "__target__": ColumnRole.TARGET,
        }
    )
    return PolarsDataset(frame, schema), g


def test_holdout_yields_one_valid_fold() -> None:
    ds = _dataset()
    folds = list(HoldoutSplitter(random_state=0).split(ds))
    assert len(folds) == 1
    validate_fold(folds[0])
    f = folds[0]
    assert f.fit_idx.size + f.test_idx.size == ds.n_rows


def test_kfold_folds_are_valid_and_partition() -> None:
    ds = _dataset()
    folds = list(StratifiedKFoldSplitter(n_splits=5, random_state=0).split(ds))
    assert len(folds) == 5
    seen: list[int] = []
    for f in folds:
        validate_fold(f)
        seen.extend(f.test_idx.tolist())
    assert sorted(seen) == list(range(ds.n_rows))  # test indices partition all rows


def test_kfold_keeps_both_classes_per_fold() -> None:
    ds = _dataset()
    y = ds.target()
    for f in StratifiedKFoldSplitter(n_splits=5, random_state=0).split(ds):
        assert set(y[f.test_idx].tolist()) == {0, 1}


def test_reproducible_with_same_seed() -> None:
    ds = _dataset()
    a = [f.test_idx.tolist() for f in StratifiedKFoldSplitter(random_state=7).split(ds)]
    b = [f.test_idx.tolist() for f in StratifiedKFoldSplitter(random_state=7).split(ds)]
    assert a == b


# --- early-stopping es carve on i.i.d. splitters (ADR-0080) ------------------


@pytest.mark.parametrize("make", [StratifiedKFoldSplitter, KFoldSplitter])
def test_iid_es_carve_is_transparent_to_the_union(make) -> None:
    # es is carved OUT of train, so fit ∪ es equals the same fold's train without the carve.
    ds = _dataset(n=120)
    plain = list(make(n_splits=4, random_state=0).split(ds))
    with_es = list(make(n_splits=4, random_state=0, es_fraction=0.2).split(ds))
    for p, e in zip(plain, with_es, strict=True):
        validate_fold(e)
        assert e.es_idx.size > 0
        assert np.array_equal(p.test_idx, e.test_idx)  # test split untouched
        union = np.sort(np.concatenate([e.fit_idx, e.es_idx]))
        assert np.array_equal(union, np.sort(p.fit_idx))  # fit ∪ es == original train


def test_holdout_es_carve_is_transparent_to_the_union() -> None:
    ds = _dataset(n=120)
    plain = next(HoldoutSplitter(random_state=0).split(ds))
    with_es = next(HoldoutSplitter(random_state=0, es_fraction=0.2).split(ds))
    validate_fold(with_es)
    assert with_es.es_idx.size > 0
    assert np.array_equal(plain.test_idx, with_es.test_idx)
    union = np.sort(np.concatenate([with_es.fit_idx, with_es.es_idx]))
    assert np.array_equal(union, np.sort(plain.fit_idx))


@pytest.mark.parametrize("make", [StratifiedKFoldSplitter, KFoldSplitter, HoldoutSplitter])
def test_iid_es_off_by_default_keeps_es_empty(make) -> None:
    ds = _dataset()
    for f in make(random_state=0).split(ds):
        assert f.es_idx.size == 0


@pytest.mark.parametrize(
    "make_plain, make_es",
    [
        (
            lambda: GroupKFoldSplitter(n_splits=4, random_state=0),
            lambda: GroupKFoldSplitter(n_splits=4, random_state=0, es_fraction=0.25),
        ),
        (
            lambda: StratifiedGroupKFoldSplitter(n_splits=4, random_state=0),
            lambda: StratifiedGroupKFoldSplitter(n_splits=4, random_state=0, es_fraction=0.25),
        ),
    ],
)
def test_group_es_carve_holds_out_whole_groups(make_plain, make_es) -> None:
    # ADR-0080 amendment: es is carved as WHOLE groups out of train — group-disjoint and transparent.
    ds, groups = _group_dataset(n=160, n_groups=24)
    plain = list(make_plain().split(ds))
    with_es = list(make_es().split(ds))
    for p, e in zip(plain, with_es, strict=True):
        validate_fold(e, groups=groups)  # fit/es/test pairwise group-disjoint
        assert e.es_idx.size > 0
        assert np.array_equal(p.test_idx, e.test_idx)  # test split untouched
        union = np.sort(np.concatenate([e.fit_idx, e.es_idx]))
        assert np.array_equal(union, np.sort(p.fit_idx))  # fit ∪ es == original train
        assert set(groups[e.fit_idx]).isdisjoint(set(groups[e.es_idx]))  # no group spans fit/es


@pytest.mark.parametrize(
    "make",
    [lambda: GroupKFoldSplitter(n_splits=4), lambda: StratifiedGroupKFoldSplitter(n_splits=4)],
)
def test_group_es_off_by_default_keeps_es_empty(make) -> None:
    ds, _ = _group_dataset()
    for f in make().split(ds):
        assert f.es_idx.size == 0


# --- M3c: plain KFold + group-aware splitters + Dataset.groups() (ADR-0023) ----


def test_plain_kfold_partitions_and_is_valid() -> None:
    ds = _dataset()
    folds = list(KFoldSplitter(n_splits=5, random_state=0).split(ds))
    assert len(folds) == 5
    seen: list[int] = []
    for f in folds:
        validate_fold(f)
        seen.extend(f.test_idx.tolist())
    assert sorted(seen) == list(range(ds.n_rows))


@pytest.mark.parametrize(
    "make",
    [
        lambda: GroupKFoldSplitter(n_splits=5),
        lambda: StratifiedGroupKFoldSplitter(n_splits=5, random_state=0),
    ],
)
def test_group_splitters_no_leakage_and_partition(make) -> None:
    ds, groups = _group_dataset()
    seen: list[int] = []
    n_folds = 0
    for f in make().split(ds):
        validate_fold(f, groups=groups)  # no group spans fit/test (anti-leakage)
        seen.extend(f.test_idx.tolist())
        n_folds += 1
    assert n_folds == 5
    assert sorted(seen) == list(range(ds.n_rows))  # test indices partition all rows


def test_stratified_group_kfold_is_reproducible() -> None:
    ds, _ = _group_dataset()
    a = [f.test_idx.tolist() for f in StratifiedGroupKFoldSplitter(random_state=7).split(ds)]
    b = [f.test_idx.tolist() for f in StratifiedGroupKFoldSplitter(random_state=7).split(ds)]
    assert a == b


def test_dataset_groups_in_row_order() -> None:
    ds, groups = _group_dataset()
    assert ds.groups() is not None
    assert ds.groups().tolist() == groups.tolist()  # row-aligned with the frame
    sub = ds.take([5, 2, 0])
    assert sub.groups().tolist() == groups[[5, 2, 0]].tolist()  # follows row selection


def test_dataset_groups_none_without_group_column() -> None:
    assert _dataset().groups() is None


def test_group_splitter_rejects_null_groups() -> None:
    # a null/NaN in the group column fails at the boundary with a domain error, not a sklearn crash
    frame = pl.DataFrame(
        {"a": [0.1, 0.2, 0.3, 0.4], "g": [0, 1, None, 1], "__target__": [0, 1, 0, 1]}
    )
    schema = FeatureSchema(
        roles={
            "a": ColumnRole.NUMERIC,
            "g": ColumnRole.GROUP,
            "__target__": ColumnRole.TARGET,
        }
    )
    ds = PolarsDataset(frame, schema)
    with pytest.raises(SchemaValidationError, match="null/NaN"):
        list(GroupKFoldSplitter(n_splits=2).split(ds))


# --- M4b: TimeSeriesSplitter (value-based, purge/embargo, es) — ADR-0027 ------


def _time_dataset(times: np.ndarray, label_times: np.ndarray | None = None) -> PolarsDataset:
    n = len(times)
    rng = np.random.default_rng(0)
    cols: dict = {
        "a": rng.normal(size=n),
        "__target__": rng.integers(0, 2, size=n),
        "__time__": times,
    }
    roles = {
        "a": ColumnRole.NUMERIC,
        "__target__": ColumnRole.TARGET,
        "__time__": ColumnRole.TIME,
    }
    label_time_col = None
    if label_times is not None:
        cols["__label_time__"] = label_times
        label_time_col = "__label_time__"
    return PolarsDataset(
        pl.DataFrame(cols), FeatureSchema(roles=roles), label_time_col=label_time_col
    )


def test_timeseries_no_time_overlap_property() -> None:
    # shuffled times -> value-based order; every fold passes the value-based overlap invariant
    rng = np.random.default_rng(1)
    times = rng.permutation(50).astype(float)
    ds = _time_dataset(times)
    t = ds.time()
    folds = list(TimeSeriesSplitter(n_splits=4, n_test=5, purge=2).split(ds))
    assert len(folds) == 4
    for f in folds:
        validate_fold(f, time_ordered=True, times=t)  # no train time inside the test interval
        assert f.es_idx.size >= 1  # non-empty es (FR-M4-8)


def test_timeseries_purge_magnitude_on_splitter() -> None:
    t = np.arange(40).astype(float)  # unique, increasing
    ds = _time_dataset(t)
    for f in TimeSeriesSplitter(n_splits=3, n_test=4, purge=3, n_es=1).split(ds):
        train_t = t[np.concatenate([f.fit_idx, f.es_idx])]
        test_t = t[f.test_idx]
        gap = [tt for tt in t if train_t.max() < tt < test_t.min()]
        assert len(gap) == 3  # exactly `purge` samples removed before the test


def test_timeseries_embargo_excludes_post_test_zone() -> None:
    t = np.arange(30).astype(float)
    ds = _time_dataset(t)
    folds = list(TimeSeriesSplitter(n_splits=3, n_test=4, embargo=3, n_es=1).split(ds))
    f2 = folds[2]  # train extends over earlier test windows; embargo after test 0 = times 22,23,24
    train_t = set(t[np.concatenate([f2.fit_idx, f2.es_idx])].tolist())
    assert {22.0, 23.0, 24.0}.isdisjoint(train_t)
    assert 25.0 in train_t  # only the embargo zone is excluded, not beyond


def test_timeseries_too_few_rows_fails_fast_with_named_need() -> None:
    """F1.6 regression pin: a short series fails BEFORE any fold is built, naming the need."""
    ds = _time_dataset(np.arange(6).astype(float))
    with pytest.raises(SchemaValidationError, match="too few rows"):
        list(TimeSeriesSplitter(n_splits=3, n_test=2, n_es=1).split(ds))


def test_timeseries_nonempty_es_and_order() -> None:
    t = np.arange(40).astype(float)
    ds = _time_dataset(t)
    for f in TimeSeriesSplitter(n_splits=3, n_test=5, n_es=2).split(ds):
        assert f.es_idx.size == 2  # non-empty es tail (FR-M4-8)
        train_t = t[np.concatenate([f.fit_idx, f.es_idx])]
        assert train_t.max() < t[f.test_idx].min()  # max(fit∪es) < min(test)
        assert t[f.fit_idx].max() < t[f.es_idx].min()  # es is the time-tail of train


def test_timeseries_label_horizon_purge() -> None:
    # a label window (t1) reaching into the test interval purges the train row (FR-M4-7)
    t = np.arange(20).astype(float)
    t1 = t.copy()
    t1[13] = 16.0  # row at time 13 has a label ending at 16 -> overlaps the test interval
    ds = _time_dataset(t, label_times=t1)
    f = next(iter(TimeSeriesSplitter(n_splits=1, n_test=4, purge=0, n_es=1).split(ds)))
    train_t = set(t[np.concatenate([f.fit_idx, f.es_idx])].tolist())
    assert 13.0 not in train_t  # purged: its label window overlaps the test
    assert 12.0 in train_t  # a row whose label ends before the test survives


def test_timeseries_reproducible() -> None:
    rng = np.random.default_rng(3)
    times = rng.permutation(50).astype(float)
    ds = _time_dataset(times)

    def splits() -> list:
        return [
            (f.fit_idx.tolist(), f.test_idx.tolist())
            for f in TimeSeriesSplitter(n_splits=4, n_test=5).split(ds)
        ]

    assert splits() == splits()  # deterministic, no RNG (NFR-M4-2)


def test_timeseries_requires_time_column() -> None:
    with pytest.raises(SchemaValidationError, match="requires a time column"):
        list(TimeSeriesSplitter(n_splits=3, n_test=4).split(_dataset()))


# --- Etap1: PeriodTimeSeriesSplitter (calendar/Δt periods, walk-forward) — ADR-0096 ----


def test_period_splitter_role_interfaces() -> None:
    # the period splitter is time-ordered (enables the value-based fold check) and reports split meta
    sp = PeriodTimeSeriesSplitter(period="month", n_splits=2, n_test=1)
    assert isinstance(sp, TimeOrderedSplitter)
    assert isinstance(sp, ReportsSplitMeta)
    assert sp.split_meta() is None  # nothing computed before split (FR-8 §4)


@pytest.mark.parametrize(
    "period, times, same, diff",
    [
        ("month", ["2021-01-05", "2021-01-28", "2021-02-03"], (0, 1), (0, 2)),
        (
            "week",
            ["2021-01-04", "2021-01-10", "2021-01-11"],
            (0, 1),
            (0, 2),
        ),  # Mon..Sun vs next Mon
        ("day", ["2021-01-05", "2021-01-05", "2021-01-06"], (0, 1), (0, 2)),
    ],
)
def test_period_key_calendar_buckets(period, times, same, diff) -> None:
    # FR-2: rows of one calendar month/ISO-week(Mon)/day share a bucket; the next period is a new bucket
    t = np.array(times, dtype="datetime64[D]")
    pid, _, _ = _period_key(t, period, None)
    assert pid[same[0]] == pid[same[1]]
    assert pid[diff[0]] != pid[diff[1]]


def test_period_key_delta_buckets_and_gaps() -> None:
    # FR-2: delta bins a numeric axis by width; empty interior buckets are densified away but counted
    t = np.array(
        [0.0, 0.5, 3.0, 3.2], dtype=float
    )  # width 1.0 -> keys 0,0,3,3 -> dense {0,1}, gaps 1,2
    pid, n_periods, n_dropped = _period_key(t, "delta", 1.0)
    assert pid.tolist() == [0, 0, 1, 1]
    assert (n_periods, n_dropped) == (2, 2)


def test_period_split_meta_reports_periods_and_gaps() -> None:
    # FR-8 §4: split_meta carries the densified period counts for the truthful manifest
    t = np.array([0.0, 0.5, 3.0, 3.2, 4.0, 4.1, 5.0, 5.5, 6.0, 7.0], dtype=float)
    ds = _time_dataset(t)
    sp = PeriodTimeSeriesSplitter(period="delta", period_size=1.0, n_splits=2, n_test=2)
    folds = list(sp.split(ds))
    assert len(folds) == 2
    assert sp.split_meta() == {
        "period": "delta",
        "n_periods": 6,
        "n_folds": 2,
        "n_dropped_empty": 2,
    }


def test_period_no_time_overlap_property() -> None:
    # FR-7/NFR-2: shuffled rows -> value-based order; every period fold passes the overlap invariant
    rng = np.random.default_rng(2)
    days = np.arange("2021-01-01", "2022-01-01", dtype="datetime64[D]")  # 12 months, daily
    ds = _time_dataset(days[rng.permutation(len(days))])
    t = ds.time()
    folds = list(
        PeriodTimeSeriesSplitter(period="month", n_splits=4, n_test=2, purge=1, n_es=2).split(ds)
    )
    assert len(folds) == 4
    for f in folds:
        validate_fold(f, time_ordered=True, times=t)  # no train time inside the test interval
        assert f.es_idx.size >= 1


def test_period_purge_drops_periods_before_test() -> None:
    # FR-3/NFR-2: purge counts PERIODS — the `purge` periods just before the test are removed from train
    t = np.arange("2021-01-01", "2021-01-21", dtype="datetime64[D]")  # 20 daily periods
    ds = _time_dataset(t)
    f = next(
        iter(
            PeriodTimeSeriesSplitter(period="day", n_splits=1, n_test=4, purge=2, n_es=1).split(ds)
        )
    )
    train_t = t[np.concatenate([f.fit_idx, f.es_idx])]
    # test = Jan17..20; purge=2 drops Jan15,16 -> latest train day is Jan14
    assert train_t.max() == np.datetime64("2021-01-14")


def test_period_embargo_excludes_post_test_zone() -> None:
    t = np.arange("2021-01-01", "2021-01-31", dtype="datetime64[D]")  # 30 daily periods
    ds = _time_dataset(t)
    folds = list(
        PeriodTimeSeriesSplitter(period="day", n_splits=3, n_test=2, embargo=2, n_es=1).split(ds)
    )
    train_t = t[np.concatenate([folds[2].fit_idx, folds[2].es_idx])]
    # fold0 test = Jan25,26; embargo=2 after it removes Jan27,28 from the later fold's train
    assert np.datetime64("2021-01-27") not in train_t
    assert np.datetime64("2021-01-28") not in train_t
    assert (
        np.datetime64("2021-01-26") in train_t
    )  # an earlier test period stays in the expanding train


def test_period_label_horizon_purge() -> None:
    # FR-7: a label window (t1) reaching into the test interval purges the train row (de Prado)
    t = np.arange("2021-01-01", "2021-01-21", dtype="datetime64[D]")  # 20 daily periods
    t1 = t.copy()
    t1[12] = np.datetime64("2021-01-17")  # Jan13's label ends at Jan17 -> overlaps the test
    ds = _time_dataset(t, label_times=t1)
    f = next(iter(PeriodTimeSeriesSplitter(period="day", n_splits=1, n_test=4, n_es=1).split(ds)))
    train_t = t[np.concatenate([f.fit_idx, f.es_idx])]
    assert np.datetime64("2021-01-13") not in train_t  # purged: label window reaches the test
    assert np.datetime64("2021-01-12") in train_t  # label ends before the test -> survives


def test_period_reproducible() -> None:
    days = np.arange("2021-01-01", "2021-07-01", dtype="datetime64[D]")  # 6 months
    rng = np.random.default_rng(3)
    ds = _time_dataset(days[rng.permutation(len(days))])

    def splits() -> list:
        return [
            (f.fit_idx.tolist(), f.test_idx.tolist())
            for f in PeriodTimeSeriesSplitter(period="month", n_splits=2, n_test=2).split(ds)
        ]

    assert splits() == splits()  # deterministic, no RNG (NFR-4)


def test_period_calendar_on_numeric_axis_rejected() -> None:
    ds = _time_dataset(np.arange(40).astype(float))
    with pytest.raises(SchemaValidationError, match="needs a datetime"):
        list(PeriodTimeSeriesSplitter(period="month", n_splits=2, n_test=2).split(ds))


def test_period_delta_on_datetime_axis_rejected() -> None:
    t = np.arange("2021-01-01", "2021-02-20", dtype="datetime64[D]")
    ds = _time_dataset(t)
    with pytest.raises(SchemaValidationError, match="needs a numeric"):
        list(
            PeriodTimeSeriesSplitter(period="delta", period_size=1.0, n_splits=2, n_test=2).split(
                ds
            )
        )


def test_period_step_gt_n_test_walks_without_overflow() -> None:
    # step_periods > n_test: controlled gaps between test windows, last window ends at the final period;
    # the feasibility gate accounts for step so no fold overflows the axis into an empty test (G13)
    t = np.arange("2021-01-01", "2021-01-31", dtype="datetime64[D]")  # 30 daily periods
    ds = _time_dataset(t)
    folds = list(
        PeriodTimeSeriesSplitter(period="day", n_splits=3, n_test=1, step_periods=4, n_es=1).split(
            ds
        )
    )
    assert len(folds) == 3
    test_starts = [t[f.test_idx].min() for f in folds]
    assert test_starts == [
        np.datetime64("2021-01-22"),
        np.datetime64("2021-01-26"),
        np.datetime64("2021-01-30"),
    ]  # windows 4 periods apart, last ends at the final period
    for f in folds:
        validate_fold(f, time_ordered=True, times=t)


def test_period_step_too_large_fails_fast() -> None:
    # an unsatisfiable step (no training period left before the first window) -> a clear domain error,
    # NOT a raw numpy ValueError from an empty test window
    t = np.arange("2021-01-01", "2021-01-11", dtype="datetime64[D]")  # 10 daily periods
    ds = _time_dataset(t)
    with pytest.raises(SchemaValidationError, match="too few periods"):
        list(PeriodTimeSeriesSplitter(period="day", n_splits=3, n_test=1, step_periods=8).split(ds))


def test_period_purge_starved_first_fold_fails_fast() -> None:
    # purge counted in periods can starve the first fold's train -> caught up-front as a domain error
    t = np.arange("2021-01-01", "2021-01-09", dtype="datetime64[D]")  # 8 daily periods
    ds = _time_dataset(t)
    with pytest.raises(SchemaValidationError, match="too few periods"):
        list(PeriodTimeSeriesSplitter(period="day", n_splits=1, n_test=4, purge=4).split(ds))


def test_period_too_few_periods_fails_fast() -> None:
    t = np.arange("2021-01-01", "2021-04-01", dtype="datetime64[D]")  # 3 months only
    ds = _time_dataset(t)
    with pytest.raises(SchemaValidationError, match="too few periods"):
        list(PeriodTimeSeriesSplitter(period="month", n_splits=2, n_test=2).split(ds))


def test_period_requires_time_column() -> None:
    with pytest.raises(SchemaValidationError, match="requires a time column"):
        list(PeriodTimeSeriesSplitter(period="month", n_splits=2, n_test=2).split(_dataset()))


# --- Etap2: Δt (wall-clock) purge/embargo + rolling max_train — ADR-0097/0099 -----


def test_timeseries_purge_delta_is_value_based() -> None:
    # FR-4: a Δt purge drops ALL rows within the time zone — on an irregular axis a VARIABLE row count
    # (4 rows in 1.0 of time here), unlike the fixed-count integer purge
    t = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 5.1, 5.2, 5.3, 6.0], dtype=float)
    ds = _time_dataset(t)
    f = next(iter(TimeSeriesSplitter(n_splits=1, n_test=1, purge_delta=1.0, n_es=1).split(ds)))
    train_t = set(t[np.concatenate([f.fit_idx, f.es_idx])].tolist())
    # test = [6.0]; zone [5.0, 6.0) removes 5.0,5.1,5.2,5.3; latest surviving train time is 4.0
    assert {5.0, 5.1, 5.2, 5.3}.isdisjoint(train_t)
    assert 4.0 in train_t


def test_timeseries_purge_delta_magnitude_on_datetime_axis() -> None:
    # FR-4: Δt is measured in the axis' storage unit (days here), so purge_delta=2 drops the last 2 days
    t = np.arange("2021-01-01", "2021-01-21", dtype="datetime64[D]")  # 20 daily rows
    ds = _time_dataset(t)
    f = next(iter(TimeSeriesSplitter(n_splits=1, n_test=4, purge_delta=2.0, n_es=1).split(ds)))
    train_t = t[np.concatenate([f.fit_idx, f.es_idx])]
    # test = Jan17..20, test_min=Jan17; Δt=2 days drops Jan15,16 -> latest train day is Jan14
    assert train_t.max() == np.datetime64("2021-01-14")


def test_timeseries_embargo_delta_excludes_post_test_zone() -> None:
    # FR-4: embargo_delta drops the half-open Δt zone after each earlier test window (value-based de Prado)
    t = np.arange(30).astype(float)
    ds = _time_dataset(t)
    folds = list(TimeSeriesSplitter(n_splits=3, n_test=4, embargo_delta=3.0, n_es=1).split(ds))
    train_t = set(t[np.concatenate([folds[2].fit_idx, folds[2].es_idx])].tolist())
    # fold0 test ends at 21; embargo zone [21, 24) removes 21,22,23 from the later fold's expanding train
    assert {21.0, 22.0, 23.0}.isdisjoint(train_t)
    assert 24.0 in train_t and 20.0 in train_t  # half-open end + rows before the zone survive


def test_timeseries_max_train_size_bounds_lookback() -> None:
    # FR-5: rolling caps the train (fit∪es) to the last max_train_size rows before the test window
    t = np.arange(40).astype(float)
    ds = _time_dataset(t)
    for f in TimeSeriesSplitter(n_splits=3, n_test=4, max_train_size=5, n_es=1).split(ds):
        train = np.concatenate([f.fit_idx, f.es_idx])
        assert train.size == 5  # exactly the last 5 rows, not the whole expanding window
        assert t[train].max() == t[f.test_idx].min() - 1  # contiguous up to the test boundary
    # contrast: without the cap the first fold's train is the whole expanding window (28 rows)
    expanding = next(iter(TimeSeriesSplitter(n_splits=3, n_test=4, n_es=1).split(ds)))
    assert np.concatenate([expanding.fit_idx, expanding.es_idx]).size == 28


def test_period_purge_delta_drops_zone_before_test() -> None:
    # FR-4: purge_delta drops the Δt zone before the test window under the period scheme too
    t = np.arange("2021-01-01", "2021-01-21", dtype="datetime64[D]")  # 20 daily periods
    ds = _time_dataset(t)
    f = next(
        iter(
            PeriodTimeSeriesSplitter(
                period="day", n_splits=1, n_test=4, purge_delta=2.0, n_es=1
            ).split(ds)
        )
    )
    train_t = t[np.concatenate([f.fit_idx, f.es_idx])]
    # test = Jan17..20, test_min=Jan17; Δt=2 days drops Jan15,16 -> latest train day is Jan14
    assert train_t.max() == np.datetime64("2021-01-14")


def test_period_embargo_delta_excludes_post_test_zone() -> None:
    # FR-4: the value-based embargo zone applies under the period scheme too (period-specific wiring)
    t = np.arange("2021-01-01", "2021-01-31", dtype="datetime64[D]")  # 30 daily periods
    ds = _time_dataset(t)
    folds = list(
        PeriodTimeSeriesSplitter(
            period="day", n_splits=3, n_test=2, embargo_delta=2.0, n_es=1
        ).split(ds)
    )
    train_t = t[np.concatenate([folds[2].fit_idx, folds[2].es_idx])]
    # fold0 test ends Jan26; embargo zone [Jan26, Jan28) removes Jan26,27 from the later fold's train
    assert np.datetime64("2021-01-26") not in train_t
    assert np.datetime64("2021-01-27") not in train_t
    assert np.datetime64("2021-01-25") in train_t  # before the zone -> stays in the expanding train


def test_period_max_train_periods_bounds_lookback() -> None:
    # FR-5: rolling caps the train to the last max_train_periods periods before each test window
    t = np.arange("2021-01-01", "2021-01-31", dtype="datetime64[D]")  # 30 daily periods, 1 row each
    ds = _time_dataset(t)
    folds = list(
        PeriodTimeSeriesSplitter(
            period="day", n_splits=3, n_test=2, max_train_periods=4, n_es=1
        ).split(ds)
    )
    assert len(folds) == 3
    for f in folds:
        train_t = t[np.concatenate([f.fit_idx, f.es_idx])]
        assert np.unique(train_t).size <= 4  # at most the 4 periods before the test window
        validate_fold(f, time_ordered=True, times=ds.time())


def test_period_rolling_starved_fold_fails_fast() -> None:
    # FR-5/R-5: too small a rolling window leaves too few train rows -> a clear domain error (fail-run)
    t = np.arange("2021-01-01", "2021-01-21", dtype="datetime64[D]")  # 20 daily periods, 1 row each
    ds = _time_dataset(t)
    with pytest.raises(SchemaValidationError, match="too few train rows"):
        list(
            PeriodTimeSeriesSplitter(
                period="day", n_splits=2, n_test=2, max_train_periods=2, n_es=1
            ).split(ds)
        )
