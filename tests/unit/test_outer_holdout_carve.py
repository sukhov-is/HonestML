"""M4c-1: scheme-aware outer-holdout carve behind the splitter port (ADR-0029 §2)."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from honestml.adapters import PolarsDataset, Reader, outer_holdout_carve
from honestml.core import ColumnRole, FeatureSchema, Task

pytestmark = pytest.mark.unit


def _dataset(n: int = 200, seed: int = 0) -> PolarsDataset:
    rng = np.random.default_rng(seed)
    frame = pl.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    y = rng.integers(0, 2, size=n)
    y[0], y[1] = 0, 1
    return Reader(Task(kind="binary")).read(frame, y)


def _group_dataset(n: int = 200, n_groups: int = 25, seed: int = 0):
    rng = np.random.default_rng(seed)
    g = rng.integers(0, n_groups, size=n)
    y = rng.integers(0, 2, size=n)
    y[0], y[1] = 0, 1
    frame = pl.DataFrame({"a": rng.normal(size=n), "g": g, "__target__": y})
    schema = FeatureSchema(
        roles={"a": ColumnRole.NUMERIC, "g": ColumnRole.GROUP, "__target__": ColumnRole.TARGET}
    )
    return PolarsDataset(frame, schema), g


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


def test_carve_disjoint_and_complete() -> None:
    """Stratified carve: dev ∪ holdout = all rows, disjoint, holdout ≈ fraction, both classes kept."""
    ds = _dataset(n=200)
    y = ds.target()
    dev, hold = outer_holdout_carve(
        ds, scheme="stratified", fraction=0.25, stratify=True, random_state=0
    )
    assert set(dev.tolist()).isdisjoint(hold.tolist())  # one-touch: holdout never in dev
    assert sorted(dev.tolist() + hold.tolist()) == list(range(200))  # complete cover
    assert hold.size == 50  # 0.25 * 200
    assert set(y[hold].tolist()) == {0, 1}  # stratified: both classes present for proba scoring


def test_period_carve_holds_out_latest_periods_no_leak() -> None:
    # ADR-0096 §3: timeseries_period holds out the latest WHOLE periods; dev is strictly earlier in time
    t = np.arange("2021-01-01", "2021-07-01", dtype="datetime64[D]")  # 6 months daily
    ds = _time_dataset(t)
    dev, hold = outer_holdout_carve(
        ds, scheme="timeseries_period", fraction=0.2, stratify=True, random_state=0, period="month"
    )
    assert set(dev.tolist()).isdisjoint(hold.tolist())  # one-touch holdout
    assert sorted(dev.tolist() + hold.tolist()) == list(range(len(t)))  # complete cover
    assert t[dev].max() < t[hold].min()  # no future leak across the period boundary


def test_period_carve_purge_drops_period_before_holdout() -> None:
    t = np.arange("2021-01-01", "2021-07-01", dtype="datetime64[D]")  # 6 months daily
    ds = _time_dataset(t)
    base = outer_holdout_carve(
        ds, scheme="timeseries_period", fraction=0.1, stratify=True, random_state=0, period="month"
    )[0]
    purged = outer_holdout_carve(
        ds,
        scheme="timeseries_period",
        fraction=0.1,
        stratify=True,
        random_state=0,
        period="month",
        purge=1,
    )[0]
    assert t[base].max() == np.datetime64("2021-05-31")  # holdout = June; dev runs through May
    assert t[purged].max() == np.datetime64("2021-04-30")  # purge=1 period drops May from dev


def test_carve_reproducible() -> None:
    ds = _dataset(n=200)
    a = outer_holdout_carve(ds, scheme="stratified", fraction=0.3, stratify=True, random_state=7)
    b = outer_holdout_carve(ds, scheme="stratified", fraction=0.3, stratify=True, random_state=7)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])


def test_carve_timeseries_late_window() -> None:
    """Time-series carve: holdout = the latest times, dev strictly before it (no future leak)."""
    times = np.random.default_rng(1).permutation(120).astype(float)
    ds = _time_dataset(times)
    t = ds.time()
    dev, hold = outer_holdout_carve(
        ds, scheme="timeseries", fraction=0.25, stratify=False, random_state=0, purge=2
    )
    assert set(dev.tolist()).isdisjoint(hold.tolist())
    assert t[dev].max() < t[hold].min()  # dev entirely before the holdout window (value-based)
    # purge drops the 2 samples just before the holdout window
    gap = [tt for tt in t if t[dev].max() < tt < t[hold].min()]
    assert len(gap) == 2


def test_carve_timeseries_label_horizon_purge() -> None:
    """F003: the outer timeseries carve must apply the de Prado label-horizon purge (t1), like the
    inner CV (FR-M4-7). A dev row strictly before the holdout in event time but whose label window t1
    reaches into the holdout interval must be dropped from dev — else future (holdout-period) info
    leaks into selection and the finalize-refit."""
    t = np.arange(20).astype(float)
    t1 = t.copy()
    t1[14] = 17.0  # row at time 14 has a label ending at 17 -> reaches into the holdout window
    ds = _time_dataset(t, label_times=t1)
    th = ds.time()
    dev, hold = outer_holdout_carve(
        ds, scheme="timeseries", fraction=0.25, stratify=False, random_state=0, purge=0
    )
    assert th[hold].min() == 15.0  # holdout = latest 5 (times 15..19)
    assert 14 not in dev.tolist()  # purged: its label window (t1=17) overlaps the holdout
    assert 13 in dev.tolist()  # a row whose label ends before the holdout survives
    assert th[dev].max() < th[hold].min()  # remaining dev still strictly before the holdout


def test_carve_group_disjoint() -> None:
    """Group carve: no group spans dev and holdout (group-disjoint outer holdout)."""
    ds, g = _group_dataset(n=200, n_groups=25)
    dev, hold = outer_holdout_carve(
        ds, scheme="group", fraction=0.25, stratify=False, random_state=0
    )
    assert set(dev.tolist()).isdisjoint(hold.tolist())
    assert set(g[dev].tolist()).isdisjoint(g[hold].tolist())  # whole groups held out


@pytest.mark.parametrize("scheme", ["stratified", "kfold", "holdout"])
def test_carve_group_aware_under_any_scheme_when_groups_present(scheme: str) -> None:
    """#11(a): a present group column makes the outer carve group-disjoint under ANY shuffling scheme,
    not only scheme='group' — a row-wise split of group-structured rows leaks (e.g. through TE)."""
    ds, g = _group_dataset(n=200, n_groups=25)
    dev, hold = outer_holdout_carve(ds, scheme=scheme, fraction=0.25, stratify=True, random_state=0)
    assert set(dev.tolist()).isdisjoint(hold.tolist())
    assert set(g[dev].tolist()).isdisjoint(g[hold].tolist())  # groups never span dev/holdout
