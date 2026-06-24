"""CVSplitter adapters (ADR-0013, extended ADR-0023).

``HoldoutSplitter`` (one fold), ``StratifiedKFoldSplitter`` / ``KFoldSplitter`` (k folds)
and the group-aware ``GroupKFoldSplitter`` / ``StratifiedGroupKFoldSplitter`` yield
``Fold(fit_idx, es_idx, test_idx)`` over row indices; every fold satisfies ``validate_fold``
(disjoint index sets; for group splitters, no group spans fit/test). Group-aware splitters
carry ``group_aware = True`` so the use-case runs the group-leakage check (ADR-0023 §2). The
``es`` tail is left empty (no early stopping until M4; the use-case trains on ``fit ∪ es``,
ADR-0010 §6). ``shuffle``/``random_state`` make the shuffling splitters reproducible;
``GroupKFoldSplitter`` is deterministic by construction (no shuffle) — NFR-4.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Iterator

import numpy as np
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    KFold,
    StratifiedGroupKFold,
    StratifiedKFold,
    train_test_split,
)

from honestml.core.dataset import Dataset
from honestml.core.exceptions import SchemaValidationError
from honestml.core.ports.splitter import Fold

_EMPTY = np.empty(0, dtype=np.int64)


def _carve_iid_es(
    fit_idx: np.ndarray,
    *,
    y: np.ndarray | None,
    es_fraction: float,
    random_state: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Carve an early-stopping tail OUT of a fold's training rows for boosting ES (ADR-0080).

    es is a subsample of fit (stratified when ``y`` is given), so ``fit ∪ es`` equals the original
    train: non-ES models that merge it back (run_slice) are byte-identical, only ES models hold it out
    as validation — OOF honesty is untouched (es never enters test). Returns ``(fit_idx, es_idx)``;
    es is empty when ``es_fraction<=0`` or the fold is too small to spare a row. Seed-deterministic.
    """
    if es_fraction <= 0.0 or fit_idx.size < 2:
        return fit_idx, _EMPTY
    stratify = y[fit_idx] if y is not None else None
    try:
        fit2, es = train_test_split(
            fit_idx,
            test_size=es_fraction,
            shuffle=True,
            random_state=random_state,
            stratify=stratify,
        )
    except ValueError:  # a class too rare to stratify the tiny es tail — carve unstratified
        fit2, es = train_test_split(
            fit_idx, test_size=es_fraction, shuffle=True, random_state=random_state
        )
    return np.sort(fit2).astype(np.int64), np.sort(es).astype(np.int64)


def _carve_group_es(
    fit_idx: np.ndarray,
    *,
    groups: np.ndarray,
    es_fraction: float,
    random_state: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Carve a GROUP-disjoint es tail from a group fold's train rows for boosting ES (ADR-0080).

    es holds out WHOLE groups (``GroupShuffleSplit``), so no entity spans fit/es: ES validation under
    the group scheme stays as honest as the outer carve (finding #11). ``fit ∪ es`` equals the original
    train, so non-ES models that merge it back are byte-identical. Returns ``(fit_idx, es_idx)``; es is
    empty when ``es_fraction<=0`` or the fold has <2 groups to spare a disjoint one. Seed-deterministic.
    """
    if es_fraction <= 0.0:
        return fit_idx, _EMPTY
    g = groups[fit_idx]
    if np.unique(g).size < 2:  # need at least two groups to hold one out disjointly
        return fit_idx, _EMPTY
    gss = GroupShuffleSplit(n_splits=1, test_size=es_fraction, random_state=random_state)
    fit_pos, es_pos = next(gss.split(np.zeros((fit_idx.size, 1)), groups=g))
    return np.sort(fit_idx[fit_pos]).astype(np.int64), np.sort(fit_idx[es_pos]).astype(np.int64)


def _target(dataset: Dataset) -> np.ndarray:
    y = dataset.target()
    if y is None:
        raise SchemaValidationError("splitter requires a target column for stratification")
    return y


def _groups(dataset: Dataset) -> np.ndarray:
    groups = dataset.groups()
    if groups is None:
        raise SchemaValidationError("group-aware cross-validation requires a group column")
    if _has_null_groups(groups):
        # fail at the boundary with a domain error, not a bare sklearn ValueError deep in split
        raise SchemaValidationError(
            "group column contains null/NaN values; groups must be complete"
        )
    return groups


def _has_null_groups(groups: np.ndarray) -> bool:
    if groups.dtype.kind == "f":
        return bool(np.isnan(groups).any())
    if groups.dtype == object:
        return any(g is None or (isinstance(g, float) and math.isnan(g)) for g in groups)
    return False


def _yield_folds(
    raw: Iterable[tuple[np.ndarray, np.ndarray]],
    carve: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
) -> Iterator[Fold]:
    """Map sklearn ``(fit_idx, test_idx)`` splits to ``Fold``s, carving each fold's es tail (ADR-0080).

    The split→carve→Fold loop shared by the four k-fold splitters; each splitter differs only in the
    sklearn object it constructs and the es-carve it injects (``_carve_iid_es`` vs ``_carve_group_es``).
    """
    for fit_idx, test_idx in raw:
        fit_i, es_i = carve(fit_idx.astype(np.int64))
        yield Fold(fit_idx=fit_i, es_idx=es_i, test_idx=test_idx.astype(np.int64))


class HoldoutSplitter:
    """Single train/test split (stratified for classification, plain for regression).

    ``stratify`` is disabled for regression (a continuous target cannot be stratified):
    composition sets ``stratify=task.is_classification`` (ADR-0020 §4 regression path).
    """

    def __init__(
        self,
        *,
        test_size: float = 0.25,
        shuffle: bool = True,
        stratify: bool = True,
        random_state: int | None = None,
        es_fraction: float = 0.0,
    ) -> None:
        self.test_size = test_size
        self.shuffle = shuffle
        self.stratify = stratify
        self.random_state = random_state
        self.es_fraction = es_fraction

    def split(self, dataset: Dataset) -> Iterator[Fold]:
        y = _target(dataset)
        idx = np.arange(dataset.n_rows)
        stratify = y if (self.shuffle and self.stratify) else None
        train_idx, test_idx = train_test_split(
            idx,
            test_size=self.test_size,
            shuffle=self.shuffle,
            random_state=self.random_state,
            stratify=stratify,
        )
        fit_idx, es_idx = _carve_iid_es(
            np.sort(train_idx).astype(np.int64),
            y=y if self.stratify else None,
            es_fraction=self.es_fraction,
            random_state=self.random_state,
        )
        yield Fold(fit_idx=fit_idx, es_idx=es_idx, test_idx=np.sort(test_idx).astype(np.int64))


class StratifiedKFoldSplitter:
    """Stratified k-fold split (default for binary classification)."""

    def __init__(
        self,
        *,
        n_splits: int = 5,
        shuffle: bool = True,
        random_state: int | None = None,
        es_fraction: float = 0.0,
    ) -> None:
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.es_fraction = es_fraction

    def split(self, dataset: Dataset) -> Iterator[Fold]:
        y = _target(dataset)
        skf = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=self.shuffle,
            random_state=self.random_state if self.shuffle else None,
        )
        return _yield_folds(
            skf.split(np.zeros((dataset.n_rows, 1)), y),
            lambda fit: _carve_iid_es(
                fit, y=y, es_fraction=self.es_fraction, random_state=self.random_state
            ),
        )


class KFoldSplitter:
    """Plain k-fold (not stratified); the default for regression (ADR-0023 §1)."""

    def __init__(
        self,
        *,
        n_splits: int = 5,
        shuffle: bool = True,
        random_state: int | None = None,
        es_fraction: float = 0.0,
    ) -> None:
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.es_fraction = es_fraction

    def split(self, dataset: Dataset) -> Iterator[Fold]:
        kf = KFold(
            n_splits=self.n_splits,
            shuffle=self.shuffle,
            random_state=self.random_state if self.shuffle else None,
        )
        return _yield_folds(
            kf.split(np.zeros((dataset.n_rows, 1))),
            lambda fit: _carve_iid_es(
                fit, y=None, es_fraction=self.es_fraction, random_state=self.random_state
            ),
        )


class GroupKFoldSplitter:
    """Group k-fold (regression): no group spans fit/test; no class stratification.

    ``random_state`` seeds only the group-disjoint es carve (ADR-0080); the GroupKFold itself stays
    deterministic by construction (no shuffle).
    """

    group_aware = True

    def __init__(
        self, *, n_splits: int = 5, random_state: int | None = None, es_fraction: float = 0.0
    ) -> None:
        self.n_splits = n_splits
        self.random_state = random_state
        self.es_fraction = es_fraction

    def split(self, dataset: Dataset) -> Iterator[Fold]:
        groups = _groups(dataset)
        gkf = GroupKFold(n_splits=self.n_splits)
        return _yield_folds(
            gkf.split(np.zeros((dataset.n_rows, 1)), groups=groups),
            lambda fit: _carve_group_es(
                fit, groups=groups, es_fraction=self.es_fraction, random_state=self.random_state
            ),
        )


class StratifiedGroupKFoldSplitter:
    """Group k-fold that also stratifies by class (classification, ADR-0023 §2)."""

    group_aware = True

    def __init__(
        self,
        *,
        n_splits: int = 5,
        shuffle: bool = True,
        random_state: int | None = None,
        es_fraction: float = 0.0,
    ) -> None:
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.es_fraction = es_fraction

    def split(self, dataset: Dataset) -> Iterator[Fold]:
        groups = _groups(dataset)
        y = _target(dataset)
        sgkf = StratifiedGroupKFold(
            n_splits=self.n_splits,
            shuffle=self.shuffle,
            random_state=self.random_state if self.shuffle else None,
        )
        return _yield_folds(
            sgkf.split(np.zeros((dataset.n_rows, 1)), y, groups),
            lambda fit: _carve_group_es(
                fit, groups=groups, es_fraction=self.es_fraction, random_state=self.random_state
            ),
        )


def outer_holdout_carve(
    dataset: Dataset,
    *,
    scheme: str,
    fraction: float,
    stratify: bool,
    random_state: int,
    purge: int = 0,
    purge_delta: float | None = None,
    period: str | None = None,
    period_size: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Scheme-aware single carve of an untouched outer holdout: ``(dev_idx, holdout_idx)`` (ADR-0029 §2).

    The leakage-sensitive carve lives behind the splitter boundary, not the composition root (fix
    R1-CA-4): ``timeseries`` holds out the latest time window (a ``purge`` gap before dev, plus the
    de Prado label-horizon purge when ``label_time`` is set, no future leak); ``timeseries_period`` does
    the same but holds out the latest whole PERIODS (period-aligned, ``purge`` counted in periods, ADR-0096
    §3) — without this branch a period scheme would fall into the shuffling split below and leak across
    time. Whenever a group column is present the carve holds out whole groups (``GroupShuffleSplit``) —
    not only under ``scheme='group'`` but under any shuffling scheme, because a row-wise split of
    group-structured rows leaks across the holdout (e.g. target encoding carries a group's outcome into
    its relatives, finding #11); only with no groups does it fall back to a single stratified/plain split
    (reusing ``HoldoutSplitter``). Dev is the remainder; the two index sets are disjoint by construction —
    the one-touch holdout invariant (NFR-M4-3). Seed-deterministic (NFR-M4-2).
    """
    if scheme == "timeseries":
        return _timeseries_carve(dataset, fraction, purge, purge_delta)
    if scheme == "timeseries_period":
        assert period is not None  # gated in composition (timeseries_period requires a period unit)
        return _period_carve(dataset, fraction, purge, purge_delta, period, period_size)
    if scheme == "group" or dataset.groups() is not None:
        return _group_carve(dataset, fraction, random_state)
    fold = next(
        HoldoutSplitter(
            test_size=fraction, shuffle=True, stratify=stratify, random_state=random_state
        ).split(dataset)
    )
    return fold.fit_idx, fold.test_idx


def _timeseries_carve(
    dataset: Dataset, fraction: float, purge: int, purge_delta: float | None
) -> tuple[np.ndarray, np.ndarray]:
    t = dataset.time()
    if t is None:
        raise SchemaValidationError(
            "timeseries outer holdout requires a time column (dataset.time())"
        )
    n = t.shape[0]
    n_holdout = max(1, round(fraction * n))
    order = np.argsort(t, kind="stable")  # original row indices in ascending-time order
    holdout = order[n - n_holdout :]
    holdout_min_t = t[holdout].min()
    dev = order[: max(0, n - n_holdout - purge)]  # drop the `purge` samples just before the holdout
    # strict time separation (purge_delta widens the cut to drop the Δt zone before the holdout, ADR-0097)
    dev = dev[_before_cut(t[dev], holdout_min_t, purge_delta)]
    t1 = dataset.label_time()
    if (
        t1 is not None
    ):  # de Prado horizon purge: a dev label window must end before the holdout (FR-M4-7)
        dev = dev[t1[dev] < holdout_min_t]
    return np.sort(dev).astype(np.int64), np.sort(holdout).astype(np.int64)


def _period_carve(
    dataset: Dataset,
    fraction: float,
    purge: int,
    purge_delta: float | None,
    period: str,
    period_size: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Period-aligned outer holdout for ``timeseries_period`` (ADR-0096 §3); mirrors :func:`_timeseries_carve`.

    Holds out the latest WHOLE periods covering >= ``fraction`` rows (no mid-period split), drops the
    ``purge`` periods just before them (or the ``purge_delta`` Δt zone, ADR-0097), then the value-based +
    ``label_time`` separation — so the holdout is honest across time even though the realized fraction snaps
    to a period boundary.
    """
    t = dataset.time()
    if t is None:
        raise SchemaValidationError(
            "timeseries_period outer holdout requires a time column (dataset.time())"
        )
    period_id, n_periods, _ = _period_key(t, period, period_size)
    n_holdout = max(1, round(fraction * t.shape[0]))
    counts = np.bincount(period_id, minlength=n_periods)
    rows_from = np.cumsum(counts[::-1])[::-1]  # rows in periods >= p (non-increasing in p)
    holdout_start = int(
        np.flatnonzero(rows_from >= n_holdout).max()
    )  # minimal set of latest periods
    holdout = np.flatnonzero(period_id >= holdout_start)
    holdout_min_t = t[holdout].min()
    dev = np.flatnonzero(
        period_id < holdout_start - purge
    )  # drop the `purge` periods before the holdout
    # strict value separation (purge_delta widens the cut to drop the Δt zone before the holdout, ADR-0097)
    dev = dev[_before_cut(t[dev], holdout_min_t, purge_delta)]
    t1 = dataset.label_time()
    if t1 is not None:  # de Prado horizon purge: a dev label window must end before the holdout
        dev = dev[t1[dev] < holdout_min_t]
    return np.sort(dev).astype(np.int64), np.sort(holdout).astype(np.int64)


def _group_carve(
    dataset: Dataset, fraction: float, random_state: int
) -> tuple[np.ndarray, np.ndarray]:
    groups = _groups(dataset)
    gss = GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=random_state)
    dev, holdout = next(gss.split(np.zeros((dataset.n_rows, 1)), groups=groups))
    return np.sort(dev).astype(np.int64), np.sort(holdout).astype(np.int64)


def _carve_es_tail(train_idx: np.ndarray, test_idx: np.ndarray, n_es: int) -> Fold:
    """Carve the es tail as the last ``n_es`` TIME-SORTED train rows; keep >= 1 fit row (ADR-0027/0080).

    Shared by both time-series splitters (NFR-7): ``train_idx`` must already be sorted by time so the tail
    is the latest rows. Raises when the fold has too few train rows to spare a non-empty es tail.
    """
    if train_idx.size < 2:
        raise SchemaValidationError(
            "timeseries fold has too few train rows to carve a non-empty es tail "
            "(a late rolling/embargo window can starve a fold's train -- reduce embargo, raise "
            "max_train_size, or lower n_es)"
        )
    n_es = min(n_es, train_idx.size - 1)  # leave at least one row for fit
    return Fold(
        fit_idx=train_idx[:-n_es].astype(np.int64),
        es_idx=train_idx[-n_es:].astype(np.int64),
        test_idx=test_idx.astype(np.int64),
    )


def _before_cut(times: np.ndarray, test_min_t: object, purge_delta: float | None) -> np.ndarray:
    """Keep-mask for train rows strictly before the test, optionally widened by a Δt purge (ADR-0097).

    ``purge_delta=None`` -> plain ``t < test_min_t`` (drops tied boundary timestamps too, current behavior);
    set -> keep ``t < test_min_t - purge_delta``, i.e. drop the half-open serial-correlation zone
    ``[test_min_t - purge_delta, test_min_t)``. Works on numeric and datetime64 axes: the gap is measured in
    the axis' own units (a datetime64 difference is a timedelta64 cast to its integer-unit count), so
    ``purge_delta`` is in those storage units.
    """
    if purge_delta is None:
        return times < test_min_t
    return (test_min_t - times).astype(np.float64) > purge_delta


def _drop_embargo_zone(
    train_idx: np.ndarray, t: np.ndarray, prev_max_t: object, embargo_delta: float
) -> np.ndarray:
    """Drop train rows in the Δt embargo zone after an earlier test window (de Prado, value-based; ADR-0097).

    Removes rows with ``prev_max_t <= t < prev_max_t + embargo_delta`` (half-open, upper bound exclusive — the
    same convention as the purge cut, F11); rows before the window or at/after its end are kept. Same unit
    handling as :func:`_before_cut`.
    """
    off = (t[train_idx] - prev_max_t).astype(np.float64)
    return train_idx[(off < 0.0) | (off >= embargo_delta)]


def _period_key(
    t: np.ndarray, period: str, period_size: float | None
) -> tuple[np.ndarray, int, int]:
    """Densified period id per row + ``(n_periods, n_dropped_empty)`` for ``timeseries_period`` (ADR-0096 §2).

    Calendar units floor the timestamp to the period start (``week`` is ISO/Monday-anchored via integer day
    arithmetic — NOT numpy ``datetime64[W]``, which anchors to Thursday); ``delta`` bins a NUMERIC axis into
    fixed ``period_size`` windows (``feature_selection.structure_labels`` pattern). A calendar unit on a
    numeric axis (or ``delta`` on datetime) fails with ``SchemaValidationError`` (R-2). Keys are
    consecutive-integer-spaced, so empty interior periods are counted (``n_dropped_empty``) and densified to
    ``0..P-1`` without shifting row indexing (R-3). Vectorized, O(n) (NFR-1).
    """
    is_datetime = np.issubdtype(t.dtype, np.datetime64)
    if period == "delta":
        if is_datetime:
            raise SchemaValidationError(
                "period='delta' needs a numeric time axis (a datetime width depends on the storage unit); "
                "use period='day'/'week'/'month' for a datetime axis"
            )
        assert period_size is not None  # guaranteed by the CVConfig validator (period='delta')
        key = ((t.astype(np.float64) - float(t.min())) / period_size).astype(np.int64)
    else:
        if not is_datetime:
            raise SchemaValidationError(
                f"period={period!r} needs a datetime time axis; use period='delta' with period_size for "
                "a numeric axis"
            )
        if period == "month":
            key = t.astype("datetime64[M]").astype(np.int64)
        else:
            days = t.astype("datetime64[D]").astype(np.int64)
            key = days if period == "day" else (days + 3) // 7  # ISO week id (Monday-anchored)
    inv = np.unique(key, return_inverse=True)[1].reshape(-1)
    n_periods = int(inv.max()) + 1
    n_dropped = int(key.max()) - int(key.min()) + 1 - n_periods
    return inv.astype(np.int64), n_periods, n_dropped


class TimeSeriesSplitter:
    """Value-based expanding-window time-series CV with purge/embargo (ADR-0027).

    Folds are ordered by the **value** of ``dataset.time()`` (not row position) and yield original
    row indices (index-aligned with ``design_matrix``, ADR-0023). The ``time_ordered`` marker makes
    ``run_slice`` validate each fold against the value-based overlap invariant. Anti-leakage gaps:
    ``purge`` drops the samples just before the test (generalized sklearn ``gap``); ``embargo`` drops
    the samples right after each earlier test window from later training (serial correlation, de Prado);
    an optional ``label_time`` (``t1``) drops train rows whose label window reaches the test (full
    de Prado purge). A non-empty ``es`` tail is carved from the end of each train window (FR-M4-8).
    ``purge_delta``/``embargo_delta`` are the value-based (Δt) analogues of the row gaps for irregular axes
    (ADR-0097; mutually exclusive with the integer gap on the same axis); ``max_train_size`` bounds the
    lookback to the last N rows (rolling; ``None`` -> expanding, ADR-0099).
    """

    time_ordered = True

    def __init__(
        self,
        *,
        n_splits: int = 5,
        n_test: int = 1,
        n_es: int = 1,
        purge: int = 0,
        embargo: int = 0,
        purge_delta: float | None = None,
        embargo_delta: float | None = None,
        max_train_size: int | None = None,
    ) -> None:
        self.n_splits = n_splits
        self.n_test = n_test
        self.n_es = n_es
        self.purge = purge
        self.embargo = embargo
        self.purge_delta = purge_delta
        self.embargo_delta = embargo_delta
        self.max_train_size = max_train_size

    def split(self, dataset: Dataset) -> Iterator[Fold]:
        t = dataset.time()
        if t is None:
            raise SchemaValidationError(
                "TimeSeriesSplitter requires a time column (dataset.time())"
            )
        t1 = dataset.label_time()
        n = t.shape[0]
        order = np.argsort(t, kind="stable")  # original row indices in ascending-time order
        total_test = self.n_splits * self.n_test
        first_test = n - total_test
        train_cap = first_test - self.purge  # rows available to the first (smallest) fold's train
        if self.max_train_size is not None:
            train_cap = min(train_cap, self.max_train_size)
        if train_cap < self.n_es + 1:
            raise SchemaValidationError(
                f"too few rows ({n}) for {self.n_splits} timeseries folds "
                f"(n_test={self.n_test}, purge={self.purge}, n_es={self.n_es}, "
                f"max_train_size={self.max_train_size})"
            )
        bounds = [
            (first_test + k * self.n_test, first_test + (k + 1) * self.n_test)
            for k in range(self.n_splits)
        ]
        for k, (ts, te) in enumerate(bounds):
            test_idx = order[ts:te]
            test_min_t = t[test_idx].min()
            hi = (
                ts - self.purge
            )  # purge drops the `purge` rows just before the test (0 when purge_delta)
            keep = np.ones(hi, dtype=bool)
            if (
                self.max_train_size is not None
            ):  # rolling: keep only the last max_train_size train rows
                lo = hi - self.max_train_size
                if lo > 0:
                    keep[:lo] = False
            if self.embargo:  # exclude the embargo zone after each earlier test window (de Prado)
                for _, prev_te in bounds[:k]:
                    if prev_te < hi:
                        keep[prev_te : prev_te + self.embargo] = False
            train_idx = order[:hi][keep]
            # strict value separation (purge_delta widens the cut to drop the Δt zone just before the test)
            train_idx = train_idx[_before_cut(t[train_idx], test_min_t, self.purge_delta)]
            if (
                self.embargo_delta is not None
            ):  # value-based embargo zone after each earlier test window
                for prev_ts, prev_te in bounds[:k]:
                    train_idx = _drop_embargo_zone(
                        train_idx, t, t[order[prev_ts:prev_te]].max(), self.embargo_delta
                    )
            if t1 is not None:  # de Prado horizon purge: label window must end before the test
                train_idx = train_idx[t1[train_idx] < test_min_t]
            yield _carve_es_tail(train_idx, test_idx, self.n_es)


class PeriodTimeSeriesSplitter:
    """Value-based walk-forward CV over calendar/Δt PERIODS (ADR-0096).

    Rows are bucketed into periods (``period`` unit, :func:`_period_key`); the window walks forward over
    the period axis — each fold tests ``n_test`` consecutive periods and trains on all strictly earlier
    ones (expanding). ``n_test``/``purge``/``embargo`` count PERIODS here (not rows, the unit rule, FR-3);
    ``n_es`` stays rows. Anti-leakage is the SAME checked mechanism as :class:`TimeSeriesSplitter`: train
    rows are value-sorted and filtered to ``t < min(t[test])`` (and ``label_time`` horizon-purged), the es
    tail is the train's time-tail, and ``time_ordered`` makes ``run_slice`` validate every fold against the
    value-based overlap invariant (FR-7). ``split_meta`` exposes the densified period counts for the
    truthful manifest (ADR-0096 §4). ``purge_delta``/``embargo_delta`` are the value-based (Δt) gap
    analogues for irregular axes (ADR-0097); ``max_train_periods`` bounds the lookback to the last N periods
    (rolling; ``None`` -> expanding, ADR-0099) -- with ``purge`` > 0 the gap is taken from those N, so the
    effective train span is ``max_train_periods - purge`` periods (the row scheme instead subtracts purge in
    addition to ``max_train_size``).
    """

    time_ordered = True

    def __init__(
        self,
        *,
        period: str,
        n_splits: int = 5,
        n_test: int = 1,
        n_es: int = 1,
        purge: int = 0,
        embargo: int = 0,
        period_size: float | None = None,
        step_periods: int | None = None,
        purge_delta: float | None = None,
        embargo_delta: float | None = None,
        max_train_periods: int | None = None,
    ) -> None:
        self.period = period
        self.n_splits = n_splits
        self.n_test = n_test
        self.n_es = n_es
        self.purge = purge
        self.embargo = embargo
        self.period_size = period_size
        self.step_periods = step_periods if step_periods is not None else n_test
        self.purge_delta = purge_delta
        self.embargo_delta = embargo_delta
        self.max_train_periods = max_train_periods
        self._meta: dict[str, object] | None = None

    def split_meta(self) -> dict[str, object] | None:
        """Diagnostics of the last :meth:`split` for the run-report ``cv`` block (ADR-0096 §4); None before."""
        return self._meta

    def split(self, dataset: Dataset) -> Iterator[Fold]:
        t = dataset.time()
        if t is None:
            raise SchemaValidationError(
                "PeriodTimeSeriesSplitter requires a time column (dataset.time())"
            )
        t1 = dataset.label_time()
        period_id, n_periods, n_dropped = _period_key(t, self.period, self.period_size)
        self._meta = {
            "period": self.period,
            "n_periods": n_periods,
            "n_folds": self.n_splits,
            "n_dropped_empty": n_dropped,
        }
        # anchor the LAST fold to end at the final period (latest data is the last test, ADR-0027) and
        # walk earlier folds back by `step`: this generalizes `first = P - n_splits*n_test` to ANY step
        # (step>n_test leaves controlled gaps, step<n_test overlaps) WITHOUT overflowing the period axis,
        # so a test window is never empty and never silently truncated.
        first_test = (n_periods - self.n_test) - (self.n_splits - 1) * self.step_periods
        if first_test - self.purge < 1:  # need >= 1 training period before the first test window
            raise SchemaValidationError(
                f"too few periods ({n_periods}) for {self.n_splits} timeseries_period folds "
                f"(n_test={self.n_test}, step={self.step_periods}, purge={self.purge} periods, "
                f"period={self.period!r}); need >= 1 training period before the first test window"
            )
        bounds = [
            (first_test + k * self.step_periods, first_test + k * self.step_periods + self.n_test)
            for k in range(self.n_splits)
        ]
        for k, (ps, pe) in enumerate(bounds):
            test_idx = np.flatnonzero((period_id >= ps) & (period_id < pe))
            test_min_t = t[test_idx].min()
            allowed = np.zeros(n_periods, dtype=bool)
            hi = ps - self.purge  # purge drops the `purge` periods just before the test window
            if hi > 0:
                allowed[:hi] = True
            if (
                self.max_train_periods is not None
            ):  # rolling: only the last max_train_periods periods
                lo = ps - self.max_train_periods
                if lo > 0:
                    allowed[:lo] = False
            if (
                self.embargo
            ):  # drop the embargo periods right after each earlier test window (de Prado)
                for _, prev_pe in bounds[:k]:
                    allowed[prev_pe : prev_pe + self.embargo] = False
            train_idx = np.flatnonzero(allowed[period_id])
            train_idx = train_idx[
                np.argsort(t[train_idx], kind="stable")
            ]  # time-sorted for the es tail
            # strict value separation (purge_delta widens the cut), then de Prado horizon purge
            train_idx = train_idx[_before_cut(t[train_idx], test_min_t, self.purge_delta)]
            if (
                self.embargo_delta is not None
            ):  # value-based embargo zone after each earlier test window
                for prev_ps, prev_pe in bounds[:k]:
                    prev_test = np.flatnonzero((period_id >= prev_ps) & (period_id < prev_pe))
                    train_idx = _drop_embargo_zone(
                        train_idx, t, t[prev_test].max(), self.embargo_delta
                    )
            if t1 is not None:
                train_idx = train_idx[t1[train_idx] < test_min_t]
            if self.max_train_periods is not None and train_idx.size < self.n_es + 2:
                # rolling can starve a fold (periods are unequal in rows); fail the run, don't skip the
                # fold silently (R-5/G5). Expanding (max_train_periods=None) keeps the byte-identical
                # behavior — its only floor is _carve_es_tail's train.size>=2 (NFR-5).
                raise SchemaValidationError(
                    f"timeseries_period rolling fold has too few train rows ({train_idx.size}) to carve "
                    f"n_es={self.n_es} + a non-empty fit; raise max_train_periods "
                    f"(={self.max_train_periods}) or lower n_es"
                )
            yield _carve_es_tail(train_idx, test_idx, self.n_es)
