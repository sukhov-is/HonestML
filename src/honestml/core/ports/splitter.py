"""The ``CVSplitter`` port and the ``Fold`` anti-leakage mechanism.

A ``Fold`` carries three disjoint index sets (``fit``/``es``/``test``):
``es`` is the held-out early-stopping tail so test metrics stay unbiased.
``validate_fold`` makes anti-leakage a *checked mechanism*, not a convention:
indices must not overlap, group sets must not leak across splits, and for
time-series the latest train time must precede the earliest test time (the
value-based overlap invariant ``max(times[fit+es]) < min(times[test])``).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from ..dataset import Dataset
from ..exceptions import SchemaValidationError


@dataclass(frozen=True)
class Fold:
    """One CV fold: fit / early-stopping / test row indices (disjoint)."""

    fit_idx: np.ndarray
    es_idx: np.ndarray
    test_idx: np.ndarray


@runtime_checkable
class CVSplitter(Protocol):
    """Yields folds for a dataset; the use-case (not the port) runs the CV loop."""

    def split(self, dataset: Dataset) -> Iterator[Fold]: ...


@runtime_checkable
class TimeOrderedSplitter(Protocol):
    """A splitter whose folds are ordered in time (role-interface, ADR-0027 §2).

    The marker is set ``True`` only by time-series splitters; the use-case uses ``isinstance`` to enable
    the value-based fold check (:func:`validate_fold`) and the expanding-window OOF encoders (ADR-0082).
    Additive capability: a splitter without the marker is plain i.i.d. (presence == the capability, no
    ``getattr`` default).
    """

    time_ordered: bool


@runtime_checkable
class ReportsSplitMeta(Protocol):
    """A splitter that exposes diagnostics of its last ``split`` for the truthful manifest (ADR-0096 §4).

    The use-case reads it via ``isinstance`` after consuming ``split`` and surfaces the dict in the
    run-report ``cv`` block. Additive role-interface (presence == the capability); ``split_meta``
    returns ``None`` before ``split`` has run.
    """

    def split_meta(self) -> dict[str, object] | None: ...


@runtime_checkable
class GroupAwareSplitter(Protocol):
    """A splitter that partitions by group, so group leakage must be checked (role-interface, ADR-0023).

    The marker is set ``True`` only by group splitters; the use-case uses ``isinstance`` to fetch
    ``dataset.groups()`` and run the group-disjointness check in :func:`validate_fold`. Additive
    (presence == the capability).
    """

    group_aware: bool


def _overlap(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.intersect1d(a, b).size > 0)


def validate_fold(
    fold: Fold,
    *,
    groups: np.ndarray | None = None,
    time_ordered: bool = False,
    times: np.ndarray | None = None,
) -> None:
    """Raise :class:`SchemaValidationError` if the fold violates anti-leakage.

    - the three index sets must be pairwise disjoint;
    - with a ``group`` role, the group sets must be pairwise disjoint (no entity
      appears in two splits);
    - for time-series (``time_ordered`` with ``times``), no train sample's **time** falls in the
      test interval: ``max(times[fit ∪ es]) < min(times[test])``. ``es`` is part of
      training (``fit ∪ es``), so it is checked too. This is a value-based overlap invariant — it
      proves order/no-overlap, not the purge/embargo magnitude (a separate splitter test).
    """
    fit_idx, es_idx, test_idx = fold.fit_idx, fold.es_idx, fold.test_idx

    if _overlap(fit_idx, es_idx) or _overlap(fit_idx, test_idx) or _overlap(es_idx, test_idx):
        raise SchemaValidationError("fold index sets overlap (leakage)")

    if groups is not None:
        gf, ge, gt = groups[fit_idx], groups[es_idx], groups[test_idx]
        if _overlap(gf, ge) or _overlap(gf, gt) or _overlap(ge, gt):
            raise SchemaValidationError("group leakage: a group spans two fold splits")

    if time_ordered and times is not None and test_idx.size:
        train_idx = np.concatenate([fit_idx, es_idx]) if es_idx.size else fit_idx
        if train_idx.size and times[train_idx].max() >= times[test_idx].min():
            raise SchemaValidationError(
                "time-series fold leaks: a train sample's time is within the test interval"
            )
