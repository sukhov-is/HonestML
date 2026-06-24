"""M1b: Fold anti-leakage is a checked mechanism, not a convention (R-6/N-4)."""

from __future__ import annotations

import numpy as np
import pytest

from honestml.adapters.splitters import _carve_es_tail
from honestml.core import Fold, SchemaValidationError, validate_fold

pytestmark = pytest.mark.unit


def test_carve_es_tail_starved_message_names_cause() -> None:
    """F103: a fold starved of train rows (a late rolling/embargo window) names embargo/max_train_size
    as the cause, instead of an es-tail message that misdirects."""
    with pytest.raises(SchemaValidationError, match="embargo"):
        _carve_es_tail(np.array([5]), np.array([6, 7]), n_es=1)  # 1 train row < 2 -> starved


def _fold(fit, es, test) -> Fold:
    return Fold(np.array(fit), np.array(es), np.array(test))


def test_disjoint_fold_passes() -> None:
    validate_fold(_fold([0, 1, 2], [3, 4], [5, 6]))  # no raise


def test_overlapping_indices_rejected() -> None:
    with pytest.raises(SchemaValidationError, match="overlap"):
        validate_fold(_fold([0, 1, 2], [2, 3], [4, 5]))  # 2 in fit and es


def test_group_leakage_rejected() -> None:
    groups = np.array([10, 10, 20, 20, 30, 30])
    # index 1 (group 10) in fit, index 0 (group 10) in test -> group 10 leaks
    with pytest.raises(SchemaValidationError, match="group leakage"):
        validate_fold(_fold([1, 2], [3], [0, 4]), groups=groups)


def test_group_disjoint_passes() -> None:
    groups = np.array([10, 10, 20, 20, 30, 30])
    validate_fold(_fold([0, 1], [2, 3], [4, 5]), groups=groups)  # groups disjoint


def test_time_order_passes() -> None:
    # value-based: every train (fit∪es) time is strictly before the test interval
    times = np.array([0, 1, 2, 3, 4, 5, 6])
    validate_fold(_fold([0, 1, 2], [3, 4], [5, 6]), time_ordered=True, times=times)


def test_time_order_violation_rejected() -> None:
    times = np.array([0, 1, 2, 3, 4, 5, 6])
    with pytest.raises(SchemaValidationError, match="time-series"):
        # fit rows carry the latest times but are placed in train, while test carries the earliest
        validate_fold(_fold([5, 6], [3, 4], [0, 1]), time_ordered=True, times=times)


def test_time_overlap_checks_es_tail() -> None:
    # an es sample whose time falls inside the test interval is leakage (es is part of training)
    times = np.array([0, 1, 2, 9, 4, 5, 6])  # index 3 (es) has time 9 > test min
    with pytest.raises(SchemaValidationError, match="time-series"):
        validate_fold(_fold([0, 1, 2], [3, 4], [5, 6]), time_ordered=True, times=times)


def test_time_ordered_without_times_is_noop() -> None:
    validate_fold(_fold([5, 6], [3, 4], [0, 1]), time_ordered=True)  # no times -> no value check
