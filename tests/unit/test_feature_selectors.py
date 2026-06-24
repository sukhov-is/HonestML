"""M6c FR-FSC-3 / ADR-0083-0084: SequentialSelector greedy backward elimination returns the trajectory."""

from __future__ import annotations

import numpy as np
import pytest

from honestml.adapters import SequentialSelector
from honestml.core import FeatureSubsetSelector, Fold

pytestmark = pytest.mark.unit

_NOCAT = np.zeros(5, dtype=bool)
_FOLD = [Fold(np.array([0, 1]), np.array([], dtype=int), np.array([2, 3]))]


def _select(sel, n_features, score_subset, *, cat=None):
    return sel.select(
        np.zeros((4, n_features)),
        np.array([0, 1, 0, 1]),
        _FOLD,
        categorical=_NOCAT[:n_features] if cat is None else cat,
        score_subset=score_subset,
        random_state=0,
    )


def test_sequential_satisfies_port() -> None:
    assert isinstance(SequentialSelector(), FeatureSubsetSelector)


def test_sequential_trajectory_is_sorted_strictly_decreasing() -> None:
    traj = _select(
        SequentialSelector(min_features=1, full_descent=True), 5, lambda idx: float(sum(idx))
    )
    assert traj[0] == (0, 1, 2, 3, 4)  # full set first
    sizes = [len(t) for t in traj]
    assert sizes == sorted(sizes, reverse=True) and len(set(sizes)) == len(
        sizes
    )  # strictly decreasing
    assert all(tuple(sorted(t)) == t for t in traj)  # each subset sorted


def test_sequential_trajectory_contains_rewarded_subset() -> None:
    # score peaks for the subset {0, 2}; under full descent the greedy path passes through it
    good = frozenset({0, 2})

    def score_subset(indices):
        s = set(indices)
        return float(len(good & s) - 0.1 * len(s - good))

    traj = _select(SequentialSelector(min_features=1, full_descent=True), 5, score_subset)
    assert good in {frozenset(t) for t in traj}


def test_sequential_respects_floor() -> None:
    # a monotone "fewer is better" scorer descends to the floor; smallest subset has >= min_features
    traj = _select(SequentialSelector(min_features=2, patience=1), 5, lambda idx: -len(list(idx)))
    assert len(traj[-1]) == 2 and traj[-1] == tuple(sorted(traj[-1]))


def test_sequential_full_descent_length() -> None:
    # full_descent ignores patience and goes to the floor: length = n - min_features + 1
    traj = _select(
        SequentialSelector(min_features=1, patience=1, full_descent=True),
        5,
        lambda idx: float(len(list(idx))),
    )
    assert len(traj) == 5 and len(traj[-1]) == 1


def test_sequential_patience_truncates_off_path() -> None:
    # off-path (full_descent=False): a never-improving scorer stops after `patience` non-improving steps
    traj = _select(
        SequentialSelector(min_features=1, patience=2, full_descent=False),
        5,
        lambda idx: float(len(list(idx))),
    )
    assert len(traj) == 3  # full + 2 non-improving drops, then stop


def test_sequential_deterministic() -> None:
    def score_subset(indices):
        return float(sum(indices))

    sel = SequentialSelector(patience=2)
    assert _select(sel, 5, score_subset) == _select(sel, 5, score_subset)


def test_sequential_calls_scorer_with_indices_only() -> None:
    seen: list = []

    def score_subset(indices):
        seen.append(list(indices))
        return float(len(list(indices)))  # wider is better -> keeps all but explores drops

    _select(SequentialSelector(patience=1), 4, score_subset)
    assert seen and all(isinstance(i, int) for call in seen for i in call)
