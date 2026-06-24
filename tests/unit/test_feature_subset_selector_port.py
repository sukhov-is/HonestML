"""M6c FR-FSC-3: the FeatureSubsetSelector port contract (runtime-checkable Protocol)."""

from __future__ import annotations

import numpy as np
import pytest

from honestml.core import FeatureSubsetSelector, Fold

pytestmark = pytest.mark.unit


class _FakeSelector:
    """A structural FeatureSubsetSelector: keep the columns whose subset maximizes score_subset."""

    name = "fake_seq"

    def select(self, x, y, folds, *, categorical, score_subset, random_state, sample_weight=None):
        n = x.shape[1]
        keep = list(range(n))
        best = score_subset(keep)
        while len(keep) > 1:
            trials = [(score_subset([c for c in keep if c != j]), j) for j in keep]
            gain, drop = max(trials)
            if gain < best:
                break
            keep = [c for c in keep if c != drop]
            best = gain
        return tuple(sorted(keep))


def test_fake_selector_satisfies_protocol() -> None:
    assert isinstance(_FakeSelector(), FeatureSubsetSelector)


def test_select_calls_score_subset_with_indices_only() -> None:
    # the adapter receives only column indices, never raw test rows (Humble Object, ADR-0046 §1)
    seen: list = []

    def score_subset(indices):
        seen.append(list(indices))
        return -len(list(indices))  # fewer columns scores higher -> drives down to one

    sel = _FakeSelector()
    out = sel.select(
        np.zeros((4, 3)),
        np.array([0, 1, 0, 1]),
        [Fold(np.array([0, 1]), np.array([], dtype=int), np.array([2, 3]))],
        categorical=np.zeros(3, dtype=bool),
        score_subset=score_subset,
        random_state=0,
    )
    assert out == (0,) or len(out) == 1  # floored to a single feature
    assert all(
        isinstance(i, int) for call in seen for i in call
    )  # only indices crossed the boundary
