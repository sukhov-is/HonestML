"""M6b FR-FS-2: the FeatureRanker port contract (runtime-checkable Protocol)."""

from __future__ import annotations

import numpy as np
import pytest

from honestml.core import FeatureRanker

pytestmark = pytest.mark.unit


class _FakeRanker:
    """A structural FeatureRanker: scores = column means (no model, no folds)."""

    name = "fake"

    def rank(self, x, y, *, categorical, random_state, sample_weight=None):
        if x.shape[0] == 0:
            raise ValueError("empty training matrix")
        return np.abs(x).mean(axis=0)

    def auto_threshold(self, n_features):
        return 1.0 / n_features


def test_fake_ranker_satisfies_protocol() -> None:
    assert isinstance(_FakeRanker(), FeatureRanker)


def test_rank_returns_per_feature_vector() -> None:
    r = _FakeRanker()
    x = np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
    out = r.rank(x, np.array([0, 1]), categorical=np.zeros(3, dtype=bool), random_state=0)
    assert out.shape == (3,)


def test_rank_empty_x_raises() -> None:
    r = _FakeRanker()
    with pytest.raises(ValueError):
        r.rank(np.empty((0, 3)), np.empty(0), categorical=np.zeros(3, dtype=bool), random_state=0)


# --- M6d FR-FSH-2: optional structure label `groups` (backward-compatible kwarg, ADR-0050 §1) ---


class _StructureRanker:
    """A ranker that accepts the M6d `groups` kwarg (structure-aware path)."""

    name = "structure"

    def rank(self, x, y, *, categorical, random_state, sample_weight=None, groups=None):
        return np.abs(x).mean(axis=0)

    def auto_threshold(self, n_features):
        return 0.0


def test_structure_ranker_satisfies_protocol() -> None:
    assert isinstance(_StructureRanker(), FeatureRanker)


def test_rank_accepts_groups_and_default_none_matches() -> None:
    r = _StructureRanker()
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    y = np.array([0, 1])
    cat = np.zeros(2, dtype=bool)
    base = r.rank(x, y, categorical=cat, random_state=0)
    with_groups = r.rank(x, y, categorical=cat, random_state=0, groups=np.array([0, 0]))
    assert np.allclose(base, with_groups)
