"""M1b property: absolute ranking — adding/removing a candidate never reorders the rest.

This is the C1 fix: with candidate-relative normalization, one extra model could
flip the ranks of all others. Absolute scoring makes each rank independent.
"""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from honestml.core import Candidate, SelectionPolicy, rank

pytestmark = pytest.mark.property

# distinct scores so the order is unambiguous
_scores = st.lists(
    st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    min_size=2,
    max_size=30,
    unique=True,
)


@given(scores=_scores, drop=st.integers(min_value=0, max_value=29))
def test_removing_a_candidate_preserves_relative_order(scores: list[float], drop: int) -> None:
    policy = SelectionPolicy()
    cands = [Candidate(id=f"c{i}", score=s) for i, s in enumerate(scores)]

    full_order = [c.id for c in rank(cands, policy)]

    drop_idx = drop % len(cands)
    removed_id = cands[drop_idx].id
    reduced = [c for c in cands if c.id != removed_id]
    reduced_order = [c.id for c in rank(reduced, policy)]

    assert reduced_order == [cid for cid in full_order if cid != removed_id]


@given(scores=_scores)
def test_adding_a_candidate_preserves_relative_order(scores: list[float]) -> None:
    policy = SelectionPolicy()
    base = [Candidate(id=f"c{i}", score=s) for i, s in enumerate(scores)]
    base_order = [c.id for c in rank(base, policy)]

    extra = Candidate(id="extra", score=2e6)  # above the generated range
    extended_order = [c.id for c in rank([*base, extra], policy)]

    assert [cid for cid in extended_order if cid != "extra"] == base_order
