"""M1a property: category codes are a stable, schema-owned function (N-1/N-2).

The same value always maps to the same code regardless of which dataset it appears
in; values unseen at fit map to the reserved unknown code; nulls map to the
reserved null code. This is what makes train and inference encoding identical.
"""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from honestml.core import CategoryTable

pytestmark = pytest.mark.property

_cats = st.lists(st.text(min_size=1, max_size=4), min_size=1, max_size=30)
_queries = st.lists(st.one_of(st.text(max_size=5), st.none()), max_size=40)


@given(train=_cats, query=_queries)
def test_codes_are_stable_and_reserved(train: list[str], query: list[str | None]) -> None:
    table = CategoryTable.fit(train)
    known = set(table.categories)
    codes = table.encode(query)

    assert len(codes) == len(query)
    for value, code in zip(query, codes):
        if value is None:
            assert code == table.null_code
        elif value in known:
            assert 0 <= code < len(table.categories)
            assert code == table.categories.index(value)
        else:
            assert code == table.unknown_code


@given(train=_cats)
def test_known_codes_are_a_bijection_with_categories(train: list[str]) -> None:
    table = CategoryTable.fit(train)
    codes = table.encode(list(table.categories))
    assert list(codes) == list(range(len(table.categories)))
