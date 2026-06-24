"""M1a property: the vectorized polars encoding equals the pure CategoryTable.encode.

PolarsDataset encodes categories with a fast polars expression (ADR-0005), while
``CategoryTable`` stays the schema-owned source of truth. This test guards against
the two diverging — they must produce identical codes for any input.
"""

from __future__ import annotations

import string

import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st

from honestml.adapters import PolarsDataset
from honestml.core import CategoryTable, ColumnRole, FeatureSchema

pytestmark = pytest.mark.property

_alphabet = string.ascii_letters + string.digits
_cats = st.lists(st.text(alphabet=_alphabet, min_size=1, max_size=4), min_size=1, max_size=20)
_query = st.lists(st.one_of(st.text(alphabet=_alphabet, max_size=4), st.none()), max_size=40)


@given(train=_cats, query=_query)
def test_vectorized_codes_match_reference(train: list[str], query: list[str | None]) -> None:
    table = CategoryTable.fit(train)
    schema = FeatureSchema(roles={"c": ColumnRole.CATEGORICAL}).with_categories({"c": table})
    frame = pl.DataFrame({"c": query}, schema={"c": pl.Utf8})

    vectorized = PolarsDataset(frame, schema).categorical_codes()[:, 0]
    reference = table.encode(query)

    assert list(vectorized) == list(reference)


_ints = st.lists(st.integers(min_value=-50, max_value=50), min_size=1, max_size=30)


@given(values=_ints)
def test_int_float_drift_codes_match(values: list[int]) -> None:
    # FR-2 (ADR-0017): with an integer source_dtype, a Float64 inference frame of the
    # same integer values yields identical codes to an Int64 frame. Uses a Float64 frame
    # (not Utf8) so a supertype-promotion regression (R-1) would fail this test.
    table = CategoryTable.fit(values, source_dtype="int64")
    schema = FeatureSchema(roles={"c": ColumnRole.CATEGORICAL}).with_categories({"c": table})
    int_frame = pl.DataFrame({"c": values}, schema={"c": pl.Int64})
    float_frame = pl.DataFrame({"c": [float(v) for v in values]}, schema={"c": pl.Float64})

    int_codes = PolarsDataset(int_frame, schema).categorical_codes()[:, 0]
    float_codes = PolarsDataset(float_frame, schema).categorical_codes()[:, 0]

    assert list(int_codes) == list(float_codes)


def test_boolean_codes_match_reference() -> None:
    # F010: a Boolean categorical must key identically at fit (core ``str``) and encode (polars
    # ``cast(Utf8)``). polars renders bool lowercase ("true"/"false") while Python str(bool) is
    # ("True"/"False"); without the ``_key`` alignment fit stored "True"/"False" and every bool row
    # collapsed to ``unknown_code`` on both OOF and inference (the feature went dead silently).
    table = CategoryTable.fit([True, False, True], source_dtype="boolean")
    schema = FeatureSchema(roles={"c": ColumnRole.CATEGORICAL}).with_categories({"c": table})
    frame = pl.DataFrame({"c": [True, False, None]}, schema={"c": pl.Boolean})

    vectorized = PolarsDataset(frame, schema).categorical_codes()[:, 0]
    reference = table.encode([True, False, None])

    assert list(vectorized) == list(reference)
    assert vectorized[0] != table.unknown_code  # True is a known category, not unknown
    assert vectorized[1] != table.unknown_code  # False is a known category, not unknown
