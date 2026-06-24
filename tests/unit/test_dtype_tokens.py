"""FR-F: canonical dtype tokens map both ways; only integers trigger coercion (ADR-0017)."""

from __future__ import annotations

import polars as pl
import pytest

from honestml.adapters.dtype_tokens import int_dtype_for_token, to_token

pytestmark = pytest.mark.unit


def test_to_token_vocabulary() -> None:
    assert to_token(pl.Series([1], dtype=pl.Int64).dtype) == "int64"
    assert to_token(pl.Series([1], dtype=pl.UInt32).dtype) == "uint32"
    assert to_token(pl.Series([1.0], dtype=pl.Float64).dtype) == "float64"
    assert to_token(pl.Series(["a"], dtype=pl.Utf8).dtype) == "string"
    assert to_token(pl.Series(["a"], dtype=pl.Categorical).dtype) == "categorical"
    assert to_token(pl.Series([True], dtype=pl.Boolean).dtype) == "boolean"
    assert to_token(pl.Series([1], dtype=pl.Date).dtype) is None  # outside vocabulary


def test_int_dtype_for_token_only_integers() -> None:
    assert int_dtype_for_token("int64") == pl.Int64
    assert int_dtype_for_token("uint8") == pl.UInt8
    assert int_dtype_for_token("float64") is None  # not coerced
    assert int_dtype_for_token("string") is None
    assert int_dtype_for_token(None) is None
