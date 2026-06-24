"""Canonical polars-dtype tokens for the schema-owned category table (ADR-0017).

The token is a stable, version-independent name of a train column's dtype, stored in
``CategoryTable.source_dtype`` so an inference column can be coerced to the train dtype
before category lookup. Kept in the adapter (polars zone): the core never interprets it.
Tokens are compared with ``==`` (not dict-hash) to match polars dtype semantics and to
stay robust across polars repr changes (e.g. ``Utf8`` -> ``String``).
"""

from __future__ import annotations

import polars as pl

# a polars dtype as a class (e.g. pl.Int64) or an instance (e.g. series.dtype)
_DType = pl.DataType | type[pl.DataType]

# integer family: the only tokens that trigger value-preserving coercion (int↔float drift)
_INT_TOKENS: list[tuple[_DType, str]] = [
    (pl.Int8, "int8"),
    (pl.Int16, "int16"),
    (pl.Int32, "int32"),
    (pl.Int64, "int64"),
    (pl.UInt8, "uint8"),
    (pl.UInt16, "uint16"),
    (pl.UInt32, "uint32"),
    (pl.UInt64, "uint64"),
]
_OTHER_TOKENS: list[tuple[_DType, str]] = [
    (pl.Float32, "float32"),
    (pl.Float64, "float64"),
    (pl.Boolean, "boolean"),
]


def to_token(dtype: _DType) -> str | None:
    """Canonical token for a polars dtype, or ``None`` if outside the vocabulary."""
    if dtype == pl.Utf8:
        return "string"
    if dtype == pl.Categorical:
        return "categorical"
    for dt, token in _INT_TOKENS + _OTHER_TOKENS:
        if dtype == dt:
            return token
    return None


def int_dtype_for_token(token: str | None) -> _DType | None:
    """Polars integer dtype for a token, or ``None`` for non-integer/unknown tokens."""
    if token is None:
        return None
    for dt, tok in _INT_TOKENS:
        if tok == token:
            return dt
    return None
