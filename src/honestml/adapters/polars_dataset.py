"""Default ``Dataset`` implementation backed by polars (ADR-0005).

polars lives only here (an adapter), never in the domain — the import-linter
contract forbids polars/pandas from ``honestml.core``. Categorical columns are
materialized as integer codes through the schema-owned table, so the model
boundary stays numpy + codes.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import polars as pl

from honestml.core.exceptions import SchemaValidationError
from honestml.core.schema import CategoryTable, FeatureSchema

from .dtype_tokens import int_dtype_for_token


class PolarsDataset:
    """A polars frame plus its :class:`FeatureSchema`.

    Satisfies the ``honestml.core.Dataset`` Protocol structurally (no nominal
    inheritance — ports are structural, ADR-0006).
    """

    def __init__(
        self,
        frame: pl.DataFrame,
        schema: FeatureSchema,
        *,
        weight_col: str | None = None,
        label_time_col: str | None = None,
    ) -> None:
        self._frame = frame
        self._schema = schema
        self._weight_col = weight_col
        self._label_time_col = label_time_col

    @property
    def schema(self) -> FeatureSchema:
        return self._schema

    @property
    def n_rows(self) -> int:
        return self._frame.height

    @property
    def columns(self) -> list[str]:
        return self._frame.columns

    def select(self, columns: Sequence[str]) -> PolarsDataset:
        cols = [c for c in columns if c in self._frame.columns]
        new_roles = {c: r for c, r in self._schema.roles.items() if c in cols}
        new_cats = {c: t for c, t in self._schema.categories.items() if c in cols}
        new_schema = self._schema.model_copy(update={"roles": new_roles, "categories": new_cats})
        weight = self._weight_col if self._weight_col in cols else None
        label_time = self._label_time_col if self._label_time_col in cols else None
        return PolarsDataset(
            self._frame.select(cols), new_schema, weight_col=weight, label_time_col=label_time
        )

    def take(self, indices: Sequence[int] | np.ndarray) -> PolarsDataset:
        idx = [int(i) for i in indices]
        return PolarsDataset(
            self._frame[idx],
            self._schema,
            weight_col=self._weight_col,
            label_time_col=self._label_time_col,
        )

    def with_selected_features(self, names: Sequence[str]) -> PolarsDataset:
        # same frame, schema gains selected_features -> design_matrix projects (ADR-0045 §1/§2)
        return PolarsDataset(
            self._frame,
            self._schema.with_selected_features(names),
            weight_col=self._weight_col,
            label_time_col=self._label_time_col,
        )

    def to_numpy(self) -> np.ndarray:
        numeric = self._schema.numeric
        if not numeric:
            return np.empty((self.n_rows, 0), dtype=np.float64)
        return self._frame.select(numeric).to_numpy().astype(np.float64, copy=False)

    def categorical_codes(self) -> np.ndarray:
        categorical = self._schema.categorical
        if not categorical:
            return np.empty((self.n_rows, 0), dtype=np.int64)
        exprs = []
        for col in categorical:
            table = self._schema.categories.get(col)
            if table is None:
                raise SchemaValidationError(
                    f"categorical column {col!r} has no fitted category table"
                )
            exprs.append(_encode_expr(col, table))
        return self._frame.select(exprs).to_numpy().astype(np.int64, copy=False)

    def target(self) -> np.ndarray | None:
        target = self._schema.target
        if target is None or target not in self._frame.columns:
            return None
        return self._frame[target].to_numpy()

    def sample_weight(self) -> np.ndarray | None:
        if self._weight_col is None:
            return None
        return self._frame[self._weight_col].to_numpy()

    def groups(self) -> np.ndarray | None:
        group = self._schema.group
        if group is None or group not in self._frame.columns:
            return None
        return self._frame[group].to_numpy()

    def time(self) -> np.ndarray | None:
        col = self._schema.time
        if col is None or col not in self._frame.columns:
            return None
        return self._frame[col].to_numpy()

    def label_time(self) -> np.ndarray | None:
        if self._label_time_col is None or self._label_time_col not in self._frame.columns:
            return None
        return self._frame[self._label_time_col].to_numpy()


def _encode_expr(col: str, table: CategoryTable) -> pl.Expr:
    """Vectorized category encoding in polars (ADR-0005: encodings in polars).

    Equivalent to :meth:`CategoryTable.encode` (guarded by a property test): null →
    ``null_code``, known → its code, unseen → ``unknown_code``. Keeps the hot path
    off the per-row Python loop while the table stays the schema-owned source of
    truth.
    """
    as_str = _key_expr(col, table.source_dtype)
    mapping = {c: i for i, c in enumerate(table.categories)}
    if mapping:
        known = as_str.replace_strict(mapping, default=table.unknown_code, return_dtype=pl.Int64)
    else:
        known = pl.lit(table.unknown_code, dtype=pl.Int64)
    return (
        pl.when(as_str.is_null())
        .then(pl.lit(table.null_code, dtype=pl.Int64))
        .otherwise(known)
        .alias(col)
    )


def _key_expr(col: str, source_dtype: str | None) -> pl.Expr:
    """String lookup key, value-preservingly coercing integer columns to the train
    dtype first (ADR-0017): csv ``1.0`` against train ``Int64`` becomes ``"1"`` and
    matches, while a fractional ``1.5`` stays unrepresentable and falls through to
    ``unknown_code`` (never a wrong known code).

    The key is built in **each branch** (both ``Utf8``) so ``when/then/otherwise`` does
    not promote ``Int64``/``Float64`` to a common ``Float64`` supertype — that would
    re-render ``1`` as ``"1.0"`` and defeat the coercion (ADR-0017 §3, R-1).
    """
    target = int_dtype_for_token(source_dtype)
    base = pl.col(col)
    if target is None:
        return base.cast(pl.Utf8)
    casted = base.cast(target, strict=False)  # fractional / overflow / NaN -> null
    preserved = casted.is_not_null() & (
        casted.cast(pl.Float64) == base.cast(pl.Float64, strict=False)
    )
    return pl.when(preserved).then(casted.cast(pl.Utf8)).otherwise(base.cast(pl.Utf8))
