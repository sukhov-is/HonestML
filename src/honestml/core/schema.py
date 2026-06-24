"""Feature schema, column roles and the schema-owned category table.

The category table is the correctness core of the data boundary: it is
fitted on training data, **serialized into the artifact** and reused at inference,
so a category always maps to the same integer code on train and inference. Codes
are passed to models as categorical features, preserving native handling
(CatBoost/LightGBM). Unseen values and nulls get dedicated reserved codes.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from enum import Enum
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class ColumnRole(str, Enum):
    """Role a column plays. The core never hard-codes domain column names."""

    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    TEXT = "text"
    TARGET = "target"
    GROUP = "group"
    TIME = "time"
    FOLD = "fold"
    IGNORE = "ignore"


def _is_null(value: object) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def te_output_name(col: str) -> str:
    """Target-encoding output column name for a source categorical."""
    return f"{col}_te"


def freq_output_name(col: str) -> str:
    """Frequency-encoding output column name for a source categorical."""
    return f"{col}_freq"


def intersection_output_name(a: str, b: str) -> str:
    """Intersection output column name for an ordered pair."""
    return f"{a}__{b}"


def categorical_positions(features: Sequence[str], categorical: Sequence[str]) -> list[int]:
    """Positions of CATEGORICAL columns within a feature list, by role membership (ADR-0088).

    The single definition of native-cat routing indices, shared by ``FeatureSchema.categorical_indices``
    and the use-case (over an already-projected feature list), so the three call sites cannot drift. By
    membership (not a ``len(numeric):`` slice) it stays correct after FE/FS reorders or drops columns (R-6).
    """
    cat = set(categorical)
    return [i for i, f in enumerate(features) if f in cat]


NativeRoutingReason = Literal["native", "high_cardinality"]


def native_routing(schema: FeatureSchema, cap: int | None) -> dict[str, NativeRoutingReason]:
    """Per-column native-routing verdict over the CATEGORICAL columns (ADR-0092/0093/0095).

    A column routes natively iff its true category count (``len(categories)``, excluding the two
    reserve codes — R-6) is ``<= cap``; otherwise it is demoted to the ordinal-codes path
    (``"high_cardinality"``). ``cap=None`` disables the gate (every categorical native — the opt-out).
    Pure function of the FROZEN schema (+ the scalar ``cap``): no ``n_rows``, no data, so the decision
    is identical on CV-fit, refit, HPO and inference and cannot drift (D-2, R-2).
    """
    cats = schema.categorical
    if cap is None:
        return {c: "native" for c in cats}
    return {
        c: ("native" if len(schema.categories[c].categories) <= cap else "high_cardinality")
        for c in cats
    }


def native_routable(schema: FeatureSchema, cap: int | None) -> list[str]:
    """CATEGORICAL names allowed to route natively — the cardinality-gated subset (ADR-0092).

    The single gate shared by every routing site (``run_slice``/``tune_estimators`` via
    ``categorical_positions`` and :meth:`FeatureSchema.categorical_indices`), so the indices cannot
    drift across CV/refit/HPO/inference (FR-2). ``cap=None`` ⇒ the full categorical set (gate off).
    """
    return [c for c, reason in native_routing(schema, cap).items() if reason == "native"]


class CategoryTable(BaseModel):
    """Frozen ordered category list owning the train↔inference code mapping.

    Codes: known categories ``0..n-1`` (the order of :attr:`categories`),
    ``null_code = n`` (explicit null category), ``unknown_code = n+1`` (any value
    unseen at fit). All codes are non-negative so they can be fed to LightGBM /
    CatBoost as categorical features.

    ``source_dtype`` is an opaque canonical token of the train column dtype, set by
    the adapter at fit. The core stores it only; the adapter uses it to coerce an
    inference column to the train dtype before key lookup, so int↔float read drift
    (csv vs parquet) yields identical codes. ``extra="ignore"`` is explicit
    so an older reader silently drops future keys (forward-compat).
    """

    model_config = ConfigDict(frozen=True, extra="ignore")

    categories: tuple[str, ...]
    source_dtype: str | None = None

    @staticmethod
    def _key(v: object) -> str:
        # canonical category key matching the adapter's polars cast(Utf8) encoder: polars renders
        # Boolean lowercase ("true"/"false") while Python str(bool) gives ("True"/"False") — without
        # this a Boolean categorical keys differently at fit (core) vs encode (polars) and collapses to
        # the unknown code (F010). All other dtypes already agree (str == polars Utf8 cast).
        return str(v).lower() if isinstance(v, bool) else str(v)

    @classmethod
    def fit(cls, values: Iterable[object], *, source_dtype: str | None = None) -> CategoryTable:
        """Build a table from training values (nulls excluded, order deterministic)."""
        uniq = {cls._key(v) for v in values if not _is_null(v)}
        return cls(categories=tuple(sorted(uniq)), source_dtype=source_dtype)

    @property
    def null_code(self) -> int:
        return len(self.categories)

    @property
    def unknown_code(self) -> int:
        return len(self.categories) + 1

    @property
    def cardinality(self) -> int:
        """Number of distinct codes including the null and unknown reserves."""
        return len(self.categories) + 2

    def _index(self) -> dict[str, int]:
        return {c: i for i, c in enumerate(self.categories)}

    def encode(self, values: Sequence[object]) -> np.ndarray:
        """Map values to integer codes using the frozen table (pure, deterministic)."""
        index = self._index()
        null_code, unknown_code = self.null_code, self.unknown_code
        out = np.empty(len(values), dtype=np.int64)
        for i, v in enumerate(values):
            if _is_null(v):
                out[i] = null_code
            else:
                out[i] = index.get(self._key(v), unknown_code)
        return out


class DatetimeDeltaSpec(BaseModel):
    """Datetime->report-date deltas in days — schema-owned, leak-safe per row.

    ``deltas`` are ordered ``(source_col, output_name)`` pairs; the output is a NUMERIC feature
    ``report_date - source`` in days. ``extra="ignore"`` keeps forward-compat (an older reader drops
    future keys), like :class:`CategoryTable`.
    """

    model_config = ConfigDict(frozen=True, extra="ignore")

    report_date: str
    deltas: tuple[tuple[str, str], ...]


class TargetEncodingSpec(BaseModel):
    """Full-train smoothed target-mean per category code.

    ``encodings`` is ``col -> {code_str -> smoothed_mean}`` keyed by ``str(<int CategoryTable code>)``;
    null/unknown/uncovered codes are NOT stored and fall back to ``global_mean`` on lookup. ``smoothing``
    is the ``k`` used at fit (kept for reproducibility/audit). Used for refit/inference/holdout — the
    OOF path (evaluation) is a separate cross-fit.
    """

    model_config = ConfigDict(frozen=True, extra="ignore")

    encodings: dict[str, dict[str, float]]
    global_mean: float
    smoothing: float


class FrequencyEncodingSpec(BaseModel):
    """Per-category relative frequency over train, keyed by ``str(code)``.

    ``frequencies`` is ``col -> {code_str -> freq}``; an unseen code maps to ``0.0`` on lookup
    (target-independent, so leak-safe at the boundary). ``extra="ignore"`` for forward-compat.
    """

    model_config = ConfigDict(frozen=True, extra="ignore")

    frequencies: dict[str, dict[str, float]]


class IntersectionSpec(BaseModel):
    """Ordered categorical-pair intersections; each ``a__b`` is its own CATEGORICAL.

    ``pairs`` are ``(a, b)`` with ``a < b`` lexicographically (``combinations(sorted(...), 2)``); the
    output column ``a__b`` gets its own :class:`CategoryTable` in ``FeatureSchema.categories``.
    """

    model_config = ConfigDict(frozen=True, extra="ignore")

    pairs: tuple[tuple[str, str], ...]


class FeatureSchema(BaseModel):
    """Typed column contract: roles + schema-owned category tables + NaN policy.

    Serializable, so the same schema (including fitted category tables and FE specs) is reused at
    inference. Built/validated by the ``Reader`` at the data boundary. The FE specs
    (``datetime_spec``/``target_encoding``/``frequency_encoding``/``intersections``) are additive and
    default ``None`` so an older artifact loads unchanged.
    """

    model_config = ConfigDict(extra="forbid")

    roles: dict[str, ColumnRole]
    categories: dict[str, CategoryTable] = Field(default_factory=dict)
    # reserved NaN policy (ADR-0015): only "keep" today (NaN flows through to NaN-capable models). Kept as
    # a single-value Literal because it is serialized into every artifact's schema.json — dropping it would
    # break loading older artifacts (FeatureSchema is extra="forbid"); reserved for a future imputation axis.
    numeric_nan: Literal["keep"] = "keep"
    datetime_spec: DatetimeDeltaSpec | None = None
    frequency_encoding: FrequencyEncodingSpec | None = None
    target_encoding: TargetEncodingSpec | None = None
    intersections: IntersectionSpec | None = None
    # feature-selection subset (ADR-0045 §1): ordered names kept by the selector; None -> all features
    # (legacy/off). `design_matrix` projects to it; additive, so an older artifact loads unchanged.
    selected_features: tuple[str, ...] | None = None

    def _cols(self, role: ColumnRole) -> list[str]:
        return [c for c, r in self.roles.items() if r == role]

    def _fe_numeric_outputs(self) -> list[str]:
        """FE-produced NUMERIC outputs in the pinned order: datetime ⊕ frequency ⊕ target_encoding."""
        out: list[str] = []
        if self.datetime_spec is not None:
            out += [name for _, name in self.datetime_spec.deltas]
        if self.frequency_encoding is not None:
            out += [freq_output_name(c) for c in self.frequency_encoding.frequencies]
        if self.target_encoding is not None:
            out += [te_output_name(c) for c in self.target_encoding.encodings]
        return out

    @property
    def numeric(self) -> list[str]:
        """NUMERIC features: ``original_numeric ⊕ datetime ⊕ frequency ⊕ target_encoding``.

        Block order is derived from the FE specs, not the roles-dict insertion order, so it is
        deterministic and identical train==inference. Without FE this is the plain role view,
        unchanged.
        """
        cols = self._cols(ColumnRole.NUMERIC)
        fe_outputs = self._fe_numeric_outputs()
        if not fe_outputs:
            return cols
        present = set(cols)
        fe_block = [c for c in fe_outputs if c in present]
        fe_set = set(fe_block)
        original = [c for c in cols if c not in fe_set]
        return original + fe_block

    @property
    def categorical(self) -> list[str]:
        """CATEGORICAL features: ``original_categorical ⊕ intersections``."""
        cols = self._cols(ColumnRole.CATEGORICAL)
        if self.intersections is None:
            return cols
        inter = [intersection_output_name(a, b) for a, b in self.intersections.pairs]
        present = set(cols)
        inter_block = [c for c in inter if c in present]
        inter_set = set(inter_block)
        original = [c for c in cols if c not in inter_set]
        return original + inter_block

    @property
    def datetime(self) -> list[str]:
        return self._cols(ColumnRole.DATETIME)

    @property
    def text(self) -> list[str]:
        return self._cols(ColumnRole.TEXT)

    @property
    def features(self) -> list[str]:
        """Model-facing features in the pinned FE block order.

        ``numeric ⊕ categorical`` where each block is itself FE-block-ordered, so this equals
        ``original_numeric ⊕ datetime ⊕ frequency ⊕ target_encoding ⊕ original_categorical ⊕
        intersections``. ``design_matrix`` materializes the numeric block then the categorical codes,
        so column ``j`` of the model input is exactly ``features[j]``. Without FE this is the
        unchanged ``numeric + categorical``.
        """
        return self.numeric + self.categorical

    def categorical_indices(self, cap: int | None = None) -> list[int]:
        """Positions of natively-routed CATEGORICAL columns in the post-FS-projection design matrix.

        Projects ``features`` to ``selected_features`` (in ``schema.features`` order, matching
        ``design_matrix``) then takes the positions of the cardinality-gated categorical names
        (:func:`native_routable`); includes intersections (``a__b``) subject to the same gate and
        excludes the FE numeric outputs (``_te``/``_freq``/datetime). ``cap=None`` keeps every
        categorical (ungated opt-out, ADR-0092/0094). Empty when the (possibly projected/gated) set
        carries no native categoricals — a legitimate native no-op.
        """
        selected = self.selected_features
        feats = (
            self.features if selected is None else [f for f in self.features if f in set(selected)]
        )
        return categorical_positions(feats, native_routable(self, cap))

    @property
    def target(self) -> str | None:
        cols = self._cols(ColumnRole.TARGET)
        return cols[0] if cols else None

    @property
    def group(self) -> str | None:
        cols = self._cols(ColumnRole.GROUP)
        return cols[0] if cols else None

    @property
    def time(self) -> str | None:
        """The TIME-role column (CV time axis), distinct from DATETIME features."""
        cols = self._cols(ColumnRole.TIME)
        return cols[0] if cols else None

    def with_categories(self, tables: dict[str, CategoryTable]) -> FeatureSchema:
        """Return a copy of the schema with the fitted category tables attached."""
        return self.model_copy(update={"categories": dict(tables)})

    def with_datetime_spec(self, spec: DatetimeDeltaSpec) -> FeatureSchema:
        """Return a copy with the fitted datetime-delta spec attached."""
        return self.model_copy(update={"datetime_spec": spec})

    def with_frequency_encoding(self, spec: FrequencyEncodingSpec) -> FeatureSchema:
        """Return a copy with the fitted frequency-encoding spec attached."""
        return self.model_copy(update={"frequency_encoding": spec})

    def with_target_encoding(self, spec: TargetEncodingSpec) -> FeatureSchema:
        """Return a copy with the fitted full-train target-encoding spec attached."""
        return self.model_copy(update={"target_encoding": spec})

    def with_intersections(self, spec: IntersectionSpec) -> FeatureSchema:
        """Return a copy with the intersection spec attached; pair tables go in categories."""
        return self.model_copy(update={"intersections": spec})

    def with_selected_features(self, names: Iterable[str]) -> FeatureSchema:
        """Return a copy carrying the selected feature subset; ``design_matrix`` projects to it."""
        return self.model_copy(update={"selected_features": tuple(names)})
