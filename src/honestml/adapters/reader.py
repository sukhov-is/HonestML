"""The ``Reader`` — public data boundary (ADR-0005, G5/NFR-13).

Accepts pandas / polars / numpy, validates the input against the ``Task``, infers
column roles (or reuses a given ``FeatureSchema`` at inference), fits the
schema-owned category tables and returns a :class:`PolarsDataset`. Invalid input
raises :class:`SchemaValidationError` with a specific reason — never a bare
``ValueError`` deep inside training.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
import polars as pl

from honestml.core.config import FEConfig
from honestml.core.exceptions import ConfigError, MissingDependencyError, SchemaValidationError
from honestml.core.logging import get_logger
from honestml.core.schema import (
    CategoryTable,
    ColumnRole,
    DatetimeDeltaSpec,
    FeatureSchema,
    FrequencyEncodingSpec,
    IntersectionSpec,
    TargetEncodingSpec,
    freq_output_name,
    intersection_output_name,
    te_output_name,
)
from honestml.core.task import Task, resolve_positive

from .dtype_tokens import to_token
from .polars_dataset import PolarsDataset, _encode_expr, _key_expr

_WEIGHT_COL = "__sample_weight__"
_GROUP_COL = "__group__"
_TIME_COL = "__time__"
_LABEL_TIME_COL = "__label_time__"
# auto-detect report-date among DATETIME columns (ADR-0018 §2); a narrow, case-insensitive name set
_REPORT_DATE_CANDIDATES = frozenset({"report_dt", "report_date", "feature_dt"})
_logger = get_logger("adapters.reader")


class TypingDecision(NamedTuple):
    """One auto-typing reclassification of a numeric column (ADR-0015).

    Diagnostic only — does not change the data. ``reason`` is a closed set:
    ``numeric_id_like`` (high-cardinality integer dropped), ``low_cardinality_int``
    (integer treated as categorical), ``string_id_like`` (near-unique string dropped),
    ``high_cardinality_string`` (high-cardinality string KEPT but flagged), ``all_null``
    (no non-null values) or ``constant`` (a single distinct value in every row). The drops
    (role ``ignore``) are ``numeric_id_like``/``string_id_like``/``all_null``/``constant``.
    """

    column: str
    source_dtype: str
    assigned_role: ColumnRole
    reason: str


class Reader:
    """Builds a validated :class:`PolarsDataset` from raw user input."""

    def __init__(self, task: Task | None = None, *, fe: FEConfig | None = None) -> None:
        self._task = task
        # feature-engineering catalog (ADR-0040 §4); drives which boundary-FE specs are fitted at
        # train. None -> only datetime deltas (driven by Task.report_date, ADR-0018) are attempted.
        # At inference the FE specs come from the given schema, so ``fe`` is unused there.
        self._fe = fe
        # diagnostic of the last read(); empty on the inference branch (schema given).
        # reflects only the most recent call — Reader is not thread-safe (ADR-0015).
        self.typing_report: list[TypingDecision] = []

    # -- public API ---------------------------------------------------------

    def read(
        self,
        X: pd.DataFrame | pl.DataFrame | np.ndarray,
        y: Any | None = None,
        *,
        schema: FeatureSchema | None = None,
        feature_names: list[str] | None = None,
        sample_weight: Any | None = None,
        groups: Any | None = None,
        time: Any | None = None,
        label_time: Any | None = None,
    ) -> PolarsDataset:
        """Build a validated dataset from raw input.

        ``groups`` (per-row group labels, e.g. for group-aware CV; ADR-0025) is row-aligned
        metadata like ``sample_weight``: it is attached as the reserved ``GROUP`` column and is
        never a feature. Null/NaN groups (incl. pandas ``pd.NA``) are rejected at this boundary.

        ``time`` (the CV time axis, ADR-0028) is attached as the reserved ``TIME`` column and
        ``label_time`` (optional label-end-time ``t1`` for full de Prado purge) as a name-based
        ``__label_time__`` column. Both must be complete (null/NaN rejected) and sortable, and
        ``label_time`` must share ``time``'s kind (both numeric or both datetime) so the purge can
        compare ``t1`` against ``time``; ``label_time`` without ``time`` is a :class:`ConfigError`.
        Neither is a feature.
        """
        self.typing_report = []
        frame = self._to_frame(X, feature_names)
        self._validate_frame(frame)

        target_col: str | None = None
        if y is not None:
            target_col = "__target__"
            frame = self._attach_column(frame, target_col, y, "y")
            self._validate_target(frame[target_col])

        weight_col: str | None = None
        if sample_weight is not None:
            weight_col = _WEIGHT_COL
            frame = self._attach_column(frame, weight_col, sample_weight, "sample_weight")

        if groups is not None:
            self._validate_groups(groups)
            frame = self._attach_column(frame, _GROUP_COL, groups, "groups")

        if time is not None:
            self._validate_time_complete(time, "time")
            frame = self._attach_column(frame, _TIME_COL, time, "time")
            self._validate_time_sortable(frame[_TIME_COL])

        label_time_col: str | None = None
        if label_time is not None:
            if time is None:
                raise ConfigError("label_time requires time (the event-time axis); pass time=")
            self._validate_time_complete(label_time, "label_time")
            frame = self._attach_column(frame, _LABEL_TIME_COL, label_time, "label_time")
            self._validate_time_sortable(frame[_LABEL_TIME_COL], "label_time")
            self._validate_time_compatible(frame[_TIME_COL], frame[_LABEL_TIME_COL])
            label_time_col = _LABEL_TIME_COL

        if schema is None:
            schema = self._infer_schema(frame, target_col)
            # source-categorical (pre-intersection) — the only TE/frequency sources (ADR-0042 §2)
            source_categorical = list(schema.categorical)
            frame, schema = self._fit_datetime_deltas(frame, schema)
            frame, schema = self._fit_intersections(frame, schema, source_categorical)
            schema = self._fit_categories(frame, schema)
            frame, schema = self._fit_frequency(frame, schema, source_categorical)
            frame, schema = self._fit_target_encoding(frame, schema, source_categorical, target_col)
        else:
            # FE materialization before validation: each transformer self-validates its sources
            # (fail-loud), since their sources are non-feature roles _validate_against_schema skips.
            frame = self._apply_datetime_deltas(frame, schema)
            frame = self._apply_intersections(frame, schema)
            frame = self._apply_frequency(frame, schema)
            frame = self._apply_target_encoding(frame, schema)
            self._validate_against_schema(frame, schema, target_col)
            self._warn_unknown_categories(frame, schema)

        return PolarsDataset(frame, schema, weight_col=weight_col, label_time_col=label_time_col)

    # -- conversion ---------------------------------------------------------

    def _to_frame(
        self,
        X: pd.DataFrame | pl.DataFrame | np.ndarray,
        feature_names: list[str] | None,
    ) -> pl.DataFrame:
        if isinstance(X, pl.DataFrame):
            self._reject_duplicate_columns(X.columns)
            return X.clone()
        if isinstance(X, pd.DataFrame):
            self._reject_duplicate_columns(list(X.columns))
            try:
                return pl.from_pandas(X)
            except ImportError as exc:
                # polars needs pyarrow to convert a pandas frame with any non-numpy-backed (string/
                # object) column — i.e. every real CSV. Surface the optional extra at the boundary
                # instead of a raw polars ImportError deep in the stack (ADR-0008, finding #3).
                raise MissingDependencyError("pyarrow") from exc
        if isinstance(X, np.ndarray):
            if X.ndim != 2:
                raise SchemaValidationError(f"X must be 2-D, got {X.ndim}-D array")
            names = feature_names or [f"f{i}" for i in range(X.shape[1])]
            if len(names) != X.shape[1]:
                raise SchemaValidationError(
                    f"feature_names length {len(names)} != n_columns {X.shape[1]}"
                )
            self._reject_duplicate_columns(names)
            return pl.DataFrame(X, schema=names)
        raise SchemaValidationError(
            f"unsupported X type {type(X).__name__}; expected pandas/polars/numpy"
        )

    @staticmethod
    def _reject_duplicate_columns(names: list[str]) -> None:
        """Fail fast on duplicate column names with a domain error instead of a raw polars one.

        Duplicate names (typical after ``pd.concat``/``merge`` with suffixes) otherwise surface as a
        bare ``ValueError``/``DuplicateError`` deep in the conversion, breaking the boundary contract
        (ADR-0008/0014).
        """
        seen: set[str] = set()
        dupes: list[str] = []
        for n in names:
            if n in seen and n not in dupes:
                dupes.append(n)
            seen.add(n)
        if dupes:
            raise SchemaValidationError(f"duplicate column names: {dupes}")

    def _attach_column(
        self, frame: pl.DataFrame, name: str, values: Any, label: str
    ) -> pl.DataFrame:
        arr = np.asarray(values).ravel()
        if len(arr) != frame.height:
            raise SchemaValidationError(f"{label} length {len(arr)} != X rows {frame.height}")
        return frame.with_columns(pl.Series(name, arr))

    # -- validation ---------------------------------------------------------

    def _validate_frame(self, frame: pl.DataFrame) -> None:
        if frame.height == 0:
            raise SchemaValidationError("X has no rows")
        if frame.width == 0:
            raise SchemaValidationError("X has no columns")

    def _validate_groups(self, groups: Any) -> None:
        """Reject null/NaN group labels at the boundary (ADR-0025 §4).

        ``pd.isna`` is pandas/polars-aware: it catches ``None``, ``np.nan`` and pandas-nullable
        ``pd.NA``/``Int64`` that the splitter-level guard misses — a null group would silently
        break group anti-leakage (DM3-4).
        """
        if bool(pd.isna(np.asarray(groups).ravel()).any()):
            raise SchemaValidationError("groups contains null/NaN; groups must be complete")

    def _validate_time_complete(self, values: Any, label: str) -> None:
        """Reject null/NaN/NaT in a time axis (ADR-0028 §2): a null breaks the time order."""
        if bool(pd.isna(np.asarray(values).ravel()).any()):
            raise SchemaValidationError(
                f"{label} contains null/NaN; the time axis must be complete"
            )

    def _validate_time_sortable(self, series: pl.Series, label: str = "time") -> None:
        """A time axis must be orderable (numeric or datetime), not free text (ADR-0028 §2)."""
        if not (series.dtype.is_numeric() or series.dtype in (pl.Date, pl.Datetime)):
            raise SchemaValidationError(
                f"{label} must be sortable (numeric or datetime), got dtype {series.dtype} "
                "(tz-aware or mixed-type datetimes arrive as Object -- pass tz-naive/UTC datetimes "
                "or a numeric axis)"
            )

    def _validate_time_compatible(self, time: pl.Series, label_time: pl.Series) -> None:
        """label_time must share time's kind so the de Prado purge can compare t1 against time (FR-M4-7)."""
        if time.dtype.is_numeric() != label_time.dtype.is_numeric():
            raise SchemaValidationError(
                f"label_time dtype {label_time.dtype} is not comparable with time dtype {time.dtype}; "
                "both must be numeric or both datetime"
            )

    def _validate_target(self, target: pl.Series) -> None:
        non_null = target.drop_nulls()
        if non_null.dtype.is_float():
            non_null = non_null.drop_nans()  # polars: NaN != null
        if non_null.len() == 0:
            raise SchemaValidationError("y is empty or all-null")
        if self._task is None:
            return
        kind = self._task.kind
        if kind == "binary":
            n_classes = non_null.n_unique()
            if n_classes > 2:
                raise SchemaValidationError(
                    f"Task.kind='binary' but y has {n_classes} distinct classes"
                )
        elif kind == "multiclass":
            if non_null.n_unique() < 2:
                raise SchemaValidationError(
                    "Task.kind='multiclass' requires at least 2 distinct classes in y"
                )
        elif kind == "regression" and not target.dtype.is_numeric():
            raise SchemaValidationError(
                f"Task.kind='regression' requires numeric y, got dtype {target.dtype}"
            )

    def _validate_against_schema(
        self, frame: pl.DataFrame, schema: FeatureSchema, target_col: str | None
    ) -> None:
        present = set(frame.columns)
        required = list(
            schema.features
        )  # keep '' too: an empty-string column name is valid in pandas
        missing = [c for c in required if c not in present]
        if missing:
            raise SchemaValidationError(f"input is missing required columns: {missing}")

    # -- role inference -----------------------------------------------------

    def _infer_schema(self, frame: pl.DataFrame, target_col: str | None) -> FeatureSchema:
        # auto-typing thresholds come from Task (single source of defaults, ADR-0015); a task-less
        # Reader() falls back to Task's own defaults — the thresholds are kind-independent.
        defaults = self._task or Task(kind="binary")
        cat_max_unique = defaults.numeric_cat_max_unique
        id_rate = defaults.numeric_id_rate
        id_min_unique = defaults.numeric_id_min_unique
        str_id_rate = defaults.string_id_rate
        str_id_min_unique = defaults.string_id_min_unique
        n = frame.height
        roles: dict[str, ColumnRole] = {}

        for col in frame.columns:
            if col == target_col:
                roles[col] = ColumnRole.TARGET
                continue
            if col == _WEIGHT_COL or col == _LABEL_TIME_COL:
                continue  # name-based metadata, no role (read via Dataset.sample_weight/label_time)
            if col == _GROUP_COL:
                roles[col] = ColumnRole.GROUP
                continue
            if col == _TIME_COL:
                roles[col] = ColumnRole.TIME
                continue
            series = frame[col]
            dtype = series.dtype
            # an all-null or constant column carries no signal -> drop as IGNORE before the dtype rules
            # (ADR-0015 ext, finding #8a). This also keeps an all-NaN numeric column out of the "NaN in
            # numeric features" gate that would otherwise evict baseline/linear (finding #6). Datetimes are
            # exempt: a constant report-date is the normal case and feeds the delta FE (ADR-0018).
            if dtype not in (pl.Date, pl.Datetime):
                degenerate = self._degenerate_role(series)
                if degenerate is not None:
                    roles[col] = ColumnRole.IGNORE
                    self._record_typing(col, dtype, ColumnRole.IGNORE, degenerate)
                    continue
            if dtype in (pl.Date, pl.Datetime):
                roles[col] = ColumnRole.DATETIME
            elif dtype == pl.Utf8 or dtype == pl.Categorical:
                role, reason = self._string_role(series, n, str_id_rate, str_id_min_unique)
                roles[col] = role
                if reason is not None:  # string id-like dropped, or high-cardinality flagged (kept)
                    self._record_typing(col, dtype, role, reason)
            elif dtype.is_numeric():
                role, reason = self._numeric_role(
                    frame[col], n, cat_max_unique, id_rate, id_min_unique, dtype
                )
                roles[col] = role
                if reason is not None:  # numeric reclassified vs naive dtype baseline
                    self._record_typing(col, dtype, role, reason)
            else:
                roles[col] = ColumnRole.CATEGORICAL

        return FeatureSchema(roles=roles)

    def _numeric_role(
        self,
        series: pl.Series,
        n: int,
        cat_max_unique: int,
        id_rate: float,
        id_min_unique: int,
        dtype: Any,
    ) -> tuple[ColumnRole, str | None]:
        """Return the role and, if reclassified from the numeric baseline, the reason."""
        if not dtype.is_integer():
            return ColumnRole.NUMERIC, None
        nuniq = series.drop_nulls().n_unique()
        rate = nuniq / n if n > 0 else 0.0
        if rate > id_rate and nuniq > id_min_unique:
            return ColumnRole.IGNORE, "numeric_id_like"
        if nuniq <= cat_max_unique:
            return ColumnRole.CATEGORICAL, "low_cardinality_int"
        return ColumnRole.NUMERIC, None

    @staticmethod
    def _string_role(
        series: pl.Series, n: int, id_rate: float, id_min_unique: int
    ) -> tuple[ColumnRole, str | None]:
        """Role for a string/categorical column (ADR-0015 ext, finding #7).

        A near-unique string (high distinct rate AND high distinct count, e.g. a Name/Ticket id) is pure
        noise as a category at inference — dropped as ``string_id_like``. A merely high-cardinality column
        (above the count floor but below the id rate) is kept but flagged ``high_cardinality_string`` so the
        user can weigh its transfer risk at fit time. Everything else is the plain categorical baseline.
        """
        nuniq = series.drop_nulls().n_unique()
        if nuniq <= id_min_unique:
            return ColumnRole.CATEGORICAL, None
        rate = nuniq / n if n > 0 else 0.0
        if rate > id_rate:
            return ColumnRole.IGNORE, "string_id_like"
        return ColumnRole.CATEGORICAL, "high_cardinality_string"

    @staticmethod
    def _degenerate_role(series: pl.Series) -> str | None:
        """``"all_null"``/``"constant"`` for a no-signal column, else ``None`` (ADR-0015 ext, finding #8a).

        ``constant`` requires a single value in EVERY row: a lone value mixed with nulls (e.g.
        ``["a", None, "a"]``) still carries a missingness signal, so it is kept.
        """
        non_null = series.drop_nulls()
        if series.dtype.is_float():
            non_null = non_null.drop_nans()  # polars: NaN is not null
        if non_null.len() == 0:
            return "all_null"
        if non_null.len() == series.len() and non_null.n_unique() <= 1:
            return "constant"
        return None

    def _record_typing(self, column: str, dtype: Any, role: ColumnRole, reason: str) -> None:
        self.typing_report.append(TypingDecision(column, str(dtype), role, reason))
        _logger.info(
            "auto-typing column=%s dtype=%s role=%s reason=%s", column, dtype, role.value, reason
        )

    def _fit_categories(self, frame: pl.DataFrame, schema: FeatureSchema) -> FeatureSchema:
        tables = {
            col: CategoryTable.fit(frame[col].to_list(), source_dtype=to_token(frame[col].dtype))
            for col in schema.categorical
        }
        return schema.with_categories(tables)

    # -- feature engineering (ADR-0040/0041/0042/0018) ----------------------

    def _codes(self, frame: pl.DataFrame, col: str, table: CategoryTable) -> np.ndarray:
        """Integer codes for a categorical column, consistent with ``categorical_codes`` (ADR-0017).

        Reuses the schema-owned, dtype-coercing polars encoder so a frequency/TE output is keyed by
        the SAME code as the source categorical — no int↔float read drift between train and inference.
        """
        return frame.select(_encode_expr(col, table)).to_numpy().ravel().astype(np.int64)

    def _resolve_report_date(self, frame: pl.DataFrame, schema: FeatureSchema) -> str | None:
        """Resolve the report-date column (ADR-0018 §2): explicit override or narrow auto-detect."""
        datetime_cols = schema.datetime
        override = self._task.report_date if self._task is not None else None
        if override is not None:
            if override not in frame.columns or override not in datetime_cols:
                raise SchemaValidationError(
                    f"Task.report_date={override!r} is not a DATETIME column in the input"
                )
            return override
        matches = [c for c in datetime_cols if c.lower() in _REPORT_DATE_CANDIDATES]
        return matches[0] if len(matches) == 1 else None

    def _delta_expr(self, report_date: str, source: str, out: str) -> pl.Expr:
        """``report_date - source`` in whole days as Float64 (ADR-0018 §3); Date-normalized first."""
        rd = pl.col(report_date).cast(pl.Date)
        return (rd - pl.col(source).cast(pl.Date)).dt.total_days().cast(pl.Float64).alias(out)

    def _fit_datetime_deltas(
        self, frame: pl.DataFrame, schema: FeatureSchema
    ) -> tuple[pl.DataFrame, FeatureSchema]:
        report_date = self._resolve_report_date(frame, schema)
        if report_date is None:
            if schema.datetime:
                _logger.warning(
                    "datetime columns %s have no report date; dropped from features "
                    "(set Task.report_date or name a report_dt/report_date/feature_dt column)",
                    schema.datetime,
                )
            return frame, schema
        sources = [c for c in schema.datetime if c != report_date]
        if not sources:
            return frame, schema
        exprs: list[pl.Expr] = []
        deltas: list[tuple[str, str]] = []
        new_roles = dict(schema.roles)
        for col in sources:
            out = f"{col}__days_to_report"
            if out in frame.columns:
                raise SchemaValidationError(
                    f"datetime delta output {out!r} collides with an existing column"
                )
            exprs.append(self._delta_expr(report_date, col, out))
            deltas.append((col, out))
            new_roles[out] = ColumnRole.NUMERIC
        frame = frame.with_columns(exprs)
        schema = schema.model_copy(update={"roles": new_roles}).with_datetime_spec(
            DatetimeDeltaSpec(report_date=report_date, deltas=tuple(deltas))
        )
        return frame, schema

    def _apply_datetime_deltas(self, frame: pl.DataFrame, schema: FeatureSchema) -> pl.DataFrame:
        spec = schema.datetime_spec
        if spec is None:
            return frame  # symmetric no-op: trained without deltas
        if spec.report_date not in frame.columns:
            raise SchemaValidationError(
                f"datetime report date {spec.report_date!r} missing at inference"
            )
        exprs: list[pl.Expr] = []
        for source, out in spec.deltas:
            if source not in frame.columns:
                raise SchemaValidationError(f"datetime source {source!r} missing at inference")
            if out in frame.columns:
                raise SchemaValidationError(
                    f"datetime delta output {out!r} collides with an existing column"
                )
            exprs.append(self._delta_expr(spec.report_date, source, out))
        return frame.with_columns(exprs)

    # drift-signal threshold (F5.8 follow-up of ADR-0017): share of UNSEEN values per column
    _UNKNOWN_WARN_SHARE = 0.10
    # group-structure signal for a target-encoding source (finding #11): a high-cardinality categorical
    # whose values still repeat across rows behaves like an undeclared group, so a row-wise holdout/CV is
    # not independent. Flag only when BOTH the distinct-value rate is high (many small groups, unlike a
    # low-cardinality real feature) AND a non-trivial share of rows actually share a value.
    _TE_GROUP_RATE = 0.5
    _TE_GROUP_DUP_SHARE = 0.2

    def _warn_unknown_categories(self, frame: pl.DataFrame, schema: FeatureSchema) -> None:
        """Warn when an inference batch carries a high share of categories unseen at train.

        Unseen values are encoded honestly (the reserved ``unknown_code`` — never a wrong
        known code), so this is observability, not correctness: a large share usually means
        schema drift or a wrong upstream join. The unseen test uses the SAME coercion-aware key as
        the real encoder (``_key_expr``/``_encode_expr``, ADR-0017), so int↔float read drift (csv
        ``1.0`` vs train ``Int64`` ``1``) is not miscounted as unseen (a naive ``cast(Utf8)`` was).
        """
        named = [
            (name, table) for name, table in schema.categories.items() if name in frame.columns
        ]
        if not named:
            return
        shares = frame.select(
            [
                (
                    pl.col(name).is_not_null()
                    & ~_key_expr(name, table.source_dtype).is_in(list(table.categories))
                )
                .mean()
                .alias(name)
                for name, table in named
            ]
        )
        for name, _ in named:
            share = float(shares[name][0] or 0.0)
            if share > self._UNKNOWN_WARN_SHARE:
                _logger.warning(
                    "inference: %.0f%% of %r values were unseen at train (encoded as the "
                    "reserved unknown_code) — possible schema drift or a wrong upstream join",
                    share * 100,
                    name,
                )

    @staticmethod
    def _intersection_expr(
        a: str, b: str, out: str, dtype_a: str | None = None, dtype_b: str | None = None
    ) -> pl.Expr:
        """Concatenate two categoricals into one ``a__b`` category string (nulls -> ``__NA__``).

        Each source is keyed value-preservingly through ``_key_expr`` (ADR-0017): an int categorical read
        as float at inference (``1.0``) yields the same key ``"1"`` it had at train, so the combined
        category matches instead of becoming an all-unseen ``"1.0__…"`` (finding #8 FE-propagation). For a
        string source ``_key_expr`` is a plain ``cast(Utf8)``, so string×string pairs are unchanged.
        """
        va = _key_expr(a, dtype_a).fill_null("__NA__")
        vb = _key_expr(b, dtype_b).fill_null("__NA__")
        return pl.concat_str([va, vb], separator="__").alias(out)

    def _fit_intersections(
        self, frame: pl.DataFrame, schema: FeatureSchema, source_categorical: list[str]
    ) -> tuple[pl.DataFrame, FeatureSchema]:
        if self._fe is None or not self._fe.intersections:
            return frame, schema
        cats = sorted(source_categorical)
        if len(cats) < 2:
            return frame, schema  # nothing to pair (no WARNING — distinct from truncation)
        all_pairs = list(combinations(cats, 2))
        pairs = all_pairs[: self._fe.max_pairs]
        if len(all_pairs) > self._fe.max_pairs:
            _logger.warning(
                "categorical intersections truncated to max_pairs=%d (of %d possible pairs)",
                self._fe.max_pairs,
                len(all_pairs),
            )
        # F2.7 residual: the combined category joins values with "__" and encodes nulls as
        # "__NA__" — a REAL value equal to the sentinel or containing the separator would
        # silently merge distinct combinations into one category. Warn, don't fail: only the
        # derived intersection feature is affected, the base columns stay intact.
        involved = sorted({name for pair in pairs for name in pair})
        flags = frame.select(
            [
                (
                    pl.col(name).cast(pl.Utf8).str.contains("__", literal=True)
                    | (pl.col(name).cast(pl.Utf8) == "__NA__")
                )
                .any()
                .alias(name)
                for name in involved
            ]
        )
        risky = [name for name in involved if bool(flags[name][0])]
        if risky:
            _logger.warning(
                "intersections: column(s) %s contain the reserved separator '__' or the "
                "sentinel '__NA__' — distinct value pairs may merge into one combined category",
                risky,
            )
        exprs: list[pl.Expr] = []
        new_roles = dict(schema.roles)
        for a, b in pairs:
            out = intersection_output_name(a, b)
            if out in frame.columns:
                raise SchemaValidationError(
                    f"intersection output {out!r} collides with an existing column"
                )
            exprs.append(
                self._intersection_expr(
                    a, b, out, to_token(frame[a].dtype), to_token(frame[b].dtype)
                )
            )
            new_roles[out] = ColumnRole.CATEGORICAL
        frame = frame.with_columns(exprs)
        schema = schema.model_copy(update={"roles": new_roles}).with_intersections(
            IntersectionSpec(pairs=tuple(pairs))
        )
        return frame, schema

    def _apply_intersections(self, frame: pl.DataFrame, schema: FeatureSchema) -> pl.DataFrame:
        spec = schema.intersections
        if spec is None:
            return frame
        exprs: list[pl.Expr] = []
        for a, b in spec.pairs:
            for src in (a, b):
                if src not in frame.columns:
                    raise SchemaValidationError(f"intersection source {src!r} missing at inference")
            out = intersection_output_name(a, b)
            if out in frame.columns:
                raise SchemaValidationError(
                    f"intersection output {out!r} collides with an existing column"
                )
            exprs.append(
                self._intersection_expr(
                    a, b, out, schema.categories[a].source_dtype, schema.categories[b].source_dtype
                )
            )
        return frame.with_columns(exprs)

    def _fit_frequency(
        self, frame: pl.DataFrame, schema: FeatureSchema, source_categorical: list[str]
    ) -> tuple[pl.DataFrame, FeatureSchema]:
        if self._fe is None or not self._fe.frequency_encoding or not source_categorical:
            return frame, schema
        n = frame.height
        exprs: list[pl.Series] = []
        new_roles = dict(schema.roles)
        frequencies: dict[str, dict[str, float]] = {}
        for col in source_categorical:
            table = schema.categories[col]
            codes = self._codes(frame, col, table)
            counts = np.bincount(codes, minlength=table.cardinality).astype(np.float64)
            freq = counts / n
            exprs.append(pl.Series(freq_output_name(col), freq[codes]))
            new_roles[freq_output_name(col)] = ColumnRole.NUMERIC
            frequencies[col] = {str(c): float(freq[c]) for c in np.unique(codes)}
        frame = frame.with_columns(exprs)
        schema = schema.model_copy(update={"roles": new_roles}).with_frequency_encoding(
            FrequencyEncodingSpec(frequencies=frequencies)
        )
        return frame, schema

    def _apply_frequency(self, frame: pl.DataFrame, schema: FeatureSchema) -> pl.DataFrame:
        spec = schema.frequency_encoding
        if spec is None:
            return frame
        exprs: list[pl.Series] = []
        for col, freq_map in spec.frequencies.items():
            if col not in frame.columns:
                raise SchemaValidationError(f"frequency source {col!r} missing at inference")
            out = freq_output_name(col)
            if out in frame.columns:
                raise SchemaValidationError(
                    f"frequency output {out!r} collides with an existing column"
                )
            codes = self._codes(frame, col, schema.categories[col])
            values = np.array([freq_map.get(str(c), 0.0) for c in codes], dtype=np.float64)
            exprs.append(pl.Series(out, values))
        return frame.with_columns(exprs)

    def _fit_target_encoding(
        self,
        frame: pl.DataFrame,
        schema: FeatureSchema,
        source_categorical: list[str],
        target_col: str | None,
    ) -> tuple[pl.DataFrame, FeatureSchema]:
        if (
            self._fe is None
            or not self._fe.target_encoding
            or not source_categorical
            or target_col is None
        ):
            return frame, schema
        k = self._fe.te_smoothing
        y_raw = frame[target_col].to_numpy()
        classes = np.unique(y_raw[~pd.isna(y_raw)])
        # binary positive indicator (facade gates TE to binary classification, ADR-0041 §4)
        positive = resolve_positive(self._task, classes) if self._task is not None else classes[-1]
        y_te = (y_raw == positive).astype(np.float64)
        global_mean = float(y_te.mean())
        exprs: list[pl.Series] = []
        new_roles = dict(schema.roles)
        encodings: dict[str, dict[str, float]] = {}
        declared_groups = _GROUP_COL in frame.columns
        group_structured: list[tuple[str, float]] = []
        for col in source_categorical:
            table = schema.categories[col]
            codes = self._codes(frame, col, table)
            count = np.bincount(codes, minlength=table.cardinality).astype(np.float64)
            sum_y = np.bincount(codes, weights=y_te, minlength=table.cardinality)
            smoothed = (sum_y + k * global_mean) / (count + k)
            full = np.full(table.cardinality, global_mean)
            real = np.arange(table.null_code)  # null/unknown reserves -> global_mean (ADR-0041 §2)
            seen = real[count[real] > 0]
            full[seen] = smoothed[seen]
            exprs.append(pl.Series(te_output_name(col), full[codes]))
            new_roles[te_output_name(col)] = ColumnRole.NUMERIC
            encodings[col] = {str(c): float(full[c]) for c in seen}
            if not declared_groups:
                real_counts = count[: table.null_code]
                n_real = float(real_counts.sum())
                if n_real > 0:
                    rate = float((real_counts > 0).sum()) / n_real
                    dup_share = float(real_counts[real_counts > 1].sum()) / n_real
                    if rate > self._TE_GROUP_RATE and dup_share > self._TE_GROUP_DUP_SHARE:
                        group_structured.append((col, dup_share))
        if group_structured:
            self._warn_te_group_structure(group_structured)
        frame = frame.with_columns(exprs)
        schema = schema.model_copy(update={"roles": new_roles}).with_target_encoding(
            TargetEncodingSpec(encodings=encodings, global_mean=global_mean, smoothing=k)
        )
        return frame, schema

    def _warn_te_group_structure(self, flagged: list[tuple[str, float]]) -> None:
        """Warn that a target-encoding source looks group-structured (finding #11).

        Diagnostic only. A high-cardinality categorical whose values repeat across rows acts like an
        undeclared group: if rows sharing a value also share the target, a row-wise outer holdout/CV is
        not independent and the encoder carries a group's outcome into its relatives — the honest score
        then over-promises. Surfaced at FIT time so the user can pass ``groups=`` before shipping.
        """
        detail = ", ".join(f"{col} ({share:.0%} of rows share a value)" for col, share in flagged)
        _logger.warning(
            "target-encoding source(s) %s have group-like structure; if rows sharing a value also "
            "share the target, a row-wise holdout/CV is not independent and the honest score may "
            "over-promise — pass groups= for a group-aware split (finding #11)",
            detail,
        )

    def _apply_target_encoding(self, frame: pl.DataFrame, schema: FeatureSchema) -> pl.DataFrame:
        spec = schema.target_encoding
        if spec is None:
            return frame
        exprs: list[pl.Series] = []
        for col, te_map in spec.encodings.items():
            if col not in frame.columns:
                raise SchemaValidationError(f"target-encoding source {col!r} missing at inference")
            out = te_output_name(col)
            if out in frame.columns:
                raise SchemaValidationError(
                    f"target-encoding output {out!r} collides with an existing column"
                )
            codes = self._codes(frame, col, schema.categories[col])
            values = np.array(
                [te_map.get(str(c), spec.global_mean) for c in codes], dtype=np.float64
            )
            exprs.append(pl.Series(out, values))
        return frame.with_columns(exprs)
