"""M1a: input validation at the boundary -> SchemaValidationError (G5/NFR-13)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

from honestml.adapters import Reader
from honestml.core import (
    ColumnRole,
    ConfigError,
    FeatureSchema,
    SchemaValidationError,
    Task,
)

pytestmark = pytest.mark.unit


def test_empty_rows_rejected() -> None:
    with pytest.raises(SchemaValidationError, match="no rows"):
        Reader().read(pl.DataFrame({"a": []}))


def test_xy_length_mismatch_rejected() -> None:
    with pytest.raises(SchemaValidationError, match="length"):
        Reader(Task(kind="binary")).read(pl.DataFrame({"a": [1, 2, 3]}), y=[0, 1])


def test_duplicate_column_names_rejected_pandas() -> None:
    """F105: duplicate columns (e.g. after concat/merge) fail with a domain error, not a raw ValueError."""
    df = pd.DataFrame([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], columns=["a", "a", "b"])
    with pytest.raises(SchemaValidationError, match="duplicate column names"):
        Reader(Task(kind="binary")).read(df, y=[0, 1])


def test_duplicate_feature_names_rejected_numpy() -> None:
    """F105: duplicate ``feature_names`` for a numpy input fail at the boundary."""
    with pytest.raises(SchemaValidationError, match="duplicate column names"):
        Reader(Task(kind="binary")).read(
            np.array([[1.0, 2.0], [3.0, 4.0]]), y=[0, 1], feature_names=["a", "a"]
        )


def test_pandas_string_reads_without_pyarrow(monkeypatch: pytest.MonkeyPatch) -> None:
    # pandas 3.0 defaults strings to an extension dtype pl.from_pandas converts only via pyarrow;
    # the reader falls back to a column-wise conversion so pyarrow stays optional (ADR-0005).
    def _boom(_frame: object) -> object:
        raise ImportError("pyarrow is required for converting a pandas dataframe to Polars")

    monkeypatch.setattr(pl, "from_pandas", _boom)
    ds = Reader(Task(kind="binary")).read(
        pd.DataFrame({"s": ["a", "b", "a", "b"], "n": [1.0, 2.0, 3.0, 4.0]}), y=[0, 1, 0, 1]
    )
    assert ds.n_rows == 4


def test_all_null_target_rejected() -> None:
    with pytest.raises(SchemaValidationError, match="all-null"):
        Reader(Task(kind="binary")).read(
            pl.DataFrame({"a": [1.0, 2.0]}), y=np.array([np.nan, np.nan])
        )


def test_binary_target_with_three_classes_rejected() -> None:
    with pytest.raises(SchemaValidationError, match="binary"):
        Reader(Task(kind="binary")).read(
            pl.DataFrame({"a": [1.0, 2.0, 3.0]}), y=np.array([0, 1, 2])
        )


def test_multiclass_single_class_rejected() -> None:
    with pytest.raises(SchemaValidationError, match="multiclass"):
        Reader(Task(kind="multiclass")).read(
            pl.DataFrame({"a": [1.0, 2.0, 3.0]}), y=np.array([7, 7, 7])
        )


def test_multiclass_three_classes_ok() -> None:
    ds = Reader(Task(kind="multiclass")).read(
        pl.DataFrame({"a": [1.0, 2.0, 3.0]}), y=np.array([0, 1, 2])
    )
    assert ds.n_rows == 3


def test_regression_target_non_numeric_rejected() -> None:
    with pytest.raises(SchemaValidationError, match="regression"):
        Reader(Task(kind="regression")).read(
            pl.DataFrame({"a": [1.0, 2.0]}), y=np.array(["x", "y"])
        )


def test_missing_required_column_at_inference_rejected() -> None:
    schema = FeatureSchema(roles={"a": ColumnRole.NUMERIC, "b": ColumnRole.NUMERIC})
    with pytest.raises(SchemaValidationError, match="missing required columns"):
        Reader().read(pl.DataFrame({"a": [1.0, 2.0]}), schema=schema)


def test_missing_empty_named_required_column_rejected() -> None:
    # F011: an empty-string column name is valid in pandas (pd.DataFrame({'': [...]})); the schema may
    # carry it as a feature, so a missing '' column at inference must still be reported, not skipped.
    schema = FeatureSchema(roles={"": ColumnRole.NUMERIC, "b": ColumnRole.NUMERIC})
    with pytest.raises(SchemaValidationError, match="missing required columns"):
        Reader().read(pl.DataFrame({"b": [1.0, 2.0]}), schema=schema)


def test_unsupported_x_type_rejected() -> None:
    with pytest.raises(SchemaValidationError, match="unsupported X type"):
        Reader().read([[1, 2], [3, 4]])  # type: ignore[arg-type]


def test_1d_numpy_rejected() -> None:
    with pytest.raises(SchemaValidationError, match="2-D"):
        Reader().read(np.array([1.0, 2.0, 3.0]))


def test_numpy_feature_names_mismatch_rejected() -> None:
    with pytest.raises(SchemaValidationError, match="feature_names length"):
        Reader().read(np.array([[1.0, 2.0]]), feature_names=["only_one"])


def test_groups_length_mismatch_rejected() -> None:
    with pytest.raises(SchemaValidationError, match="length"):
        Reader(Task(kind="binary")).read(
            pl.DataFrame({"a": [1.0, 2.0, 3.0]}), y=np.array([0, 1, 0]), groups=np.array([1, 2])
        )


@pytest.mark.parametrize(
    "groups",
    [
        np.array([1.0, np.nan, 2.0]),  # plain float NaN
        pd.array([1, pd.NA, 2], dtype="Int64"),  # pandas-nullable pd.NA (ADR-0025 §4, R-5)
    ],
)
def test_groups_null_including_pandas_na_rejected(groups: object) -> None:
    # a null/NaN group must fail at the boundary, never silently break group anti-leakage
    with pytest.raises(SchemaValidationError, match="null/NaN"):
        Reader(Task(kind="binary")).read(
            pl.DataFrame({"a": [1.0, 2.0, 3.0]}), y=np.array([0, 1, 0]), groups=groups
        )


# --- M4b: public time axis (ADR-0028) ---------------------------------------


def _xy():
    return pl.DataFrame({"a": [1.0, 2.0, 3.0]}), np.array([0, 1, 0])


def test_time_kwarg_assigns_time_role() -> None:
    X, y = _xy()
    ds = Reader(Task(kind="binary")).read(X, y, time=np.array([10, 20, 30]))
    assert ds.schema.time == "__time__"
    assert "__time__" not in ds.schema.features  # the time axis is not a model feature
    assert ds.time().tolist() == [10, 20, 30]  # row-aligned


def test_time_null_rejected() -> None:
    X, y = _xy()
    with pytest.raises(SchemaValidationError, match="time axis must be complete"):
        Reader(Task(kind="binary")).read(X, y, time=np.array([1.0, np.nan, 3.0]))


def test_time_non_sortable_rejected() -> None:
    X, y = _xy()
    with pytest.raises(SchemaValidationError, match="sortable"):
        Reader(Task(kind="binary")).read(X, y, time=np.array(["x", "y", "z"]))


def test_label_time_without_time_raises() -> None:
    X, y = _xy()
    with pytest.raises(ConfigError, match="label_time requires time"):
        Reader(Task(kind="binary")).read(X, y, label_time=np.array([1, 2, 3]))


def test_label_time_attached_name_based() -> None:
    X, y = _xy()
    ds = Reader(Task(kind="binary")).read(
        X, y, time=np.array([1, 2, 3]), label_time=np.array([2, 3, 4])
    )
    assert ds.label_time().tolist() == [2, 3, 4]
    assert "__label_time__" not in ds.schema.roles  # name-based metadata, not a role


def test_datetime_time_axis_accepted() -> None:
    X, y = _xy()
    times = pd.to_datetime(["2021-01-03", "2021-01-01", "2021-01-02"]).to_numpy()
    ds = Reader(Task(kind="binary")).read(X, y, time=times)
    assert ds.schema.time == "__time__" and ds.time().shape == (3,)


def test_tz_aware_time_axis_rejected() -> None:
    # F106: a tz-aware time axis arrives as polars Object; reject it at the boundary with a message that
    # names tz-awareness as the cause (not a bare "got dtype Object").
    X, y = _xy()
    times = pd.to_datetime(["2021-01-01", "2021-01-02", "2021-01-03"], utc=True)  # tz-aware
    with pytest.raises(SchemaValidationError, match="tz-aware"):
        Reader(Task(kind="binary")).read(X, y, time=times)


def test_label_time_non_sortable_rejected() -> None:
    # F003-adjacent: label_time (t1) is compared against time in the de Prado purge; a free-text t1 would
    # break that comparison deep in the splitter, so it must be rejected at the boundary like time itself.
    X, y = _xy()
    with pytest.raises(SchemaValidationError, match="label_time must be sortable"):
        Reader(Task(kind="binary")).read(
            X, y, time=np.array([1, 2, 3]), label_time=np.array(["x", "y", "z"])
        )


def test_label_time_dtype_incompatible_with_time_rejected() -> None:
    # F003-adjacent: a numeric t1 against a datetime time axis cannot be ordered together (t1 < time
    # raises/misbehaves in the purge); the kinds must match at the boundary, not crash in the carve.
    X, y = _xy()
    times = pd.to_datetime(["2021-01-01", "2021-01-02", "2021-01-03"]).to_numpy()
    with pytest.raises(SchemaValidationError, match="not comparable with time"):
        Reader(Task(kind="binary")).read(X, y, time=times, label_time=np.array([1, 2, 3]))
