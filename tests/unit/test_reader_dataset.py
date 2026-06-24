"""M1a: Reader builds a Dataset from pandas/polars/numpy; codes are train↔inference stable."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import polars as pl
import pytest

from honestml.adapters import PolarsDataset, Reader
from honestml.core import ColumnRole, Dataset, Task

pytestmark = pytest.mark.unit


def _train_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "num": [1.0, 2.0, 3.0, 4.0],
            "cat": ["a", "b", "a", "c"],
            "low_card_int": [0, 1, 0, 1],
        }
    )


def test_reads_polars() -> None:
    ds = Reader(Task(kind="binary")).read(_train_frame(), y=np.array([0, 1, 0, 1]))
    assert isinstance(ds, Dataset)
    assert isinstance(ds, PolarsDataset)
    assert ds.n_rows == 4
    assert ds.schema.roles["num"] == ColumnRole.NUMERIC
    assert ds.schema.roles["cat"] == ColumnRole.CATEGORICAL
    assert ds.schema.roles["low_card_int"] == ColumnRole.CATEGORICAL  # auto-typed


def test_reads_pandas() -> None:
    pdf = pd.DataFrame(
        {"num": [1.0, 2.0, 3.0, 4.0], "cat": ["a", "b", "a", "c"], "low_card_int": [0, 1, 0, 1]}
    )
    ds = Reader(Task(kind="binary")).read(pdf, y=[0, 1, 0, 1])
    assert ds.n_rows == 4
    np.testing.assert_array_equal(ds.target(), np.array([0, 1, 0, 1]))


def test_reads_numpy_with_feature_names() -> None:
    X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    ds = Reader(Task(kind="regression")).read(X, feature_names=["a", "b"])
    assert ds.columns == ["a", "b"]
    assert ds.to_numpy().shape == (3, 2)


def test_to_numpy_and_categorical_codes_blocks() -> None:
    ds = Reader(Task(kind="binary")).read(_train_frame(), y=np.array([0, 1, 0, 1]))
    # numeric block: only "num" (low_card_int auto-typed to categorical)
    assert ds.to_numpy().shape == (4, 1)
    np.testing.assert_array_equal(ds.to_numpy().ravel(), [1.0, 2.0, 3.0, 4.0])
    # categorical block: cat + low_card_int
    codes = ds.categorical_codes()
    assert codes.shape == (4, 2)
    # "cat": a,b,a,c -> 0,1,0,2
    np.testing.assert_array_equal(codes[:, 0], [0, 1, 0, 2])


def test_sample_weight_passthrough() -> None:
    ds = Reader(Task(kind="binary")).read(
        _train_frame(), y=np.array([0, 1, 0, 1]), sample_weight=np.array([1.0, 2, 1, 1])
    )
    np.testing.assert_array_equal(ds.sample_weight(), [1.0, 2.0, 1.0, 1.0])


def test_groups_kwarg_assigns_group_role_row_aligned() -> None:
    # FR-4b: public groups= declares a GROUP column (mirrors sample_weight); not a feature
    g = np.array([10, 10, 20, 20])
    ds = Reader(Task(kind="binary")).read(_train_frame(), y=np.array([0, 1, 0, 1]), groups=g)
    assert ds.schema.group == "__group__"
    assert ds.schema.roles["__group__"] == ColumnRole.GROUP
    assert "__group__" not in ds.schema.features
    np.testing.assert_array_equal(ds.groups(), g)


def test_select_and_take() -> None:
    ds = Reader(Task(kind="binary")).read(_train_frame(), y=np.array([0, 1, 0, 1]))
    sub = ds.select(["num"])
    assert sub.schema.numeric == ["num"] and sub.schema.categorical == []
    rows = ds.take([0, 2])
    assert rows.n_rows == 2
    np.testing.assert_array_equal(rows.to_numpy().ravel(), [1.0, 3.0])


# --- F1.7: auto-typing thresholds in Task + observable typing_report (ADR-0015) ---


def test_typing_report_records_only_reclassifications() -> None:
    reader = Reader(Task(kind="binary"))
    reader.read(_train_frame(), y=np.array([0, 1, 0, 1]))
    reasons = {d.column: d.reason for d in reader.typing_report}
    # only the low-cardinality int is reported; native float/string are baseline
    assert reasons == {"low_card_int": "low_cardinality_int"}


def test_typing_report_id_like_ignored() -> None:
    n = 200
    frame = pl.DataFrame({"id": list(range(n)), "x": [0.0] * n})
    reader = Reader(Task(kind="binary"))
    ds = reader.read(frame, y=np.array([0, 1] * (n // 2)))
    assert ds.schema.roles["id"] == ColumnRole.IGNORE
    rec = [d for d in reader.typing_report if d.column == "id"]
    assert rec and rec[0].reason == "numeric_id_like" and rec[0].assigned_role == ColumnRole.IGNORE


def test_numeric_id_min_unique_threshold_configurable() -> None:
    n = 50  # 50 unique ints, rate=1.0; >cat_max(20) so not categorical
    frame = pl.DataFrame({"k": list(range(n)), "x": [0.0] * n})
    y = np.array([0, 1] * (n // 2))
    default_ds = Reader(Task(kind="binary")).read(frame, y=y)
    assert default_ds.schema.roles["k"] == ColumnRole.NUMERIC  # 50 <= default 100 -> not id
    low_ds = Reader(Task(kind="binary", numeric_id_min_unique=40)).read(frame, y=y)
    assert low_ds.schema.roles["k"] == ColumnRole.IGNORE  # 50 > 40 -> id-like


def test_typing_report_empty_on_inference_branch() -> None:
    reader = Reader(Task(kind="binary"))
    schema = reader.read(_train_frame(), y=np.array([0, 1, 0, 1])).schema
    reader2 = Reader()
    reader2.read(_train_frame(), schema=schema)
    assert reader2.typing_report == []


def test_typing_report_reset_between_reads() -> None:
    reader = Reader(Task(kind="binary"))
    reader.read(_train_frame(), y=np.array([0, 1, 0, 1]))
    assert reader.typing_report  # low_card_int recorded
    plain = pl.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [5.0, 6.0, 7.0, 8.0]})
    reader.read(plain, y=np.array([0, 1, 0, 1]))
    assert reader.typing_report == []


def test_typing_report_info_logged(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="honestml"):
        Reader(Task(kind="binary")).read(_train_frame(), y=np.array([0, 1, 0, 1]))
    assert any("auto-typing" in r.getMessage() for r in caplog.records)


# --- finding #8a: all-null / constant columns -> role IGNORE (ADR-0015 ext) ---


def test_all_nan_numeric_column_dropped_as_ignore() -> None:
    # the #6 cascade: an all-NaN numeric column must be dropped (IGNORE), not kept NUMERIC where it would
    # trip the "NaN in numeric features" gate and evict baseline/linear from the zoo.
    pdf = pd.DataFrame({"num": [1.0, 2.0, 3.0, 4.0], "allnan": [np.nan, np.nan, np.nan, np.nan]})
    reader = Reader(Task(kind="binary"))
    ds = reader.read(pdf, y=np.array([0, 1, 0, 1]))
    assert ds.schema.roles["allnan"] == ColumnRole.IGNORE
    assert "allnan" not in ds.schema.features
    assert any(d.column == "allnan" and d.reason == "all_null" for d in reader.typing_report)


def test_constant_column_dropped_as_ignore() -> None:
    frame = pl.DataFrame({"num": [1.0, 2.0, 3.0, 4.0], "const": [7, 7, 7, 7]})
    reader = Reader(Task(kind="binary"))
    ds = reader.read(frame, y=np.array([0, 1, 0, 1]))
    assert ds.schema.roles["const"] == ColumnRole.IGNORE  # constant wins over low_cardinality_int
    assert any(d.column == "const" and d.reason == "constant" for d in reader.typing_report)


def test_all_null_string_column_dropped_as_ignore() -> None:
    frame = pl.DataFrame(
        {"num": [1.0, 2.0], "blank": pl.Series("blank", [None, None], dtype=pl.Utf8)}
    )
    reader = Reader(Task(kind="binary"))
    ds = reader.read(frame, y=np.array([0, 1]))
    assert ds.schema.roles["blank"] == ColumnRole.IGNORE
    assert any(d.column == "blank" and d.reason == "all_null" for d in reader.typing_report)


# --- finding #7: near-unique string columns -> string_id_like / high_cardinality_string ---


def test_near_unique_string_dropped_as_id() -> None:
    n = 150
    frame = pl.DataFrame(
        {"num": [float(i) for i in range(n)], "name": [f"id_{i}" for i in range(n)]}
    )
    reader = Reader(Task(kind="binary"))
    ds = reader.read(frame, y=np.array([0, 1] * (n // 2)))
    assert ds.schema.roles["name"] == ColumnRole.IGNORE  # 100% unique, > floor -> pure-noise id
    assert "name" not in ds.schema.features
    assert any(d.column == "name" and d.reason == "string_id_like" for d in reader.typing_report)


def test_high_cardinality_string_kept_but_flagged() -> None:
    n = 300
    tickets = [
        f"t{i}" for i in range(n // 2)
    ] * 2  # 150 distinct, each twice -> rate 0.5, count > floor
    frame = pl.DataFrame({"num": [float(i) for i in range(n)], "ticket": tickets})
    reader = Reader(Task(kind="binary"))
    ds = reader.read(frame, y=np.array([0, 1] * (n // 2)))
    assert ds.schema.roles["ticket"] == ColumnRole.CATEGORICAL  # kept, not dropped
    assert "ticket" in ds.schema.features
    assert any(
        d.column == "ticket" and d.reason == "high_cardinality_string" for d in reader.typing_report
    )


def test_low_cardinality_string_unflagged() -> None:
    # the common case: a few categories -> plain categorical, no typing-report noise (default preserved)
    reader = Reader(Task(kind="binary"))
    reader.read(_train_frame(), y=np.array([0, 1, 0, 1]))
    assert not any(d.column == "cat" for d in reader.typing_report)


def test_string_id_min_unique_configurable() -> None:
    frame = pl.DataFrame(
        {"num": [float(i) for i in range(40)], "code": [f"c{i}" for i in range(40)]}
    )
    ds = Reader(Task(kind="binary", string_id_min_unique=20)).read(frame, y=np.array([0, 1] * 20))
    assert (
        ds.schema.roles["code"] == ColumnRole.IGNORE
    )  # 40 distinct > 20 floor, rate 1.0 -> id-like


# --- FR-F: source-dtype token + value-preserving coercion (ADR-0017) ---


def test_fit_records_source_dtype_token() -> None:
    # FR-1: each fitted category table carries the train column's canonical dtype token.
    ds = Reader(Task(kind="binary")).read(_train_frame(), y=np.array([0, 1, 0, 1]))
    cats = ds.schema.categories
    assert cats["low_card_int"].source_dtype == "int64"
    assert cats["cat"].source_dtype == "string"


def test_fractional_against_integer_train_maps_to_unknown() -> None:
    # FR-3: a value not representable in the train integer dtype falls to unknown_code,
    # never to a wrong known code (1.5 must not become the code of category "1").
    train = pl.DataFrame({"k": [0, 1, 2, 0], "n": [1.0, 2.0, 3.0, 4.0]})
    schema = Reader(Task(kind="binary")).read(train, y=np.array([0, 1, 0, 1])).schema
    table = schema.categories["k"]
    infer = pl.DataFrame({"k": [1.5, 1.0], "n": [0.0, 0.0]})
    codes = (
        Reader().read(infer, schema=schema).categorical_codes()[:, schema.categorical.index("k")]
    )
    assert list(codes) == [table.unknown_code, table.categories.index("1")]


def test_int_float_read_drift_not_flagged_as_unseen(caplog: pytest.LogCaptureFixture) -> None:
    # C5 regression: a categorical fitted on Int64 then read as Float64 at inference (csv vs parquet
    # drift) carries the SAME values; the unseen-share warning must coerce like the encoder (ADR-0017)
    # and stay silent — a naive cast(Utf8) read "1.0" != train "1" and falsely warned ~100% unseen.
    train = pl.DataFrame({"k": [0, 1, 2, 0, 1, 2], "n": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
    schema = Reader(Task(kind="binary")).read(train, y=np.array([0, 1, 0, 1, 0, 1])).schema
    infer = pl.DataFrame({"k": [1.0, 2.0, 0.0], "n": [0.0, 0.0, 0.0]})
    with caplog.at_level(logging.WARNING):
        Reader().read(infer, schema=schema)
    assert not any("unseen at train" in r.message for r in caplog.records)


def test_codes_consistent_train_to_inference() -> None:
    reader = Reader(Task(kind="binary"))
    ds_train = reader.read(_train_frame(), y=np.array([0, 1, 0, 1]))
    schema = ds_train.schema

    infer = pl.DataFrame(
        {"num": [9.0, 9.0, 9.0], "cat": ["a", "z", None], "low_card_int": [0, 0, 0]}
    )
    ds_infer = Reader().read(infer, schema=schema)
    cat_table = schema.categories["cat"]
    codes = ds_infer.categorical_codes()[:, 0]
    # a -> same code as train (0); unseen "z" -> unknown_code; null -> null_code
    assert list(codes) == [0, cat_table.unknown_code, cat_table.null_code]
