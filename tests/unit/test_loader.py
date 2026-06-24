"""F1.3/F1.9: load_table reads parquet/csv/folder; failures are domain errors (ADR-0014)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from honestml.adapters import Reader, load_table
from honestml.core import InputError, SchemaValidationError, Task

pytestmark = pytest.mark.unit


def _frame() -> pl.DataFrame:
    return pl.DataFrame(
        {"cat": ["a", "b", "a", "c"], "lc_int": [0, 1, 0, 1], "num": [1.0, 2.0, 3.0, 4.0]}
    )


def test_load_parquet(tmp_path: Path) -> None:
    p = tmp_path / "t.parquet"
    _frame().write_parquet(p)
    assert load_table(p).equals(_frame())


def test_load_csv(tmp_path: Path) -> None:
    p = tmp_path / "t.csv"
    _frame().write_csv(p)
    out = load_table(p)
    assert out.columns == _frame().columns
    assert out.height == 4


def test_load_folder_concatenates(tmp_path: Path) -> None:
    _frame().write_parquet(tmp_path / "a.parquet")
    _frame().write_parquet(tmp_path / "b.parquet")
    assert load_table(tmp_path).height == 8


def test_missing_path_raises_input_error(tmp_path: Path) -> None:
    with pytest.raises(InputError, match="does not exist"):
        load_table(tmp_path / "nope.parquet")


def test_unsupported_format_raises_input_error(tmp_path: Path) -> None:
    p = tmp_path / "t.txt"
    p.write_text("x")
    with pytest.raises(InputError, match="unsupported file format"):
        load_table(p)


def test_empty_folder_raises_input_error(tmp_path: Path) -> None:
    with pytest.raises(InputError, match="no .parquet/.csv"):
        load_table(tmp_path)


def test_folder_schema_mismatch_raises_schema_error(tmp_path: Path) -> None:
    _frame().write_parquet(tmp_path / "a.parquet")
    pl.DataFrame({"other": [1, 2]}).write_parquet(tmp_path / "b.parquet")
    with pytest.raises(SchemaValidationError, match="schema mismatch"):
        load_table(tmp_path)


def test_csv_inference_codes_match_when_dtypes_agree(tmp_path: Path) -> None:
    # ADR-0014: when polars infers the same dtypes from parquet and csv (clean int/str
    # data), train(parquet) and inference(csv) produce identical categorical codes.
    pq, csv = tmp_path / "t.parquet", tmp_path / "t.csv"
    _frame().write_parquet(pq)
    _frame().write_csv(csv)
    y = np.array([0, 1, 0, 1])
    ds_train = Reader(Task(kind="binary")).read(load_table(pq), y=y)
    ds_infer = Reader().read(load_table(csv), schema=ds_train.schema)
    np.testing.assert_array_equal(ds_train.categorical_codes(), ds_infer.categorical_codes())


def test_csv_int_float_dtype_drift_known_limitation(tmp_path: Path) -> None:
    # ADR-0017: source-dtype coercion now makes csv Float64 read match train parquet Int64.
    # The int column written as floats is read as Float64 from csv while the train parquet
    # kept Int64 → without coercion codes would diverge; with it they stay identical.
    pq, csv = tmp_path / "train.parquet", tmp_path / "infer.csv"
    _frame().write_parquet(pq)  # lc_int: Int64
    pl.DataFrame(
        {"cat": ["a", "b", "a", "c"], "lc_int": [0.0, 1.0, 0.0, 1.0], "num": [1.0, 2.0, 3.0, 4.0]}
    ).write_csv(csv)  # lc_int read back as Float64
    y = np.array([0, 1, 0, 1])
    ds_train = Reader(Task(kind="binary")).read(load_table(pq), y=y)
    ds_infer = Reader().read(load_table(csv), schema=ds_train.schema)
    np.testing.assert_array_equal(ds_train.categorical_codes(), ds_infer.categorical_codes())
