"""M6a-3: boundary feature engineering in the Reader (ADR-0018/0042) — fit + train==inference."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import polars as pl
import pytest

from honestml.adapters import Reader
from honestml.application import design_matrix
from honestml.core import FEConfig, SchemaValidationError, Task
from honestml.core.schema import intersection_output_name

pytestmark = pytest.mark.unit


def _binary_frame() -> tuple[pd.DataFrame, list[int]]:
    df = pd.DataFrame(
        {
            "num": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "cat": ["a", "a", "b", "b", "a", "b"],
        }
    )
    y = [0, 1, 0, 1, 1, 0]
    return df, y


# -- datetime deltas (ADR-0018) --------------------------------------------


def test_datetime_delta_autodetect_and_in_design_matrix() -> None:
    df = pd.DataFrame(
        {
            "event_dt": pd.to_datetime(["2021-01-01", "2021-01-05"]),
            "report_dt": pd.to_datetime(["2021-01-11", "2021-01-11"]),
            "num": [1.0, 2.0],
        }
    )
    ds = Reader(Task(kind="binary")).read(df, [0, 1])
    spec = ds.schema.datetime_spec
    assert spec is not None and spec.report_date == "report_dt"
    assert spec.deltas == (("event_dt", "event_dt__days_to_report"),)
    assert "event_dt__days_to_report" in ds.schema.features
    assert "report_dt" not in ds.schema.features and "event_dt" not in ds.schema.features
    col = ds.schema.features.index("event_dt__days_to_report")
    assert list(design_matrix(ds)[:, col]) == [10.0, 6.0]


def test_datetime_date_equals_datetime_input() -> None:
    # R-M2: Date-normalization makes a Date column and a Datetime-with-time column give the same delta
    as_date = pl.DataFrame(
        {
            "event_dt": pl.Series([dt.date(2021, 1, 1)]),
            "report_dt": pl.Series([dt.date(2021, 1, 4)]),
        }
    )
    as_dt = pl.DataFrame(
        {
            "event_dt": pl.Series([dt.datetime(2021, 1, 1, 12, 0)]),
            "report_dt": pl.Series([dt.datetime(2021, 1, 4, 9, 0)]),
        }
    )
    d1 = Reader(Task(kind="binary")).read(as_date, [0])
    d2 = Reader(Task(kind="binary")).read(as_dt, [0])
    i = d1.schema.features.index("event_dt__days_to_report")
    assert design_matrix(d1)[0, i] == design_matrix(d2)[0, i] == 3.0


def test_datetime_explicit_override_missing_raises() -> None:
    df = pd.DataFrame({"num": [1.0, 2.0], "d": pd.to_datetime(["2021-01-01", "2021-01-02"])})
    with pytest.raises(SchemaValidationError, match="report_date"):
        Reader(Task(kind="binary", report_date="absent")).read(df, [0, 1])


def test_datetime_no_report_date_warns_and_drops(caplog: pytest.LogCaptureFixture) -> None:
    df = pd.DataFrame({"num": [1.0, 2.0], "some_dt": pd.to_datetime(["2021-01-01", "2021-01-02"])})
    with caplog.at_level("WARNING"):
        ds = Reader(Task(kind="binary")).read(df, [0, 1])
    assert ds.schema.datetime_spec is None
    assert "some_dt" not in ds.schema.features
    assert any("report date" in r.message for r in caplog.records)


def test_datetime_inference_missing_source_raises() -> None:
    df = pd.DataFrame(
        {"event_dt": pd.to_datetime(["2021-01-01"]), "report_dt": pd.to_datetime(["2021-01-03"])}
    )
    ds = Reader(Task(kind="binary")).read(df, [0])
    bad = pd.DataFrame({"report_dt": pd.to_datetime(["2021-01-03"])})  # event_dt missing
    with pytest.raises(SchemaValidationError, match="event_dt"):
        Reader(Task(kind="binary")).read(bad, schema=ds.schema)


# -- frequency encoding -----------------------------------------------------


def test_frequency_in_design_matrix_and_unseen_zero() -> None:
    df, y = _binary_frame()
    fe = FEConfig(frequency_encoding=True)
    ds = Reader(Task(kind="binary"), fe=fe).read(df, y)
    assert "cat_freq" in ds.schema.features
    i = ds.schema.features.index("cat_freq")
    freq = design_matrix(ds)[:, i]
    assert freq[0] == pytest.approx(3 / 6) and freq[2] == pytest.approx(3 / 6)  # a:3, b:3
    # inference with an unseen category -> 0.0
    infer = pd.DataFrame({"num": [9.0], "cat": ["zzz"]})
    ds2 = Reader(Task(kind="binary")).read(infer, schema=ds.schema)
    assert design_matrix(ds2)[0, i] == 0.0


# -- intersections ----------------------------------------------------------


def test_intersections_deterministic_and_truncated(caplog: pytest.LogCaptureFixture) -> None:
    df = pd.DataFrame({"c3": ["x", "y"], "c1": ["a", "b"], "c2": ["p", "q"]})
    fe = FEConfig(intersections=True, max_pairs=2)
    with caplog.at_level("WARNING"):
        ds = Reader(Task(kind="binary"), fe=fe).read(df, [0, 1])
    # combinations(sorted([c1,c2,c3]), 2)[:2] = (c1,c2),(c1,c3)
    assert ds.schema.intersections is not None
    assert ds.schema.intersections.pairs == (("c1", "c2"), ("c1", "c3"))
    assert "c1__c2" in ds.schema.features and "c1__c3" in ds.schema.features
    assert "c2__c3" not in ds.schema.features
    assert any("truncated" in r.message for r in caplog.records)


def test_intersections_under_two_categoricals_is_noop() -> None:
    df = pd.DataFrame({"num": [1.0, 2.0], "cat": ["a", "b"]})
    ds = Reader(Task(kind="binary"), fe=FEConfig(intersections=True)).read(df, [0, 1])
    assert ds.schema.intersections is None


def test_intersection_output_collision_raises() -> None:
    df = pd.DataFrame({"c1": ["a", "b"], "c2": ["p", "q"], "c1__c2": ["z", "z"]})
    with pytest.raises(SchemaValidationError, match="collides"):
        Reader(Task(kind="binary"), fe=FEConfig(intersections=True)).read(df, [0, 1])


# -- target encoding (full-train spec) --------------------------------------


def test_target_encoding_full_train_spec_and_train_eq_inference() -> None:
    df, y = _binary_frame()
    fe = FEConfig(target_encoding=True, te_smoothing=1.0)
    ds = Reader(Task(kind="binary"), fe=fe).read(df, y)
    spec = ds.schema.target_encoding
    assert spec is not None and "cat" in spec.encodings
    # P(y=1): a -> {0,1,1}=2/3, b -> {0,1,0}=1/3; global=1/2; smoothed with k=1
    gm = 0.5
    a_smoothed = (2 + 1 * gm) / (3 + 1)
    i = ds.schema.features.index("cat_te")
    train_mat = design_matrix(ds)
    assert train_mat[0, i] == pytest.approx(a_smoothed)
    # inference on the same X (schema reused, no y) -> identical TE column (train==inference)
    ds_inf = Reader(Task(kind="binary")).read(df.drop(columns=[]), schema=ds.schema)
    assert np.allclose(design_matrix(ds_inf)[:, i], train_mat[:, i])


def test_target_encoding_unseen_category_gets_global_mean() -> None:
    df, y = _binary_frame()
    ds = Reader(Task(kind="binary"), fe=FEConfig(target_encoding=True)).read(df, y)
    spec = ds.schema.target_encoding
    assert spec is not None
    infer = pd.DataFrame({"num": [9.0], "cat": ["zzz"]})
    ds2 = Reader(Task(kind="binary")).read(infer, schema=ds.schema)
    i = ds.schema.features.index("cat_te")
    assert design_matrix(ds2)[0, i] == pytest.approx(spec.global_mean)


def test_intersection_int_categorical_float_drift_matches_train() -> None:
    # finding #8 (FE-propagation): an int categorical read as float at inference must still produce the
    # SAME intersection category — the combined key is built value-preservingly, not via a raw cast(Utf8).
    train = pl.DataFrame({"bath": [1, 2, 1, 2], "region": ["x", "y", "x", "y"]})
    schema = (
        Reader(Task(kind="binary"), fe=FEConfig(intersections=True))
        .read(train, [0, 1, 0, 1])
        .schema
    )
    out = intersection_output_name("bath", "region")
    table = schema.categories[out]
    infer = pl.DataFrame(
        {"bath": [1.0, 2.0], "region": ["x", "y"]}
    )  # bath drifted int64 -> float64
    ds = Reader().read(infer, schema=schema)
    codes = ds.categorical_codes()[:, schema.categorical.index(out)]
    assert table.unknown_code not in codes.tolist()  # not nulled to unknown by a "1.0__x" miss
    assert codes.tolist() == [table.categories.index("1__x"), table.categories.index("2__y")]


def test_inference_output_name_collision_raises() -> None:
    # FR-FE-6 / ADR-0042 §2: an inference column colliding with an FE output fails loud, like the fit path
    df, y = _binary_frame()
    ds = Reader(Task(kind="binary"), fe=FEConfig(frequency_encoding=True)).read(df, y)
    bad = pd.DataFrame(
        {"num": [1.0], "cat": ["a"], "cat_freq": [9.9]}
    )  # collides with the FE output
    with pytest.raises(SchemaValidationError, match="collides"):
        Reader(Task(kind="binary")).read(bad, schema=ds.schema)


def test_target_encoding_null_category_matches_global_mean_oof_and_inference() -> None:
    # ADR-0041 §2: a null category encodes to global_mean at inference (full-train spec), the same
    # value the OOF reserve-masking produces — no null-bucket divergence between eval and ship.
    df = pl.DataFrame(
        {"num": [1.0, 2.0, 3.0, 4.0], "cat": pl.Series(["a", None, "a", None], dtype=pl.Utf8)}
    )
    y = [1, 0, 1, 0]
    ds = Reader(Task(kind="binary"), fe=FEConfig(target_encoding=True)).read(df, y)
    spec = ds.schema.target_encoding
    assert spec is not None
    infer = pl.DataFrame({"num": [9.0], "cat": pl.Series([None], dtype=pl.Utf8)})  # null category
    ds2 = Reader(Task(kind="binary")).read(infer, schema=ds.schema)
    i = ds.schema.features.index("cat_te")
    assert design_matrix(ds2)[0, i] == pytest.approx(spec.global_mean)


def test_fe_off_default_leaves_schema_like_m5() -> None:
    df, y = _binary_frame()
    ds = Reader(Task(kind="binary")).read(df, y)
    assert ds.schema.target_encoding is None and ds.schema.frequency_encoding is None
    assert ds.schema.intersections is None and ds.schema.datetime_spec is None
    assert ds.schema.features == ["num", "cat"]


# -- F-audit residual closures (F2.7 / F5.8) --------------------------------


def test_intersection_warns_on_reserved_separator_collision(caplog) -> None:
    """F2.7 residual: a real value carrying "__"/"__NA__" can merge distinct pairs in the
    DERIVED intersection category — the fit warns instead of silently merging."""
    df = pd.DataFrame(
        {
            "c1": ["a__b", "a", "x", "a__b", "x", "a"],
            "c2": ["c", "b__c", "y", "c", "y", "b__c"],
        }
    )
    y = [0, 1, 0, 1, 1, 0]
    with caplog.at_level("WARNING", logger="honestml"):
        Reader(Task(kind="binary"), fe=FEConfig(intersections=True)).read(df, y)
    assert any("reserved separator" in r.message for r in caplog.records)
    caplog.clear()
    clean, y2 = _binary_frame()
    with caplog.at_level("WARNING", logger="honestml"):
        Reader(Task(kind="binary"), fe=FEConfig(intersections=True)).read(clean, y2)
    assert not any("reserved separator" in r.message for r in caplog.records)


def test_inference_warns_on_high_unknown_category_share(caplog) -> None:
    """F5.8 follow-up: a high share of train-unseen category values is a drift signal —
    WARNING at the inference boundary (the encoding itself stays honest: unknown_code)."""
    df, y = _binary_frame()
    reader = Reader(Task(kind="binary"))
    schema = reader.read(df, y).schema
    drifted = pd.DataFrame({"num": [1.0, 2.0, 3.0], "cat": ["zz", "qq", "a"]})
    with caplog.at_level("WARNING", logger="honestml"):
        Reader().read(drifted, schema=schema)
    assert any("unseen at train" in r.message for r in caplog.records)
    caplog.clear()
    seen = pd.DataFrame({"num": [1.0], "cat": ["a"]})
    with caplog.at_level("WARNING", logger="honestml"):
        Reader().read(seen, schema=schema)
    assert not any("unseen at train" in r.message for r in caplog.records)


def _group_te_frame(n_pairs: int = 20, n_singletons: int = 20) -> tuple[pd.DataFrame, list[int]]:
    # a high-cardinality categorical that still repeats across rows (group-like, e.g. a ticket id)
    tickets = [f"t{i}" for i in range(n_pairs)] * 2 + [f"u{i}" for i in range(n_singletons)]
    n = len(tickets)
    df = pd.DataFrame({"num": np.arange(n, dtype=float), "ticket": tickets})
    return df, [i % 2 for i in range(n)]


def test_te_group_structure_warns_at_fit(caplog) -> None:
    # finding #11(b): a high-cardinality TE source whose values repeat acts like an undeclared group —
    # warn at FIT time so the user can pass groups= before a row-wise holdout over-promises.
    df, y = _group_te_frame()
    with caplog.at_level("WARNING", logger="honestml"):
        Reader(Task(kind="binary"), fe=FEConfig(target_encoding=True)).read(df, y)
    assert any("group-like structure" in r.message for r in caplog.records)


def test_te_group_structure_silent_for_low_cardinality(caplog) -> None:
    # a low-cardinality real feature (3 cities) feeding TE is NOT a group — no false-positive warning.
    n = 60
    df = pd.DataFrame(
        {"num": np.arange(n, dtype=float), "city": [["x", "y", "z"][i % 3] for i in range(n)]}
    )
    y = [i % 2 for i in range(n)]
    with caplog.at_level("WARNING", logger="honestml"):
        Reader(Task(kind="binary"), fe=FEConfig(target_encoding=True)).read(df, y)
    assert not any("group-like structure" in r.message for r in caplog.records)


def test_te_group_structure_silent_when_groups_declared(caplog) -> None:
    # groups already declared -> the carve/CV is group-aware (finding #11a), so no advisory is needed.
    df, y = _group_te_frame()
    groups = df["ticket"].to_numpy()
    with caplog.at_level("WARNING", logger="honestml"):
        Reader(Task(kind="binary"), fe=FEConfig(target_encoding=True)).read(df, y, groups=groups)
    assert not any("group-like structure" in r.message for r in caplog.records)
