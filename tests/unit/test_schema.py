"""M1a: FeatureSchema + schema-owned CategoryTable (ADR-0005/0006)."""

from __future__ import annotations

import pytest

from honestml.core import (
    CategoryTable,
    ColumnRole,
    DatetimeDeltaSpec,
    FeatureSchema,
    FrequencyEncodingSpec,
    IntersectionSpec,
    TargetEncodingSpec,
    Task,
)
from honestml.core.schema import native_routable, native_routing

pytestmark = pytest.mark.unit


def test_category_table_fit_is_deterministic_and_sorted() -> None:
    a = CategoryTable.fit(["b", "a", "c", "a", None])
    b = CategoryTable.fit(["c", None, "a", "b"])
    assert a.categories == ("a", "b", "c")
    assert a == b  # order-independent, null-excluded


def test_category_table_codes() -> None:
    table = CategoryTable.fit(["a", "b", "c"])
    codes = table.encode(["a", "b", "c", None, "unseen"])
    assert list(codes) == [0, 1, 2, table.null_code, table.unknown_code]
    assert table.null_code == 3
    assert table.unknown_code == 4
    assert table.cardinality == 5


def test_category_table_codes_are_non_negative() -> None:
    table = CategoryTable.fit(["x", "y"])
    codes = table.encode(["x", None, "z"])
    assert (codes >= 0).all()  # LightGBM/CatBoost-friendly


def test_category_table_json_round_trip() -> None:
    table = CategoryTable.fit(["a", "b"])
    restored = CategoryTable.model_validate_json(table.model_dump_json())
    assert restored == table
    assert list(restored.encode(["a", "b", "q"])) == [0, 1, table.unknown_code]


def test_time_role_distinct_from_features_and_datetime() -> None:
    # ADR-0028: TIME is the CV axis, excluded from features, distinct from DATETIME
    schema = FeatureSchema(
        roles={
            "x": ColumnRole.NUMERIC,
            "__time__": ColumnRole.TIME,
            "d": ColumnRole.DATETIME,
        }
    )
    assert schema.time == "__time__"
    assert "__time__" not in schema.features  # time axis is not a model feature
    assert schema.datetime == ["d"] and schema.time != "d"


def test_time_role_serializes() -> None:
    schema = FeatureSchema(roles={"x": ColumnRole.NUMERIC, "__time__": ColumnRole.TIME})
    restored = FeatureSchema.model_validate_json(schema.model_dump_json())
    assert restored.time == "__time__"  # TIME is a valid serialized enum member (NFR-M4-5)


def test_category_table_source_dtype_round_trip() -> None:
    # FR-1: the train dtype token is carried and survives JSON round-trip.
    table = CategoryTable.fit(["0", "1"], source_dtype="int64")
    assert table.source_dtype == "int64"
    restored = CategoryTable.model_validate_json(table.model_dump_json())
    assert restored == table and restored.source_dtype == "int64"


def test_category_table_legacy_json_without_source_dtype_loads() -> None:
    # FR-4 backward-compat: old artifacts have no source_dtype -> None, behaves as before.
    table = CategoryTable.model_validate_json('{"categories": ["a", "b"]}')
    assert table.source_dtype is None
    assert list(table.encode(["a", "b", "q"])) == [0, 1, table.unknown_code]


def test_category_table_ignores_unknown_keys_forward_compat() -> None:
    # FR-4 forward-compat: explicit extra="ignore" -> an old reader drops unknown future keys.
    table = CategoryTable.model_validate_json(
        '{"categories": ["a"], "source_dtype": "int64", "future_field": 7}'
    )
    assert table.categories == ("a",) and table.source_dtype == "int64"


def test_feature_schema_role_views_and_features_order() -> None:
    schema = FeatureSchema(
        roles={
            "n1": ColumnRole.NUMERIC,
            "c1": ColumnRole.CATEGORICAL,
            "n2": ColumnRole.NUMERIC,
            "y": ColumnRole.TARGET,
            "d": ColumnRole.DATETIME,
        }
    )
    assert schema.numeric == ["n1", "n2"]
    assert schema.categorical == ["c1"]
    assert schema.features == ["n1", "n2", "c1"]  # numeric block then categorical
    assert schema.target == "y"
    assert schema.datetime == ["d"]


def test_feature_schema_with_categories_round_trip() -> None:
    schema = FeatureSchema(roles={"c": ColumnRole.CATEGORICAL})
    fitted = schema.with_categories({"c": CategoryTable.fit(["a", "b"])})
    restored = FeatureSchema.model_validate_json(fitted.model_dump_json())
    assert restored.categories["c"].categories == ("a", "b")


# --- M6a FE specs: serialization + pinned block order (ADR-0042 §1/§5, NFR-FE-3) ---


def _fe_schema() -> FeatureSchema:
    schema = FeatureSchema(
        roles={
            "n1": ColumnRole.NUMERIC,
            "c1": ColumnRole.CATEGORICAL,
            "c2": ColumnRole.CATEGORICAL,
            "d": ColumnRole.DATETIME,
            "rd": ColumnRole.DATETIME,
            "d__days_to_report": ColumnRole.NUMERIC,
            "c1_freq": ColumnRole.NUMERIC,
            "c1_te": ColumnRole.NUMERIC,
            "c1__c2": ColumnRole.CATEGORICAL,
        }
    )
    return (
        schema.with_datetime_spec(
            DatetimeDeltaSpec(report_date="rd", deltas=(("d", "d__days_to_report"),))
        )
        .with_frequency_encoding(FrequencyEncodingSpec(frequencies={"c1": {"0": 0.5}}))
        .with_target_encoding(
            TargetEncodingSpec(encodings={"c1": {"0": 0.7}}, global_mean=0.5, smoothing=10.0)
        )
        .with_intersections(IntersectionSpec(pairs=(("c1", "c2"),)))
    )


def test_fe_features_pinned_block_order() -> None:
    # original_numeric ⊕ datetime ⊕ frequency ⊕ target_encoding ⊕ original_categorical ⊕ intersections
    schema = _fe_schema()
    assert schema.numeric == ["n1", "d__days_to_report", "c1_freq", "c1_te"]
    assert schema.categorical == ["c1", "c2", "c1__c2"]
    assert schema.features == ["n1", "d__days_to_report", "c1_freq", "c1_te", "c1", "c2", "c1__c2"]
    # datetime sources stay DATETIME, excluded from features (ADR-0018)
    assert "d" not in schema.features and "rd" not in schema.features


def test_fe_features_consistent_with_design_matrix_split() -> None:
    # design_matrix is hstack(numeric, categorical_codes); features must equal that concatenation
    schema = _fe_schema()
    assert schema.features == schema.numeric + schema.categorical


def test_fe_specs_json_round_trip() -> None:
    restored = FeatureSchema.model_validate_json(_fe_schema().model_dump_json())
    assert restored.datetime_spec is not None and restored.datetime_spec.report_date == "rd"
    assert restored.target_encoding is not None
    assert restored.target_encoding.encodings["c1"]["0"] == 0.7  # str-keyed, no coercion drift
    assert restored.frequency_encoding is not None
    assert restored.intersections is not None and restored.intersections.pairs == (("c1", "c2"),)
    assert restored.features == _fe_schema().features


def test_legacy_schema_without_fe_specs_loads_and_is_unchanged() -> None:
    # additive None defaults -> a pre-M6a schema (no FE keys) loads, features == numeric + categorical
    schema = FeatureSchema.model_validate_json('{"roles": {"n1": "numeric", "c1": "categorical"}}')
    assert schema.datetime_spec is None and schema.target_encoding is None
    assert schema.features == ["n1", "c1"]


# --- M6b feature selection: additive FeatureSchema.selected_features (ADR-0045 §1, FR-FS-4) ---


def test_selected_features_default_none_and_round_trips() -> None:
    schema = FeatureSchema(roles={"n1": ColumnRole.NUMERIC, "c1": ColumnRole.CATEGORICAL})
    assert schema.selected_features is None
    sel = schema.with_selected_features(["n1"])
    assert sel.selected_features == ("n1",)
    restored = FeatureSchema.model_validate_json(sel.model_dump_json())
    assert restored.selected_features == ("n1",)


def test_legacy_schema_without_selected_loads() -> None:
    # additive None default -> a pre-M6b artifact (no key) loads as "all features"
    schema = FeatureSchema.model_validate_json('{"roles": {"n1": "numeric", "c1": "categorical"}}')
    assert schema.selected_features is None


# --- WS-A native categorical: FeatureSchema.categorical_indices (ADR-0088, FR-3, R-6) ---


def test_categorical_indices_is_fe_aware() -> None:
    # role-membership over the FE feature list: intersections (a__b) included, _te/_freq/datetime out
    schema = _fe_schema()
    # features: ["n1", "d__days_to_report", "c1_freq", "c1_te", "c1", "c2", "c1__c2"]
    assert schema.categorical_indices() == [4, 5, 6]  # c1, c2, c1__c2


def test_categorical_indices_shifts_with_fs_projection_not_naive_slice() -> None:
    # dropping a NUMERIC FE output shifts the categorical block left; a `len(numeric):` slice
    # (original numeric count) would mis-index after projection (R-6).
    schema = _fe_schema()
    projected = schema.with_selected_features(
        ["n1", "d__days_to_report", "c1_te", "c1", "c2", "c1__c2"]  # c1_freq removed
    )
    assert projected.categorical_indices() == [3, 4, 5]
    naive = list(
        range(len(schema.numeric), len(schema.features))
    )  # the wrong `len(numeric):` slice
    assert projected.categorical_indices() != naive


def test_categorical_indices_follows_schema_features_order_not_subset_order() -> None:
    # design_matrix projects in schema.features order regardless of how the subset is stored
    # (slice.py:204-206); categorical_indices() must match that, not the tuple order.
    schema = _fe_schema()
    shuffled = schema.with_selected_features(("c1__c2", "c1", "n1", "c2"))
    # projected in schema.features order -> ["n1", "c1", "c2", "c1__c2"], categoricals at 1,2,3
    assert shuffled.categorical_indices() == [1, 2, 3]


def test_categorical_indices_empty_without_categoricals() -> None:
    schema = FeatureSchema(roles={"n1": ColumnRole.NUMERIC, "n2": ColumnRole.NUMERIC})
    assert schema.categorical_indices() == []


# --- native-categorical cardinality gate (ADR-0092/0093, FR-1/FR-4/NFR-5) ---


def _gated_schema() -> FeatureSchema:
    # lo: 5 true categories, hi: 50 — a low-card and a high-card categorical plus a numeric
    schema = FeatureSchema(
        roles={
            "n1": ColumnRole.NUMERIC,
            "lo": ColumnRole.CATEGORICAL,
            "hi": ColumnRole.CATEGORICAL,
        }
    )
    return schema.with_categories(
        {
            "lo": CategoryTable.fit([str(i) for i in range(5)]),
            "hi": CategoryTable.fit([str(i) for i in range(50)]),
        }
    )


def test_native_routing_verdict_by_cardinality() -> None:
    # FR-1: a column is native iff len(categories) <= cap; richer ones are demoted to codes
    schema = _gated_schema()
    assert native_routing(schema, 10) == {"lo": "native", "hi": "high_cardinality"}
    assert native_routable(schema, 10) == ["lo"]


def test_native_routing_boundary_is_inclusive() -> None:
    # ADR-0093: cardinality == cap routes natively; cap-1 demotes it (boundary on len(categories))
    schema = _gated_schema()  # lo card 5
    assert native_routing(schema, 5)["lo"] == "native"
    assert native_routing(schema, 4)["lo"] == "high_cardinality"


def test_native_routing_uses_true_category_count_not_cardinality_reserves() -> None:
    # R-6: the gate measures len(categories) (5), NOT CategoryTable.cardinality (5 + 2 reserves = 7)
    schema = _gated_schema()
    assert schema.categories["lo"].cardinality == 7  # includes null/unknown reserves
    assert native_routing(schema, 5)["lo"] == "native"  # gated on the 5 true categories, not 7


def test_native_cat_max_unique_none_disables_gate() -> None:
    # FR-4/NFR-3: cap=None ⇒ every categorical native ⇒ routing identical to the ungated path
    schema = _gated_schema()
    assert native_routing(schema, None) == {"lo": "native", "hi": "native"}
    assert (
        schema.categorical_indices(None) == schema.categorical_indices()
    )  # ungated == default arg
    assert schema.categorical_indices(None) == [1, 2]  # both categoricals routed


def test_categorical_indices_gate_demotes_high_card_position() -> None:
    # FR-1/NFR-5: with a cap below hi's cardinality, only lo's position survives; the high-card
    # column is excluded (it rides the codes path), so n_native strictly drops vs the ungated set.
    schema = _gated_schema()  # features: [n1, lo, hi]
    assert schema.categorical_indices(10) == [1]  # lo only; hi demoted
    assert len(schema.categorical_indices(10)) < len(schema.categorical_indices(None))


def test_native_routing_gates_intersections_too() -> None:
    # R-4: an a__b intersection is a first-class categorical with its own (here high) cardinality and
    # is gated the same way — the expected, documented behavioural change.
    schema = (
        FeatureSchema(
            roles={
                "a": ColumnRole.CATEGORICAL,
                "b": ColumnRole.CATEGORICAL,
                "a__b": ColumnRole.CATEGORICAL,
            }
        )
        .with_categories(
            {
                "a": CategoryTable.fit([str(i) for i in range(4)]),
                "b": CategoryTable.fit([str(i) for i in range(4)]),
                "a__b": CategoryTable.fit([str(i) for i in range(40)]),
            }
        )
        .with_intersections(IntersectionSpec(pairs=(("a", "b"),)))
    )
    verdict = native_routing(schema, 10)
    assert verdict == {"a": "native", "b": "native", "a__b": "high_cardinality"}


def test_default_cap_is_in_the_justified_band_and_demotes_high_card() -> None:
    # ADR-0094 (falsifiable): the shipped default sits in the justified band — at least the auto-typing
    # int->categorical threshold (numeric_cat_max_unique), at most a tens-scale upper bound — AND it
    # actually demotes a high-card fixture, so the gate is not a silent no-op. The exact value is pinned
    # empirically by benchmarks/native_cat_gate.py; this locks that the default does its job.
    task = Task(kind="binary")
    cap = task.native_cat_max_unique
    assert cap is not None and task.numeric_cat_max_unique <= cap <= 100
    schema = FeatureSchema(
        roles={"lo": ColumnRole.CATEGORICAL, "hi": ColumnRole.CATEGORICAL}
    ).with_categories(
        {
            "lo": CategoryTable.fit([str(i) for i in range(5)]),
            "hi": CategoryTable.fit([str(i) for i in range(200)]),
        }
    )
    assert native_routing(schema, cap) == {"lo": "native", "hi": "high_cardinality"}
