"""M1a: Task domain object (ADR-0006) — generalization root."""

from __future__ import annotations

import numpy as np
import pytest

from honestml.core import ConfigError, Task, resolve_positive

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("kind", "metric"),
    [("binary", "roc_auc"), ("multiclass", "log_loss"), ("regression", "rmse")],
)
def test_default_target_metric_per_kind(kind: str, metric: str) -> None:
    assert Task(kind=kind).target_metric == metric


def test_explicit_metric_overrides_default() -> None:
    assert Task(kind="binary", metric="pr_auc").target_metric == "pr_auc"


@pytest.mark.parametrize(
    ("kind", "expected"),
    [("binary", True), ("multiclass", True), ("regression", False)],
)
def test_is_classification(kind: str, expected: bool) -> None:
    assert Task(kind=kind).is_classification is expected


def test_default_cv_scheme() -> None:
    assert Task(kind="binary").default_cv_scheme == "stratified"
    assert Task(kind="regression").default_cv_scheme == "kfold"


def test_task_json_round_trip() -> None:
    t = Task(kind="multiclass", metric="f1_macro", numeric_cat_max_unique=10)
    assert Task.model_validate_json(t.model_dump_json()) == t


def test_invalid_kind_rejected() -> None:
    with pytest.raises(Exception):
        Task(kind="ranking")  # type: ignore[arg-type]


# --- M6a: additive Task.report_date (ADR-0018 §5) + resolve_positive relocated to core ---


def test_report_date_defaults_none_and_round_trips() -> None:
    assert Task(kind="binary").report_date is None
    t = Task(kind="binary", report_date="report_dt")
    assert Task.model_validate_json(t.model_dump_json()) == t


def test_resolve_positive_rule() -> None:
    # canonical {0,1}: label 1 is positive; else the greatest; explicit override honored
    assert resolve_positive(Task(kind="binary"), np.array([0, 1])) == 1
    assert resolve_positive(Task(kind="binary"), np.array(["a", "b"])) == "b"
    assert resolve_positive(Task(kind="binary", positive_label="a"), np.array(["a", "b"])) == "a"


def test_resolve_positive_unknown_override_raises() -> None:
    with pytest.raises(ConfigError):
        resolve_positive(Task(kind="binary", positive_label=9), np.array([0, 1]))
