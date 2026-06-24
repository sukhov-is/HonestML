"""M0-3: library logging — namespaced logger, NullHandler by default, structured stage."""

from __future__ import annotations

import logging

import pytest

from honestml.core import get_logger, log_stage

pytestmark = pytest.mark.unit


def test_root_logger_is_namespaced_with_null_handler() -> None:
    root = get_logger()
    assert root.name == "honestml"
    assert any(isinstance(h, logging.NullHandler) for h in root.handlers)


def test_child_logger_name() -> None:
    assert get_logger("data").name == "honestml.data"


def test_log_stage_emits_structured_fields(caplog: pytest.LogCaptureFixture) -> None:
    log = get_logger("test")
    with caplog.at_level(logging.INFO, logger="honestml"):
        log_stage(log, step="lightgbm", stage="eval", roc_auc=0.81)
    assert "step=lightgbm" in caplog.text
    assert "stage=eval" in caplog.text
    assert "roc_auc=0.81" in caplog.text
