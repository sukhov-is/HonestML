"""M0 / ADR-0008: exception taxonomy base — callers catch specific types."""

from __future__ import annotations

import pytest

from honestml.core import AutoMLError, ConfigError, MissingDependencyError

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("exc_type", [ConfigError, MissingDependencyError])
def test_subclasses_of_automl_error(exc_type: type[AutoMLError]) -> None:
    assert issubclass(exc_type, AutoMLError)


def test_missing_dependency_message_is_actionable() -> None:
    err = MissingDependencyError("xgboost")
    msg = str(err)
    assert "xgboost" in msg
    assert "pip install honestml[xgboost]" in msg
    assert err.extra == "xgboost"


def test_can_catch_specific_then_base() -> None:
    with pytest.raises(AutoMLError):
        raise ConfigError("bad config")
