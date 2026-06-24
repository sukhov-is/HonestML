"""M7a-A: the Tuner port + backend-neutral SearchSpace (ADR-0061)."""

from __future__ import annotations

import pytest

from honestml.core import ConfigError, TuneOutcome, Tuner, parse_search_space
from honestml.core.ports.tuner import CategoricalParam, FloatParam, IntParam

pytestmark = pytest.mark.unit


def test_parse_search_space_typed_entries() -> None:
    space = parse_search_space(
        {
            "depth": {"type": "int", "low": 2, "high": 10},
            "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
            "grow_policy": {"type": "categorical", "choices": ["sym", "depthwise"]},
        }
    )
    assert isinstance(space["depth"], IntParam)
    assert isinstance(space["learning_rate"], FloatParam) and space["learning_rate"].log
    assert isinstance(space["grow_policy"], CategoricalParam)


def test_parse_search_space_empty_is_empty() -> None:
    assert parse_search_space({}) == {}


def test_parse_search_space_rejects_unknown_type() -> None:
    with pytest.raises(ConfigError):
        parse_search_space({"x": {"type": "loguniform", "low": 1, "high": 2}})


@pytest.mark.parametrize(
    "bad",
    [
        {"type": "int", "low": 5, "high": 5},
        {"type": "float", "low": 1.0, "high": 0.5},
        {"type": "float", "low": 0.0, "high": 1.0, "log": True},
        {"type": "categorical", "choices": []},
    ],
)
def test_parse_search_space_rejects_bad_bounds(bad: dict) -> None:
    with pytest.raises(ConfigError):
        parse_search_space({"p": bad})


def test_tuner_is_runtime_checkable() -> None:
    class _FakeTuner:
        name = "fake"

        def tune(
            self, search_space, score, *, max_trials, timeout_s, greater_is_better, random_state
        ):
            pick = max if greater_is_better else min  # adapter sets direction (raw metric score)
            best = pick(({"v": i} for i in range(max_trials)), key=lambda p: score(p))
            return TuneOutcome(best_params=best, n_trials_run=max_trials, best_score=score(best))

    t = _FakeTuner()
    assert isinstance(t, Tuner)
    out = t.tune(
        {},
        lambda p: -((p["v"] - 3) ** 2),
        max_trials=6,
        timeout_s=None,
        greater_is_better=True,
        random_state=0,
    )
    assert out.best_params == {"v": 3} and out.n_trials_run == 6
