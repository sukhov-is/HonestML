"""M7a-C: the Optuna Tuner adapter (ADR-0061 §3) — determinism, optimum, native scalars."""

from __future__ import annotations

import pytest

pytest.importorskip("optuna")  # the heavy `hpo` extra; skip when not installed

from honestml.adapters import OptunaTuner  # noqa: E402
from honestml.core import Tuner, parse_search_space  # noqa: E402

pytestmark = pytest.mark.unit

_SPACE = parse_search_space(
    {
        "x": {"type": "float", "low": -5.0, "high": 5.0},
        "k": {"type": "categorical", "choices": ["a", "b"]},
    }
)


def _score(params: dict) -> float:
    # maximized at x=2, k="b"
    return -((params["x"] - 2.0) ** 2) + (1.0 if params["k"] == "b" else 0.0)


def test_optuna_tuner_is_a_tuner() -> None:
    assert isinstance(OptunaTuner(), Tuner)


def test_finds_quadratic_optimum() -> None:
    out = OptunaTuner().tune(
        _SPACE, _score, max_trials=60, timeout_s=None, greater_is_better=True, random_state=0
    )
    assert out.n_trials_run == 60
    assert abs(out.best_params["x"] - 2.0) < 0.5
    assert out.best_params["k"] == "b"


def test_seed_determinism_identical_params() -> None:
    kw = dict(max_trials=40, timeout_s=None, greater_is_better=True, random_state=42)
    a = OptunaTuner().tune(_SPACE, _score, **kw)
    b = OptunaTuner().tune(_SPACE, _score, **kw)
    assert a.best_params == b.best_params and a.best_score == b.best_score


def test_different_seed_may_differ() -> None:
    a = OptunaTuner().tune(
        _SPACE, _score, max_trials=15, timeout_s=None, greater_is_better=True, random_state=1
    )
    b = OptunaTuner().tune(
        _SPACE, _score, max_trials=15, timeout_s=None, greater_is_better=True, random_state=2
    )
    # the sampler actually consumes the seed (not a hard guarantee of inequality, but stable here)
    assert a.best_params["x"] != b.best_params["x"]


def test_best_params_are_native_scalars() -> None:
    out = OptunaTuner().tune(
        parse_search_space({"n": {"type": "int", "low": 1, "high": 9}}),
        lambda p: -((p["n"] - 5) ** 2),
        max_trials=20,
        timeout_s=None,
        greater_is_better=True,
        random_state=0,
    )
    v = out.best_params["n"]
    assert type(v) is int  # python-native, not np.int64 (ADR-0061 §2)


def test_minimize_direction() -> None:
    # greater_is_better=False -> the adapter minimizes the raw score (loss-like objective)
    out = OptunaTuner().tune(
        parse_search_space({"x": {"type": "float", "low": -5.0, "high": 5.0}}),
        lambda p: (p["x"] - 1.0) ** 2,
        max_trials=50,
        timeout_s=None,
        greater_is_better=False,
        random_state=0,
    )
    assert abs(out.best_params["x"] - 1.0) < 0.5
