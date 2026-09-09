"""M7a-C: the Optuna Tuner adapter (ADR-0061 §3) — determinism, optimum, native scalars."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

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


def _cached_tuner(
    tmp_path: Path, *, fingerprint: str = "data", model: str = "model"
) -> OptunaTuner:
    tuner = OptunaTuner()
    tuner.configure_cache(str(tmp_path), fingerprint)
    tuner.set_search_context(model, ("f0", "f1"))
    return tuner


def test_checkpoint_replays_completed_hpo_without_objective_calls(tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []

    def score(params: dict[str, Any]) -> float:
        calls.append(dict(params))
        return _score(params)

    kwargs = dict(max_trials=14, timeout_s=None, greater_is_better=True, random_state=43)
    cold = _cached_tuner(tmp_path).tune(_SPACE, score, **kwargs)
    calls.clear()
    warm = _cached_tuner(tmp_path).tune(_SPACE, score, **kwargs)
    assert not calls
    assert warm.best_params == cold.best_params and warm.best_score == cold.best_score
    assert warm.completed and warm.reused_trials == cold.n_trials_run == 14


def test_interrupted_hpo_reconstructs_sampler_and_matches_continuous(tmp_path: Path) -> None:
    prefix: list[dict[str, Any]] = []
    suffix: list[dict[str, Any]] = []
    continuous: list[dict[str, Any]] = []

    def interrupted(params: dict[str, Any]) -> float:
        if len(prefix) == 12:
            raise KeyboardInterrupt
        prefix.append(dict(params))
        return _score(params)

    def resumed(params: dict[str, Any]) -> float:
        suffix.append(dict(params))
        return _score(params)

    def baseline(params: dict[str, Any]) -> float:
        continuous.append(dict(params))
        return _score(params)

    kwargs = dict(max_trials=18, timeout_s=None, greater_is_better=True, random_state=43)
    with pytest.raises(KeyboardInterrupt):
        _cached_tuner(tmp_path).tune(_SPACE, interrupted, **kwargs)
    result = _cached_tuner(tmp_path).tune(_SPACE, resumed, **kwargs)
    expected = OptunaTuner().tune(_SPACE, baseline, **kwargs)
    assert prefix + suffix == continuous
    assert result.best_params == expected.best_params and result.best_score == expected.best_score
    assert result.reused_trials == 12 and len(suffix) == 6 and result.completed


@pytest.mark.parametrize("change", ["data", "model", "features", "seed", "space"])
def test_checkpoint_context_changes_invalidate_trials(tmp_path: Path, change: str) -> None:
    kwargs = dict(max_trials=3, timeout_s=None, greater_is_better=True, random_state=43)
    _cached_tuner(tmp_path).tune(_SPACE, _score, **kwargs)
    tuner = _cached_tuner(
        tmp_path,
        fingerprint="changed" if change == "data" else "data",
        model="other" if change == "model" else "model",
    )
    if change == "features":
        tuner.set_search_context("model", ("f1", "f0"))
    if change == "seed":
        kwargs["random_state"] = 44
    space = dict(reversed(list(_SPACE.items()))) if change == "space" else _SPACE
    result = tuner.tune(space, _score, **kwargs)
    assert result.reused_trials == 0


def test_incompatible_checkpoint_prefix_recomputes_all_trials(tmp_path: Path) -> None:
    kwargs = dict(max_trials=4, timeout_s=None, greater_is_better=True, random_state=43)
    expected = _cached_tuner(tmp_path).tune(_SPACE, _score, **kwargs)
    path = next(tmp_path.rglob("*.json"))
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["trials"][2]["params"]["x"] = 1000
    path.write_text(json.dumps(payload), encoding="utf-8")
    calls: list[dict[str, Any]] = []

    def score(params: dict[str, Any]) -> float:
        calls.append(dict(params))
        return _score(params)

    result = _cached_tuner(tmp_path).tune(_SPACE, score, **kwargs)
    assert len(calls) == 4 and result.reused_trials == 0
    assert result.best_params == expected.best_params


def test_uncommitted_checkpoint_does_not_count_as_completed(tmp_path: Path) -> None:
    root = tmp_path / "data" / "hpo"
    root.mkdir(parents=True)
    (root / ".interrupted.tmp").write_text("{", encoding="utf-8")
    result = _cached_tuner(tmp_path).tune(
        _SPACE, _score, max_trials=4, timeout_s=0, greater_is_better=True, random_state=43
    )
    assert result.n_trials_run == result.reused_trials == 0
    assert not result.completed


def test_budget_stop_keeps_completed_trials_and_resumes_partial_objective(tmp_path: Path) -> None:
    from honestml.core.exceptions import BudgetExhaustedError

    tuner = _cached_tuner(tmp_path)
    trial_numbers: list[int | None] = []

    def limited(params: dict[str, Any]) -> float:
        trial_numbers.append(tuner.current_trial_number)
        if len(trial_numbers) == 3:
            raise BudgetExhaustedError("time", completed=2, skipped=1, failed=0)
        return _score(params)

    kwargs = dict(max_trials=4, timeout_s=None, greater_is_better=True, random_state=43)
    partial = tuner.tune(_SPACE, limited, **kwargs)
    assert partial.n_trials_run == 2 and not partial.completed
    assert trial_numbers == [0, 1, 2] and tuner.current_trial_number is None
    trial_numbers.clear()

    def finish(params: dict[str, Any]) -> float:
        trial_numbers.append(tuner.current_trial_number)
        return _score(params)

    result = tuner.tune(_SPACE, finish, **kwargs)
    assert result.completed and result.n_trials_run == 4 and result.reused_trials == 2
    assert trial_numbers == [2, 3]
    expected = OptunaTuner().tune(_SPACE, _score, **kwargs)
    assert result.best_params == expected.best_params and result.best_score == expected.best_score


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("greater_is_better", [True, False])
def test_invalid_trial_never_completes_a_model_or_retries_on_resume(
    tmp_path: Path, invalid: float, greater_is_better: bool
) -> None:
    calls = 0

    def score(params: dict[str, Any]) -> float:
        nonlocal calls
        calls += 1
        return invalid

    kwargs = dict(
        max_trials=1, timeout_s=None, greater_is_better=greater_is_better, random_state=43
    )
    cold = _cached_tuner(tmp_path).tune(_SPACE, score, **kwargs)
    assert not cold.completed
    assert cold.best_params == {} and math.isnan(cold.best_score)
    assert cold.n_trials_run == cold.failed_trials == 1 and cold.successful_trials == 0
    warm = _cached_tuner(tmp_path).tune(_SPACE, score, **kwargs)
    assert calls == 1 and warm.reused_trials == 1
    assert not warm.completed and warm.failed_trials == 1 and warm.best_params == {}
    checkpoint = json.loads(next(tmp_path.rglob("*.json")).read_text(encoding="utf-8"))
    assert checkpoint["trials"][0]["status"] == "failed"


def test_failed_trial_prefix_reconstructs_same_sampler_and_finite_winner(tmp_path: Path) -> None:
    tuner = _cached_tuner(tmp_path)
    seen: list[dict[str, Any]] = []

    def score(params: dict[str, Any]) -> float:
        assert tuner.current_trial_number is not None
        seen.append(dict(params))
        return float("inf") if tuner.current_trial_number % 3 == 0 else _score(params)

    def interrupted(params: dict[str, Any]) -> float:
        if tuner.current_trial_number == 12:
            raise KeyboardInterrupt
        return score(params)

    kwargs = dict(max_trials=18, timeout_s=None, greater_is_better=True, random_state=43)
    with pytest.raises(KeyboardInterrupt):
        tuner.tune(_SPACE, interrupted, **kwargs)
    result = tuner.tune(_SPACE, score, **kwargs)
    resumed_params = list(seen)
    tuner = OptunaTuner()
    seen.clear()
    expected = tuner.tune(_SPACE, score, **kwargs)
    assert resumed_params == seen
    assert result.best_params == expected.best_params and result.best_score == expected.best_score
    assert math.isfinite(result.best_score) and result.completed
    assert (
        result.n_trials_run == 18 and result.failed_trials == 6 and result.successful_trials == 12
    )
    assert result.reused_trials == 12
