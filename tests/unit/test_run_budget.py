"""M5a-engine: the RunBudget adapter on a fake clock (ADR-0032 §2, NFR-M5-1/3).

The budget logic is a Humble Object: exercised on an injected fake clock, no training,
no real wall-clock. ``mode="trials"`` exhaustion depends on the number of ``consume``
calls (not their value); ``mode="time"`` starts ``t0`` lazily and captures it once.
"""

from __future__ import annotations

import logging

import pytest

from honestml.adapters import RunBudget
from honestml.core import BudgetConfig, MissingDependencyError

pytestmark = pytest.mark.unit


class _Clock:
    """A controllable fake clock: ``t`` is read on every call (no auto-advance)."""

    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


class _Rss:
    """A controllable fake RSS probe in MB (injected like the clock — no psutil, Humble Object)."""

    def __init__(self, mb: float) -> None:
        self.mb = mb

    def __call__(self) -> float:
        return self.mb


def test_none_never_exhausts() -> None:
    b = RunBudget(BudgetConfig())  # default mode="none"
    assert b.exhausted is False
    assert b.time_left() == float("inf")
    b.consume(123.0)  # no-op
    assert b.exhausted is False
    assert b.memory_left() is None


def test_trials_exhausts_after_k() -> None:
    b = RunBudget(BudgetConfig(mode="trials", n_trials=2))
    assert b.exhausted is False
    b.consume(0.1)
    assert b.exhausted is False
    b.consume(0.1)
    assert b.exhausted is True


def test_trials_ignores_seconds_value() -> None:
    """Exhaustion depends on the count of consume calls, never the seconds argument."""
    b = RunBudget(BudgetConfig(mode="trials", n_trials=2))
    b.consume(9999.0)
    assert b.exhausted is False  # huge value, but only one trial counted
    b.consume(0.0)
    assert b.exhausted is True


def test_time_exhausts_on_fake_clock() -> None:
    clock = _Clock()
    b = RunBudget(BudgetConfig(mode="time", time_budget_s=10.0), clock=clock)
    assert b.exhausted is False  # t0 captured at 0.0
    clock.t = 5.0
    assert b.exhausted is False
    clock.t = 11.0
    assert b.exhausted is True


def test_time_lazy_start_idempotent() -> None:
    """t0 is captured on the first read (setup not billed) and only once (stable across reads)."""
    clock = _Clock()
    b = RunBudget(BudgetConfig(mode="time", time_budget_s=10.0), clock=clock)
    clock.t = 100.0  # advance BEFORE the first read -> t0 must be 100, not the construction time
    assert b.exhausted is False  # 100 - 100 = 0 < 10
    clock.t = 105.0
    assert b.exhausted is False  # 105 - 100 = 5 < 10 (t0 not re-captured)
    clock.t = 111.0
    assert b.exhausted is True  # 111 - 100 = 11 >= 10
    assert b.time_left() < 0.0


def test_time_left_positive_within_budget() -> None:
    clock = _Clock()
    b = RunBudget(BudgetConfig(mode="time", time_budget_s=10.0), clock=clock)
    assert b.exhausted is False  # lazy start fixes t0 at 0.0
    clock.t = 3.0
    assert b.time_left() == pytest.approx(7.0)


# --- M5 memory-enforce: orthogonal cooperative RSS limit on a fake probe (ADR-0039, FR-MEM-1) ---


def test_none_exhausted_reason_is_none() -> None:
    b = RunBudget(BudgetConfig())  # mode="none", no memory
    assert b.exhausted is False
    assert b.exhausted_reason is None


def test_memory_gate_exhausts_on_fake_probe() -> None:
    rss = _Rss(50.0)
    b = RunBudget(BudgetConfig(memory_limit_mb=100), mem_probe=rss)
    assert b.exhausted is False  # 50 < 100
    rss.mb = 150.0
    assert b.exhausted is True
    assert b.exhausted_reason == "memory"


def test_memory_left_value_and_negative_overshoot() -> None:
    rss = _Rss(120.0)
    b = RunBudget(BudgetConfig(memory_limit_mb=100), mem_probe=rss)
    assert b.memory_left() == pytest.approx(-20.0)  # overshoot valid, not clamped (fix m2)
    rss.mb = 40.0
    assert b.memory_left() == pytest.approx(60.0)


def test_memory_left_none_without_limit() -> None:
    assert RunBudget(BudgetConfig(), mem_probe=_Rss(10.0)).memory_left() is None


def test_memory_composes_with_time() -> None:
    clock, rss = _Clock(), _Rss(10.0)
    b = RunBudget(
        BudgetConfig(mode="time", time_budget_s=10.0, memory_limit_mb=100),
        clock=clock,
        mem_probe=rss,
    )
    assert b.exhausted is False  # both within budget
    rss.mb = 150.0  # memory blows first, time still fine
    assert b.exhausted is True
    assert b.exhausted_reason == "memory"


def test_memory_composes_with_trials() -> None:
    rss = _Rss(150.0)
    b = RunBudget(BudgetConfig(mode="trials", n_trials=5, memory_limit_mb=100), mem_probe=rss)
    assert b.exhausted is True and b.exhausted_reason == "memory"  # trials not used up, memory is


def test_reason_priority_mode_first() -> None:
    """time AND memory both exhausted -> reason is the explicit mode axis (time), not memory (fix M1)."""
    clock, rss = _Clock(), _Rss(10.0)
    b = RunBudget(
        BudgetConfig(mode="time", time_budget_s=10.0, memory_limit_mb=100),
        clock=clock,
        mem_probe=rss,
    )
    assert b.exhausted is False  # first read fixes t0 at 0.0 (lazy start), both within budget
    clock.t = 20.0  # time exhausted
    rss.mb = 150.0  # memory exhausted too
    assert b.exhausted is True
    assert b.exhausted_reason == "time"


def test_none_plus_memory_is_active() -> None:
    rss = _Rss(150.0)
    b = RunBudget(BudgetConfig(mode="none", memory_limit_mb=100), mem_probe=rss)
    assert b.exhausted is True and b.exhausted_reason == "memory"  # memory-only run (R-MEM-NONE)


def test_trials_reason_when_exhausted() -> None:
    b = RunBudget(BudgetConfig(mode="trials", n_trials=1))
    assert b.exhausted_reason is None
    b.consume(0.1)
    assert b.exhausted_reason == "trials"


def test_baseline_below_limit_warns(caplog: pytest.LogCaptureFixture) -> None:
    """The limit is already unsatisfiable at startup -> actionable WARNING, not a hard error (R-MEM-BASELINE)."""
    with caplog.at_level(logging.WARNING, logger="honestml"):
        RunBudget(BudgetConfig(memory_limit_mb=100), mem_probe=_Rss(200.0))
    assert any("baseline" in r.getMessage().lower() for r in caplog.records)


def test_baseline_below_limit_does_not_warn(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="honestml"):
        RunBudget(BudgetConfig(memory_limit_mb=100), mem_probe=_Rss(10.0))
    assert not any("baseline" in r.getMessage().lower() for r in caplog.records)


def test_memory_limit_requires_psutil(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default probe + a limit imports psutil at construction; absent -> MissingDependencyError (B2)."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        if name == "psutil":
            raise ImportError("No module named 'psutil'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(MissingDependencyError) as ei:
        RunBudget(BudgetConfig(memory_limit_mb=100))
    assert ei.value.extra == "memory" and ei.value.package == "psutil"


def test_injected_probe_needs_no_psutil() -> None:
    # an injected probe never imports psutil even with a limit set (psutil-free tests, NFR-RM-2)
    b = RunBudget(BudgetConfig(memory_limit_mb=100), mem_probe=lambda: 10.0)
    assert b.exhausted is False
    assert b.memory_left() == pytest.approx(90.0)
