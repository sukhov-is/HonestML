"""The ``RunBudget`` adapter: cooperative time/trials enforcement + an orthogonal memory limit.

Implements the :class:`honestml.core.Budget` port (ADR-0032 §2, ADR-0039). Mode covers the *time*
axis, memory is an **orthogonal** axis composed on top of any mode:

- ``"none"`` (unbounded, default): never mode-exhausted, ``time_left`` is ``inf``.
- ``"trials"``: counts ``consume`` calls as completed trials; the ``seconds`` argument is advisory.
- ``"time"``: the clock starts **lazily** on the first ``exhausted`` read (entry to the candidate
  loop), so setup/carve is not billed; ``t0`` is captured exactly once.
- memory (``memory_limit_mb``): when set, the run is also exhausted once process RSS ``>=`` the limit
  (checked *before* a candidate starts). Composes with every mode (a run can be time- **and**
  memory-bounded, or memory-only under ``mode="none"``).

The ``clock`` and ``mem_probe`` are injected so the budget logic is synchronously testable on fakes
without training or psutil (Humble Object, NFR-RM-2); the concrete ``time.perf_counter``/``psutil`` RSS
live only here (NFR-RM-4). ``psutil`` is an optional ``[memory]`` extra, imported lazily and only when
a memory limit is set with the default probe — never at top-level import (lightweight core, ADR-0001).
"""

from __future__ import annotations

import time
from collections.abc import Callable

from honestml.core import BudgetConfig, MissingDependencyError, get_logger

logger = get_logger("adapters.run_budget")


def _default_rss_mb() -> Callable[[], float]:
    """Build the default process-RSS probe, importing ``psutil`` once (ADR-0039 §3).

    Constructs ``psutil.Process()`` a single time (fix m3); the returned closure reads RSS in MB on
    each call. Raises :class:`MissingDependencyError` when psutil is absent — the single failure
    point, at budget construction, not at top-level import.
    """
    try:
        import psutil
    except ImportError as exc:
        raise MissingDependencyError("memory", package="psutil") from exc
    process = psutil.Process()
    return lambda: process.memory_info().rss / 1024**2


class RunBudget:
    """Cooperative per-candidate budget over wall-clock time / trial count, plus a memory limit."""

    def __init__(
        self,
        config: BudgetConfig,
        *,
        clock: Callable[[], float] = time.perf_counter,
        mem_probe: Callable[[], float] | None = None,
    ) -> None:
        self.mode = config.mode
        self._time_budget_s = config.time_budget_s
        self._n_trials = config.n_trials
        self._memory_limit_mb = config.memory_limit_mb
        self._clock = clock
        self._t0: float | None = None
        self._trials_done = 0
        # the RSS probe is built lazily and ONLY when a memory limit is set; an injected probe (tests)
        # needs no psutil. This is the single psutil import point (ADR-0039 §3, fix M4).
        if config.memory_limit_mb is not None and mem_probe is None:
            mem_probe = _default_rss_mb()
        self._mem_probe = mem_probe
        # actionable WARNING when the limit is already unsatisfiable at startup: no candidate can ever
        # start (ADR-0039 §2, R-MEM-BASELINE) — diagnostic, not a flaky hard ConfigError.
        if config.memory_limit_mb is not None and mem_probe is not None:
            baseline = mem_probe()
            if baseline >= config.memory_limit_mb:
                logger.warning(
                    "memory_limit_mb=%s below process baseline ~%.0f MB; no candidate can start",
                    config.memory_limit_mb,
                    baseline,
                )

    def time_left(self) -> float:
        if self.mode != "time":
            return float("inf")
        # mode="time" guarantees a numeric budget (BudgetConfig validator, boundary invariant)
        assert self._time_budget_s is not None
        return self._time_budget_s - (self._clock() - self._start())

    def consume(self, seconds: float) -> None:
        # trials: count the call as one completed trial (the seconds value is advisory, ignored).
        # time/none: a no-op — exhaustion is clock-/mode-derived, not consume-driven (ADR-0032 §2).
        if self.mode == "trials":
            self._trials_done += 1

    @property
    def exhausted(self) -> bool:
        # memory is orthogonal: exhausted if the mode is out OR the memory limit is hit (ADR-0039 §1)
        return self._mode_exhausted() or self._memory_exhausted()

    @property
    def exhausted_reason(self) -> str | None:
        # mode-first priority: an explicit user budget (time/trials) outranks the secondary memory
        # guard (fix M1), consistent with the `exhausted` short-circuit order and the c4 flowchart.
        if self._mode_exhausted():
            return self.mode
        if self._memory_exhausted():
            return "memory"
        return None

    def memory_left(self) -> float | None:
        if self._memory_limit_mb is None or self._mem_probe is None:
            return None
        # negative is valid (overshoot) and informative — not clamped, symmetric with time_left()<0 (m2)
        return self._memory_limit_mb - self._mem_probe()

    def _mode_exhausted(self) -> bool:
        if self.mode == "none":
            return False
        if self.mode == "trials":
            assert self._n_trials is not None
            return self._trials_done >= self._n_trials
        return self.time_left() <= 0.0

    def _memory_exhausted(self) -> bool:
        if self._memory_limit_mb is None or self._mem_probe is None:
            return False
        return self._mem_probe() >= self._memory_limit_mb

    def _start(self) -> float:
        """Capture ``t0`` once on first use (lazy start); subsequent reads reuse it."""
        if self._t0 is None:
            self._t0 = self._clock()
        return self._t0
