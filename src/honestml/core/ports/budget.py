"""The ``Budget`` port.

A run budget covers wall-clock **and** memory; the use-case consumes it and, on
exhaustion, returns the best result so far (graceful degradation).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Budget(Protocol):
    """Tracks remaining wall-clock time and (optionally) memory headroom."""

    def time_left(self) -> float:
        """Seconds remaining (``inf`` if unbounded)."""
        ...

    def consume(self, seconds: float) -> None:
        """Account for elapsed work."""
        ...

    @property
    def exhausted(self) -> bool:
        """True once no time (or memory) headroom remains."""
        ...

    @property
    def exhausted_reason(self) -> str | None:
        """Which axis is exhausted (``"time"``/``"trials"``/``"memory"``), or ``None`` while headroom
        remains. Mode-first: an explicit time/trials budget outranks the orthogonal memory guard.
        Read by the use-case for a truthful ``BudgetExhaustedError``/run-report."""
        ...

    def memory_left(self) -> float | None:
        """MB of memory headroom, or ``None`` if memory is not tracked."""
        ...
