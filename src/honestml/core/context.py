"""Per-run context.

A fresh ``RunContext`` is created on each run and passed explicitly, so
concurrent runs in one process stay independent (reentrancy/reproducibility).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from .config import RunConfig
from .logging import get_logger


@dataclass
class RunContext:
    """Mutable state of a single run: timings, logger, config."""

    run_config: RunConfig = field(default_factory=RunConfig)
    logger: logging.Logger = field(default_factory=get_logger)
    timings: dict[str, dict[str, float]] = field(default_factory=dict)

    @contextmanager
    def timed_stage(self, key: str, stage: str) -> Iterator[None]:
        """Time a stage and record the elapsed seconds under ``timings[key][stage]``."""
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed = round(time.perf_counter() - t0, 1)
            self.timings.setdefault(key, {})[stage] = elapsed
            self.logger.info("stage key=%s stage=%s elapsed=%.1fs", key, stage, elapsed)

    def record_stage_time(self, key: str, stage: str, elapsed: float) -> None:
        """Record a stage time loaded from cache (no timer)."""
        self.timings.setdefault(key, {})[stage] = elapsed

    def total_time(self, key: str) -> float:
        """Sum of all stage times recorded for ``key``."""
        return sum(self.timings.get(key, {}).values())

    def manifest(self) -> dict[str, Any]:
        """Serializable run manifest (config + timings) — basis for replay."""
        return {
            "config": self.run_config.model_dump(mode="json"),
            "timings": self.timings,
        }
