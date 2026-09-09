"""Per-run timing and aggregate work accounting, isolated between concurrent runs."""

from __future__ import annotations

import copy
import logging
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from .config import RunConfig
from .logging import get_logger


@dataclass
class _Stage:
    id: int
    started_at: float
    child_seconds: float = 0.0
    peak_rss_mb: float | None = None


@dataclass(frozen=True)
class _FitAttribution:
    recipe: str | None = None
    fold: int | None = None
    trial: int | None = None


@dataclass
class RunContext:
    """Record inclusive totals, exclusive stage events and aggregate training work.

    Stage scopes are synchronous and nested. Each invocation survives in ``stages``;
    repeated names accumulate in ``timings``. Wall time is measured independently,
    with time outside completed stage scopes attributed to overhead. Cached historical
    durations never count towards current wall time.
    """

    run_config: RunConfig = field(default_factory=RunConfig)
    logger: logging.Logger = field(default_factory=get_logger)
    timings: dict[str, dict[str, float]] = field(default_factory=dict)
    rss_probe: Callable[[], float] | None = field(default=None, repr=False)
    environment: dict[str, Any] = field(default_factory=dict)
    before_fit: Callable[[str], None] | None = field(default=None, repr=False)
    _started_at: float = field(default_factory=lambda: time.perf_counter(), repr=False)
    _finished_at: float | None = field(default=None, init=False, repr=False)
    _stack: list[_Stage] = field(default_factory=list, init=False, repr=False)
    _events: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)
    _work: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)
    _next_id: int = field(default=0, init=False, repr=False)
    _fit_attribution: _FitAttribution = field(
        default_factory=_FitAttribution, init=False, repr=False
    )
    _peak_rss_mb: float | None = field(default=None, init=False, repr=False)

    def start_run(self, started_at: float | None = None) -> None:
        """Set the monotonic run boundary, optionally including work before context creation."""
        self._started_at = time.perf_counter() if started_at is None else started_at
        self._finished_at = None
        self.sample_memory()

    def finish_run(self) -> None:
        """Freeze the independently measured end boundary for subsequent report snapshots."""
        self.sample_memory()
        self._finished_at = time.perf_counter()

    def sample_memory(self) -> float | None:
        """Sample process RSS at an operation boundary; this is not a continuous peak."""
        if self.rss_probe is None:
            return None
        rss = float(self.rss_probe())
        self._peak_rss_mb = max(rss, self._peak_rss_mb or 0.0)
        for active in self._stack:
            active.peak_rss_mb = max(rss, active.peak_rss_mb or 0.0)
        return rss

    @contextmanager
    def timed_stage(self, key: str, stage: str) -> Iterator[None]:
        """Measure one stage, subtracting nested measured intervals from its exclusive time."""
        parent = self._stack[-1].id if self._stack else None
        active = _Stage(self._next_id, time.perf_counter())
        self._next_id += 1
        self._stack.append(active)
        self.sample_memory()
        completed = False
        try:
            yield
            completed = True
        finally:
            self.sample_memory()
            elapsed = time.perf_counter() - active.started_at
            self._stack.pop()
            if self._stack:
                self._stack[-1].child_seconds += elapsed
            times = self.timings.setdefault(key, {})
            times[stage] = times.get(stage, 0.0) + elapsed
            self._events.append(
                {
                    "id": active.id,
                    "parent_id": parent,
                    "key": key,
                    "stage": stage,
                    "elapsed_s": elapsed,
                    "exclusive_s": max(0.0, elapsed - active.child_seconds),
                    "source": "measured",
                    "status": "completed" if completed else "failed",
                    "sampled_peak_rss_mb": active.peak_rss_mb,
                }
            )
            self.logger.info("stage key=%s stage=%s elapsed=%.3fs", key, stage, elapsed)

    def record_stage_time(self, key: str, stage: str, elapsed: float) -> None:
        """Record historical cached work separately from measured current-run duration."""
        times = self.timings.setdefault(key, {})
        times[stage] = times.get(stage, 0.0) + elapsed
        self._events.append(
            {
                "id": self._next_id,
                "parent_id": None,
                "key": key,
                "stage": stage,
                "elapsed_s": elapsed,
                "exclusive_s": 0.0,
                "source": "cache",
                "status": "reused",
                "sampled_peak_rss_mb": None,
            }
        )
        self._next_id += 1

    def total_time(self, key: str) -> float:
        """Total exclusive measured plus historical cached stage work for a context key."""
        return sum(
            float(e["elapsed_s"] if e["source"] == "cache" else e["exclusive_s"])
            for e in self._events
            if e["key"] == key
        )

    @contextmanager
    def fit_attribution(
        self,
        *,
        recipe: str | None = None,
        fold: int | None = None,
        trial: int | None = None,
    ) -> Iterator[None]:
        """Attribute nested backend fits to their recipe and local fold/trial without changing work.

        Unspecified labels inherit the enclosing scope. Scopes are restored after errors and are
        isolated per run; explicit labels on ``timed_fit`` override the inherited defaults.
        """
        previous = self._fit_attribution
        self._fit_attribution = _FitAttribution(
            previous.recipe if recipe is None else recipe,
            previous.fold if fold is None else fold,
            previous.trial if trial is None else trial,
        )
        try:
            yield
        finally:
            self._fit_attribution = previous

    def record_fit(
        self,
        stage: str,
        *,
        model_id: str,
        elapsed_s: float,
        rows: int,
        columns: int,
        fold: int | None = None,
        trial: int | None = None,
        recipe: str | None = None,
        tree_budget: int | None = None,
        iterations: int | None = None,
        status: str = "completed",
    ) -> None:
        """Record a fit attempt using dimensions and resource counts, without row values."""
        self._work.append(
            {
                "stage": stage,
                "model_id": model_id,
                "elapsed_s": float(elapsed_s),
                "rows": int(rows),
                "columns": int(columns),
                "fold": self._fit_attribution.fold if fold is None else fold,
                "trial": self._fit_attribution.trial if trial is None else trial,
                "recipe": self._fit_attribution.recipe if recipe is None else recipe,
                "tree_budget": tree_budget,
                "iterations": iterations,
                "status": status,
                "sampled_rss_mb": self.sample_memory(),
            }
        )

    @contextmanager
    def timed_fit(
        self,
        stage: str,
        *,
        model_id: str,
        rows: int,
        columns: int,
        fold: int | None = None,
        trial: int | None = None,
        recipe: str | None = None,
    ) -> Iterator[dict[str, int | None]]:
        """Measure a fit attempt; fill yielded tree budget/iterations after backend completion.

        Fit work is an attribution channel inside stages, not an additional interval
        in the wall-time sum. Exceptions propagate and record a failed attempt.
        """
        if self.before_fit is not None:
            self.before_fit(stage)
        resources: dict[str, int | None] = {"tree_budget": None, "iterations": None}
        started_at = time.perf_counter()
        self.sample_memory()
        completed = False
        try:
            yield resources
            completed = True
        finally:
            self.record_fit(
                stage,
                model_id=model_id,
                elapsed_s=time.perf_counter() - started_at,
                rows=rows,
                columns=columns,
                fold=fold,
                trial=trial,
                recipe=recipe,
                tree_budget=resources["tree_budget"],
                iterations=resources["iterations"],
                status="completed" if completed else "failed",
            )

    def cost_report(self) -> dict[str, Any]:
        """Return a detached JSON-compatible snapshot of full-run cost and resources."""
        end = self._finished_at if self._finished_at is not None else time.perf_counter()
        total = end - self._started_at
        exclusive: dict[str, dict[str, float]] = {}
        for event in self._events:
            if event["source"] == "measured":
                times = exclusive.setdefault(event["key"], {})
                times[event["stage"]] = times.get(event["stage"], 0.0) + event["exclusive_s"]
        attributed = sum(sum(stages.values()) for stages in exclusive.values())
        completed = sum(w["status"] == "completed" for w in self._work)
        failed = sum(w["status"] == "failed" for w in self._work)
        return copy.deepcopy(
            {
                "schema_version": 1,
                "total_wall_s": total,
                "attributed_wall_s": attributed,
                "overhead_s": total - attributed,
                "exclusive_timings": exclusive,
                "stages": sorted(self._events, key=lambda e: e["id"]),
                "work": self._work,
                "fit_counts": {
                    "attempted": len(self._work),
                    "completed": completed,
                    "failed": failed,
                },
                "sampled_peak_rss_mb": self._peak_rss_mb,
                "memory_measurement": "stage_boundaries" if self.rss_probe else "unavailable",
                "environment": self.environment,
            }
        )

    def manifest(self) -> dict[str, Any]:
        """Serializable run configuration, accumulated timings and cost accounting."""
        return {
            "config": self.run_config.model_dump(mode="json"),
            "timings": copy.deepcopy(self.timings),
            "cost": self.cost_report(),
        }
