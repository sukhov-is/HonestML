"""The ``CandidateCache`` port.

A durable per-candidate stage cache. ``get`` returns a completed :class:`Candidate` (with its OOF)
or ``None`` on a miss / incompatible / corrupt entry (**fail-closed** — any uncertainty is a miss,
never a false hit, never an exception out); ``put`` persists a completed candidate durably. The
adapter scopes the cache to a run-fingerprint (constructed with ``cache_dir`` + ``fingerprint``), so
the port key is just ``candidate_id`` (minimal surface).

The use-case (``run_slice``) reads/writes per candidate only — the port is deliberately ``get``/
``put`` **forever**: GC / clear / eviction live **outside** the port, at the filesystem level,
so they never enter the use-case layer. The concrete disk/joblib
store is ``JoblibCandidateCache`` in ``adapters``; the use-case never names it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from honestml.core.selection_policy import Candidate


@runtime_checkable
class CandidateCache(Protocol):
    """Durable per-candidate stage cache, scoped to a run-fingerprint by the adapter."""

    def get(self, candidate_id: str) -> Candidate | None:
        """The cached candidate (with OOF), or ``None`` on miss/incompatible/corrupt (fail-closed)."""
        ...

    def put(self, candidate_id: str, candidate: Candidate) -> None:
        """Persist a completed candidate durably and atomically (commit-marker written last)."""
        ...


@runtime_checkable
class StageCache(Protocol):
    """Atomic completed-stage storage within the adapter's versioned run-fingerprint scope."""

    def get_stage(self, stage: str) -> object | None:
        """Return a completed compatible payload, or None on uncertain/corrupt/incomplete data."""
        ...

    def put_stage(self, stage: str, value: object) -> None:
        """Persist a completed stage atomically within the trusted local cache."""
        ...
