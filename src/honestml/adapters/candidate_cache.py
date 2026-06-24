"""The ``JoblibCandidateCache`` adapter: durable, atomic per-candidate stage cache (ADR-0036 §2).

Implements the :class:`honestml.core.CandidateCache` port over the filesystem + joblib. Layout:
``cache_dir/<fingerprint>/<candidate_id>/{entry.joblib, meta.json}``. ``entry.joblib`` holds the whole
``Candidate`` (scalars + numpy OOF; joblib serializes numpy natively). ``meta.json`` is the
**commit-marker, written LAST**: a crash between the two leaves an entry without meta, so ``get``
treats it as a miss and recomputes — a half-entry is never falsely valid (R-CRASH).

Writes are **atomic** (``tmp`` + ``os.replace`` on one volume; tmp names uniquified by PID + uuid so
concurrent processes never collide on tmp — a safety net for the one-process-per-dir contract,
R-CONCURRENT). ``get`` is **fail-closed**: missing meta, an incompatible ``CACHE_VERSION`` or any
deserialization error -> ``None`` (+ WARNING), never an exception out (NFR-RC-1/4).

SECURITY: ``entry.joblib`` is deserialized via joblib/pickle — the SAME trust boundary as
``load_artifact``: load a cache only from a **trusted directory**. The ``CACHE_VERSION`` check is a
forward-compatibility gate, **not** an integrity/authenticity check; signing/integrity is M8.
``candidate_id`` is confined to its directory via ``Path(...).name`` (anti-traversal); the
``<fingerprint>`` segment is a hex digest (NFR-RC-5).
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from honestml.core import get_logger

if TYPE_CHECKING:
    from honestml.core import Candidate

CACHE_VERSION = 1
_ENTRY_FILE = "entry.joblib"
_META_FILE = "meta.json"

logger = get_logger("adapters.candidate_cache")


class JoblibCandidateCache:
    """Durable per-candidate cache under ``cache_dir/<fingerprint>/`` (one process per directory)."""

    def __init__(self, cache_dir: str | Path, fingerprint: str) -> None:
        # <fingerprint> is a hex digest (safe); confine it with .name defensively all the same
        self._root = Path(cache_dir) / Path(fingerprint).name

    def get(self, candidate_id: str) -> Candidate | None:
        import joblib

        entry_dir = self._candidate_dir(candidate_id)
        meta_path = entry_dir / _META_FILE
        if not meta_path.exists():
            return None  # no commit-marker -> miss (covers crash-before-meta), fail-closed
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("cache_version") != CACHE_VERSION:
                return None  # incompatible entry format -> miss, not a crash
            candidate: Candidate = joblib.load(entry_dir / _ENTRY_FILE)
        except Exception as exc:  # corrupt/unreadable entry -> fail-closed miss + WARNING
            logger.warning("cache entry %r unreadable (%s); recomputing", candidate_id, exc)
            return None
        return candidate

    def put(self, candidate_id: str, candidate: Candidate) -> None:
        import joblib

        entry_dir = self._candidate_dir(candidate_id)
        entry_dir.mkdir(parents=True, exist_ok=True)
        # entry first (atomic); meta.json (commit-marker) LAST (atomic) — see module docstring
        tmp_entry = self._tmp(entry_dir, _ENTRY_FILE)
        joblib.dump(candidate, tmp_entry)
        os.replace(tmp_entry, entry_dir / _ENTRY_FILE)
        self._commit_meta(entry_dir, candidate_id)

    def _commit_meta(self, entry_dir: Path, candidate_id: str) -> None:
        """Write ``meta.json`` (the commit-marker) atomically and LAST — a testable crash seam.

        Isolating this lets a test crash ``put`` *after* ``entry.joblib`` but *before* ``meta.json``
        and assert ``get`` -> ``None`` through the real code path (ADR-0036 §2)."""
        meta = {
            "cache_version": CACHE_VERSION,
            "candidate_id": candidate_id,
            "fingerprint": self._root.name,
        }
        tmp_meta = self._tmp(entry_dir, _META_FILE)
        tmp_meta.write_text(json.dumps(meta), encoding="utf-8")
        os.replace(tmp_meta, entry_dir / _META_FILE)

    def _candidate_dir(self, candidate_id: str) -> Path:
        # confine to the cache root: a candidate id contributes only its basename (anti-traversal)
        return self._root / Path(candidate_id).name

    @staticmethod
    def _tmp(entry_dir: Path, name: str) -> Path:
        # PID + uuid: concurrent tmp writes never collide (R-CONCURRENT safety net)
        return entry_dir / f".{name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
