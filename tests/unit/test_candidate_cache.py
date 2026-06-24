"""M5-resume RC-b: the ``JoblibCandidateCache`` adapter (ADR-0036 §2, FR-RC-2/3/5, NFR-RC-1/4/5).

Durable, atomic, fail-closed per-candidate cache on disk. Round-trips a ``Candidate`` (with numpy
OOF), treats a missing/incompatible/corrupt entry as a miss (never an exception), commits the meta
marker last (a crash before it is a miss), and confines candidate ids to the cache directory.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from honestml.adapters import CACHE_VERSION, JoblibCandidateCache
from honestml.core import Candidate

pytestmark = pytest.mark.unit


def _candidate(id_: str = "m", score: float = 0.8) -> Candidate:
    n = 6
    return Candidate(
        id=id_,
        score=score,
        n_features=3,
        train_time=0.12,
        oof_pred=np.linspace(0.0, 1.0, n),
        oof_mask=np.ones(n, dtype=bool),
        oof_proba=np.linspace(0.0, 1.0, n),
    )


def test_put_get_round_trips_oof(tmp_path) -> None:
    cache = JoblibCandidateCache(tmp_path, "fp")
    cand = _candidate()
    cache.put("m", cand)
    got = cache.get("m")
    assert got is not None
    assert got.id == cand.id and got.score == cand.score and got.n_features == cand.n_features
    np.testing.assert_array_equal(got.oof_pred, cand.oof_pred)
    np.testing.assert_array_equal(got.oof_mask, cand.oof_mask)
    np.testing.assert_array_equal(got.oof_proba, cand.oof_proba)


def test_miss_on_absent_id(tmp_path) -> None:
    assert JoblibCandidateCache(tmp_path, "fp").get("never-written") is None


def test_no_meta_is_miss(tmp_path) -> None:
    cache = JoblibCandidateCache(tmp_path, "fp")
    cache.put("m", _candidate())
    (tmp_path / "fp" / "m" / "meta.json").unlink()  # entry.joblib without commit-marker
    assert cache.get("m") is None


def test_crash_before_meta_is_miss(tmp_path) -> None:
    """Crash after entry.joblib, before meta.json -> get -> None through the real path (R-CRASH)."""
    cache = JoblibCandidateCache(tmp_path, "fp")

    def boom(*_a, **_k):
        raise RuntimeError("crash after entry, before meta")

    cache._commit_meta = boom  # type: ignore[method-assign]
    with pytest.raises(RuntimeError):
        cache.put("m", _candidate())
    assert (tmp_path / "fp" / "m" / "entry.joblib").exists()  # entry was written
    assert JoblibCandidateCache(tmp_path, "fp").get("m") is None  # but no commit -> miss


def test_version_mismatch_is_miss(tmp_path) -> None:
    cache = JoblibCandidateCache(tmp_path, "fp")
    cache.put("m", _candidate())
    (tmp_path / "fp" / "m" / "meta.json").write_text(
        json.dumps({"cache_version": CACHE_VERSION + 999}), encoding="utf-8"
    )
    assert cache.get("m") is None  # incompatible format -> miss, not a crash


def test_corrupt_entry_is_miss(tmp_path) -> None:
    cache = JoblibCandidateCache(tmp_path, "fp")
    cache.put("m", _candidate())
    (tmp_path / "fp" / "m" / "entry.joblib").write_bytes(b"not a joblib payload")
    assert cache.get("m") is None  # fail-closed; deserialization error swallowed, no exception out


def test_reput_overwrites_both_files(tmp_path) -> None:
    cache = JoblibCandidateCache(tmp_path, "fp")
    cache.put("m", _candidate(score=0.5))
    cache.put("m", _candidate(score=0.9))
    got = cache.get("m")
    assert got is not None and got.score == 0.9
    d = tmp_path / "fp" / "m"
    assert (d / "entry.joblib").exists() and (d / "meta.json").exists()


def test_fingerprint_scopes_entries(tmp_path) -> None:
    JoblibCandidateCache(tmp_path, "fpA").put("m", _candidate())
    assert (
        JoblibCandidateCache(tmp_path, "fpB").get("m") is None
    )  # different fp -> different subdir


def test_id_path_traversal_confined(tmp_path) -> None:
    cache = JoblibCandidateCache(tmp_path, "fp")
    cache.put("../evil", _candidate())
    assert not (tmp_path / "evil").exists()  # nothing escaped the cache root
    assert (tmp_path / "fp" / "evil").exists()  # confined under <fingerprint>/
    assert cache.get("../evil") is not None  # same basename -> round-trips


def test_no_tmp_files_left_after_put(tmp_path) -> None:
    cache = JoblibCandidateCache(tmp_path, "fp")
    cache.put("m", _candidate())
    leftovers = list((tmp_path / "fp" / "m").glob(".*.tmp"))
    assert leftovers == []  # os.replace consumed both tmp files (atomic)
