"""Completed stage cache is atomic and incompatible state is a miss."""

from pathlib import Path

import joblib
import pytest

from honestml.adapters.candidate_cache import JoblibCandidateCache


@pytest.mark.parametrize("error", [AttributeError, ImportError, ModuleNotFoundError])
def test_unavailable_saved_class_is_a_miss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: type[Exception]
) -> None:
    cache = JoblibCandidateCache(tmp_path, "run")
    cache.put_stage("prepared", {"selected": ("a",)})

    def incompatible(path: object) -> object:
        raise error("saved class unavailable")

    monkeypatch.setattr(joblib, "load", incompatible)
    assert cache.get_stage("prepared") is None


def test_interrupted_stage_replacement_is_a_miss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = JoblibCandidateCache(tmp_path, "run")
    cache.put_stage("prepared", {"selected": ("a",)})

    def interrupted(*args: object) -> None:
        raise OSError("interrupted before marker")

    monkeypatch.setattr(cache, "_commit_meta", interrupted)
    with pytest.raises(OSError, match="interrupted"):
        cache.put_stage("prepared", {"selected": ("b",)})
    assert cache.get_stage("prepared") is None


def test_stage_scope_and_corrupt_marker(tmp_path: Path) -> None:
    cache = JoblibCandidateCache(tmp_path, "run")
    cache.put_stage("prepared", {"selected": ("a",)})
    assert cache.get_stage("prepared") == {"selected": ("a",)}
    assert JoblibCandidateCache(tmp_path, "other").get_stage("prepared") is None
    marker = tmp_path / "run" / "_stages" / "prepared" / "meta.json"
    marker.write_text("[]", encoding="utf-8")
    assert cache.get_stage("prepared") is None
