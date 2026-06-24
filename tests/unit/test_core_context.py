"""M0-2: RunContext replaces module globals — two runs in one process are independent."""

from __future__ import annotations

import json

import pytest

from honestml.core import RunContext

pytestmark = pytest.mark.unit


def test_two_contexts_are_independent() -> None:
    a, b = RunContext(), RunContext()
    with a.timed_stage("combo", "stage"):
        pass
    assert a.timings
    # b shares no state with a (no module-level globals)
    assert not b.timings


def test_timed_stage_records_elapsed() -> None:
    ctx = RunContext()
    with ctx.timed_stage("k", "s"):
        pass
    assert "s" in ctx.timings["k"]
    assert ctx.timings["k"]["s"] >= 0.0


def test_total_time_sums_stages() -> None:
    ctx = RunContext()
    ctx.record_stage_time("k", "a", 1.5)
    ctx.record_stage_time("k", "b", 2.0)
    assert ctx.total_time("k") == pytest.approx(3.5)


def test_manifest_is_json_serializable() -> None:
    ctx = RunContext()
    ctx.record_stage_time("k", "s", 1.0)
    manifest = ctx.manifest()
    dumped = json.dumps(manifest)  # must not raise
    assert "config" in manifest and "timings" in manifest
    assert json.loads(dumped)["timings"]["k"]["s"] == 1.0
