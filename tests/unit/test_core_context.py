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


def test_nested_repeated_stages_reconcile_with_independent_total(monkeypatch) -> None:
    now = [10.0]
    monkeypatch.setattr("honestml.core.context.time.perf_counter", lambda: now[0])
    ctx = RunContext()
    now[0] = 11.0
    with ctx.timed_stage("run", "selection"):
        now[0] = 12.0
        with ctx.timed_stage("run", "refit"):
            now[0] = 14.0
        now[0] = 15.0
    with ctx.timed_stage("run", "refit"):
        now[0] = 18.0
    now[0] = 20.0
    ctx.finish_run()
    cost = ctx.cost_report()
    assert ctx.timings["run"] == {"selection": 4.0, "refit": 5.0}
    assert cost["exclusive_timings"]["run"] == {"selection": 2.0, "refit": 5.0}
    assert cost["total_wall_s"] == 10.0
    assert cost["overhead_s"] == 3.0
    assert len(cost["stages"]) == 3
    assert ctx.total_time("run") == 7.0
    now[0] = 30.0
    assert ctx.cost_report()["total_wall_s"] == 10.0


def test_failed_fit_records_only_aggregate_resources(monkeypatch) -> None:
    now = [0.0]
    monkeypatch.setattr("honestml.core.context.time.perf_counter", lambda: now[0])
    ctx = RunContext()
    with pytest.raises(ValueError, match="training failed"):
        with ctx.timed_fit("hpo", model_id="linear", rows=30, columns=4, trial=2, fold=1):
            now[0] = 0.25
            raise ValueError("training failed")
    cost = ctx.cost_report()
    assert cost["fit_counts"] == {"attempted": 1, "completed": 0, "failed": 1}
    assert cost["work"][0]["elapsed_s"] == 0.25
    assert cost["work"][0]["rows"] == 30
    assert cost["work"][0]["trial"] == 2
    assert "training failed" not in json.dumps(cost)


def test_cached_time_is_not_current_wall_time(monkeypatch) -> None:
    now = [0.0]
    monkeypatch.setattr("honestml.core.context.time.perf_counter", lambda: now[0])
    ctx = RunContext()
    ctx.record_stage_time("model", "cv", 100.0)
    now[0] = 1.0
    cost = ctx.cost_report()
    assert cost["total_wall_s"] == cost["overhead_s"] == 1.0
    assert cost["attributed_wall_s"] == 0.0
    assert cost["stages"][0]["source"] == "cache"


def test_cost_snapshot_is_independent_and_tracks_sampled_rss(monkeypatch) -> None:
    now = [0.0]
    monkeypatch.setattr("honestml.core.context.time.perf_counter", lambda: now[0])
    rss = [50.0]
    ctx = RunContext(rss_probe=lambda: rss[0])
    with ctx.timed_fit("cv", model_id="tree", rows=40, columns=2) as resources:
        resources["tree_budget"] = 100
        resources["iterations"] = 15
        rss[0] = 80.0
        now[0] = 0.1
    cost = ctx.cost_report()
    assert cost["sampled_peak_rss_mb"] == 80.0
    assert cost["work"][0]["iterations"] == 15
    cost["work"][0]["iterations"] = 999
    assert ctx.cost_report()["work"][0]["iterations"] == 15


def test_rejected_before_fit_hook_does_not_record_an_attempt() -> None:
    def reject(stage: str) -> None:
        assert stage == "fs"
        raise RuntimeError("budget exhausted")

    ctx = RunContext(before_fit=reject)
    with pytest.raises(RuntimeError, match="budget exhausted"):
        with ctx.timed_fit("fs", model_id="ranker", rows=30, columns=2):
            pytest.fail("a rejected fit must not start")
    assert ctx.cost_report()["fit_counts"] == {"attempted": 0, "completed": 0, "failed": 0}


def test_fit_attribution_is_nested_exception_safe_and_context_local() -> None:
    ctx, other = RunContext(), RunContext()
    with ctx.fit_attribution(recipe="sequential", fold=2):
        with pytest.raises(ValueError):
            with ctx.fit_attribution(recipe="prefilter", fold=0):
                with ctx.timed_fit("fs", model_id="proxy", rows=10, columns=4):
                    raise ValueError("failed fit")
        with ctx.timed_fit("fs", model_id="proxy", rows=10, columns=3):
            pass
        with other.timed_fit("fs", model_id="proxy", rows=10, columns=3):
            pass
    with ctx.timed_fit("cv", model_id="model", rows=10, columns=3, fold=1):
        pass
    work = ctx.cost_report()["work"]
    assert [(row["recipe"], row["fold"]) for row in work] == [
        ("prefilter", 0),
        ("sequential", 2),
        (None, 1),
    ]
    assert other.cost_report()["work"][0]["recipe"] is None
    assert work[0]["status"] == "failed"
