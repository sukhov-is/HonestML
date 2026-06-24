"""M9-2: human-readable run-report rendering (ADR-0075, FR-DLV-3, NFR-DLV-1/4).

The renderer is a pure consumer of the run-report JSON: markdown needs only the
stdlib, HTML embeds matplotlib charts (extra ``report``) and degrades gracefully
without them; unknown keys and legacy reports (missing additive keys) never break it.
"""

from __future__ import annotations

import pytest
from sklearn.datasets import make_classification

from honestml import AutoML, ConfigError, render_report, save_run_report
from honestml.composition import run_report as rr

pytestmark = pytest.mark.unit

_MINIMAL = {
    "run_manifest_version": 1,
    "honestml_version": "0.1.0",
    "config": {"seed": 0},
    "timings": {"run": {"selection": 1.25}},
    "winner": "linear",
    "holdout_score": 0.77,
    "leaderboard": [
        {"model_id": "linear", "score": 0.91, "rank": 1},
        {"model_id": "baseline", "score": 0.5, "rank": 2},
    ],
    "band": {"member_ids": ["linear"], "unstable": False, "width": 1, "winner_by_tiebreak": False},
    "budget": {"mode": "none", "exhausted": False, "skipped": [], "exhausted_by": None},
    "significance": "bootstrap",
}


def test_md_renders_v1_minimum_without_matplotlib(tmp_path) -> None:
    """NFR-DLV-4: a v1-minimum (legacy) report renders; missing additive keys -> n/a."""
    path = render_report(_MINIMAL, tmp_path, fmt="md")
    text = path.read_text(encoding="utf-8")
    assert path.name == "run_report.md"
    assert "linear (winner)" in text
    assert "| task | n/a |" in text  # legacy report: no additive task/metric keys
    assert "| holdout_score | 0.77 |" in text  # FR-DLV-3: the honest estimate is rendered
    assert "0.91" in text and "bootstrap" in text


def test_md_renders_native_routing_and_cv_blocks(tmp_path) -> None:
    """F113: native_routing and period-CV diagnostics are surfaced in the human-readable report,
    not only the JSON (ADR-0095 D-4 / ADR-0096 §4)."""
    report = dict(_MINIMAL)
    report["cv"] = {"period": "month", "n_periods": 12, "n_folds": 5, "n_dropped_empty": 1}
    report["native_routing"] = {
        "native": ["city"],
        "demoted_to_codes": [{"column": "user_id", "reason": "high_cardinality"}],
    }
    text = render_report(report, tmp_path, fmt="md").read_text(encoding="utf-8")
    assert "CV split" in text and "n_dropped_empty" in text
    assert "Native routing" in text and "user_id (high_cardinality)" in text


def test_md_escapes_user_named_models(tmp_path) -> None:
    """ADR-0075 §2: user names must not break md tables or smuggle raw HTML."""
    report = dict(_MINIMAL)
    report["winner"] = "evil|<script>alert(1)</script>"
    report["leaderboard"] = [{"model_id": report["winner"], "score": 0.9, "rank": 1}]
    text = render_report(report, tmp_path / "r.md", fmt="md").read_text(encoding="utf-8")
    assert "<script>" not in text
    assert "evil\\|&lt;script&gt;" in text


def test_md_escapes_backslash_pipe(tmp_path) -> None:
    """A user backslash must not re-open the pipe escape (GFM: \\\\ + raw |)."""
    report = dict(_MINIMAL)
    report["winner"] = "evil\\|x"
    text = render_report(report, tmp_path / "r.md", fmt="md").read_text(encoding="utf-8")
    assert "evil\\\\\\|x" in text  # backslash escaped first, then the pipe


def test_html_charts_survive_mathtext_names(tmp_path) -> None:
    """A '$...$' model/metric name must not crash matplotlib's mathtext parser."""
    pytest.importorskip("matplotlib")
    report = dict(_MINIMAL)
    report["metric"] = "profit $ per $row$"
    report["leaderboard"] = [{"model_id": "price $\\foo$ model", "score": 0.9, "rank": 1}]
    text = render_report(report, tmp_path, fmt="html").read_text(encoding="utf-8")
    assert "data:image/png;base64," in text


def test_failed_candidates_rendered(tmp_path) -> None:
    """F4.2: per-candidate failures are part of the human-readable summary."""
    report = dict(_MINIMAL)
    report["failed"] = [{"model_id": "xgb", "reason": "native crash"}]
    text = render_report(report, tmp_path / "f2.md", fmt="md").read_text(encoding="utf-8")
    assert "Failed candidates" in text and "native crash" in text
    clean = render_report(_MINIMAL, tmp_path / "clean.md", fmt="md").read_text(encoding="utf-8")
    assert "Failed candidates" not in clean


def test_round_trip_from_saved_json(tmp_path) -> None:
    json_path = save_run_report(dict(_MINIMAL), tmp_path)
    text = render_report(json_path, tmp_path, fmt="md").read_text(encoding="utf-8")
    assert "linear (winner)" in text


def test_unknown_keys_and_bad_fmt(tmp_path) -> None:
    report = {**_MINIMAL, "future_block": {"x": 1}}
    assert render_report(report, tmp_path / "f.md", fmt="md").exists()
    with pytest.raises(ConfigError, match="fmt must be"):
        render_report(_MINIMAL, tmp_path, fmt="pdf")


def test_html_without_matplotlib_degrades(tmp_path, monkeypatch, caplog) -> None:
    """FR-DLV-3 graceful degradation: chart-less HTML + WARNING, not a failure."""
    import logging

    monkeypatch.setattr(rr, "find_spec", lambda name: None)
    with caplog.at_level(logging.WARNING, logger="honestml"):
        text = render_report(_MINIMAL, tmp_path, fmt="html").read_text(encoding="utf-8")
    assert "data:image/png" not in text
    assert "linear (winner)" in text
    assert any("without charts" in r.message for r in caplog.records)


def test_html_is_self_contained_and_escaped(tmp_path) -> None:
    pytest.importorskip("matplotlib")
    report = dict(_MINIMAL)
    report["winner"] = "<img src=x onerror=alert(1)>"
    report["leaderboard"] = [{"model_id": report["winner"], "score": 0.9, "rank": 1}]
    text = render_report(report, tmp_path, fmt="html").read_text(encoding="utf-8")
    assert "data:image/png;base64," in text  # charts embedded
    assert "<img src=x" not in text  # user string escaped
    assert "&lt;img src=x" in text
    assert "http://" not in text and "https://" not in text and "<script" not in text


def test_facade_report_renders_with_task_metric_and_preset(tmp_path) -> None:
    """FR-DLV-3 + ADR-0075 §2: the additive task/metric keys arrive through a real fit."""
    X, y = make_classification(
        n_samples=60, n_features=6, n_informative=4, n_redundant=0, random_state=0
    )
    model = AutoML(task="binary", models=("baseline", "linear"), random_state=0, preset="fast")
    model.fit(X, y)
    assert model.run_report_["task"] == "binary"
    assert isinstance(model.run_report_["metric"], str)
    text = render_report(model.run_report_, tmp_path, fmt="md").read_text(encoding="utf-8")
    assert "| task | binary |" in text
    assert f"| metric | {model.run_report_['metric']} |" in text
    assert "fast (applied: cv)" in text
