"""Run-report I/O forms: JSON serialization + human-readable rendering.

Distinct from ``save_artifact`` (which writes a multi-file model directory): the run
report is one tracker-independent JSON document, assembled by the pure
``honestml.application.build_run_report`` and exposed as ``AutoML.run_report_``.
``render_report`` is a pure CONSUMER of that JSON (markdown always — stdlib only;
self-contained HTML with matplotlib charts when the ``report`` extra is installed,
chart-less HTML with a WARNING otherwise): it reads known keys via ``.get`` and
ignores unknown ones, so the report's additive evolution never breaks rendering.
"""

from __future__ import annotations

import base64
import html as _html
import io
import json
from collections.abc import Mapping
from importlib.util import find_spec
from pathlib import Path
from typing import Any

from honestml.core import ConfigError, get_logger

logger = get_logger("composition.run_report")

_RUN_REPORT_FILE = "run_report.json"


def save_run_report(report: dict[str, Any], path: str | Path, *, overwrite: bool = True) -> Path:
    """Write *report* as indented UTF-8 JSON, returning the written file path.

    If *path* is an existing directory, the report is written to
    ``path/run_report.json``; otherwise *path* is the file itself. With
    ``overwrite=False`` an existing target raises :class:`FileExistsError`.
    """
    target = Path(path)
    if target.is_dir():
        target = target / _RUN_REPORT_FILE
    if not overwrite and target.exists():
        raise FileExistsError(f"run report already exists at {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return target


def render_report(
    report: Mapping[str, Any] | str | Path, path: str | Path, *, fmt: str = "md"
) -> Path:
    """Render the run report as markdown or self-contained HTML.

    *report* is the ``run_report_`` mapping or a path to a saved ``run_report.json``
    (round-trip with :func:`save_run_report`). ``fmt="md"`` needs nothing beyond the
    stdlib; ``fmt="html"`` embeds matplotlib charts as base64 PNG when the ``report``
    extra is installed and degrades gracefully (WARNING, no charts) when it is not.
    If *path* is an existing directory the file is ``path/run_report.<fmt>``.
    """
    if fmt not in ("md", "html"):
        raise ConfigError(f"fmt must be 'md' or 'html', got {fmt!r}")
    if not isinstance(report, (Mapping, str, Path)):
        raise ConfigError(
            f"report must be a mapping or a path to run_report.json, got {type(report).__name__}"
        )
    data: Mapping[str, Any] = (
        report
        if isinstance(report, Mapping)
        else json.loads(Path(report).read_text(encoding="utf-8"))
    )
    target = Path(path)
    if target.is_dir():
        target = target / f"run_report.{fmt}"
    target.parent.mkdir(parents=True, exist_ok=True)
    text = _render_md(data) if fmt == "md" else _render_html(data)
    target.write_text(text, encoding="utf-8")
    return target


# --- shared section model (one content source for both formats) ---------------------------


def _cell(value: Any) -> str:
    """One displayable cell: compact floats, flat containers, 'n/a' for missing."""
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (list, tuple)):
        return ", ".join(_cell(item) for item in value) or "n/a"
    return str(value)


def _sections(report: Mapping[str, Any]) -> list[tuple[str, list[tuple[str, str]]]]:
    """The report as (title, [(label, value-cell)]) sections — keys read via .get only."""
    preset = report.get("preset")
    sections: list[tuple[str, list[tuple[str, str]]]] = [
        (
            "Run",
            [
                ("task", _cell(report.get("task"))),
                ("metric", _cell(report.get("metric"))),
                ("winner", _cell(report.get("winner"))),
                ("holdout_score", _cell(report.get("holdout_score"))),
                ("honestml_version", _cell(report.get("honestml_version"))),
                ("run_fingerprint", _cell(report.get("run_fingerprint"))),
                ("significance", _cell(report.get("significance"))),
                (
                    "preset",
                    "n/a"
                    if preset is None
                    else f"{preset.get('name') or '<custom>'} (applied: {_cell(preset.get('applied'))})",
                ),
            ],
        )
    ]
    band = report.get("band") or {}
    if band:
        sections.append(
            (
                "Equivalence band",
                [
                    ("members", _cell(band.get("member_ids"))),
                    ("width", _cell(band.get("width"))),
                    ("unstable", _cell(band.get("unstable"))),
                    ("winner_by_tiebreak", _cell(band.get("winner_by_tiebreak"))),
                ],
            )
        )
    optimism = report.get("holdout_optimism")
    if optimism:
        sections.append(
            (
                "Holdout diagnostic",
                [
                    ("winner_oof", _cell(optimism.get("winner_oof"))),
                    ("holdout", _cell(optimism.get("holdout"))),
                    ("relative_optimism", _cell(optimism.get("relative_optimism"))),
                    ("note", _cell(optimism.get("message"))),
                ],
            )
        )
    budget = report.get("budget") or {}
    if budget:
        sections.append(
            (
                "Budget",
                [
                    ("mode", _cell(budget.get("mode"))),
                    ("exhausted", _cell(budget.get("exhausted"))),
                    ("exhausted_by", _cell(budget.get("exhausted_by"))),
                    ("skipped", _cell(budget.get("skipped"))),
                ],
            )
        )
    cv = report.get("cv")
    if cv:
        sections.append(
            (
                "CV split",
                [
                    ("period", _cell(cv.get("period"))),
                    ("n_periods", _cell(cv.get("n_periods"))),
                    ("n_folds", _cell(cv.get("n_folds"))),
                    ("n_dropped_empty", _cell(cv.get("n_dropped_empty"))),
                ],
            )
        )
    fs = report.get("feature_selection")
    if fs:
        sections.append(
            (
                "Feature selection",
                [
                    ("strategy", _cell(fs.get("strategy"))),
                    ("n_selected", _cell(fs.get("n_selected"))),
                    ("winner", _cell(fs.get("winner"))),
                ],
            )
        )
    routing = report.get("native_routing")
    if routing:
        demoted = routing.get("demoted_to_codes") or []
        sections.append(
            (
                "Native routing",
                [
                    ("native", _cell(routing.get("native"))),
                    (
                        "demoted_to_codes",
                        _cell([f"{d.get('column')} ({d.get('reason')})" for d in demoted]),
                    ),
                ],
            )
        )
    hpo = report.get("hpo")
    if hpo:
        tuned = hpo.get("tuned") or {}
        sections.append(
            (
                "HPO",
                [
                    ("backend", _cell(hpo.get("backend"))),
                    ("tuned models", _cell(sorted(tuned))),
                    ("cost_estimate_fits", _cell(hpo.get("cost_estimate_fits"))),
                    ("deterministic", _cell(hpo.get("deterministic"))),
                ],
            )
        )
    ensemble = report.get("ensemble")
    if ensemble:
        sections.append(
            (
                "Ensemble",
                [
                    ("applied", _cell(ensemble.get("applied"))),
                    ("method", _cell(ensemble.get("method"))),
                    ("members", _cell(ensemble.get("member_ids"))),
                    ("gate_reason", _cell(ensemble.get("gate_reason"))),
                    ("oof_delta", _cell(ensemble.get("oof_delta"))),
                ],
            )
        )
    serving = report.get("serving")
    if serving:
        sections.append(
            (
                "Serving",
                [
                    ("finalize", _cell(serving.get("finalize"))),
                    ("shipped_on", _cell(serving.get("shipped_on"))),
                    ("outer_holdout", _cell(serving.get("outer_holdout"))),
                ],
            )
        )
    failed = report.get("failed") or []
    if failed:
        sections.append(
            (
                "Failed candidates",
                [(str(f.get("model_id")), _cell(f.get("reason"))) for f in failed],
            )
        )
    cache = report.get("cache") or {}
    if cache.get("enabled"):
        sections.append(
            (
                "Cache",
                [
                    ("reused", _cell(len(cache.get("reused") or []))),
                    ("computed", _cell(len(cache.get("computed") or []))),
                ],
            )
        )
    return sections


def _timing_rows(report: Mapping[str, Any]) -> list[tuple[str, float]]:
    return [
        (f"{group}.{stage}", float(elapsed))
        for group, stages in (report.get("timings") or {}).items()
        for stage, elapsed in stages.items()
    ]


def _leaderboard_rows(report: Mapping[str, Any]) -> list[tuple[str, str, str]]:
    winner = report.get("winner")
    return [
        (
            _cell(entry.get("rank")),
            str(entry.get("model_id")) + (" (winner)" if entry.get("model_id") == winner else ""),
            _cell(entry.get("score")),
        )
        for entry in report.get("leaderboard") or []
    ]


# --- markdown ------------------------------------------------------------------------------


def _md(value: str) -> str:
    # md-cell escaping (ADR-0075 §2): user-named models/features must not break tables
    # or smuggle raw HTML into viewers (GitHub renders inline HTML in .md); the backslash
    # goes FIRST or a user "\|" would re-emerge as an escaped backslash + raw pipe
    return (
        value.replace("\\", "\\\\")
        .replace("|", "\\|")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", " ")
    )


def _render_md(report: Mapping[str, Any]) -> str:
    lines = ["# AutoML run report", ""]
    for title, rows in _sections(report):
        lines += [f"## {title}", "", "| | |", "|---|---|"]
        lines += [f"| {_md(label)} | {_md(value)} |" for label, value in rows]
        lines.append("")
    lines += ["## Leaderboard", "", "| rank | model | score |", "|---|---|---|"]
    lines += [f"| {_md(r)} | {_md(m)} | {_md(s)} |" for r, m, s in _leaderboard_rows(report)]
    lines.append("")
    timings = _timing_rows(report)
    if timings:
        lines += ["## Timings (s)", "", "| stage | elapsed |", "|---|---|"]
        lines += [f"| {_md(stage)} | {_md(_cell(elapsed))} |" for stage, elapsed in timings]
        lines.append("")
    lines += [
        "## Resolved config",
        "",
        "```json",
        json.dumps(report.get("config") or {}, indent=2),
        "```",
        "",
    ]
    return "\n".join(lines)


# --- html (self-contained; user strings only in text nodes / quoted attributes) -----------

_CSS = (
    "body{font-family:system-ui,sans-serif;margin:2rem;max-width:60rem}"
    "table{border-collapse:collapse;margin:0.5rem 0}"
    "td,th{border:1px solid #ccc;padding:0.25rem 0.6rem;text-align:left}"
    "h1,h2{color:#223}img{max-width:100%}pre{background:#f6f6f6;padding:0.6rem;overflow-x:auto}"
)


def _render_html(report: Mapping[str, Any]) -> str:
    e = _html.escape
    parts = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        f"<title>AutoML run report</title><style>{_CSS}</style></head><body>",
        "<h1>AutoML run report</h1>",
    ]
    for title, rows in _sections(report):
        parts.append(f"<h2>{e(title)}</h2><table>")
        parts += [f"<tr><th>{e(label)}</th><td>{e(value)}</td></tr>" for label, value in rows]
        parts.append("</table>")
    parts.append("<h2>Leaderboard</h2><table><tr><th>rank</th><th>model</th><th>score</th></tr>")
    parts += [
        f"<tr><td>{e(r)}</td><td>{e(m)}</td><td>{e(s)}</td></tr>"
        for r, m, s in _leaderboard_rows(report)
    ]
    parts.append("</table>")
    for chart_title, png_b64 in _charts(report):
        parts.append(
            f"<h2>{e(chart_title)}</h2>"
            f'<img alt="{e(chart_title)}" src="data:image/png;base64,{png_b64}"/>'
        )
    timings = _timing_rows(report)
    if timings:
        parts.append("<h2>Timings (s)</h2><table><tr><th>stage</th><th>elapsed</th></tr>")
        parts += [f"<tr><td>{e(s)}</td><td>{e(_cell(t))}</td></tr>" for s, t in timings]
        parts.append("</table>")
    config_json = json.dumps(report.get("config") or {}, indent=2)
    parts += [f"<h2>Resolved config</h2><pre>{e(config_json)}</pre>", "</body></html>"]
    return "".join(parts)


def _chart_label(value: Any) -> str:
    # "$...$" would trigger matplotlib's mathtext parser and CRASH the render on a
    # user-named model/metric — neutralize the math semantics, keep the characters
    return str(value).replace("$", r"\$")


def _charts(report: Mapping[str, Any]) -> list[tuple[str, str]]:
    """Base64-PNG charts via matplotlib (extra ``report``); [] + WARNING when absent.

    Uses the object API only (``Figure`` + per-figure Agg canvas): no ``pyplot``, no
    ``matplotlib.use(...)`` — rendering a report must not flip the global backend of
    the calling process (e.g. a notebook's inline backend).
    """
    if find_spec("matplotlib") is None:
        logger.warning(
            "matplotlib is not installed (pip install honestml[report]): "
            "rendering the HTML report without charts"
        )
        return []
    from matplotlib.figure import Figure

    charts: list[tuple[str, str]] = []
    leaderboard = report.get("leaderboard") or []
    if leaderboard:
        fig = Figure(figsize=(7, 0.5 + 0.4 * len(leaderboard)))
        ax = fig.subplots()
        names = [_chart_label(entry.get("model_id")) for entry in leaderboard]
        scores = [float(entry.get("score", 0.0)) for entry in leaderboard]
        ax.barh(names[::-1], scores[::-1], color="#4a7ebb")
        ax.set_xlabel(_chart_label(report.get("metric") or "score"))
        charts.append(("Leaderboard scores", _png_b64(fig)))
    timings = _timing_rows(report)
    if timings:
        fig = Figure(figsize=(7, 0.5 + 0.4 * len(timings)))
        ax = fig.subplots()
        ax.barh(
            [_chart_label(stage) for stage, _ in timings][::-1],
            [elapsed for _, elapsed in timings][::-1],
            color="#7bb074",
        )
        ax.set_xlabel("elapsed, s")
        charts.append(("Stage timings", _png_b64(fig)))
    return charts


def _png_b64(fig: Any) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight", dpi=110)
    return base64.b64encode(buffer.getvalue()).decode("ascii")
