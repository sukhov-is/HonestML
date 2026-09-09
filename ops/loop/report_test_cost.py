"""Rank what the last test run cost, so a suite growing expensive is visible while it is one case.

Cost concentrates. A suite is rarely slow because it holds many cases; it is slow because a few
of them are expensive, and the cheap ones stay cheap however many accumulate. The share column
is what turns a ranking into a decision: a case holding a large part of the work is worth
reading, one holding a fraction of a percent is not, whatever its absolute time.

A case over the threshold is reported unless the budget accepts it. ops/loop/test-cost-budget.json
declares, per pytest id, the ceiling a case may cost and the reason that cost is honest — a full
pipeline it has to fit, a fold matrix it has to replay. One entry reads
"<pytest id>": {"seconds": <ceiling>, "reason": "<what the cost buys>"}. Accepted cases are
counted rather than listed, so what stays on screen is what nobody has judged yet, or what grew
past the ceiling its reason justified. A ceiling carries headroom over the observed cost: it is a
tripwire for a regression, not a target to tune against.

Every report carries the command that replays one case, so confirming that a fix removed the
cost is one call rather than a full suite.

    uv run python ops/loop/report_test_cost.py [--threshold-seconds 2]
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "ops/loop/logs/pytest-junit.xml"
BUDGET = ROOT / "ops/loop/test-cost-budget.json"


def node_id(case: ET.Element) -> str:
    """The pytest id of a case, in the form pytest itself takes back."""
    path = (case.get("file") or "").replace("\\", "/")
    parts = (case.get("classname") or "").split(".")
    stem = Path(path).stem
    classes = parts[parts.index(stem) + 1 :] if stem in parts else []
    return "::".join([path, *classes, case.get("name") or ""])


def seconds(case: ET.Element) -> float:
    return float(case.get("time") or 0.0)


def load_budget() -> dict[str, dict[str, object]]:
    if not BUDGET.is_file():
        return {}
    accepted: dict[str, dict[str, object]] = json.loads(BUDGET.read_text(encoding="utf-8"))
    return accepted


def verdict(node: str, cost: float, budget: dict[str, dict[str, object]]) -> str | None:
    """Why this case asks for a decision, or None when the budget already answered."""
    accepted = budget.get(node)
    if accepted is None:
        return "not budgeted"
    ceiling = float(str(accepted["seconds"]))
    if cost < ceiling:
        return None
    return f"over its {ceiling:g}s ceiling for: {accepted['reason']}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--threshold-seconds",
        type=float,
        default=2.0,
        help="cost above which a case is reported; raise it when the list grows past acting on",
    )
    threshold = parser.parse_args().threshold_seconds
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    if not RESULTS.is_file():
        print(f"No {RESULTS.relative_to(ROOT).as_posix()} — run ops/loop/gates.ps1 first.")
        return 0
    suite = ET.parse(RESULTS).getroot().find("testsuite")
    if suite is None:
        print(f"{RESULTS.relative_to(ROOT).as_posix()} records no suite.")
        return 0
    cases = suite.findall("testcase")
    if not cases:
        print(f"{RESULTS.relative_to(ROOT).as_posix()} records no cases.")
        return 0

    budget = load_budget()
    if not budget:
        print(
            "ops/loop/test-cost-budget.json accepts nothing yet — "
            "every case over the threshold is unbudgeted."
        )
    wall = float(suite.get("time") or 0.0)
    case_time = sum(seconds(case) for case in cases)
    print(
        f"--- {suite.get('name', '?')} run {suite.get('timestamp', '?')} | {len(cases)} cases "
        f"| wall {wall:.1f}s | case time {case_time:.1f}s | threshold {threshold:g}s"
    )

    reportable: list[tuple[float, str, str]] = []
    within = 0
    for case in sorted(cases, key=seconds, reverse=True):
        cost = seconds(case)
        if cost < threshold:
            break
        node = node_id(case)
        if (reason := verdict(node, cost, budget)) is None:
            within += 1
        else:
            reportable.append((cost, node, reason))

    if not reportable:
        print(f"    nothing over {threshold:g}s beyond the budget ({within} accepted)")
        return 0
    for cost, node, reason in reportable:
        print(f"  {cost:8.1f}s {cost / case_time * 100:5.1f}%  {node} — {reason}")
    print(
        f"    {len(reportable)} of {len(cases)} cases ask for a decision, "
        f"{within} accepted by the budget"
    )
    print(f'    replay one: uv run pytest "{reportable[0][1]}" -q')
    return 0


if __name__ == "__main__":
    sys.exit(main())
