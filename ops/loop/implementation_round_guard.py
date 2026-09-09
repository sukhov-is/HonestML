"""Hook guard for completing autonomous implementation rounds.

The markers below are state signals, not evidence. What a round actually proved is checked by
ops/loop/gates.ps1. This guard only refuses to let a round stop while its own bookkeeping still
says the work is unfinished.
"""

from __future__ import annotations

import io
import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STATE_PATH = ROOT / "docs/loop/state.md"
BLOCKERS_PATH = ROOT / "docs/loop/blockers.md"
ACTIVE_ROUND_PATH = ROOT / "ops/loop/logs/active-round.json"
NEEDS_FIX = "needs-fix"
# The only obstacle that ends an IMPLEMENT round short of done: something outside the round's
# reach. It is declared, in the round's own note, at the point the round claims it.
EXTERNAL_BLOCKER = "EXTERNAL-BLOCKER:"
ACCEPTED_RISK = re.compile(r"accepted[- ]risk", re.IGNORECASE)
RISK_REFERENCE = re.compile(r"\b(?:RISK|R)-[A-Z0-9][A-Z0-9-]*\b", re.IGNORECASE)
EMPTY_EVIDENCE = {"", "-", "—", "planned"}
# Requirement, driver, ADR, implementation, verification, status — the narrowest matrix row.
MATRIX_ROW_CELLS = 5
# Chronicle limits of the two ledger tables (LOOP.md §5); the ledger and its hook share them.
ROUNDS_NOTE_LIMIT = 500
ITEMS_NOTE_LIMIT = 200
# every tool either driver can write the ledger with
WRITE_TOOLS = ("Bash", "Edit", "Write", "PowerShell", "shell_command", "apply_patch")

# Completion markers this guard enforces. The implementation skill owns them and instructs the
# model to emit these strings; it ships once per runtime, so both copies must agree with the
# guard and with each other. contract_drift() binds them.
MILESTONES_MARKER = "Implementation milestones complete: yes"
REVIEW_COLLECTION_MODE_MARKER = "Review collection mode: provider-native, non-polling"
REVIEW_STATE_HEADING = "## Review state"
REVIEW_PASSES_HEADING = "## Review passes"
REVIEW_FINDINGS_HEADING = "## Review findings"
FULL_REVIEW_WAVES_MARKER = "Full review waves: 1"
REGRESSION_REVIEW_WAVES_MARKER = "Fresh regression review waves: 1"
REVIEW_PROTOCOL_MARKER = "Review protocol: complete"
VERDICT_MARKER = "Final verdict: ACCEPT"
FINDINGS_MARKER = "Open actionable findings: 0"
REVIEW_MARKERS = (
    REVIEW_STATE_HEADING,
    REVIEW_PASSES_HEADING,
    REVIEW_FINDINGS_HEADING,
    FULL_REVIEW_WAVES_MARKER,
    REGRESSION_REVIEW_WAVES_MARKER,
    REVIEW_PROTOCOL_MARKER,
    VERDICT_MARKER,
    FINDINGS_MARKER,
)
CONTRACT_MARKERS = (MILESTONES_MARKER, REVIEW_COLLECTION_MODE_MARKER, *REVIEW_MARKERS)
CONTRACT_SOURCES = (
    ".CLAUDE/skills/implementation/SKILL.md",
    ".agents/skills/implementation/SKILL.md",
)


def state_path() -> Path:
    """Where this round's ledger lives. One loop, one ledger — the seam is here for when
    that stops being true, so nothing else has to learn a second answer."""
    return STATE_PATH


def blockers_path() -> Path:
    return BLOCKERS_PATH


def split_cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def table_rows(text: str) -> list[list[str]]:
    return [
        cells
        for line in text.splitlines()
        if line.strip().startswith("|")
        if (cells := split_cells(line))
        if not all(char in "-: " for char in "".join(cells))
    ]


def declares_external_blocker(note: str) -> bool:
    return note.lstrip("*_` ").startswith(EXTERNAL_BLOCKER)


def added_status_violations(lines: list[str]) -> list[str]:
    """What a single freshly written ledger line can be judged on by itself.

    The round row and the item row arrive as two separate ledger calls, so their agreement is
    only checkable once the round stops. What IS self-contained is the reason: a round claiming
    needs-fix must name the obstacle in its own note, on the line that claims it.
    """
    violations: list[str] = []
    for cells in table_rows("\n".join(lines)):
        if cells[0].isdigit() and len(cells) >= 9 and cells[7] == NEEDS_FIX:
            if not declares_external_blocker(cells[8]):
                violations.append(
                    f"round {cells[0]} records {NEEDS_FIX} without declaring an obstacle: "
                    f"start the note with {EXTERNAL_BLOCKER!r}. Running out of budget or turns "
                    f"is not an obstacle — finish the round."
                )
    return violations


def latest_round(rows: list[list[str]]) -> list[str] | None:
    rounds = [cells for cells in rows if cells[0].isdigit() and len(cells) >= 8]
    return rounds[-1] if rounds else None


def item_row(rows: list[list[str]], item_id: str) -> list[str] | None:
    return next((cells for cells in rows if cells[0] == item_id and len(cells) >= 4), None)


def architecture_package(cells: list[str]) -> Path | None:
    match = re.search(r"(docs/architecture/[^\s`]+/)", cells[2])
    return ROOT / match.group(1) if match else None


def is_accepted_risk(cells: list[str]) -> bool:
    return bool(ACCEPTED_RISK.search(" ".join(cells)))


def has_evidence(cell: str) -> bool:
    return cell.strip().lower() not in EMPTY_EVIDENCE


def requirement_id(cell: str) -> str | None:
    """The FR/NFR id a matrix cell names, whatever emphasis it wears."""
    token = cell.strip().lstrip("*_`~ ").split(" ")[0].strip("*_`~")
    return token if token.startswith(("FR-", "NFR-")) else None


def is_closed_mark(cell: str) -> bool:
    """Whether a status cell says the row is closed.

    A matrix names the slice that closed the row next to the mark — `☑ impl-2`, `☑ B-impl-1`,
    `☑ · parity deferred (onnx extra)`. The annotation is the row's history; the mark is its state.
    """
    return cell.strip().startswith("☑")


def is_status_cell(cell: str) -> bool:
    """A matrix row ends in a status; rows of the package's other tables end in prose.

    Coverage and handoff tables list the same FR/NFR ids with an owning slice instead of a
    status, so reading them as matrix rows reports requirements that were never in the matrix.
    """
    value = cell.strip()
    return value in {"planned", "accepted-risk"} or value.startswith(("☑", "☐", "◐"))


def is_matrix_row(cells: list[str]) -> bool:
    """Whether a table row belongs to the traceability matrix.

    A matrix row carries requirement, driver, ADR, implementation, verification and status.
    Narrower tables in the same file — sibling registries, coverage and handoff lists — name the
    same ids without proving them, so width and a status cell are what tell the matrix apart.
    """
    return (
        len(cells) >= MATRIX_ROW_CELLS
        and is_status_cell(cells[-1])
        and any(requirement_id(cell) for cell in cells[:2])
    )


def unchecked_requirements(traceability: str) -> list[str]:
    unchecked: list[str] = []
    for cells in table_rows(traceability):
        if not is_matrix_row(cells):
            continue
        requirement_index = next(
            (index for index, cell in enumerate(cells[:2]) if requirement_id(cell)), None
        )
        if requirement_index is None:
            continue
        requirement = requirement_id(cells[requirement_index])
        if requirement_index == 1:
            if len(cells) >= 7:
                implementation_cell, verification_cell, status = cells[-3:]
                completed = (
                    status.startswith("☑ done")
                    and has_evidence(implementation_cell)
                    and has_evidence(verification_cell)
                )
                accepted_risk = (
                    status == "accepted-risk"
                    and has_evidence(verification_cell)
                    and RISK_REFERENCE.search(verification_cell) is not None
                )
                if completed or accepted_risk:
                    continue
        # legacy packages stay readable; the source-first schema requires evidence
        elif is_closed_mark(cells[-1]) or is_accepted_risk(cells):
            continue
        unchecked.append(str(requirement))
    return unchecked


def matrix_state(package: Path) -> tuple[int, int] | None:
    """(closed, total) requirements of a package matrix, or None when it has none."""
    matrix = package / "06-traceability.md"
    if not matrix.is_file():
        return None
    text = matrix.read_text(encoding="utf-8")
    rows = [cells for cells in table_rows(text) if is_matrix_row(cells)]
    if not rows:
        return None
    return len(rows) - len(unchecked_requirements(text)), len(rows)


def states_marker(text: str, marker: str) -> bool:
    return re.search(rf"(?m)^{re.escape(marker)}\s*$", text) is not None


def review_is_accepted(review: str) -> bool:
    return all(states_marker(review, marker) for marker in REVIEW_MARKERS)


def milestones_are_complete(plan: str) -> bool:
    return states_marker(plan, MILESTONES_MARKER)


def contract_drift() -> list[str]:
    """Sources that stopped stating a marker this guard blocks on.

    A marker edited in one place only turns the Stop gate into a demand the round is never told
    how to satisfy, which costs a full-context turn per attempt until the run is killed.
    """
    drift: list[str] = []
    for source in CONTRACT_SOURCES:
        path = ROOT / source
        if not path.is_file():
            drift.append(f"{source} is missing")
            continue
        text = path.read_text(encoding="utf-8")
        drift.extend(f"{source} no longer states {m!r}" for m in CONTRACT_MARKERS if m not in text)
    return drift


def touches_ledger(tool_name: str, tool_input: Mapping[str, object]) -> bool:
    """Cheap guard before the git subprocess: only a ledger write can break the status rule."""
    if tool_name not in WRITE_TOOLS:
        return False
    payload = " ".join(str(value) for value in tool_input.values())
    return "docs/loop/state.md" in payload.replace("\\", "/").lower()


def session_round_kind(environ: Mapping[str, str] = os.environ) -> str | None:
    """Round kind of this session only: the driver marks its round processes via env."""
    if environ.get("HONESTML_LOOP_ROUND"):
        return environ.get("HONESTML_LOOP_KIND")
    return None


def active_round_kind(
    marker_path: Path = ACTIVE_ROUND_PATH, *, now: float | None = None
) -> str | None:
    if os.environ.get("HONESTML_LOOP_ROUND"):
        return session_round_kind()
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        expires_at = float(marker["expires_at"])
        kind = str(marker["kind"])
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
    return kind if (time.time() if now is None else now) <= expires_at else None


def git(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={ROOT}",
            "-C",
            str(ROOT),
            *arguments,
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def added_lines_vs_head() -> list[str] | None:
    result = git("diff", "-U0", "HEAD", "--", "docs/loop/state.md")
    if result.returncode != 0:
        return None
    return [
        line[1:]
        for line in result.stdout.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    ]


def stop_violations() -> list[str]:
    rows = table_rows(state_path().read_text(encoding="utf-8"))
    current_round = latest_round(rows)
    if current_round is None:
        return ["docs/loop/state.md has no round row"]

    item_id = current_round[1].split()[0]
    round_type = current_round[2]
    status = current_round[7]
    current_item = item_row(rows, item_id)
    if round_type != "impl":
        return [f"latest ledger row is {round_type}, expected impl"]
    if status == "blocked":
        return blocked_violations(item_id, current_item)
    if status == NEEDS_FIX:
        return needs_fix_violations(item_id, current_round, current_item)
    if status != "done":
        return [
            f"implementation status is {status!r}, expected 'done'",
            "continue the same round; do not roll back WIP",
        ]
    if current_item is None or current_item[1] != "done":
        return [f"item {item_id} is not marked done in the item table"]

    package = architecture_package(current_item)
    if package is None:
        return [f"item {item_id} has no architecture package path"]
    plan_path = package / "08-plan.md"
    traceability_path = package / "06-traceability.md"
    review_path = package / "09-review.md"
    if not plan_path.is_file():
        return [f"missing {plan_path.relative_to(ROOT)}"]
    if not milestones_are_complete(plan_path.read_text(encoding="utf-8")):
        return ["08-plan.md does not confirm that every implementation milestone is complete"]
    if not traceability_path.is_file():
        return [f"missing {traceability_path.relative_to(ROOT)}"]
    unchecked = unchecked_requirements(traceability_path.read_text(encoding="utf-8"))
    if unchecked:
        preview = ", ".join(unchecked[:8])
        suffix = "..." if len(unchecked) > 8 else ""
        return [f"unchecked traceability requirements: {preview}{suffix}"]
    if not review_path.is_file():
        return [f"missing {review_path.relative_to(ROOT)}"]
    if not review_is_accepted(review_path.read_text(encoding="utf-8")):
        return ["09-review.md has no completed finite review protocol"]
    return final_git_violations(item_id, "impl")


def round_moved_the_product() -> bool:
    """Whether the round's commit changed anything but bookkeeping."""
    result = git("show", "--name-only", "--format=", "HEAD")
    if result.returncode != 0:
        return False
    return any(path.strip() and not path.startswith("docs/") for path in result.stdout.splitlines())


def needs_fix_violations(
    item_id: str, current_round: list[str], current_item: list[str] | None
) -> list[str]:
    """What an IMPLEMENT round owes before it may stop at needs-fix.

    An obstacle outside the round's reach is the only legitimate reason. Everything else — an
    unfinished milestone, a red test, an open review finding, an exhausted budget — is work this
    same round still owns. The round must also leave the product further along than it found it:
    without that, needs-fix is a free exit that costs a round and buys nothing.
    """
    if current_item is None or current_item[1] != NEEDS_FIX:
        return [f"round records {NEEDS_FIX} but item {item_id} does not"]
    note = current_round[8] if len(current_round) > 8 else ""
    if not declares_external_blocker(note):
        return [
            f"{NEEDS_FIX} is legitimate only for a declared external obstacle: start the round "
            f"note with {EXTERNAL_BLOCKER!r} and say what is blocked and by whom."
        ]
    package = architecture_package(current_item)
    if package is None:
        return [f"item {item_id} has no architecture package path"]
    plan_path = package / "08-plan.md"
    if plan_path.is_file() and milestones_are_complete(plan_path.read_text(encoding="utf-8")):
        return ["08-plan.md says every milestone is complete; finish this round as done"]
    if violations := final_git_violations(item_id, NEEDS_FIX):
        return violations
    if not round_moved_the_product():
        return [
            f"{NEEDS_FIX} without progress: the round's commit touches only docs/. "
            "Declare the obstacle after the work it actually blocks, not instead of it."
        ]
    return []


def blocked_violations(item_id: str, current_item: list[str] | None) -> list[str]:
    if current_item is None or current_item[1] != "blocked":
        return [f"blocked round did not mark item {item_id} blocked"]
    blockers = blockers_path().read_text(encoding="utf-8")
    open_heading = re.compile(rf"^## .*\b{re.escape(item_id)}\b.*\bopen\b", re.MULTILINE)
    if not open_heading.search(blockers):
        return [f"blocked implementation has no open external blocker for {item_id}"]
    return final_git_violations(item_id, "blocked")


def final_git_violations(item_id: str, status: str) -> list[str]:
    dirty = git("status", "--porcelain")
    if dirty.returncode != 0:
        return ["git status failed"]
    if dirty.stdout.strip():
        return ["working tree is dirty; commit the accepted implementation before stopping"]
    subject = git("log", "-1", "--format=%s")
    expected = f"loop({item_id}): {status}"
    if subject.returncode != 0 or expected not in subject.stdout:
        return [f"latest commit subject must contain {expected!r}"]
    return []


def write_block(reason: list[str]) -> int:
    stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    stderr.write(
        "IMPLEMENT round completion blocked:\n- "
        + "\n- ".join(reason)
        + "\nFix every item in this same round, re-review, run final gates, "
        + "and stop only after done.\n"
    )
    stderr.flush()
    return 2


def self_test() -> int:
    # marker checks read the round env, so run them with the driver's env stripped:
    # a self-test invoked inside a round must assert the marker logic, not the round
    saved = {key: os.environ.get(key) for key in ("HONESTML_LOOP_ROUND", "HONESTML_LOOP_KIND")}
    for key in saved:
        os.environ.pop(key, None)
    marker = ROOT / "ops/loop/logs/implementation-round-guard-self-test.json"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text('{"kind":"impl","expires_at":200}', encoding="utf-8")
    live_marker_kind = active_round_kind(marker, now=100)
    expired_marker_kind = active_round_kind(marker, now=300)
    marker.unlink(missing_ok=True)
    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    checks = {
        "reject undeclared needs-fix": bool(
            added_status_violations(["| 10 | FR-X | impl | no-go | red | 0 | - | needs-fix | x |"])
        ),
        "accept declared needs-fix": not added_status_violations(
            ["| 10 | FR-X | impl | no-go | red | 0 | - | needs-fix | EXTERNAL-BLOCKER: pypiToken |"]
        ),
        "declaration survives bold": declares_external_blocker("**EXTERNAL-BLOCKER:** pypiToken"),
        "budget is not an obstacle": bool(
            added_status_violations(
                ["| 10 | FR-X | impl | no-go | red | 0 | - | needs-fix | ran out of budget |"]
            )
        ),
        "item row alone is not judged here": not added_status_violations(
            ["| FR-X | needs-fix | `docs/architecture/x/` | x |"]
        ),
        "needs-fix pairing is required": bool(
            needs_fix_violations(
                "FR-X",
                ["10", "FR-X", "impl", "no", "red", "0", "-", NEEDS_FIX, "EXTERNAL-BLOCKER: x"],
                ["FR-X", "done", "`docs/architecture/x/`", ""],
            )
        ),
        "accept checked traceability": not unchecked_requirements(
            "| FR-X | D-1 | ADR | src/x.py | test_x | ☑ |\n"
            "| NFR-X | D-1 | ADR | src/x.py | test_x | ☑ |"
        ),
        "reject unchecked traceability": unchecked_requirements(
            "| FR-X | D-1 | ADR | src/x.py | test_x | ☐ |"
        )
        == ["FR-X"],
        "accept done status in legacy schema": not unchecked_requirements(
            "| FR-X | D-1 | ADR | src/x.py | test_x | ☑ done |"
        ),
        "accept a closed mark carrying its slice": not unchecked_requirements(
            "| FR-X | D-1 | ADR | src/x.py | test_x | ☑ B-impl-1 |"
        ),
        "reject a partial mark": unchecked_requirements(
            "| FR-X | D-1 | ADR | src/x.py | test_x | ◐ N-36 |"
        )
        == ["FR-X"],
        "read a requirement id through its emphasis": unchecked_requirements(
            "| **FR-X** cascade sizing | D-1 | ADR | src/x.py | test_x | ☐ |"
        )
        == ["FR-X"],
        "strip emphasis off a requirement id": requirement_id("**FR-FS-1** отбор") == "FR-FS-1",
        "a prose cell names no requirement": requirement_id("unit: парсинг валид/невалид") is None,
        "accept evidence traceability": not unchecked_requirements(
            "| spec | FR-Z | D | ADR | src/x.py | test_x | ☑ done |"
        ),
        "reject planned traceability": unchecked_requirements(
            "| spec | FR-Z | D | ADR | src/x.py | test_x | planned |"
        )
        == ["FR-Z"],
        "reject done without evidence": unchecked_requirements(
            "| spec | FR-Z | D | ADR | — | — | ☑ done |"
        )
        == ["FR-Z"],
        "reject bare check in evidence schema": unchecked_requirements(
            "| spec | FR-Z | D | ADR | src/x.py | test_x | ☑ |"
        )
        == ["FR-Z"],
        "a row narrower than the matrix schema is not judged": not unchecked_requirements(
            "| spec | FR-Z | ☑ |"
        ),
        "a sibling registry row is not a requirement": not unchecked_requirements(
            "| `FR-024-onnx-export` | ONNX exporter for the fitted artifact | ☐ |"
        ),
        "accept documented risk in evidence schema": not unchecked_requirements(
            "| spec | NFR-Z | D | ADR | — | risk:R-1 approved | accepted-risk |"
        ),
        "reject undocumented risk in evidence schema": unchecked_requirements(
            "| spec | NFR-Z | D | ADR | — | — | accepted-risk |"
        )
        == ["NFR-Z"],
        "reject unreferenced risk in evidence schema": unchecked_requirements(
            "| spec | NFR-Z | D | ADR | — | approved by owner | accepted-risk |"
        )
        == ["NFR-Z"],
        "ignore coverage row that names an owning slice": not unchecked_requirements(
            "| FR-002 | reader boundary foundation | FR-002-B owns the parity gate |"
        ),
        "ignore handoff row without a status": not unchecked_requirements(
            "| NFR-047 | no onnxruntime in this slice | FR-EXP-9 defines the parity test |"
        ),
        "accept accepted-risk cell": not unchecked_requirements(
            "| NFR-X (accepted-risk) | D | ADR | impl | test | ☐ accepted-risk |"
        ),
        "accept accepted-risk note": not unchecked_requirements(
            "| NFR-Y | D | ADR | impl | not measured - accepted risk | ☐ |"
        ),
        "accept completed milestones": milestones_are_complete(
            "Implementation milestones complete: yes\n"
        ),
        "reject incomplete milestones": not milestones_are_complete(
            "Implementation milestones complete: no\n"
        ),
        "activate live round marker": live_marker_kind == "impl",
        "ignore expired round marker": expired_marker_kind is None,
        "session kind from driver env": session_round_kind(
            {"HONESTML_LOOP_ROUND": "1", "HONESTML_LOOP_KIND": "impl"}
        )
        == "impl",
        "foreign session has no round kind": session_round_kind({}) is None,
        "accept review verdict": review_is_accepted(
            "## Review state\n"
            "## Review passes\n"
            "## Review findings\n"
            "Full review waves: 1\n"
            "Fresh regression review waves: 1\n"
            "Review protocol: complete\n"
            "Final verdict: ACCEPT\n"
            "Open actionable findings: 0\n"
        ),
        "reject accept without review protocol": not review_is_accepted(
            "Final verdict: ACCEPT\nOpen actionable findings: 0\n"
        ),
        "reject multiple full review waves": not review_is_accepted(
            "## Review state\n"
            "## Review passes\n"
            "## Review findings\n"
            "Full review waves: 2\n"
            "Fresh regression review waves: 1\n"
            "Review protocol: complete\n"
            "Final verdict: ACCEPT\n"
            "Open actionable findings: 0\n"
        ),
        "reject multiple regression review waves": not review_is_accepted(
            "## Review state\n"
            "## Review passes\n"
            "## Review findings\n"
            "Full review waves: 1\n"
            "Fresh regression review waves: 2\n"
            "Review protocol: complete\n"
            "Final verdict: ACCEPT\n"
            "Open actionable findings: 0\n"
        ),
        "reject review no-go": not review_is_accepted(
            "Final verdict: NO-GO\nOpen actionable findings: 3\n"
        ),
        "contract markers still stated by every source": not contract_drift(),
        "ledger prefilter accepts a ledger edit": touches_ledger(
            "Edit", {"file_path": str(STATE_PATH)}
        ),
        "ledger prefilter accepts a ledger write via bash": touches_ledger(
            "Bash", {"command": "printf '| 9 |' >> docs/loop/state.md"}
        ),
        "ledger prefilter accepts a ledger patch": touches_ledger(
            "apply_patch", {"input": "*** Update File: docs/loop/state.md"}
        ),
        "ledger prefilter skips unrelated edits": not touches_ledger(
            "Edit", {"file_path": str(ROOT / "src/honestml/core/config.py")}
        ),
        "ledger prefilter skips unrelated commands": not touches_ledger(
            "Bash", {"command": "uv run pytest tests/unit -q"}
        ),
        "ledger prefilter ignores other tools": not touches_ledger(
            "Read", {"file_path": str(STATE_PATH)}
        ),
    }
    for reason in contract_drift():
        print(f"     drift: {reason}")
    failures = 0
    for name, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'} {name}")
        failures += int(not passed)
    return 1 if failures else 0


def main() -> int:
    if "--self-test" in sys.argv:
        return self_test()
    if active_round_kind() != "impl":
        return 0

    data = json.load(io.TextIOWrapper(sys.stdin.buffer, encoding="utf-8-sig"))
    tool_name = data.get("tool_name")
    if tool_name:
        if not touches_ledger(tool_name, data.get("tool_input") or {}):
            return 0
        added = added_lines_vs_head()
        violations = added_status_violations(added or [])
        return write_block(violations) if violations else 0
    # stop enforcement binds only to the round's own session, marked by the driver via env;
    # a live marker alone covers concurrent sessions' ledger writes, not their stops
    if session_round_kind() != "impl":
        return 0
    violations = stop_violations()
    return write_block(violations) if violations else 0


if __name__ == "__main__":
    sys.exit(main())
