"""Append a round row and update an item row in the loop ledger.

The ledger is bookkeeping with a fixed shape: the round number follows the previous row, the
round goes to the bottom of its table, a new item goes to the top of its own, and both notes
have length limits. Hand-writing that costs a full-context turn per round and drifts; here it
is one call whose arguments are the facts of the round.

Traceability counts come from the package matrix, not from memory: --package prefixes the note
with the closed/total ratio the guard checks anyway.

    python ops/loop/ledger.py round --item FR-021-cascade --type impl --gate go --dod green \
        --tests +54 --status done --package docs/architecture/adaptive-cascade-sizing/ \
        --note "non-inferiority + mass_floor; ревью 4‖ ACCEPT."
    python ops/loop/ledger.py item --item FR-021-cascade --phase done \
        --package docs/architecture/adaptive-cascade-sizing/ --note "DoD green; матрица 14/14."
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Callable
from datetime import datetime, timedelta

from implementation_round_guard import (
    ITEMS_NOTE_LIMIT,
    NEEDS_FIX,
    ROOT,
    ROUNDS_NOTE_LIMIT,
    blockers_path,
    matrix_state,
    split_cells,
    state_path,
    table_rows,
)

ROUND_TYPES = ("design", "impl", "blocked-only")
# Where an item can come to rest. A round's status is one of these too: the round says what
# happened to the item, so the two vocabularies are the same one.
ITEM_PHASES = ("designed", "done", "needs-fix", "blocked")
GATES_VERDICT_PATH = ROOT / "ops" / "loop" / "logs" / "gates-verdict.json"
# Older than this and the verdict describes a tree that has since moved on.
VERDICT_MAX_AGE_HOURS = 24
DOD_NOT_RUN = "—"
# An obstacle that survives this many rounds in a row is not one the loop can work around.
NEEDS_FIX_ESCALATION = 5


def state_lines() -> list[str]:
    return state_path().read_text(encoding="utf-8").splitlines()


def write_state(lines: list[str]) -> None:
    state_path().write_text("\n".join(lines) + "\n", encoding="utf-8")


def clamp(note: str, limit: int) -> str:
    return note if len(note) <= limit else note[: limit - 1].rstrip() + "…"


def matrix_ratio(package: str) -> str | None:
    """closed/total of the package matrix, or None when there is no matrix to read."""
    state = matrix_state(ROOT / package.strip("/"))
    return None if state is None else f"{state[0]}/{state[1]}"


def table_bounds(lines: list[str], header_start: str) -> tuple[int, int]:
    """Index of the header row and of the last data row of the table it opens."""
    header = next(i for i, line in enumerate(lines) if line.startswith(header_start))
    last = header + 1  # separator row
    while last + 1 < len(lines) and lines[last + 1].startswith("|"):
        last += 1
    return header, last


def compose_note(arguments: argparse.Namespace) -> str:
    note = arguments.note or ""
    if not arguments.package:
        return note
    ratio = matrix_ratio(arguments.package)
    prefix = f"{ratio} traceability; " if ratio else ""
    return f"{prefix}{note} Пакет {arguments.package.strip('/')}/.".strip()


def gates_verdict() -> str | None:
    """Verdict of the last gates run, or None when no usable one exists."""
    try:
        record = json.loads(GATES_VERDICT_PATH.read_text(encoding="utf-8"))
        stamp = datetime.fromisoformat(re.sub(r"(\.\d{6})\d+", r"\1", record["ts"]))
    except (OSError, ValueError, KeyError, TypeError):
        return None
    now = datetime.now(stamp.tzinfo)
    if now - stamp > timedelta(hours=VERDICT_MAX_AGE_HOURS):
        return None
    verdict = record.get("verdict")
    return verdict if isinstance(verdict, str) else None


def dod_violation(claimed: str) -> str | None:
    """Why the claimed DoD column contradicts what the gates actually reported."""
    if claimed == DOD_NOT_RUN:
        return None
    verdict = gates_verdict()
    if verdict is None:
        return (
            f"--dod {claimed} claims a verdict, but no gates run from the last "
            f"{VERDICT_MAX_AGE_HOURS}h is on record. Run ops/loop/gates.ps1, or record "
            f"--dod {DOD_NOT_RUN}."
        )
    if claimed != verdict:
        return f"--dod {claimed} contradicts the recorded gates verdict ({verdict})."
    return None


def trailing_needs_fix(item_id: str) -> int:
    """How many of the ledger's last rounds ended at needs-fix on this item, unbroken."""
    rows = table_rows(state_path().read_text(encoding="utf-8"))
    rounds = [cells for cells in rows if cells[0].isdigit()]
    streak = 0
    for cells in reversed(rounds):
        if len(cells) < 8 or cells[7] != NEEDS_FIX or cells[1].split()[0] != item_id:
            break
        streak += 1
    return streak


def next_blocker_number() -> int:
    text = blockers_path().read_text(encoding="utf-8")
    issued = [int(match) for match in re.findall(r"^##\s*BLK-(\d+)\b", text, re.MULTILINE)]
    return max(issued) + 1 if issued else 1


def escalate_needs_fix_streak(item_id: str, note: str) -> str | None:
    """File a blocker once an obstacle has outlasted NEEDS_FIX_ESCALATION rounds."""
    if trailing_needs_fix(item_id) < NEEDS_FIX_ESCALATION:
        return None
    marker = f"needs-fix-серия {NEEDS_FIX_ESCALATION}× · {item_id}"
    text = blockers_path().read_text(encoding="utf-8")
    if marker in text:
        return None
    number = next_blocker_number()
    entry = (
        f"\n## BLK-{number} · {item_id} · open\n"
        f"- Вопрос: препятствие не снимается работой раунда — нужно решение владельца.\n"
        f"- Почему блокирует: {marker}; причина — {clamp(note, 140)}\n"
        f"- Варианты: снять препятствие решением владельца; снять пункт с работы.\n"
        f"- Рекомендация: решение владельца по названной причине.\n"
    )
    blockers_path().write_text(text.rstrip("\n") + "\n" + entry, encoding="utf-8")
    return f"BLK-{number}"


def add_round(arguments: argparse.Namespace) -> int:
    # The DoD column is a fact about a run, not a word: it is checked before the row exists.
    if violation := dod_violation(arguments.dod):
        print(violation, file=sys.stderr)
        return 1
    lines = state_lines()
    header, last = table_bounds(lines, "| round |")
    numbers = [
        int(cells[0])
        for cells in table_rows("\n".join(lines[header : last + 1]))
        if cells[0].isdigit()
    ]
    note = compose_note(arguments)
    row = " | ".join(
        [
            "",
            str(max(numbers) + 1 if numbers else 1),
            arguments.item,
            arguments.type,
            arguments.gate,
            arguments.dod,
            arguments.tests,
            "—",
            arguments.status,
            clamp(note, ROUNDS_NOTE_LIMIT),
            "",
        ]
    ).strip()
    lines.insert(last + 1, row)
    write_state(lines)
    print(row)
    # the round says what became of the item; the item table should not need a second call
    if arguments.type != "blocked-only":
        print(upsert_item(arguments.item.split()[0], arguments.status, arguments.package))
    if arguments.status == NEEDS_FIX:
        if blocker := escalate_needs_fix_streak(arguments.item.split()[0], arguments.note or ""):
            print(f"{blocker} filed: the obstacle outlasted {NEEDS_FIX_ESCALATION} rounds")
    return 0


def upsert_item(item_id: str, phase: str, package: str = "", note: str = "") -> str:
    """Set an item's phase, keeping whatever the call does not carry.

    A round that only moves an item forward should not have to restate its package path and
    note to avoid erasing them.
    """
    lines = state_lines()
    header, last = table_bounds(lines, "| id пункта |")
    existing = next(
        (
            cells
            for index in range(header + 2, last + 1)
            if (cells := split_cells(lines[index]))[0] == item_id
        ),
        None,
    )
    package_cell = f"`{package.strip('/')}/`" if package else (existing[2] if existing else "—")
    note_cell = clamp(note, ITEMS_NOTE_LIMIT) if note else (existing[3] if existing else "")
    row = f"| {item_id} | {phase} | {package_cell} | {note_cell} |"
    for index in range(header + 2, last + 1):
        if split_cells(lines[index])[0] == item_id:
            lines[index] = row
            break
    else:
        lines.insert(header + 2, row)
    write_state(lines)
    return row


def add_item(arguments: argparse.Namespace) -> int:
    print(upsert_item(arguments.item, arguments.phase, arguments.package, arguments.note or ""))
    return 0


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    round_parser = sub.add_parser("round", help="append a round row")
    round_parser.add_argument("--item", required=True)
    round_parser.add_argument("--type", required=True, choices=ROUND_TYPES)
    round_parser.add_argument("--gate", default="—")
    round_parser.add_argument("--dod", default="—")
    round_parser.add_argument("--tests", default="—")
    round_parser.add_argument("--status", required=True, choices=ITEM_PHASES)
    round_parser.add_argument("--package", default="")
    round_parser.add_argument("--note", default="")
    round_parser.set_defaults(handler=add_round)

    item_parser = sub.add_parser("item", help="add or update an item row")
    item_parser.add_argument("--item", required=True)
    item_parser.add_argument("--phase", required=True, choices=ITEM_PHASES)
    item_parser.add_argument("--package", default="")
    item_parser.add_argument("--note", default="")
    item_parser.set_defaults(handler=add_item)

    arguments = parser.parse_args()
    handler: Callable[[argparse.Namespace], int] = arguments.handler
    return handler(arguments)


if __name__ == "__main__":
    sys.exit(main())
