"""Check that the loop's registers agree with each other and with the package matrices.

A slice's status lives in three places at once: the ledger item row a round writes by hand
(docs/loop/state.md), the slice line in docs/loop/backlog.md and the matrix of the package itself.
Only the matrix is evidence — the other two are navigation, written at the end of a round, and a
hand-written navigator drifts silently. This reads the matrices and reports every place the
ledger or the backlog disagrees with them.

It never rewrites the registers: their notes (what was closed, what remains, why) are authored,
not derived, so a generator would destroy what it cannot reproduce.

The backlog is judged as a navigator, not a journal: a slice is one line of a closed grammar
with a bounded body, and a candidate is a registration no round may pick. What a round added
is judged against HEAD — the tree already carries its history.

It also fails on a repeated identity. Two rows for one item are not a conflict any tool reports:
the ledger keeps updating the first and the second stays frozen at whatever it said, so a
repeated identity is the only trace the split leaves.

    uv run python ops/loop/check_registers.py [--item <id пункта>] [--self-test]
"""

from __future__ import annotations

import argparse
import collections
import io
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from implementation_round_guard import (
    ROOT,
    blockers_path,
    git,
    matrix_state,
    state_path,
    table_rows,
)

ARCHITECTURE = ROOT / "docs/architecture"
BACKLOG = ROOT / "docs/loop/backlog.md"
# A package pointer carries every segment below docs/architecture: a slice package lives inside
# its umbrella, so `automl-productionization/M4-ml-correctness` is one package, not two.
PACKAGE_PATTERN = re.compile(r"docs/architecture/([\w-]+(?:/[\w-]+)*)/")
BLOCKER_HEADING = re.compile(r"(?m)^## (BLK-\d+) · ([^ ·]+) · (open|answered)\b")
# Header cells of the two ledger tables. A header row has an item row's shape, so the id column
# is what tells them apart.
LEDGER_HEADERS = frozenset({"round", "id пункта"})

# --- backlog grammar -------------------------------------------------------------------------
# A slice is one list item at column 0; its body runs to the next column-0 list item or heading.
SLICE_LINE = re.compile(r"^- \*\*Срез `([^`]+)` \(([^)]*)\):\*\*(.*)$")
FIELD_LINE = re.compile(r"^- \*\*([^*`]+?):\*\*(.*)$")
NESTED_ENTRY = re.compile(r"^\s+- `([^`]+)` — (.*)$")
EPIC_HEADING = re.compile(r"^### (EPIC-[A-Z0-9]+)\b(.*)$")
SECTION_HEADING = re.compile(r"^## (.+?)\s*$")
SUMMARY_ROW = re.compile(r"^\| (EPIC-[A-Z0-9]+) \|")
COUNTER_PATTERN = re.compile(
    r"Всего эпиков: \*\*(\d+)\*\* · done: \*\*(\d+)\*\* · partial: \*\*(\d+)\*\* · "
    r"todo: \*\*(\d+)\*\*"
)
# The closed status vocabulary. Progress marks are confirmed by the matrix; a process state
# names why the work stands (a blocker, an owner's decision) rather than how much is closed.
STATUS_PATTERN = re.compile(r"^(☑|◐|☐|⛔ \S.*|⏸ \S.*|✂ \S.*)$")
PROGRESS_MARKS = frozenset({"☑", "◐", "☐"})
REQUIRED_FIELDS = ("Состав", "Требования", "Зависимости", "Блокеры")
CANDIDATES_FIELD = "Кандидаты (не выбираются)"
DEFERRED_FIELD_PREFIX = "Отложено"
CLOSES_PATTERN = re.compile(r"закрывает: ((?:FR|NFR)-[A-Z0-9-]+|—)")
# What a navigator line never carries: it is a fact about a state, and these are facts about a
# history — the changelog and the package already hold them.
CHRONICLE = re.compile(
    r"\bр\.\s?\d|\bраунд\w*\s+\d|\bpytest\b|\bmypy\b|\bDoD\b|(?<![-\w])blocker\b|(?<![-\w])major\b|"
    r"матриц\w*\s+\d+\s*/\s*\d+|\d+\s*/\s*\d+\s*☑|\bADR-|\bdesign-gate\b|\bревью\b|Δ|"
    r"\b(?:IMPLEMENTED|DONE)\b|дельт\w*[^.;]{0,80}нулев",
    re.IGNORECASE,
)
SLICE_BODY_LIMIT = 400
ENTRY_LIMIT = 300
SUMMARY_CELL_LIMIT = 200
NEW_CANDIDATES_LIMIT = 3
SUFFIX_DEPTH_LIMIT = 2
# Item ids of another contour: an audit finding worked by a loop round is registered in the
# audit registers, not as a backlog slice.
FOREIGN_ITEM = re.compile(r"^[NHML]-\d+$")
IMPL_PHASE = re.compile(r"-impl-\d+[a-z]?$")
# The requirement a slice id belongs to, and the suffix that names the slice under it.
SLICE_PREFIX = re.compile(
    r"^((?:FR|NFR)-\d+[a-z]?|(?:FR|NFR)-[A-Z]+(?:-[A-Z]+)*-\d+[a-z]?|EPIC-[A-Z0-9]+)(?:-(.+))?$"
)


@dataclass
class Entry:
    id: str
    text: str
    line: int


@dataclass
class Slice:
    id: str
    status: str
    body: str
    line: int
    epic: str


@dataclass
class Epic:
    id: str
    line: int
    fields: set[str] = field(default_factory=set)
    slices: list[Slice] = field(default_factory=list)
    candidates: list[Entry] = field(default_factory=list)
    deferred: list[Entry] = field(default_factory=list)


@dataclass
class Backlog:
    counter: tuple[int, int, int, int] | None
    summary: list[list[str]]
    epics: list[Epic]
    has_summary: bool

    @property
    def slices(self) -> list[Slice]:
        return [s for epic in self.epics for s in epic.slices]

    @property
    def candidates(self) -> list[Entry]:
        return [c for epic in self.epics for c in epic.candidates]

    @property
    def deferred(self) -> list[Entry]:
        return [d for epic in self.epics for d in epic.deferred]


def parse_backlog(text: str) -> Backlog:
    counter = COUNTER_PATTERN.search(text)
    lines = text.splitlines()
    epics: list[Epic] = []
    summary: list[list[str]] = []
    has_summary = False
    current: Epic | None = None
    block = ""  # which field the nested entries below belong to
    slice_open: Slice | None = None
    in_summary = False
    for number, line in enumerate(lines, start=1):
        section = SECTION_HEADING.match(line)
        if section:
            in_summary = section.group(1).strip() == "Сводка"
            has_summary = has_summary or in_summary
            current, block, slice_open = None, "", None
            continue
        if in_summary and SUMMARY_ROW.match(line):
            summary.append([cell.strip() for cell in line.strip().strip("|").split("|")])
            continue
        heading = EPIC_HEADING.match(line)
        if heading:
            current = Epic(heading.group(1), number)
            epics.append(current)
            block, slice_open = "", None
            continue
        if current is None:
            continue
        slice_match = SLICE_LINE.match(line)
        if slice_match:
            slice_open = Slice(
                slice_match.group(1),
                slice_match.group(2).strip(),
                slice_match.group(3),
                number,
                current.id,
            )
            current.slices.append(slice_open)
            block = ""
            continue
        field_match = FIELD_LINE.match(line)
        if field_match:
            name = field_match.group(1).strip()
            current.fields.add(name)
            block = name
            slice_open = None
            continue
        nested = NESTED_ENTRY.match(line)
        if nested and block:
            entry = Entry(nested.group(1), nested.group(2), number)
            if block == CANDIDATES_FIELD:
                current.candidates.append(entry)
            elif block.startswith(DEFERRED_FIELD_PREFIX):
                current.deferred.append(entry)
            continue
        if slice_open is not None and line.startswith("  "):
            slice_open.body += " " + line.strip()
        elif line.startswith("- "):
            slice_open, block = None, ""
    return Backlog(
        counter=tuple(int(v) for v in counter.groups()) if counter else None,  # type: ignore[arg-type]
        summary=summary,
        epics=epics,
        has_summary=has_summary,
    )


def ledger_rows(state_text: str) -> tuple[list[list[str]], list[list[str]]]:
    """(round rows, item rows) of the ledger, headers and separators dropped."""
    rounds: list[list[str]] = []
    items: list[list[str]] = []
    for cells in table_rows(state_text):
        if cells[0].isdigit():
            rounds.append(cells)
        elif len(cells) >= 4 and cells[0] not in LEDGER_HEADERS:
            items.append(cells)
    return rounds, items


def extends(umbrella: str, candidate: str) -> bool:
    """Whether a slice id names a child of an umbrella: `FR-SER-1-a` extends `FR-SER-1` and
    `FR-4b` extends `FR-4`."""
    if candidate == umbrella or not candidate.startswith(umbrella):
        return False
    tail = candidate[len(umbrella) :]
    return (
        tail[0] == "-"
        or (umbrella[-1].isdigit() and tail[0].isalpha())
        or (umbrella[-1].isalpha() and tail[0].isdigit())
    )


def expected_mark(closed: int, total: int) -> str:
    if closed == total:
        return "☑"
    return "☐" if closed == 0 else "◐"


def navigator_length(text: str) -> int:
    """Length of a navigator line without its package pointers and collapsed whitespace."""
    return len(" ".join(PACKAGE_PATTERN.sub("", text.replace("`", "")).split()))


def slice_prefix(slice_id: str) -> tuple[str, list[str]] | None:
    match = SLICE_PREFIX.match(slice_id)
    if not match:
        return None
    suffix = match.group(2)
    return match.group(1), suffix.split("-") if suffix else []


# --- backlog shape and content ----------------------------------------------------------------
def check_backlog_shape(backlog: Backlog) -> list[str]:
    problems: list[str] = []
    if not backlog.has_summary:
        problems.append("backlog has no `## Сводка` section — the navigator grammar is missing")
    if backlog.counter is None:
        problems.append("backlog header has no epic counter line")
    for epic in backlog.epics:
        missing = [name for name in REQUIRED_FIELDS if name not in epic.fields]
        if missing:
            problems.append(f"epic {epic.id}: section lacks field(s) {', '.join(missing)}")
    for item in backlog.slices:
        if not STATUS_PATTERN.match(item.status):
            problems.append(
                f"slice {item.id}: status {item.status!r} is outside the vocabulary "
                f"☑ · ◐ · ☐ · ⛔ <ref> · ⏸ <decision> · ✂ <decision>"
            )
    return problems


def check_navigator_bodies(backlog: Backlog) -> list[str]:
    """A navigator line states a state; length and chronicle tokens are how a journal creeps in."""
    problems: list[str] = []
    for item in backlog.slices:
        length = navigator_length(item.body)
        if length > SLICE_BODY_LIMIT:
            problems.append(
                f"slice {item.id}: body is {length} chars, limit {SLICE_BODY_LIMIT}; "
                f"the chronicle belongs to the changelog and the package"
            )
        hit = CHRONICLE.search(item.body)
        if hit:
            problems.append(f"slice {item.id}: body carries chronicle ({hit.group(0)!r})")
    for kind, entries in (("candidate", backlog.candidates), ("deferred", backlog.deferred)):
        for entry in entries:
            if navigator_length(entry.text) > ENTRY_LIMIT:
                problems.append(f"{kind} {entry.id}: entry longer than {ENTRY_LIMIT} chars")
            if kind == "candidate" and not CLOSES_PATTERN.search(entry.text):
                problems.append(
                    f"candidate {entry.id}: entry names no `закрывает: <FR/NFR> | —` clause"
                )
    for cells in backlog.summary:
        for cell in cells[4:6]:
            if len(cell) > SUMMARY_CELL_LIMIT:
                problems.append(
                    f"summary {cells[0]}: cell longer than {SUMMARY_CELL_LIMIT} chars "
                    f"({len(cell)}); the summary points, it does not narrate"
                )
            hit = CHRONICLE.search(cell)
            if hit:
                problems.append(f"summary {cells[0]}: cell carries chronicle ({hit.group(0)!r})")
    return problems


def check_backlog_slices(
    backlog: Backlog, architecture: Path = ARCHITECTURE
) -> tuple[list[str], int, int]:
    """(problems, compared slices, slices in a process state)."""
    problems: list[str] = []
    compared = skipped = 0
    for item in backlog.slices:
        packages = PACKAGE_PATTERN.findall(item.body)
        if item.status not in PROGRESS_MARKS:
            skipped += 1
            continue
        if not packages:
            # a slice without a package is an umbrella: its mark follows the slices whose ids
            # extend its own, and an umbrella with no children has nothing to prove ☑ with
            children = [s.status for s in backlog.slices if extends(item.id, s.id)]
            if children:
                closed = sum(1 for status in children if status == "☑")
                expected = expected_mark(closed, len(children))
                if item.status != expected:
                    problems.append(
                        f"slice {item.id}: umbrella says {item.status!r}, its {len(children)} "
                        f"children give {expected} ({closed} ☑)"
                    )
            elif item.status == "☑":
                problems.append(
                    f"slice {item.id}: marked ☑ without a package pointer or child slices"
                )
            continue
        compared += 1
        state = matrix_state(architecture / packages[0])
        if state is None:
            # a not-started slice may name the package it will create
            if item.status == "☑":
                problems.append(f"slice {item.id}: marked ☑, but {packages[0]} has no matrix")
            continue
        closed, total = state
        expected = expected_mark(closed, total)
        if item.status != expected:
            problems.append(
                f"slice {item.id}: backlog says {item.status!r}, matrix is {closed}/{total} "
                f"({expected})"
            )
    return problems, compared, skipped


def check_backlog_identities(backlog: Backlog) -> list[str]:
    problems: list[str] = []
    for kind, found in (
        ("epic section", [epic.id for epic in backlog.epics]),
        ("summary row", [cells[0] for cells in backlog.summary]),
        ("slice", [item.id for item in backlog.slices]),
        ("candidate", [entry.id for entry in backlog.candidates]),
        ("deferred entry", [entry.id for entry in backlog.deferred]),
    ):
        for identity, count in sorted(collections.Counter(found).items()):
            if count > 1:
                problems.append(f"{kind} {identity} appears {count} times")
    slices = {item.id for item in backlog.slices}
    candidates = {entry.id for entry in backlog.candidates}
    deferred = {entry.id for entry in backlog.deferred}
    for identity in sorted(slices & candidates):
        problems.append(f"slice {identity} is still listed as a candidate")
    for identity in sorted(slices & deferred):
        problems.append(f"slice {identity} is still listed as deferred")
    for identity in sorted(candidates & deferred):
        problems.append(f"candidate {identity} is also listed as deferred")
    sections = {epic.id for epic in backlog.epics}
    rows = {cells[0] for cells in backlog.summary}
    for identity in sorted(rows - sections):
        problems.append(f"summary row {identity} has no `### {identity}` section")
    for identity in sorted(sections - rows):
        problems.append(f"epic section {identity} has no row in `## Сводка`")
    return problems


def check_epic_counters(backlog: Backlog) -> list[str]:
    if backlog.counter is None:
        return []
    total, done, partial, todo = backlog.counter
    counted = {"done": 0, "partial": 0, "todo": 0}
    for cells in backlog.summary:
        value = cells[3] if len(cells) > 3 else ""
        key = "done" if value.startswith("☑") else "partial" if value.startswith("◐") else "todo"
        counted[key] += 1
    problems: list[str] = []
    if len(backlog.summary) != total:
        problems.append(
            f"epic counter says {total} epics, summary table lists {len(backlog.summary)}"
        )
    for key, declared in (("done", done), ("partial", partial), ("todo", todo)):
        if counted[key] != declared:
            problems.append(f"epic counter says {key}={declared}, summary table has {counted[key]}")
    return problems


# --- ledger against matrices and backlog ------------------------------------------------------
def check_ledger_items(
    items: list[list[str]],
    backlog: Backlog | None = None,
    architecture: Path = ARCHITECTURE,
    awaiting_owner: frozenset[str] = frozenset(),
) -> list[str]:
    """Item phases the package matrices and the backlog do not support.

    `awaiting_owner` names the items an OPEN blocker still holds. Only for those does a `blocked`
    phase have to read as ⛔ in the backlog: once the owner has answered, the phase is stale by
    design — it stays until a round consumes the decision (LOOP.md §1) — while the backlog already
    tells the truth about readiness, and check_unblocked_items is what reports the stale phase.
    """
    problems: list[str] = []
    marks = {item.id: item.status for item in backlog.slices} if backlog else {}
    for cells in items:
        item, phase = cells[0], cells[1]
        # an implementation phase of a slice (`X-impl-2`) is a ledger row of its own, but the
        # backlog navigates by the slice: a phase row without a line of its own is judged
        # against its parent's line
        parent = item if item in marks else IMPL_PHASE.sub("", item)
        if backlog is not None and not FOREIGN_ITEM.match(item) and parent not in marks:
            problems.append(
                f"item {item}: ledger names it, but the backlog has no slice line for it — "
                f"a round works only on slices; a candidate is promoted first"
            )
        if phase == "done" and parent in marks and marks[parent] != "☑":
            problems.append(f"item {item}: ledger says done, backlog says {marks[parent]!r}")
        if (
            phase == "blocked"
            and item in awaiting_owner
            and parent in marks
            and not marks[parent].startswith("⛔")
        ):
            problems.append(f"item {item}: ledger says blocked, backlog says {marks[parent]!r}")
        packages = PACKAGE_PATTERN.findall(cells[2])
        # an audit finding's package keeps the audit contour's schema and is judged by the
        # audit registers, not by this matrix reading
        if not packages or FOREIGN_ITEM.match(item):
            continue
        state = matrix_state(architecture / packages[0])
        if state is None:
            # a design-gate no-go leaves a blocked package as research plus a findings
            # register, legitimately without a traceability matrix (LOOP.md §2a)
            if phase != "blocked":
                problems.append(f"item {item}: package {packages[0]} has no readable matrix")
            continue
        closed, total = state
        if phase == "done" and closed != total:
            problems.append(f"item {item}: ledger says done, matrix is {closed}/{total}")
    return problems


def check_register_identities(
    rounds: list[list[str]], items: list[list[str]], blockers_text: str
) -> list[str]:
    """Identities carried by more than one row of their own register."""
    problems: list[str] = []
    for kind, found in (
        ("ledger item", [cells[0] for cells in items]),
        ("ledger round", [cells[0] for cells in rounds]),
        ("blocker", [blocker for blocker, _, _ in BLOCKER_HEADING.findall(blockers_text)]),
    ):
        for identity, count in sorted(collections.Counter(found).items()):
            if count > 1:
                problems.append(f"{kind} {identity} appears {count} times")
    return problems


def items_awaiting_owner(blockers_text: str) -> frozenset[str]:
    """Items an open blocker still holds — the only ones whose `blocked` phase is current."""
    return frozenset(
        item for _blocker, item, state in BLOCKER_HEADING.findall(blockers_text) if state == "open"
    )


def check_unblocked_items(items: list[list[str]], blockers_text: str) -> list[str]:
    """Blocked items that nothing can release.

    A blocked phase means the owner owes a decision, and the decision lives in `blockers.md`.
    An item whose blockers are all answered waits for nobody; an item with no blocker at all was
    never waiting for the owner in the first place — its phase is doing the job of a note. Both
    are invisible from here on: a blocked item is not offered to a round, so neither comes back
    on its own, and the round table rotates into the archive, so the phase is the only thing
    left saying anything about it.
    """
    states: dict[str, set[str]] = collections.defaultdict(set)
    for _blocker, item, state in BLOCKER_HEADING.findall(blockers_text):
        states[item].add(state)
    problems: list[str] = []
    for cells in items:
        item = cells[0]
        if cells[1] != "blocked":
            continue
        seen = states.get(item)
        if seen is None:
            problems.append(
                f"item {item}: the phase is blocked, but no blocker in blockers.md names it; "
                f"nothing can release it"
            )
        elif "open" not in seen:
            problems.append(
                f"item {item}: every blocker is answered, but the phase is still blocked; "
                f"the round that consumes the decision is pending"
            )
    return problems


# --- what this round added, judged against HEAD -----------------------------------------------
def check_new_slices(current: Backlog, head: Backlog) -> list[str]:
    """Slice lines that exist now and did not at HEAD: an id stays within a bounded depth under
    the requirement it belongs to."""
    problems: list[str] = []
    head_slices = {item.id for item in head.slices}
    for item in current.slices:
        if item.id in head_slices:
            continue
        parts = slice_prefix(item.id)
        if parts is None:
            continue
        requirement, suffix = parts
        if len(suffix) > SUFFIX_DEPTH_LIMIT:
            problems.append(
                f"slice {item.id}: id is {len(suffix)} segments below {requirement}, limit "
                f"{SUFFIX_DEPTH_LIMIT}; a deeper need restates the parent under its own id"
            )
    return problems


def check_new_candidates(
    current: Backlog, head: Backlog, in_round: bool | None = None
) -> list[str]:
    """Candidates registered since HEAD, bounded per round. The bound belongs to a round — a
    design package registers what its review found and no more — so it is judged only inside
    one; an owner's session registers as much as the owner decides."""
    if in_round is None:
        in_round = bool(os.environ.get("HONESTML_LOOP_ROUND"))
    if not in_round:
        return []
    head_ids = {entry.id for entry in head.candidates}
    added = [entry.id for entry in current.candidates if entry.id not in head_ids]
    if len(added) > NEW_CANDIDATES_LIMIT:
        return [
            f"candidates: {len(added)} registered since HEAD ({', '.join(added)}), limit "
            f"{NEW_CANDIDATES_LIMIT} per package"
        ]
    return []


def head_text(relative: str) -> str | None:
    result = git("show", f"HEAD:{relative}")
    return result.stdout if result.returncode == 0 else None


# --- self-test ---------------------------------------------------------------------------------
FIXTURE_BACKLOG = """# Backlog
Всего эпиков: **2** · done: **1** · partial: **1** · todo: **0**

## Сводка
| Эпик | Результат | Требования | Статус | Осталось | Блокеры |
|---|---|---|---|---|---|
| EPIC-01 | one | FR-001 | ☑ | — | — |
| EPIC-02 | two | FR-002 | ◐ | `FR-002-b` | BLK-1 |

## Эпики
### EPIC-01 — one
- **Состав:** a.
- **Требования:** FR-001.
- **Зависимости:** нет.
- **Блокеры:** нет.
- **Срез `FR-001-a` (☑):** outcome a. `docs/architecture/fx-closed/`

### EPIC-02 — two
- **Состав:** b.
- **Требования:** FR-002.
- **Зависимости:** EPIC-01.
- **Блокеры:** BLK-1.
- **Срез `FR-002-a` (◐):** outcome b,
  continued on a second line. `docs/architecture/fx-half/`
- **Срез `FR-002-b` (⛔ BLK-1):** waits for the owner.
- **Кандидаты (не выбираются):**
  - `FR-002-c` — a finding; источник: ревью `FR-002-a`; закрывает: FR-002
  - `FR-002-d` — another; источник: ревью `FR-002-a`; закрывает: —
- **Отложено (⏸ решение владельца 2026-01-01):**
  - `FR-002-e` — parked.
"""


def self_test_checks() -> dict[str, bool]:
    """Every claim this module makes about the shape of a register, judged on fixtures.

    A register check that stopped parsing reports nothing and reads as green, so the parser is
    re-proved on every run rather than only when someone asks for it.
    """
    architecture = ROOT / "ops/loop/logs/check-registers-self-test"
    matrix = (
        "| Требование | Драйвер | ADR | Реализация | Проверка | Статус |\n"
        "|---|---|---|---|---|---|\n"
        "| FR-A | D | ADR | `src/a.py` | `test_a` | ☑ done |\n"
        "| FR-B | D | ADR | `src/b.py` | `test_b` | {b} |\n"
    )
    for name, mark in (("fx-closed", "☑"), ("fx-half", "☐")):
        package = architecture / name
        package.mkdir(parents=True, exist_ok=True)
        (package / "06-traceability.md").write_text(matrix.format(b=mark), encoding="utf-8")
    backlog = parse_backlog(FIXTURE_BACKLOG)
    ledger = (
        "| round | item | type | gate | DoD | Δtests | commit | status | note |\n"
        "|---|---|---|---|---|---|---|---|---|\n"
        "| 7 | FR-001-a | impl | go | green | +1 | — | done | ok |\n"
        "| id пункта | фаза | дизайн-пакет | заметка |\n"
        "|---|---|---|---|\n"
        "| FR-001-a | done | `docs/architecture/fx-closed/` | note with a | pipe |\n"
    )
    rounds, items = ledger_rows(ledger)
    blockers = "## BLK-9 · ITEM-2 · answered: Вариант 1\n## BLK-8 · ITEM-3 · open\n"
    blocked_phase = [["ITEM-2", "blocked", "—", ""], ["ITEM-3", "blocked", "—", ""]]
    slice_problems, compared, skipped = check_backlog_slices(backlog, architecture)
    long_body = parse_backlog(
        FIXTURE_BACKLOG.replace("outcome a.", "outcome a. " + "x" * SLICE_BODY_LIMIT)
    )
    chronicle = parse_backlog(FIXTURE_BACKLOG.replace("outcome a.", "outcome a, DoD green р.12."))
    odd_status = parse_backlog(FIXTURE_BACKLOG.replace("(⛔ BLK-1)", "(blocked)"))
    duplicate = parse_backlog(
        FIXTURE_BACKLOG.replace("`FR-002-e` — parked.", "`FR-002-c` — parked.")
    )
    deep = parse_backlog(
        FIXTURE_BACKLOG.replace(
            "- **Срез `FR-002-b` (⛔ BLK-1):** waits for the owner.",
            "- **Срез `FR-002-b` (⛔ BLK-1):** waits for the owner.\n"
            "- **Срез `FR-002-a-1-x` (☐):** too deep.",
        )
    )
    many = parse_backlog(
        FIXTURE_BACKLOG.replace(
            "  - `FR-002-e` — parked.",
            "  - `FR-002-e` — parked.\n- **Кандидаты (не выбираются):**\n"
            + "".join(f"  - `FR-002-n{i}` — x; закрывает: —\n" for i in range(4)),
        )
    )
    checks: dict[str, bool] = {
        "round rows are told from item rows": ([c[0] for c in rounds], [c[0] for c in items])
        == (["7"], ["FR-001-a"]),
        "a note carrying a pipe stays one item row": len(items) == 1,
        "the fixture backlog parses into its epics, slices and registrations": (
            [e.id for e in backlog.epics],
            [s.id for s in backlog.slices],
            [c.id for c in backlog.candidates],
            [d.id for d in backlog.deferred],
        )
        == (
            ["EPIC-01", "EPIC-02"],
            ["FR-001-a", "FR-002-a", "FR-002-b"],
            ["FR-002-c", "FR-002-d"],
            ["FR-002-e"],
        ),
        "a continued slice line joins its body": "second line" in backlog.slices[1].body,
        "the fixture backlog is well-formed": not (
            check_backlog_shape(backlog)
            + check_navigator_bodies(backlog)
            + check_backlog_identities(backlog)
            + check_epic_counters(backlog)
        ),
        "matrices confirm the fixture marks": (slice_problems, compared, skipped) == ([], 2, 1),
        "a mark the matrix denies is reported": check_backlog_slices(
            parse_backlog(FIXTURE_BACKLOG.replace("(☑):** outcome a", "(☐):** outcome a")),
            architecture,
        )[0]
        == ["slice FR-001-a: backlog says '☐', matrix is 2/2 (☑)"],
        "a status outside the vocabulary is reported": check_backlog_shape(odd_status)
        == [
            "slice FR-002-b: status 'blocked' is outside the vocabulary "
            "☑ · ◐ · ☐ · ⛔ <ref> · ⏸ <decision> · ✂ <decision>"
        ],
        "a missing epic field is reported": check_backlog_shape(
            parse_backlog(FIXTURE_BACKLOG.replace("- **Блокеры:** нет.\n", ""))
        )
        == ["epic EPIC-01: section lacks field(s) Блокеры"],
        "a body over the limit is reported": any(
            "limit 400" in p for p in check_navigator_bodies(long_body)
        ),
        "a chronicle token in a body is reported": check_navigator_bodies(chronicle)
        == ["slice FR-001-a: body carries chronicle ('DoD')"],
        "a candidate without a closes clause is reported": check_navigator_bodies(
            parse_backlog(FIXTURE_BACKLOG.replace("; закрывает: —", ""))
        )
        == ["candidate FR-002-d: entry names no `закрывает: <FR/NFR> | —` clause"],
        "a summary cell over the limit is reported": any(
            "summary EPIC-02" in p
            for p in check_navigator_bodies(
                parse_backlog(FIXTURE_BACKLOG.replace("| `FR-002-b` |", "| " + "y" * 201 + " |"))
            )
        ),
        "an id listed twice is reported": check_backlog_identities(duplicate)
        == ["candidate FR-002-c is also listed as deferred"],
        "a summary row without a section is reported": check_backlog_identities(
            parse_backlog(FIXTURE_BACKLOG.replace("### EPIC-01 — one", "### EPIC-03 — one"))
        )
        == [
            "summary row EPIC-01 has no `### EPIC-01` section",
            "epic section EPIC-03 has no row in `## Сводка`",
        ],
        "a counter the summary denies is reported": check_epic_counters(
            parse_backlog(FIXTURE_BACKLOG.replace("done: **1**", "done: **2**"))
        )
        == ["epic counter says done=2, summary table has 1"],
        "done against a half-closed matrix is reported": check_ledger_items(
            [["FR-002-a", "done", "`docs/architecture/fx-half/`", "note"]], backlog, architecture
        )
        == [
            "item FR-002-a: ledger says done, backlog says '◐'",
            "item FR-002-a: ledger says done, matrix is 1/2",
        ],
        "a ledger item that is not a slice is reported": check_ledger_items(
            [["FR-002-c", "done", "—", ""]], backlog, architecture
        )
        == [
            "item FR-002-c: ledger names it, but the backlog has no slice line for it — "
            "a round works only on slices; a candidate is promoted first"
        ],
        "an audit item needs no slice": not check_ledger_items(
            [["N-21", "done", "—", ""]], backlog, architecture
        ),
        "a blocked phase reads as blocked while a blocker is open": check_ledger_items(
            [["FR-002-a", "blocked", "—", ""]],
            backlog,
            architecture,
            awaiting_owner=frozenset({"FR-002-a"}),
        )
        == ["item FR-002-a: ledger says blocked, backlog says '◐'"],
        "an answered blocker leaves the stale phase to its own check": not check_ledger_items(
            [["FR-002-a", "blocked", "—", ""]], backlog, architecture
        ),
        "a missing matrix is reported for a done item": check_ledger_items(
            [["FR-001-a", "done", "`docs/architecture/no-such-package/`", ""]],
            backlog,
            architecture,
        )
        == ["item FR-001-a: package no-such-package has no readable matrix"],
        "a blocked item may have no matrix": not check_ledger_items(
            [["FR-002-b", "blocked", "`docs/architecture/no-such-package/`", ""]],
            backlog,
            architecture,
        ),
        "a repeated item identity is reported": check_register_identities(
            [], [["ITEM-1", "done", "—", ""], ["ITEM-1", "blocked", "—", ""]], ""
        )
        == ["ledger item ITEM-1 appears 2 times"],
        "a repeated blocker identity is reported": check_register_identities(
            [], [], "## BLK-9 · A · open\n## BLK-9 · B · open\n"
        )
        == ["blocker BLK-9 appears 2 times"],
        "an answered blocker leaves no blocked phase behind": check_unblocked_items(
            blocked_phase, blockers
        )
        == [
            "item ITEM-2: every blocker is answered, but the phase is still blocked; "
            "the round that consumes the decision is pending"
        ],
        "an open blocker keeps its item blocked": not check_unblocked_items(
            [["ITEM-3", "blocked", "—", ""]], blockers
        ),
        "a blocked item with no blocker is reported": check_unblocked_items(
            [["ITEM-9", "blocked", "—", ""]], blockers
        )
        == [
            "item ITEM-9: the phase is blocked, but no blocker in blockers.md names it; "
            "nothing can release it"
        ],
        "a settled item needs no blocker": not check_unblocked_items(
            [["ITEM-9", "done", "—", ""]], blockers
        ),
        "an id below the depth limit is reported": any(
            p.startswith("slice FR-002-a-1-x: id is 3 segments below FR-002")
            for p in check_new_slices(deep, backlog)
        ),
        "outside a round the candidate bound does not apply": not check_new_candidates(
            many, backlog, in_round=False
        ),
        "too many new candidates are reported": check_new_candidates(many, backlog, in_round=True)
        == [
            "candidates: 4 registered since HEAD (FR-002-n0, FR-002-n1, FR-002-n2, FR-002-n3), "
            "limit 3 per package"
        ],
    }
    for name in ("fx-closed", "fx-half"):
        (architecture / name / "06-traceability.md").unlink()
        (architecture / name).rmdir()
    architecture.rmdir()
    return checks


def self_test() -> int:
    """Report through an explicit UTF-8 stream: the gate runs on a cp1251 console, and a check
    named after a mark it judges must not kill the run that was about to prove it."""
    out = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    failures = 0
    for name, passed in self_test_checks().items():
        out.write(f"{'PASS' if passed else 'FAIL'} {name}\n")
        failures += int(not passed)
    out.flush()
    return 1 if failures else 0


def report(
    out: io.TextIOWrapper,
    problems: list[str],
    notes: list[str],
    items: list[list[str]],
    backlog: Backlog,
    compared: int,
    skipped: int,
    item: str,
) -> int:
    # A round is answerable for its own item: someone else's disagreement is a fact to report, not
    # a reason to hold this round's commit, and a gate that fails on it gets ignored.
    fatal = [
        problem
        for problem in problems
        if not item or re.search(rf"(?<![\w-]){re.escape(item)}(?![\w-])", problem)
    ]
    for note in notes:
        out.write(f"NOTE registers: {note}\n")
    for problem in problems:
        out.write(f"{'FAIL' if problem in fatal else 'WARN'} registers: {problem}\n")
    if fatal:
        out.flush()
        return 1
    matrices = sum(1 for cells in items if PACKAGE_PATTERN.search(cells[2]))
    out.write(
        f"PASS registers agree with the package matrices ({matrices} items, {compared} slices "
        f"compared, {skipped} in a process state, {len(backlog.epics)} epics)\n"
    )
    out.flush()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument(
        "--item",
        default="",
        help="fail only on rows of this item; disagreements elsewhere are reported as warnings",
    )
    arguments = parser.parse_args()
    if arguments.self_test:
        return self_test()

    broken = [name for name, passed in self_test_checks().items() if not passed]
    rounds, items = ledger_rows(state_path().read_text(encoding="utf-8"))
    blockers_text = blockers_path().read_text(encoding="utf-8")
    backlog = parse_backlog(BACKLOG.read_text(encoding="utf-8"))
    slice_problems, compared, skipped = check_backlog_slices(backlog)
    problems = (
        [f"this check no longer reads its own fixtures: {name}" for name in broken]
        + check_backlog_shape(backlog)
        + check_navigator_bodies(backlog)
        + check_backlog_identities(backlog)
        + check_epic_counters(backlog)
        + slice_problems
        + check_register_identities(rounds, items, blockers_text)
        + check_ledger_items(items, backlog, awaiting_owner=items_awaiting_owner(blockers_text))
        + check_unblocked_items(items, blockers_text)
    )
    # what this round added is judged against HEAD; a HEAD without the grammar has nothing
    # to compare with and says so instead of judging every line as new
    head_backlog_text = head_text("docs/loop/backlog.md")
    head_backlog = parse_backlog(head_backlog_text) if head_backlog_text is not None else None
    notes: list[str] = []
    if head_backlog is None or not head_backlog.has_summary:
        notes.append("HEAD backlog is not in the navigator grammar; checks against HEAD skipped")
    else:
        problems += check_new_slices(backlog, head_backlog)
        problems += check_new_candidates(backlog, head_backlog)
    out = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    return report(out, problems, notes, items, backlog, compared, skipped, arguments.item)


if __name__ == "__main__":
    sys.exit(main())
