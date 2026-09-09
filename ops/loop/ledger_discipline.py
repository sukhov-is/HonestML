"""PostToolUse hook: chronicle limits for docs/loop/state.md (LOOP.md §5).

One file serves both drivers: the tool that wrote the ledger differs per provider,
the rule does not. Exit 2 feeds the violation back to the model; exit 0 is silent
(zero tokens). Validates the RESULTING FILE: lines added relative to HEAD (git
diff), so every write path is covered — structured edits, whole-file writes and
shell appends alike. Legacy over-limit rows already in HEAD never block anything.
Violations keep re-blocking on every subsequent touch of the file until fixed —
by design.

Calls are pre-filtered by a cheap guard: a write tool whose payload names the
ledger. Every invocation is appended to ops/loop/hook-audit.log (git-ignored) so
the "did the hook even run" question is answerable by reading one file.

Rounds row  (numeric first cell): note (9th cell)  <= 500 chars.
Items row   (non-numeric first): заметка (4th cell) <= 200 chars.
"""

import datetime
import io
import json
import subprocess
import sys

from implementation_round_guard import (
    ITEMS_NOTE_LIMIT,
    ROOT,
    ROUNDS_NOTE_LIMIT,
    STATE_PATH,
    WRITE_TOOLS,
    split_cells,
)
from no_background_in_rounds import READERS

STATE_REL = STATE_PATH.relative_to(ROOT).as_posix()


def audit(tool: str, verdict: str, detail: str = "") -> None:
    try:
        timestamp = datetime.datetime.now().isoformat(timespec="seconds")
        line = f"{timestamp} | {tool} | {verdict} | {detail}\n"
        with open(ROOT / "ops/loop/hook-audit.log", "a", encoding="utf-8") as fh:
            fh.write(line)
    except OSError:
        pass


def touches_ledger(tool_name: str, tool_input: dict) -> bool:
    """A write tool naming the ledger anywhere in its payload — path, command or patch.

    A shell that only reads the ledger changes nothing in it, so it is not judged against the
    chronicle limits: grepping the ledger is how a round orients itself.
    """
    if tool_name not in WRITE_TOOLS:
        return False
    command = str(tool_input.get("command") or "")
    if READERS.match(command):
        return False
    payload = " ".join(str(value) for value in tool_input.values())
    return "docs/loop/state.md" in payload.replace("\\", "/").lower()


def is_table_noise(line: str, cells: list[str]) -> bool:
    stripped = line.strip()
    if not stripped.startswith("|"):
        return True
    if all(ch in "-: |" for ch in stripped):  # separator row
        return True
    return cells[0] in ("round", "id пункта")  # header rows


def violations(lines: list[str]) -> list[str]:
    found: list[str] = []
    for line in lines:
        cells = split_cells(line)
        if is_table_noise(line, cells):
            continue
        if cells[0].isdigit() and len(cells) >= 9:  # rounds row
            note = " | ".join(cells[8:])
            if len(note) > ROUNDS_NOTE_LIMIT:
                found.append(
                    f"строка раунда {cells[0]}: note {len(note)} симв. > {ROUNDS_NOTE_LIMIT}"
                )
        elif not cells[0].isdigit() and len(cells) >= 4:  # items row
            note = " | ".join(cells[3:])
            if len(note) > ITEMS_NOTE_LIMIT:
                found.append(
                    f"строка пункта {cells[0]}: заметка {len(note)} симв. > {ITEMS_NOTE_LIMIT}"
                )
    return found


def added_lines_vs_head() -> list[str] | None:
    result = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={ROOT}",
            "-C",
            str(ROOT),
            "diff",
            "-U0",
            "HEAD",
            "--",
            STATE_REL,
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        return None
    return [
        ln[1:]
        for ln in result.stdout.splitlines()
        if ln.startswith("+") and not ln.startswith("+++")
    ]


def main() -> int:
    data = json.load(io.TextIOWrapper(sys.stdin.buffer, encoding="utf-8"))
    tool_name = data.get("tool_name", "")
    if not touches_ledger(tool_name, data.get("tool_input") or {}):
        return 0

    added = added_lines_vs_head()
    if added is None:  # git unavailable -> fail open, never block
        audit(tool_name, "git-fail")
        return 0

    found = violations(added)
    if not found:
        audit(tool_name, "pass", f"added_lines={len(added)}")
        return 0

    audit(tool_name, "BLOCK", "; ".join(found)[:200])
    err = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    err.write(
        "LOOP.md §5, лимиты хроники нарушены:\n  "
        + "\n  ".join(found)
        + "\nСократи до статус-фактов и указателей (срез, вердикт гейта, Δtests, "
        "путь пакета, «следующий раунд»). Полную хронику несёт "
        "docs/implementation/changelog.md / дизайн-пакет — в гроссбухе она не нужна.\n"
    )
    err.flush()
    return 2


if __name__ == "__main__":
    sys.exit(main())
