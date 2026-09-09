"""PreToolUse hook: deny background execution inside loop rounds.

One file serves both drivers. Active ONLY when HONESTML_LOOP_ROUND is set (the
driver exports it for round sessions); interactive sessions are untouched. Exit 2
blocks the call before it runs and returns guidance; exit 0 is silent.

Deterministic on purpose: LOOP.md §2b/§5 already forbid background gates and
poll loops, but instruction-level bans lose to habit. A leftover background task
wakes the finished round (rewriting the expired cache), and every poll turn
re-reads the whole context. Foreground with an adequate timeout costs one turn.

Detaching arrives two ways and both are caught: the tool's own background flag,
and a command that detaches itself (trailing `&`, nohup, Start-Job, Start-Process).
A read-only command is judged by what it runs, not by what it prints: grepping the
sources for "Start-Process" is not a detach, and denying it teaches nothing.

Polling is the same waste wearing a wait: a watch or a task read that returns well
inside the round's own timeout gets asked again and again, and every answer costs
the whole context. A wait earns its turn only when it is long enough to be the last
one — see LONG_WAIT_MIN_SECONDS.

Every invocation that gets past the env guard is appended to
ops/loop/hook-audit.log (git-ignored), which keeps its recent tail and no more.
"""

import datetime
import io
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DETACHING = re.compile(r"&\s*$|\b(?:nohup|Start-Job|Start-Process)\b", re.IGNORECASE)
# a command whose head is one of these only reads; its arguments may quote anything
READERS = re.compile(
    r"^\s*(?:grep|rg|cat|head|tail|sed|awk|find|ls|wc|diff|findstr|"
    r"Select-String|Get-Content|Get-ChildItem|git\s+(?:log|show|diff|status|blame|ls-files))\b",
    re.IGNORECASE,
)
# below this a wait is a poll: the round pays its whole context for another look
LONG_WAIT_MIN_SECONDS = 600
# how each waiting tool spells "how long", in milliseconds, and its no-deadline escape
WAIT_TOOLS = {"Monitor": ("timeout_ms", "persistent"), "TaskOutput": ("timeout", None)}
# the audit log is a tail, not an archive: past this size the older half goes
AUDIT_MAX_BYTES = 1 << 20


def audit(tool: str, verdict: str, detail: str = "") -> None:
    path = ROOT / "ops/loop/hook-audit.log"
    try:
        if path.exists() and path.stat().st_size > AUDIT_MAX_BYTES:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
            path.write_text("".join(lines[len(lines) // 2 :]), encoding="utf-8")
        timestamp = datetime.datetime.now().isoformat(timespec="seconds")
        line = f"{timestamp} | pre:{tool} | {verdict} | {detail}\n"
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(line)
    except OSError:
        pass


def launches_background(tool_input: dict[str, object]) -> bool:
    if tool_input.get("run_in_background"):
        return True
    command = str(tool_input.get("command") or "")
    if READERS.match(command):
        return False
    return bool(DETACHING.search(command))


def short_wait(tool_name: str, tool_input: dict[str, object]) -> bool:
    """Whether this call waits too briefly to be anything but a poll."""
    if tool_name not in WAIT_TOOLS:
        return False
    field, forever = WAIT_TOOLS[tool_name]
    if forever and tool_input.get(forever):
        return False
    if tool_name == "TaskOutput" and not tool_input.get("block", True):
        return True
    deadline = tool_input.get(field)
    if not isinstance(deadline, (int, float, str)):
        return True  # every waiting tool here defaults to less than the floor
    try:
        millis = int(deadline)
    except ValueError:
        return True
    return millis < LONG_WAIT_MIN_SECONDS * 1000


def deny(tool_name: str, head: str, message: str) -> int:
    audit(tool_name, "DENY", head)
    err = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    err.write(message)
    err.flush()
    return 2


def main() -> int:
    if not os.environ.get("HONESTML_LOOP_ROUND"):
        return 0
    data = json.load(io.TextIOWrapper(sys.stdin.buffer, encoding="utf-8"))
    tool_input = data.get("tool_input") or {}
    tool_name = data.get("tool_name", "?")
    head = str(tool_input.get("command") or tool_input.get("prompt") or "")[:120]
    if launches_background(tool_input):
        return deny(
            tool_name,
            head,
            "Фоновые задачи в раундах петли запрещены (LOOP.md §2b/§5): оставленная "
            "задача будит завершённый раунд и пересоздаёт протухший кэш, а каждый "
            "poll-ход заново оплачивает весь контекст. Запусти то же самое FOREGROUND "
            "одним вызовом с достаточным timeout (потолок Bash-вызова — 60 минут; полный "
            "DoD — `powershell -NoProfile -File ops/loop/gates.ps1`).\n",
        )
    if short_wait(tool_name, tool_input):
        return deny(
            tool_name,
            head,
            "Короткое ожидание в раунде петли — это poll: ответ вернётся раньше, чем "
            "работа закончится, и следующий ход заново оплатит весь контекст. Жди один "
            f"раз и достаточно долго (от {LONG_WAIT_MIN_SECONDS} с; Monitor — persistent), "
            "либо запусти работу FOREGROUND одним вызовом.\n",
        )
    # a silent hook and an unmounted one look the same in a log that records only refusals
    audit(tool_name, "pass", head)
    return 0


if __name__ == "__main__":
    sys.exit(main())
