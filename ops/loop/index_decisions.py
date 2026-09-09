"""Project the architecture packages and their decisions into one greppable index.

A round that needs "what did we already decide about the honest holdout" has to open packages one
by one: the answer lives in 04-adr/ of 29 directories, so a bare reference is unresolvable without
knowing its package first. This writes docs/architecture/INDEX.md, where one line per package and
one per decision carry the identity, status and title.

The index is a projection, never a second source of truth: it is derived from the packages on
every write, and --check tells whether the file on disk still matches what the packages say, the
way `ruff format --check` does for formatting.

    uv run python ops/loop/index_decisions.py --write
    uv run python ops/loop/index_decisions.py --check
"""

from __future__ import annotations

import argparse
import io
import re
import sys
from pathlib import Path

from implementation_round_guard import ROOT, matrix_state

PACKAGES = ROOT / "docs/architecture"
INDEX = PACKAGES / "INDEX.md"
# An id carries its package tag (ADR-BTEG-0001) or is bare (ADR-0001); the title follows after a
# dash, a colon or a middle dot. The status line is a bullet in some packages and a bare bold
# field in others, and it may carry date and drivers after the first middle dot.
ADR_TITLE = re.compile(r"^#\s*(ADR-(?:[A-Za-z0-9]+-)*\d+)\s*[:—·.-]\s*(.+?)\s*$", re.MULTILINE)
ADR_STATUS = re.compile(r"^-?\s*\*{0,2}Статус:?\*{0,2}:?\s*(.+?)\s*(?:·|$)", re.MULTILINE)
HEADER = (
    "<!-- порождается ops/loop/index_decisions.py --write; правки вносятся в сами пакеты -->\n"
    "# Индекс архитектурных решений\n"
)


def decisions(package: Path) -> list[tuple[str, str, str]]:
    """(id, status, title) of every ADR in the package, in file order."""
    found: list[tuple[str, str, str]] = []
    for adr in sorted((package / "04-adr").glob("ADR-*.md")):
        text = adr.read_text(encoding="utf-8", errors="replace")
        title_match = ADR_TITLE.search(text)
        status_match = ADR_STATUS.search(text)
        found.append(
            (
                title_match.group(1) if title_match else adr.stem,
                status_match.group(1).strip("* ") if status_match else "?",
                title_match.group(2) if title_match else adr.stem,
            )
        )
    return found


def packages() -> list[Path]:
    """Every design package under docs/architecture, at whatever depth it sits.

    A package that grew slices holds them as packages of its own, so identity is the path below
    docs/architecture and not a single directory name; what makes a directory a package is that
    it carries decisions or a matrix, not where it is nested."""
    return sorted(
        directory
        for directory in PACKAGES.rglob("*")
        if directory.is_dir()
        if (directory / "04-adr").is_dir() or (directory / "06-traceability.md").is_file()
    )


def render() -> str:
    lines = [HEADER]
    found = packages()
    lines.append(f"\nПакетов: {len(found)}.\n")
    for package in found:
        name = package.relative_to(PACKAGES).as_posix()
        state = matrix_state(package)
        matrix = f"{state[0]}/{state[1]}" if state else "нет матрицы"
        adrs = decisions(package)
        lines.append(f"\n## {name}\n")
        lines.append(f"матрица {matrix} · решений {len(adrs)}\n")
        for identity, status, title in adrs:
            lines.append(f"- `{name}/{identity}` · {status} · {title}\n")
    return "".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help="regenerate the index")
    mode.add_argument("--check", action="store_true", help="fail when the index is out of date")
    arguments = parser.parse_args()

    out = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    if not PACKAGES.is_dir():
        out.write(f"SKIP index: {PACKAGES.relative_to(ROOT)} does not exist here\n")
        out.flush()
        return 0

    projection = render()
    if arguments.write:
        INDEX.write_text(projection, encoding="utf-8")
        rel = INDEX.relative_to(ROOT)
        out.write(f"index: wrote {rel} ({len(projection.splitlines())} lines)\n")
        out.flush()
        return 0

    current = INDEX.read_text(encoding="utf-8") if INDEX.is_file() else ""
    if current == projection:
        out.write(f"PASS index matches the packages ({len(projection.splitlines())} lines)\n")
        out.flush()
        return 0
    out.write(
        "FAIL index is out of date - run: uv run python ops/loop/index_decisions.py --write\n"
    )
    out.flush()
    return 1


if __name__ == "__main__":
    sys.exit(main())
