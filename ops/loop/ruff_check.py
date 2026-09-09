"""PostToolUse hook: run ruff on the sources a tool call changed.

One file serves both drivers: structured edits carry a file path, patches carry
their paths in the patch body, and both resolve to the same ruff run.

Exit 2 feeds violations back to the model; exit 0 is silent (zero tokens).
Scoped to src/tests/notebooks — the trees ruff owns — so skill fixtures and docs
are never blocked. Notebooks are linted like any other source: ruff reads .ipynb
directly, and pyproject grants them only the print allowance.
"""

import json
import subprocess
import sys
from pathlib import Path

PATCH_PATH_MARKERS = (
    "*** Add File: ",
    "*** Update File: ",
    "*** Delete File: ",
    "*** Move to: ",
)
LINTED_SUFFIXES = (".py", ".ipynb")
LINTED_ROOTS = ("src", "tests", "notebooks")


def changed_paths(data: dict[str, object]) -> list[str]:
    tool_input = data.get("tool_input") or {}
    if not isinstance(tool_input, dict):
        return []

    tool_name = data.get("tool_name")
    if tool_name == "apply_patch":
        command = tool_input.get("command", "")
        if not isinstance(command, str):
            return []
        return [
            line.removeprefix(marker).strip()
            for line in command.splitlines()
            for marker in PATCH_PATH_MARKERS
            if line.startswith(marker)
        ]

    raw_path = tool_input.get("file_path") or tool_input.get("notebook_path", "")
    return [raw_path] if isinstance(raw_path, str) and raw_path else []


def scoped_lint_paths(raw_paths: list[str], root: Path) -> list[Path]:
    found: set[Path] = set()
    for raw_path in raw_paths:
        if not raw_path.endswith(LINTED_SUFFIXES):
            continue
        path = Path(raw_path)
        resolved = (path if path.is_absolute() else root / path).resolve()
        try:
            rel = resolved.relative_to(root)
        except ValueError:
            continue
        if rel.parts and rel.parts[0] in LINTED_ROOTS and resolved.is_file():
            found.add(rel)
    return sorted(found)


def main() -> int:
    data = json.load(sys.stdin)
    root = Path(__file__).resolve().parents[2]
    paths = scoped_lint_paths(changed_paths(data), root)
    if not paths:
        return 0
    result = subprocess.run(
        [
            "uv",
            "run",
            "ruff",
            "check",
            *map(str, paths),
            "--output-format=concise",
        ],
        capture_output=True,
        text=True,
        cwd=root,
    )
    if result.returncode != 0:
        sys.stderr.write(result.stdout + result.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
