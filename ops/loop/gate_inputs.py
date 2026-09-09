"""Digest of everything a gate's outcome can depend on.

A gate that reruns over a tree it already judged costs the round half an hour and learns
nothing. The digest is what lets ops/loop/gates.ps1 tell "nothing moved" from "nothing was
measured": same digest, same tree, so the recorded verdict still describes it.

The closure is content-addressed, not stamped. Size and modification time miss an edit that
keeps both, and a cache that skips a changed tree records a verdict for a run that never
happened — the one failure this must not have. Reading and hashing the closure costs about
a second against a suite that costs half an hour, so there is nothing to buy by trusting a
stamp.

The environment counts as input: uv.lock pins the libraries the suite runs against, and a
resolution that moves without pyproject moving changes outcomes with no source edit.

    uv run python ops/loop/gate_inputs.py [--list]
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# Everything the suite reads: the code under test, the tests, the corpora and scripts they load,
# the docs and release metadata they assert against, the tool configuration, and the dependency
# resolution the run happens in.
INPUTS = (
    "src",
    "tests",
    "benchmarks",
    "scripts",
    "docs",
    "audits",
    ".github/workflows",
    "mkdocs.yml",
    "README.md",
    "LICENSE",
    "pyproject.toml",
    "uv.lock",
    ".importlinter",
    ".pre-commit-config.yaml",
    "ops/loop/gates.ps1",
    "ops/loop/gate_inputs.py",
)
# Kept in the docs tree but out of the closure: the decision registers, the loop's own ledger and
# the project's historical journal. Only the registers and check-index gates read any of this, and
# they are never served from cache; the suite's verdict cannot turn on it, while hashing it would
# drop the cache on every note a round takes.
NOT_INPUTS = (
    "docs/adr",
    "docs/architecture",
    "docs/audit",
    "docs/backlog.md",
    "docs/implementation",
    "docs/loop",
)
# Written by the very runs this digest gates, and after it is taken.
IGNORED_DIRS = frozenset({"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"})
IGNORED_SUFFIXES = frozenset({".pyc", ".pyo"})


def in_closure(path: Path) -> bool:
    """Whether this file's content can change what a gate decides."""
    if IGNORED_DIRS & set(path.parts) or path.suffix in IGNORED_SUFFIXES:
        return False
    relative = path.relative_to(ROOT).as_posix()
    return not any(relative == skip or relative.startswith(f"{skip}/") for skip in NOT_INPUTS)


def root_files(name: str) -> list[Path]:
    root = ROOT / name
    if root.is_file():
        return [root]
    return sorted(path for path in root.rglob("*") if path.is_file() and in_closure(path))


def input_files() -> list[Path]:
    """Every file in the closure, or empty when a root contributes nothing.

    A root that yields no files has moved or been renamed, and the digest would then describe
    a smaller tree than the gate actually judges — silently, and identically on every run.
    """
    files: list[Path] = []
    for name in INPUTS:
        found = root_files(name)
        if not found:
            return []
        files.extend(found)
    return files


def digest(files: list[Path]) -> str:
    """Hash of the closure as the tools read it — path and content, line endings normalised.

    No gate's verdict turns on whether a file ends its lines with CRLF or LF, and this tree
    carries both. Hashing the raw bytes would drop the cache whenever a checkout rewrote a
    file it did not change.
    """
    accumulator = hashlib.blake2b(digest_size=16)
    for path in files:
        accumulator.update(path.relative_to(ROOT).as_posix().encode())
        accumulator.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return accumulator.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--list", action="store_true", help="report the closure instead of hashing")
    arguments = parser.parse_args()

    files = input_files()
    if not files:
        missing = ", ".join(name for name in INPUTS if not root_files(name))
        print(f"gate inputs are incomplete: {missing} contributed no files", file=sys.stderr)
        return 1
    if arguments.list:
        print(f"{len(files)} files across {', '.join(INPUTS)}")
        for path in files:
            print(f"  {path.relative_to(ROOT).as_posix()}")
        return 0
    print(digest(files))
    return 0


if __name__ == "__main__":
    sys.exit(main())
