"""Release version gate (ADR-0077 §2): tag == pyproject.version == honestml.__version__.

There are deliberately TWO version sources in the repo (the static
``pyproject.toml`` field and the ``__version__`` constant pinned by
``test_public_api``); a release tag must match BOTH — bumping one without the
other would publish a wheel with a lying ``honestml.__version__``. Pure function +
CLI so the same check runs in ``release.yml`` and in the unit suite.
"""
# ruff: noqa: T201  (a CLI gate script: stdout/stderr IS the interface)

from __future__ import annotations

import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent


def check_tag_version(tag: str, pyproject_text: str, init_text: str) -> str:
    """Return the verified version or raise ``ValueError`` on any mismatch."""
    # regex instead of tomllib: the gate must run on py3.10 (unit suite) too. Anchored to
    # the [project] block — a `version` in some [tool.*] section must not shadow it.
    block = re.search(
        r"^\[project\]\n(.*?)(?=^\[|\Z)", pyproject_text, flags=re.MULTILINE | re.DOTALL
    )
    project = re.search(
        r'^version = "([^"]+)"', block.group(1) if block else "", flags=re.MULTILINE
    )
    if project is None:
        raise ValueError("pyproject.toml carries no static [project] version field")
    version = project.group(1)
    match = re.search(r'^__version__ = "([^"]+)"', init_text, flags=re.MULTILINE)
    if match is None:
        raise ValueError("honestml/__init__.py carries no __version__ constant")
    dunder = match.group(1)
    tag_version = tag.removeprefix("refs/tags/").removeprefix("v")
    if not (tag_version == version == dunder):
        raise ValueError(
            f"version mismatch: tag={tag_version!r}, pyproject={version!r}, "
            f"honestml.__version__={dunder!r} — all three must be equal"
        )
    return version


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: check_tag_version.py <tag>", file=sys.stderr)
        return 2
    try:
        version = check_tag_version(
            argv[1],
            (_ROOT / "pyproject.toml").read_text(encoding="utf-8"),
            (_ROOT / "src" / "honestml" / "__init__.py").read_text(encoding="utf-8"),
        )
    except ValueError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    print(f"OK: releasing version {version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
