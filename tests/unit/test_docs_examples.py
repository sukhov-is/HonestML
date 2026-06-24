"""M9-3 (NFR-DLV-6): the quickstart examples are executed, not trusted.

Authoring contract (ADR-0077 §4): every ```python block in docs/quickstart.md is
self-sufficient in sequence and runs in ONE shared namespace; illustrative
non-executable snippets use ```text fences.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

# slow (not unit): one job, not the 3 OS x 3 Python matrix (ADR-0077 §4)
pytestmark = pytest.mark.slow

_QUICKSTART = Path(__file__).resolve().parents[2] / "docs" / "quickstart.md"


def test_quickstart_python_blocks_execute() -> None:
    text = _QUICKSTART.read_text(encoding="utf-8")
    blocks = re.findall(r"```python\n(.*?)```", text, flags=re.DOTALL)
    assert len(blocks) >= 3, "quickstart lost its executable examples"
    namespace: dict = {}
    for i, block in enumerate(blocks):
        try:
            exec(compile(block, f"<quickstart block {i}>", "exec"), namespace)  # noqa: S102
        except Exception as exc:  # pragma: no cover - the assertion message is the point
            pytest.fail(f"quickstart block {i} failed: {exc}\n---\n{block}")


_GUIDE_PAGES = sorted((_QUICKSTART.parent / "guide").glob("*.md"))


@pytest.mark.parametrize("page", _GUIDE_PAGES, ids=lambda p: p.stem)
def test_guide_python_blocks_are_one_shot(page: Path) -> None:
    """Guide contract: every ```python block is self-contained (fresh namespace each)."""
    text = page.read_text(encoding="utf-8")
    blocks = re.findall(r"```python\n(.*?)```", text, flags=re.DOTALL)
    assert blocks, f"{page.name} has no executable examples"
    for i, block in enumerate(blocks):
        try:
            exec(compile(block, f"<{page.stem} block {i}>", "exec"), {})  # noqa: S102
        except Exception as exc:  # pragma: no cover - the assertion message is the point
            pytest.fail(f"{page.name} block {i} is not one-shot: {exc}\n---\n{block}")
