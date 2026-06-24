"""Generate ``llms.txt`` and ``llms-full.txt`` for the published docs site.

``llms.txt`` (https://llmstxt.org) is a machine-readable index of the documentation;
``llms-full.txt`` additionally inlines the full markdown source of every nav page.
Both are derived from the ``nav`` of ``mkdocs.yml`` and written into the built site
directory, so they are served from the site root next to the rendered pages.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

_ROOT = Path(__file__).resolve().parents[1]


def _nav_pages(nav: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """Flatten the mkdocs ``nav`` into ordered ``(title, docs-relative path)`` pairs."""
    pages: list[tuple[str, str]] = []
    for item in nav:
        for title, value in item.items():
            if isinstance(value, str):
                pages.append((title, value))
            else:
                pages.extend(_nav_pages(value))
    return pages


def _page_url(site_url: str, path: str) -> str:
    """Public URL of a nav page under mkdocs ``use_directory_urls`` (the default)."""
    base = site_url.rstrip("/") + "/"
    if path == "index.md":
        return base
    return base + path.removesuffix(".md") + "/"


def _first_paragraph(text: str) -> str:
    """The first plain prose line of a page — a one-line note for the llms.txt index."""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith(("#", "<", "`", "-", "[", "!", ">", "|")):
            return stripped
    return ""


def build(site_dir: Path) -> tuple[Path, Path]:
    """Write ``llms.txt`` and ``llms-full.txt`` into *site_dir*; return their paths."""
    config = yaml.safe_load((_ROOT / "mkdocs.yml").read_text(encoding="utf-8"))
    site_url = config["site_url"]
    pages = [
        (title, path, (_ROOT / "docs" / path).read_text(encoding="utf-8"))
        for title, path in _nav_pages(config["nav"])
    ]

    head = f"# {config['site_name']}\n\n> {config['site_description']}\n"
    index_lines = [head, "\n## Documentation\n"]
    for title, path, text in pages:
        note = _first_paragraph(text)
        suffix = f": {note}" if note else ""
        index_lines.append(f"- [{title}]({_page_url(site_url, path)}){suffix}")

    full_parts = [head]
    for title, path, text in pages:
        full_parts.append(
            f"\n---\n\n# {title}\nURL: {_page_url(site_url, path)}\n\n{text.strip()}\n"
        )

    site_dir.mkdir(parents=True, exist_ok=True)
    llms = site_dir / "llms.txt"
    llms_full = site_dir / "llms-full.txt"
    llms.write_text("\n".join(index_lines) + "\n", encoding="utf-8")
    llms_full.write_text("".join(full_parts), encoding="utf-8")
    return llms, llms_full


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-dir", type=Path, default=_ROOT / "site")
    build(parser.parse_args().site_dir)


if __name__ == "__main__":
    main()
