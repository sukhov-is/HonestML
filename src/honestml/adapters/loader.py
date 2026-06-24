"""Thin input loader: a file/folder path → ``polars.DataFrame`` (ADR-0014).

Optional convenience so the user can pass a path instead of reading the data
themselves; the core and the ``AutoML`` facade stay in-memory. Supports parquet and
csv, and a folder of single-schema files (concatenated). I/O failures surface as
:class:`InputError` with the path; a folder whose files disagree on schema raises
:class:`SchemaValidationError`.

Note: csv carries no typed schema (polars infers dtypes), so for guaranteed
train↔inference code stability prefer parquet or pass an explicit ``FeatureSchema``
(ADR-0014 §best-effort).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import polars as pl

from honestml.core.exceptions import InputError, SchemaValidationError

_READERS: dict[str, Callable[..., pl.DataFrame]] = {
    ".parquet": pl.read_parquet,
    ".csv": pl.read_csv,
}


def load_table(source: str | Path) -> pl.DataFrame:
    """Read a parquet/csv file, or a folder of single-schema files, into a frame."""
    path = Path(source)
    if not path.exists():
        raise InputError(f"input path does not exist: {path}")
    return _load_dir(path) if path.is_dir() else _load_file(path)


def _load_file(path: Path) -> pl.DataFrame:
    reader = _READERS.get(path.suffix.lower())
    if reader is None:
        raise InputError(
            f"unsupported file format {path.suffix!r} for {path}; expected .parquet or .csv"
        )
    try:
        return reader(path)
    except Exception as exc:  # boundary: translate engine I/O/parse errors into a domain error
        raise InputError(f"failed to read {path}: {exc}") from exc


def _load_dir(path: Path) -> pl.DataFrame:
    files = sorted(p for p in path.iterdir() if p.suffix.lower() in _READERS)
    if not files:
        raise InputError(f"no .parquet/.csv files in directory: {path}")
    frames = [_load_file(p) for p in files]
    reference = frames[0].schema
    for file, frame in zip(files[1:], frames[1:], strict=True):
        if frame.schema != reference:
            raise SchemaValidationError(
                f"schema mismatch in folder {path}: {file.name} differs from {files[0].name}"
            )
    return pl.concat(frames)
