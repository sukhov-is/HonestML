"""Library logging.

Canonical library pattern: a namespaced logger with a ``NullHandler`` so nothing
is emitted until the consumer configures handlers. No ``print`` anywhere in the
library (enforced by ruff ``T20``).
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("honestml")
logger.addHandler(logging.NullHandler())


def get_logger(name: str | None = None) -> logging.Logger:
    """Return the library logger, or a child logger ``honestml.<name>``."""
    return logger if name is None else logger.getChild(name)


def log_stage(
    log: logging.Logger,
    *,
    step: str | None = None,
    stage: str | None = None,
    level: int = logging.INFO,
    **fields: Any,
) -> None:
    """Emit a structured ``key=value`` progress line for machine parsing."""
    parts = []
    if step is not None:
        parts.append(f"step={step}")
    if stage is not None:
        parts.append(f"stage={stage}")
    parts.extend(f"{k}={v}" for k, v in fields.items())
    log.log(level, " ".join(parts))
