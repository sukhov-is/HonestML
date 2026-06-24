"""The ``Task`` domain object.

A ``Task`` parameterizes the whole run by problem kind and ties together the
target metric, the default split scheme and the auto-typing policy. Adding a new
kind does not touch the ports.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from .exceptions import ConfigError

TaskKind = Literal["binary", "multiclass", "regression"]

_DEFAULT_METRIC: dict[str, str] = {
    "binary": "roc_auc",
    "multiclass": "log_loss",
    "regression": "rmse",
}


class Task(BaseModel):
    """Problem definition: kind + target metric name + split/typing policy."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: TaskKind
    metric: str | None = None
    positive_label: int | str | None = None
    # explicit report-date column for datetime->days_to_report deltas (ADR-0018 §5); None -> auto-detect
    # among DATETIME columns in the Reader. Additive frozen field; round-trips in the run manifest.
    report_date: str | None = None
    numeric_cat_max_unique: int = Field(default=20, ge=1)
    numeric_id_rate: float = Field(default=0.95, gt=0.0, le=1.0)
    numeric_id_min_unique: int = Field(default=100, ge=1)
    # string id-like typing (ADR-0015 ext, finding #7): a near-unique string column (a Name/Ticket id) is
    # pure noise as a category at inference; drop it when BOTH its distinct rate exceeds string_id_rate and
    # its distinct count exceeds string_id_min_unique. High-cardinality-but-below-rate columns are flagged,
    # not dropped. Defaults mirror the numeric id rule.
    string_id_rate: float = Field(default=0.95, gt=0.0, le=1.0)
    string_id_min_unique: int = Field(default=100, ge=1)
    # native-categorical cardinality gate (ADR-0092/0093): a CATEGORICAL column routes natively into
    # CatBoost/LightGBM only when its true category count is <= this cap; richer columns (high-card /
    # id-like / a__b intersections — the cost & overfit surface) ride the existing ordinal-codes path.
    # `None` disables the gate (every categorical native — the opt-out). Above numeric_cat_max_unique so
    # typical useful categoricals pass; the value is pinned empirically by benchmarks/native_cat_gate.py.
    native_cat_max_unique: int | None = Field(default=64, ge=1)

    @property
    def is_classification(self) -> bool:
        return self.kind in ("binary", "multiclass")

    @property
    def target_metric(self) -> str:
        """The declared target metric name, or the default for this kind."""
        return self.metric or _DEFAULT_METRIC[self.kind]

    @property
    def default_cv_scheme(self) -> str:
        """Default cross-validation scheme when the user does not override it."""
        return "stratified" if self.is_classification else "kfold"


def resolve_positive(task: Task, classes: np.ndarray) -> object:
    """Resolve the positive class label — pure domain rule.

    ``Task.positive_label`` if set; otherwise label ``1`` when present (the canonical ``{0, 1}``
    encoding), else the greatest label. Lives in the core so both the use-case (OOF scoring) and
    the adapter boundary (full-train target-encoding fit) share one source of truth without
    crossing the layer rule.
    """
    labels = classes.tolist()
    if task.positive_label is not None:
        if task.positive_label not in labels:
            raise ConfigError(
                f"positive_label={task.positive_label!r} is not among classes {labels}"
            )
        return task.positive_label
    return 1 if 1 in labels else labels[-1]
