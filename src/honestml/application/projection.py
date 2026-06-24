"""Metric projection and scorer prologue (ADR-0010/0021) — leaf application helpers.

The projection / proba-alignment primitives shared by ``run_slice``, the feature-compare scorers and the
ensemble scorer. They live in a leaf module (importing only ``core``) so ``slice`` and ``feature_compare``
can both depend on them without the import cycle they used to form.
"""

from __future__ import annotations

import numpy as np

from honestml.core import ConfigError, Metric, Task, resolve_positive
from honestml.core.task import TaskKind

_PROBA_NEEDS = ("proba", "threshold")
_PROBA_EPS = 1e-6


def project_for_metric(
    metric: Metric, *, proba: np.ndarray | None, pred: np.ndarray, kind: TaskKind = "binary"
) -> np.ndarray:
    """Project model output to what ``metric.needs`` consumes (ADR-0010 §3, ADR-0021 §3).

    The caller has already shaped ``proba`` for ``kind`` (binary → 1-D ``P(positive)``;
    multiclass → ``(n, K)`` aligned to the global class order). This selects the array the
    metric scores and guards the multiclass proba shape; for ``class``/``value`` metrics it
    returns ``pred``.
    """
    if metric.needs in _PROBA_NEEDS:
        if proba is None:
            raise ConfigError(
                f"metric {metric.name!r} needs probabilities but the model does not provide them"
            )
        if kind == "multiclass" and proba.ndim != 2:
            raise ConfigError(
                f"multiclass metric {metric.name!r} requires a 2-D (n, K) probability matrix"
            )
        return proba
    return pred


def align_proba(
    proba_fold: np.ndarray, est_classes: np.ndarray, global_classes: np.ndarray
) -> np.ndarray:
    """Reindex a fold's ``predict_proba`` to the global class order (ADR-0021 §2).

    Each ``est_classes`` column is placed at its label's global position; a class absent
    from this fold's model gets a small ε mass (``1e-6``, not literal 0), then every row is
    renormalized to sum 1 — so ``log_loss`` never hits ``-log(0)`` and each row stays a valid
    distribution. Multiclass-only; binary keeps its direct ``P(positive)`` column.
    """
    n = proba_fold.shape[0]
    aligned = np.full((n, global_classes.size), _PROBA_EPS, dtype=np.float64)
    positions = {label: j for j, label in enumerate(global_classes.tolist())}
    for src, label in enumerate(est_classes.tolist()):
        target = positions.get(label)
        if target is not None:
            aligned[:, target] = proba_fold[:, src]
    aligned /= aligned.sum(axis=1, keepdims=True)
    return aligned


def _scorer_setup(
    task: Task,
    metric: Metric,
    y: np.ndarray,
    *,
    global_classes: np.ndarray | None = None,
) -> tuple[np.ndarray | None, object | None, bool, float]:
    """Shared scorer prologue (ADR-0021): ``(classes, positive, need_proba, sign)``.

    ``classes`` is the class order proba aligns to — the given ``global_classes`` if any, else
    ``np.unique(y)`` for classification (``None`` for regression); ``positive`` is the binary positive
    label; ``need_proba`` whether the metric consumes probabilities; ``sign`` flips a lower-is-better
    metric to higher-is-better. One source so a metric-rule change cannot drift across the scorers.
    """
    classes = (
        global_classes
        if global_classes is not None
        else (np.unique(y) if task.is_classification else None)
    )
    positive = (
        resolve_positive(task, classes) if task.kind == "binary" and classes is not None else None
    )
    need_proba = metric.needs in _PROBA_NEEDS
    sign = 1.0 if metric.greater_is_better else -1.0
    return classes, positive, need_proba, sign
