"""Metric adapters (ADR-0013, extended ADR-0021).

Thin wrappers over ``sklearn.metrics`` implementing the ``Metric`` port. ``needs``
declares the projection of the model output the metric consumes; the use-case projects
accordingly (ADR-0010 §3). For binary, proba-metrics receive ``P(positive)`` as a 1-D
array (bit-exact with M2 — no ``labels``/``average`` passed). For multiclass they receive
an ``(n, K)`` matrix aligned to the global class order, and ``labels``/``average`` come from
the metric's own fields, set once at construction by composition (single source of truth,
ADR-0021 §4). Regression metrics (``rmse``/``mae``) consume ``value``. ``sample_weight`` is
forwarded (G2).
"""

from __future__ import annotations

from typing import cast

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)

from honestml.core import Metric, MetricNeeds
from honestml.core.exceptions import ConfigError


class _ScorerBase:
    """Common construction: optional global ``classes`` (labels) and ``average`` mode."""

    average: str | None
    proper_proba: bool = False  # proper loss changed by calibration (ADR-0031 §2); overridden below

    def __init__(
        self,
        *,
        classes: np.ndarray | None = None,
        average: str | None = None,
        positive: object | None = None,
    ) -> None:
        self.labels = np.asarray(classes) if classes is not None else None
        self.average = average
        self.positive = positive

    def _orient_binary(self, y_true: np.ndarray) -> np.ndarray:
        """Relabel a binary target to ``y == positive`` so a 1-D ``P(positive)`` is read as P(class 1).

        sklearn's 1-D proba metrics treat the score as ``P(greatest label)``; the use-case feeds
        ``P(positive)`` (``Task.positive_label``-aware), so without this the orientation inverts
        whenever ``positive`` is not the greatest label (F111). A no-op when ``positive`` is unset.
        """
        return y_true == self.positive if self.positive is not None else y_true


class RocAuc(_ScorerBase):
    """Area under the ROC curve (default for binary; OvR for multiclass)."""

    name = "roc_auc"
    greater_is_better = True
    needs: MetricNeeds = "proba"
    optimum = 1.0

    def score(
        self, y_true: np.ndarray, y_pred: np.ndarray, sample_weight: np.ndarray | None = None
    ) -> float:
        if y_pred.ndim == 2:
            return float(
                roc_auc_score(
                    y_true,
                    y_pred,
                    multi_class="ovr",
                    average=self.average or "macro",
                    labels=self.labels,
                    sample_weight=sample_weight,
                )
            )
        return float(
            roc_auc_score(self._orient_binary(y_true), y_pred, sample_weight=sample_weight)
        )


class PrAuc(_ScorerBase):
    """Average precision (area under the precision-recall curve); binary only."""

    name = "pr_auc"
    greater_is_better = True
    needs: MetricNeeds = "proba"
    optimum = 1.0

    def score(
        self, y_true: np.ndarray, y_pred: np.ndarray, sample_weight: np.ndarray | None = None
    ) -> float:
        if y_pred.ndim == 2:
            raise ConfigError("pr_auc is not supported for multiclass (n, K) probabilities")
        return float(
            average_precision_score(
                self._orient_binary(y_true), y_pred, sample_weight=sample_weight
            )
        )


class Accuracy(_ScorerBase):
    """Fraction of correctly classified samples (consumes hard class labels)."""

    name = "accuracy"
    greater_is_better = True
    needs: MetricNeeds = "class"
    optimum = 1.0

    def score(
        self, y_true: np.ndarray, y_pred: np.ndarray, sample_weight: np.ndarray | None = None
    ) -> float:
        return float(accuracy_score(y_true, y_pred, sample_weight=sample_weight))


class LogLoss(_ScorerBase):
    """Logistic loss (lower is better)."""

    name = "log_loss"
    greater_is_better = False
    needs: MetricNeeds = "proba"
    optimum = 0.0
    proper_proba = True

    def score(
        self, y_true: np.ndarray, y_pred: np.ndarray, sample_weight: np.ndarray | None = None
    ) -> float:
        if y_pred.ndim == 2:
            return float(log_loss(y_true, y_pred, labels=self.labels, sample_weight=sample_weight))
        return float(log_loss(self._orient_binary(y_true), y_pred, sample_weight=sample_weight))


class Brier(_ScorerBase):
    """Brier score (proper; lower is better). Binary 1-D ``P(pos)``; multiclass mean row sum-sq."""

    name = "brier"
    greater_is_better = False
    needs: MetricNeeds = "proba"
    optimum = 0.0
    proper_proba = True

    def score(
        self, y_true: np.ndarray, y_pred: np.ndarray, sample_weight: np.ndarray | None = None
    ) -> float:
        if y_pred.ndim == 2:
            onehot = np.zeros_like(y_pred, dtype=np.float64)
            onehot[np.arange(y_true.shape[0]), self._class_index(y_true)] = 1.0
            row = ((y_pred - onehot) ** 2).sum(axis=1)
            return float(np.average(row, weights=sample_weight))
        return float(
            brier_score_loss(self._orient_binary(y_true), y_pred, sample_weight=sample_weight)
        )

    def _class_index(self, y_true: np.ndarray) -> np.ndarray:
        if self.labels is not None:
            return np.searchsorted(self.labels, y_true)
        return y_true.astype(np.intp)


class Ece(_ScorerBase):
    """Top-label (confidence) Expected Calibration Error: binned ``|accuracy − confidence|``.

    Binary uses ``P(pos)``; multiclass uses max-probability confidence and ``argmax`` prediction
    (AutoGluon's top-label plank; class-wise ECE is future). Uniform bins on ``[0, 1]``,
    weight-aware, empty bins contribute 0 (ADR-0030 §5). Not proper → ``proper_proba=False``,
    not a sole selection gate.
    """

    name = "ece"
    greater_is_better = False
    needs: MetricNeeds = "proba"
    optimum = 0.0
    proper_proba = False

    def __init__(
        self,
        *,
        classes: np.ndarray | None = None,
        average: str | None = None,
        positive: object | None = None,
        n_bins: int = 10,
    ) -> None:
        super().__init__(classes=classes, average=average, positive=positive)
        self.n_bins = n_bins

    def score(
        self, y_true: np.ndarray, y_pred: np.ndarray, sample_weight: np.ndarray | None = None
    ) -> float:
        conf, correct = self._confidence_correct(y_true, y_pred)
        w = np.ones_like(conf) if sample_weight is None else np.asarray(sample_weight, np.float64)
        total = float(w.sum())
        edges = np.linspace(0.0, 1.0, self.n_bins + 1)
        bin_id = np.clip(np.digitize(conf, edges[1:-1]), 0, self.n_bins - 1)
        ece = 0.0
        for m in range(self.n_bins):
            sel = bin_id == m
            wm = float(w[sel].sum())
            if wm == 0.0:  # empty bin contributes 0 weight (ADR-0030 §5)
                continue
            acc = float(np.average(correct[sel], weights=w[sel]))
            cnf = float(np.average(conf[sel], weights=w[sel]))
            ece += (wm / total) * abs(acc - cnf)
        return ece

    def _confidence_correct(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if y_pred.ndim == 2:
            pred_idx = y_pred.argmax(axis=1)
            true_idx = (
                np.searchsorted(self.labels, y_true)
                if self.labels is not None
                else y_true.astype(np.intp)
            )
            return y_pred.max(axis=1), (pred_idx == true_idx).astype(np.float64)
        if self.positive is not None:
            pos: object = self.positive
        else:
            pos = self.labels[-1] if self.labels is not None else y_true.max()
        pred_pos = y_pred >= 0.5
        conf = np.where(pred_pos, y_pred, 1.0 - y_pred)
        return conf, (pred_pos == (y_true == pos)).astype(np.float64)


class Rmse(_ScorerBase):
    """Root mean squared error (regression; lower is better)."""

    name = "rmse"
    greater_is_better = False
    needs: MetricNeeds = "value"
    optimum = 0.0

    def score(
        self, y_true: np.ndarray, y_pred: np.ndarray, sample_weight: np.ndarray | None = None
    ) -> float:
        return float(np.sqrt(mean_squared_error(y_true, y_pred, sample_weight=sample_weight)))


class Mae(_ScorerBase):
    """Mean absolute error (regression; lower is better)."""

    name = "mae"
    greater_is_better = False
    needs: MetricNeeds = "value"
    optimum = 0.0

    def score(
        self, y_true: np.ndarray, y_pred: np.ndarray, sample_weight: np.ndarray | None = None
    ) -> float:
        return float(mean_absolute_error(y_true, y_pred, sample_weight=sample_weight))


_REGISTRY: dict[str, type[_ScorerBase]] = {
    RocAuc.name: RocAuc,
    PrAuc.name: PrAuc,
    Accuracy.name: Accuracy,
    LogLoss.name: LogLoss,
    Brier.name: Brier,
    Ece.name: Ece,
    Rmse.name: Rmse,
    Mae.name: Mae,
}
# sklearn-accepted averaging modes; validated at the boundary (a manifest/config is untrusted)
_AVERAGES = frozenset({None, "macro", "micro", "weighted", "samples"})


def resolve_metric(
    name: str,
    *,
    classes: np.ndarray | None = None,
    average: str | None = None,
    positive: object | None = None,
) -> Metric:
    """Return a metric instance by name (``ConfigError`` on miss), carrying labels/average/positive.

    Back-compat: callable without kwargs (``classes=None, average=None, positive=None``) — the binary
    path then keeps the greatest-label orientation, so a pre-ADR-0021 call is unchanged. ``positive``
    (``Task.positive_label``-aware) orients the binary proba metrics on ``P(positive)`` (F111).
    ``average`` is validated here because it can arrive from an artifact manifest (external input).
    """
    cls = _REGISTRY.get(name)
    if cls is None:
        raise ConfigError(f"unknown metric {name!r}; available: {sorted(_REGISTRY)}")
    if average not in _AVERAGES:
        allowed = sorted(a for a in _AVERAGES if a is not None)
        raise ConfigError(f"unknown average {average!r} for metric {name!r}; allowed: {allowed}")
    return cast(Metric, cls(classes=classes, average=average, positive=positive))
