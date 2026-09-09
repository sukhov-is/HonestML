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

import hashlib
import sys
from collections import OrderedDict
from collections.abc import Callable
from typing import cast

import numpy as np
import sklearn
from scipy.special import xlogy
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.preprocessing import LabelBinarizer

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

    def _prepare_ensemble_score(
        self, y_true: np.ndarray, sample_weight: np.ndarray | None
    ) -> Callable[[np.ndarray], float] | None:
        metric = cast(Metric, self)
        if not _uses_builtin_score(metric):
            return None
        target = y_true.copy()
        target.setflags(write=False)
        weights = sample_weight.copy() if sample_weight is not None else None
        if weights is not None:
            weights.setflags(write=False)
        return _memoized_ensemble_score(lambda pred: metric.score(target, pred, weights))

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


_ENSEMBLE_SCORERS: dict[type[_ScorerBase], object] = {
    LogLoss: LogLoss.score,
    Accuracy: Accuracy.score,
    Mae: Mae.score,
    Rmse: Rmse.score,
    RocAuc: RocAuc.score,
    PrAuc: PrAuc.score,
}
_ENSEMBLE_ORIENT_BINARY = _ScorerBase._orient_binary
_ENSEMBLE_CACHE_MAX_ENTRIES = 1024
_ENSEMBLE_CACHE_MAX_BYTES = 256 * 1024
_EnsembleScoreKey = tuple[str, tuple[int, ...], tuple[int, ...], bytes]


def _memoized_ensemble_score(
    score: Callable[[np.ndarray], float],
) -> Callable[[np.ndarray], float]:
    """Cache pure scalar scores; retained keys are bounded, array contents are not retained."""
    cache: OrderedDict[_EnsembleScoreKey, tuple[float, int]] = OrderedDict()
    retained_bytes = 0

    def cached(pred: np.ndarray) -> float:
        nonlocal retained_bytes
        if (
            type(pred) is not np.ndarray
            or pred.dtype.hasobject
            or _ENSEMBLE_CACHE_MAX_ENTRIES < 1
            or _ENSEMBLE_CACHE_MAX_BYTES <= sys.getsizeof(cache)
        ):
            return score(pred)
        key = (
            pred.dtype.str,
            pred.shape,
            pred.strides,
            hashlib.sha256(np.ascontiguousarray(pred)).digest(),
        )
        existing = cache.get(key)
        if existing is not None:
            cache.move_to_end(key)
            return existing[0]
        value = score(pred)
        if not np.isfinite(value):
            return value
        size = (
            sys.getsizeof(key)
            + sum(sys.getsizeof(part) for part in key)
            + sum(sys.getsizeof(n) for n in (*key[1], *key[2]))
            + sys.getsizeof(value)
            + sys.getsizeof((value, 0))
            + sys.getsizeof(0)
        )
        cache[key] = (value, size)
        retained_bytes += size
        while cache and (
            len(cache) > _ENSEMBLE_CACHE_MAX_ENTRIES
            or retained_bytes + sys.getsizeof(cache) > _ENSEMBLE_CACHE_MAX_BYTES
        ):
            _, (_, removed_bytes) = cache.popitem(last=False)
            retained_bytes -= removed_bytes
        return value

    return cached


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


def _uses_builtin_score(metric: Metric) -> bool:
    scorer = cast(_ScorerBase, metric)
    original = _ENSEMBLE_SCORERS.get(type(scorer))
    if (
        original is None
        or "score" in vars(scorer)
        or getattr(metric.score, "__func__", None) is not original
        or getattr(metric.score, "__self__", None) is not metric
    ):
        return False
    return not (
        type(scorer) in (LogLoss, RocAuc, PrAuc)
        and (
            "_orient_binary" in vars(scorer)
            or getattr(scorer._orient_binary, "__func__", None) is not _ENSEMBLE_ORIENT_BINARY
            or getattr(scorer._orient_binary, "__self__", None) is not scorer
        )
    )


def _prepare_ranking_bootstrap_score(
    metric: Metric,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weight: np.ndarray | None,
) -> Callable[[np.ndarray], float] | None:
    if (
        sklearn.__version__ != "1.7.2"
        or sample_weight is not None
        or not _uses_builtin_score(metric)
        or type(y_true) is not np.ndarray
        or type(y_pred) is not np.ndarray
        or y_true.ndim != 1
        or y_pred.ndim != 1
        or y_pred.dtype not in (np.float32, np.float64)
        or y_true.dtype.kind not in "biufUS"
        or not np.isfinite(y_pred).all()
        or np.unique(y_true).size != 2
    ):
        return None
    ranking_metric = cast(RocAuc | PrAuc, metric)
    target = ranking_metric._orient_binary(y_true)
    labels = np.unique(target)
    if labels.size != 2:
        return None
    try:
        original = metric.score(y_true, y_pred, None)
    except ValueError:
        return None
    if not np.isfinite(original):
        return None
    positive = target == (labels[-1] if type(metric) is RocAuc else 1)
    order = np.argsort(y_pred, kind="mergesort")[::-1]
    ordered_scores = y_pred[order]
    starts = np.r_[0, np.flatnonzero(ordered_scores[1:] != ordered_scores[:-1]) + 1]
    group_ids = np.empty(y_pred.size, dtype=np.intp)
    group_ids[order] = np.repeat(np.arange(starts.size), np.diff(np.r_[starts, y_pred.size]))

    def score(indices: np.ndarray) -> float:
        selected_positive = positive[indices]
        positives = int(selected_positive.sum())
        if positives == 0 or positives == indices.size:
            return metric.score(y_true[indices], y_pred[indices], None)
        selected_groups = group_ids[indices]
        counts = np.bincount(selected_groups, minlength=starts.size)
        positive_counts = np.bincount(selected_groups[selected_positive], minlength=starts.size)
        present = counts != 0
        # unweighted integer counts reproduce sklearn's float64 stable_cumsum exactly.
        tps = np.cumsum(positive_counts[present], dtype=np.float64)
        fps = np.cumsum(counts[present], dtype=np.int64) - tps
        if type(metric) is RocAuc:
            # mirror roc_curve(drop_intermediate=True) before normalizing and integrating.
            if len(fps) > 2:
                keep = np.r_[True, np.logical_or(np.diff(fps, 2), np.diff(tps, 2)), True]
                fps, tps = fps[keep], tps[keep]
            fps, tps = np.r_[0, fps], np.r_[0, tps]
            return float(auc(fps / fps[-1], tps / tps[-1]))
        ps = tps + fps
        precision = np.zeros_like(tps)
        np.divide(tps, ps, out=precision, where=ps != 0)
        recall = tps / tps[-1]
        precision, recall = np.hstack((precision[::-1], 1)), np.hstack((recall[::-1], 0))
        return float(max(0.0, -np.sum(np.diff(recall) * np.array(precision)[:-1])))

    if score(np.arange(y_true.size)) != original:
        return None
    return score


def _prepare_bootstrap_score(
    metric: Metric,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weight: np.ndarray | None,
) -> Callable[[np.ndarray], float] | None:
    """Prepare exact built-in resample scoring; unsupported inputs use the public scorer."""
    if type(metric) in (RocAuc, PrAuc):
        return _prepare_ranking_bootstrap_score(metric, y_true, y_pred, sample_weight)
    # log_loss validation and clipping follow the audited sklearn implementation.
    if type(metric) is LogLoss and sklearn.__version__ != "1.7.2":
        return None
    if (
        type(metric) not in (LogLoss, Accuracy, Mae, Rmse)
        or y_true.ndim != 1
        or not _uses_builtin_score(metric)
    ):
        return None
    if type(metric) in (Mae, Rmse) and (
        y_pred.ndim != 1 or y_true.dtype != np.float64 or y_pred.dtype != np.float64
    ):
        return None
    if type(metric) is Accuracy and y_pred.ndim != 1:
        return None
    if y_pred.ndim not in (1, 2):
        return None
    try:
        original = metric.score(y_true, y_pred, sample_weight)
    except (ValueError, ZeroDivisionError):
        return None
    if not np.isfinite(original):
        return None
    required_classes: np.ndarray | None = None
    class_codes: np.ndarray | None = None
    if type(metric) is LogLoss:
        # match the public scorer's binary orientation and explicit multiclass label contract.
        loss_metric = metric
        target = loss_metric._orient_binary(y_true) if y_pred.ndim == 1 else y_true
        labels = loss_metric.labels if y_pred.ndim == 2 else None
        binarizer = LabelBinarizer().fit(target if labels is None else labels)
        transformed = binarizer.transform(target)
        if transformed.shape[1] == 1:
            transformed = np.append(1 - transformed, transformed, axis=1)
        probabilities = np.asarray(y_pred)
        if probabilities.dtype not in (np.float16, np.float32, np.float64):
            probabilities = probabilities.astype(np.float64)
        if probabilities.ndim == 1:
            probabilities = probabilities[:, None]
        if probabilities.shape[1] == 1:
            probabilities = np.append(1 - probabilities, probabilities, axis=1)
        eps = np.finfo(probabilities.dtype).eps
        probabilities = np.clip(probabilities, eps, 1 - eps)
        rows = -xlogy(transformed, probabilities).sum(axis=1)
        if labels is None:
            required_classes, class_codes = np.unique(target, return_inverse=True)
    elif type(metric) is Accuracy:
        rows = y_true == y_pred
    else:
        residual = y_true - y_pred
        rows = residual**2 if type(metric) is Rmse else np.abs(residual)
    root = type(metric) is Rmse

    def score(indices: np.ndarray) -> float:
        if required_classes is not None and class_codes is not None:
            if np.unique(class_codes[indices]).size != required_classes.size:
                return float("nan")
        weights = sample_weight[indices] if sample_weight is not None else None
        value = np.average(rows[indices], weights=weights)
        return float(np.sqrt(value) if root else value)

    # retain the general path when full-input numeric validation does not match row reduction.
    if score(np.arange(y_true.size)) != original:
        return None
    return score
