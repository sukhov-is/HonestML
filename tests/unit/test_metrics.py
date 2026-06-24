"""M2-2: metric adapters declare needs/direction and wrap sklearn (ADR-0013)."""

from __future__ import annotations

import numpy as np
import pytest

from honestml.adapters import (
    Accuracy,
    Brier,
    Ece,
    LogLoss,
    Mae,
    PrAuc,
    Rmse,
    RocAuc,
    resolve_metric,
)
from honestml.core import ConfigError, Metric

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("metric", [RocAuc(), PrAuc(), Accuracy(), LogLoss(), Rmse(), Mae()])
def test_implements_metric_port(metric) -> None:
    assert isinstance(metric, Metric)
    assert isinstance(metric.name, str)
    assert metric.needs in ("proba", "threshold", "class", "value")
    assert metric.average is None  # default (binary / unset)


def test_directions_and_needs() -> None:
    assert RocAuc().greater_is_better is True and RocAuc().needs == "proba"
    assert LogLoss().greater_is_better is False and LogLoss().needs == "proba"
    assert Accuracy().needs == "class"


def test_perfect_separation_scores() -> None:
    y = np.array([0, 0, 1, 1])
    proba = np.array([0.1, 0.2, 0.8, 0.9])
    assert RocAuc().score(y, proba) == 1.0
    assert PrAuc().score(y, proba) == 1.0
    assert Accuracy().score(y, (proba >= 0.5).astype(int)) == 1.0


def test_sample_weight_is_forwarded() -> None:
    y = np.array([0, 1, 0, 1])
    proba = np.array([0.4, 0.6, 0.6, 0.4])
    w = np.array([1.0, 1.0, 0.0, 0.0])
    # zero-weighting the two wrong rows must lift weighted accuracy to 1.0
    assert Accuracy().score(y, (proba >= 0.5).astype(int), sample_weight=w) == 1.0


def test_resolve_metric() -> None:
    assert resolve_metric("roc_auc").name == "roc_auc"
    with pytest.raises(ConfigError, match="unknown metric"):
        resolve_metric("nope")


def test_resolve_metric_legacy_without_average() -> None:
    """Pre-ADR-0021 call style (no classes/average kwargs) still resolves (NFR-3)."""
    m = resolve_metric("roc_auc")
    assert m.average is None
    m2 = resolve_metric("log_loss", classes=np.array([0, 1, 2]), average="weighted")
    assert m2.average == "weighted"


# --- M3b: regression + multiclass (ADR-0021) -------------------------------


def test_regression_metrics_score() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0])
    pred = np.array([1.0, 2.0, 3.0, 4.0])
    assert Rmse().score(y, pred) == 0.0
    assert Mae().score(y, pred) == 0.0
    off = np.array([2.0, 2.0, 3.0, 4.0])  # one unit of error on the first row
    assert Rmse().score(y, off) == pytest.approx(0.5)
    assert Mae().score(y, off) == pytest.approx(0.25)
    assert Rmse().greater_is_better is False and Rmse().needs == "value"


def test_roc_auc_multiclass_ovr() -> None:
    classes = np.array([0, 1, 2])
    y = np.array([0, 1, 2, 0, 1, 2])
    # near-perfect per-class probabilities -> OvR macro AUC ~ 1.0
    proba = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
            [0.7, 0.2, 0.1],
            [0.2, 0.7, 0.1],
            [0.1, 0.2, 0.7],
        ]
    )
    score = RocAuc(classes=classes, average="macro").score(y, proba)
    assert score == pytest.approx(1.0)


def test_log_loss_multiclass_bounded() -> None:
    classes = np.array([0, 1, 2])
    y = np.array([0, 1, 2])
    proba = np.full((3, 3), 1.0 / 3.0)
    score = LogLoss(classes=classes).score(y, proba)
    assert np.isfinite(score) and score == pytest.approx(np.log(3), rel=1e-6)


def test_pr_auc_multiclass_raises() -> None:
    proba = np.full((4, 3), 1.0 / 3.0)
    with pytest.raises(ConfigError, match="pr_auc"):
        PrAuc(classes=np.array([0, 1, 2])).score(np.array([0, 1, 2, 0]), proba)


def test_binary_metrics_bit_exact_without_labels() -> None:
    """Binary metrics ignore labels/average -> bit-exact with the pre-ADR-0021 path."""
    y = np.array([0, 0, 1, 1])
    proba = np.array([0.1, 0.2, 0.8, 0.9])
    assert RocAuc(classes=np.array([0, 1])).score(y, proba) == RocAuc().score(y, proba) == 1.0
    assert LogLoss(classes=np.array([0, 1])).score(y, proba) == LogLoss().score(y, proba)


def test_binary_metrics_respect_non_greatest_positive() -> None:
    """F111: proba metrics must orient on ``positive`` even when it is not the greatest label.

    The use-case feeds ``P(positive)``; a perfect model with ``positive=0`` (or ``"churn"``) must
    score near-optimal, not inverted (without this, sklearn reads the 1-D score as ``P(greatest)``).
    """
    classes = np.array([0, 1])
    y = np.array([0, 0, 0, 1, 1, 1])
    p_pos = np.array([0.95, 0.9, 0.85, 0.1, 0.05, 0.0])  # perfect P(positive=0)
    assert RocAuc(classes=classes, positive=0).score(y, p_pos) == pytest.approx(1.0)
    assert PrAuc(classes=classes, positive=0).score(y, p_pos) == pytest.approx(1.0)
    assert Brier(classes=classes, positive=0).score(y, p_pos) < 0.02
    assert LogLoss(classes=classes, positive=0).score(y, p_pos) < 0.2
    assert Ece(classes=classes, positive=0).score(y, p_pos) < 0.1

    # string labels: sorted order ["churn", "stay"], positive="churn" is NOT the greatest
    sclasses = np.array(["churn", "stay"])
    ys = np.array(["churn", "churn", "stay", "stay"])
    p_churn = np.array([0.95, 0.9, 0.05, 0.1])  # perfect P("churn")
    assert RocAuc(classes=sclasses, positive="churn").score(ys, p_churn) == pytest.approx(1.0)
    assert Brier(classes=sclasses, positive="churn").score(ys, p_churn) < 0.02


def test_binary_orientation_noop_when_positive_is_greatest() -> None:
    """The orientation fix is a no-op when ``positive`` is the greatest label (default {0, 1})."""
    y = np.array([0, 0, 1, 1])
    proba = np.array([0.1, 0.2, 0.8, 0.9])  # P(positive=1)
    classes = np.array([0, 1])
    assert RocAuc(classes=classes, positive=1).score(y, proba) == RocAuc().score(y, proba)
    assert Brier(classes=classes, positive=1).score(y, proba) == pytest.approx(
        Brier().score(y, proba)
    )
