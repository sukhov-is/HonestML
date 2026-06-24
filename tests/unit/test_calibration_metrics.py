"""M4d-1: Brier and ECE metrics (ADR-0030 §5, FR-M4-11) + ``proper_proba`` gate (ADR-0031 §2)."""

from __future__ import annotations

import numpy as np
import pytest

from honestml.adapters import Ece, resolve_metric

pytestmark = pytest.mark.unit


def test_brier_resolves_and_scores() -> None:
    """``brier`` resolves; binary 1-D P(pos) matches the textbook value; proper_proba=True."""
    metric = resolve_metric("brier")
    assert metric.proper_proba is True
    assert metric.greater_is_better is False
    assert metric.score(np.array([0, 1]), np.array([0.0, 1.0])) == 0.0
    assert metric.score(np.array([0, 1]), np.array([0.5, 0.5])) == pytest.approx(0.25)


def test_brier_multiclass_perfect_is_zero() -> None:
    """Multiclass Brier = mean row sum-of-squares; a perfect one-hot proba scores 0."""
    classes = np.array([0, 1, 2])
    metric = resolve_metric("brier", classes=classes)
    y = np.array([0, 1, 2])
    perfect = np.eye(3)
    assert metric.score(y, perfect) == pytest.approx(0.0)
    uniform = np.full((3, 3), 1 / 3)
    assert metric.score(y, uniform) == pytest.approx(2 / 3)


def test_ece_resolves_not_proper() -> None:
    """``ece`` resolves; it is NOT a proper loss (proper_proba=False) — not a sole gate."""
    metric = resolve_metric("ece")
    assert metric.proper_proba is False
    assert metric.greater_is_better is False


def test_ece_binned_overconfident() -> None:
    """Over-confident constant proba: |accuracy − confidence| in the single populated bin."""
    y = np.array([1, 1, 0, 0])
    proba = np.array([0.9, 0.9, 0.9, 0.9])  # confident positive, but only 50% correct
    assert Ece().score(y, proba) == pytest.approx(0.4)


def test_ece_perfectly_calibrated_is_zero() -> None:
    y = np.array([1, 1, 0, 0])
    proba = np.array([0.5, 0.5, 0.5, 0.5])  # conf 0.5 == acc 0.5 in the bin
    assert Ece().score(y, proba) == pytest.approx(0.0)


def test_ece_bins_configurable_and_weighted() -> None:
    """ECE bin count is configurable; sample_weight reweights the bin accuracy/confidence."""
    y = np.array([1, 1, 0, 0])
    proba = np.array([0.9, 0.9, 0.9, 0.9])
    assert Ece(n_bins=5).n_bins == 5
    # upweight the correct (positive) rows -> weighted accuracy rises toward confidence
    w = np.array([3.0, 3.0, 1.0, 1.0])
    weighted = Ece().score(y, proba, sample_weight=w)
    assert weighted == pytest.approx(abs(0.75 - 0.9))  # acc = 6/8 = 0.75, conf 0.9


def test_ece_multiclass_top_label() -> None:
    """Multiclass ECE uses max-proba confidence and argmax prediction (top-label)."""
    classes = np.array([0, 1, 2])
    metric = resolve_metric("ece", classes=classes)
    y = np.array([0, 1, 2, 0])
    # confident and correct on 3/4, confident wrong on the last -> conf 0.8, acc 0.75 in one bin
    proba = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8], [0.1, 0.8, 0.1]])
    assert metric.score(y, proba) == pytest.approx(abs(0.75 - 0.8))


def test_ranking_and_regression_metrics_not_proper() -> None:
    """roc_auc/accuracy/rmse carry proper_proba=False (refinement no-op by gate, ADR-0031 §2)."""
    for name in ("roc_auc", "accuracy", "rmse"):
        assert resolve_metric(name).proper_proba is False
    assert resolve_metric("log_loss").proper_proba is True
