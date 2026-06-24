"""M4d-3: probability calibration + refinement selection end-to-end through ``AutoML`` (ADR-0030/0031)."""

from __future__ import annotations

import numpy as np
import pytest

from honestml import AutoML
from honestml.core import ConfigError, CVConfig

pytestmark = pytest.mark.unit


def _data(n=200):
    from sklearn.datasets import make_classification

    return make_classification(
        n_samples=n, n_features=6, n_informative=4, n_redundant=0, random_state=0
    )


def test_calibrate_off_default_unchanged() -> None:
    """Default (calibrate='off'): no calibration report/curve, raw selection — M3/M4a behavior."""
    X, y = _data()
    model = AutoML(task="binary", random_state=0).fit(X, y)
    assert model.calibration_ is None and model.reliability_curve_ is None
    assert model.selection_mode_ == "raw"


def test_calibrate_gate_runs_and_reports() -> None:
    """calibrate='sigmoid' runs the cross-fit gate and reports raw/calibrated Brier & ECE."""
    X, y = _data()
    model = AutoML(task="binary", random_state=0, cv=CVConfig(calibrate="sigmoid")).fit(X, y)
    rep = model.calibration_
    assert rep is not None
    assert {"brier_raw", "brier_calibrated", "ece_raw", "ece_calibrated"} <= set(rep)
    assert isinstance(rep["applied"], bool)
    if rep["applied"]:
        # the gate only attaches when calibrated Brier does not exceed raw (ADR-0030 §3)
        assert rep["brier_calibrated"] <= rep["brier_raw"]
        assert model.reliability_curve_ is not None
        p = model.predict_proba(X)
        assert p.shape == (len(y), 2) and np.allclose(p.sum(axis=1), 1.0)


def test_regression_calibrate_raises() -> None:
    """Calibration is classification-only -> a regression task with calibrate!='off' fails fast."""
    from sklearn.datasets import make_regression

    X, y = make_regression(n_samples=80, n_features=5, random_state=0)
    with pytest.raises(ConfigError, match="classification-only"):
        AutoML(task="regression", random_state=0, cv=CVConfig(calibrate="sigmoid")).fit(X, y)


def test_timeseries_disables_calibration() -> None:
    """Production calibration is disabled under time-series CV (M4): reported, not attached."""
    X, y = _data(n=200)
    t = np.arange(len(y))
    model = AutoML(
        task="binary",
        cv=CVConfig(scheme="timeseries", n_splits=3, n_test=20, calibrate="sigmoid"),
        random_state=0,
    ).fit(X, y, time=t)
    assert model.calibration_ is not None and model.calibration_["applied"] is False
    assert model.calibration_["reason"] == "time-series"
    assert model.fitted_.calibrator is None


def test_refinement_selection_end_to_end() -> None:
    """selection='refinement' with a proper-proba metric ranks on calibrated OOF loss end-to-end."""
    X, y = _data(n=200)
    model = AutoML(
        task="binary",
        metric="log_loss",
        random_state=0,
        cv=CVConfig(selection="refinement", refinement_min_oof=10),
    ).fit(X, y)
    assert model.selection_mode_ == "refinement"
    p = model.predict_proba(X)
    assert p.shape == (len(y), 2) and np.allclose(p.sum(axis=1), 1.0)
