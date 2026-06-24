"""M4d-1: probability calibrators (ADR-0030 §1) — fit/transform, 0/1-clip, OvR renorm."""

from __future__ import annotations

import numpy as np
import pytest

from honestml.adapters import IsotonicCalibrator, SigmoidCalibrator, resolve_calibrator
from honestml.core import ConfigError

pytestmark = pytest.mark.unit

_EPS = 1e-6


def test_isotonic_binary_clips_off_zero_and_one() -> None:
    """Isotonic on a separable set would map to exactly 0/1; the clip keeps it strictly inside."""
    proba = np.array([0.0, 0.0, 1.0, 1.0])
    y = np.array([0, 0, 1, 1])
    cal = IsotonicCalibrator()
    cal.fit(proba, y)
    out = cal.transform(np.array([0.0, 1.0]))
    assert np.all(out > 0.0) and np.all(out < 1.0)
    assert out[0] == pytest.approx(_EPS) and out[1] == pytest.approx(1.0 - _EPS)


def test_sigmoid_binary_monotone_in_unit_interval() -> None:
    """Platt sigmoid stays in (0, 1) and is monotone non-decreasing in the raw score."""
    rng = np.random.default_rng(0)
    proba = rng.uniform(size=400)
    y = (rng.uniform(size=400) < proba).astype(int)  # well-correlated with the score
    cal = SigmoidCalibrator()
    cal.fit(proba, y)
    out = cal.transform(np.array([0.1, 0.5, 0.9]))
    assert np.all(out > 0.0) and np.all(out < 1.0)
    assert out[0] <= out[1] <= out[2]


def test_multiclass_per_class_renormalized() -> None:
    """Multiclass OvR calibration returns strictly-positive rows that sum to 1."""
    rng = np.random.default_rng(1)
    n, k = 300, 3
    y = rng.integers(0, k, size=n)
    proba = np.full((n, k), 0.2)
    proba[np.arange(n), y] = 0.6
    proba = proba / proba.sum(axis=1, keepdims=True)
    cal = IsotonicCalibrator()
    cal.fit(proba, y)
    out = cal.transform(proba)
    assert out.shape == (n, k)
    assert np.all(out > 0.0)
    assert np.allclose(out.sum(axis=1), 1.0)


def test_resolve_calibrator_methods_and_auto() -> None:
    """``resolve_calibrator`` maps methods; ``auto`` picks isotonic only with ample n_calib."""
    assert resolve_calibrator("sigmoid") is SigmoidCalibrator
    assert resolve_calibrator("isotonic") is IsotonicCalibrator
    assert resolve_calibrator("auto", n_calib=50) is SigmoidCalibrator
    assert resolve_calibrator("auto", n_calib=5000) is IsotonicCalibrator
    assert resolve_calibrator("auto", n_calib=None) is SigmoidCalibrator  # selection path
    with pytest.raises(ConfigError, match="unknown calibrate method"):
        resolve_calibrator("platt")
