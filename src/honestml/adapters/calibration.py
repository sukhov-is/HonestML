"""Probability calibrators (ADR-0030 §1) — implement the ``Calibrator`` port over sklearn.

Binary = Platt ``_SigmoidCalibration`` / ``IsotonicRegression``; multiclass = per-class OvR
+ renormalization. Output is clipped to ``[ε, 1-ε]`` (and row-normalized for multiclass):
isotonic can emit exactly 0/1 on a small calibration set, which makes ``log_loss`` infinite
(paper Apx D, arXiv:2501.19195) — the clip is the library's Laplace-style smoothing.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# private sklearn API (chosen over CalibratedClassifierCV, ADR-0030 §1): no stability guarantee across
# majors — revisit on sklearn 2.x, where the import can break at runtime (tech debt, F024).
from sklearn.calibration import _SigmoidCalibration
from sklearn.isotonic import IsotonicRegression

from honestml.core import CalibratorFactory, ConfigError

_EPS = 1e-6
_AUTO_ISOTONIC_MIN = 1000  # 'auto' uses isotonic only when calibration data is ample (ADR-0030 §1)


class _PerColumnCalibrator:
    """Per-column 1-D calibration: one map for binary ``P(pos)``, K OvR maps for multiclass."""

    def __init__(self) -> None:
        self._maps: list[Any] = []
        self._binary = True

    def _base(self) -> Any:  # overridden by concrete calibrators
        raise NotImplementedError

    def fit(
        self, proba: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None
    ) -> None:
        if proba.ndim == 1:
            self._binary = True
            m = self._base()
            m.fit(proba, y, sample_weight)
            self._maps = [m]
            return
        self._binary = False
        self._maps = []
        for j in range(proba.shape[1]):
            m = self._base()
            m.fit(proba[:, j], (y == j).astype(np.float64), sample_weight)
            self._maps.append(m)

    def transform(self, proba: np.ndarray) -> np.ndarray:
        if self._binary:
            out = np.asarray(self._maps[0].predict(proba), dtype=np.float64)
            return np.clip(out, _EPS, 1.0 - _EPS)
        cols = [
            np.asarray(self._maps[j].predict(proba[:, j]), dtype=np.float64)
            for j in range(proba.shape[1])
        ]
        cal = np.clip(np.column_stack(cols), _EPS, None)
        return cal / cal.sum(axis=1, keepdims=True)


class SigmoidCalibrator(_PerColumnCalibrator):
    """Platt sigmoid calibration (low-DOF; the refinement-selection default, ADR-0031 §5)."""

    def _base(self) -> Any:
        return _SigmoidCalibration()


class IsotonicCalibrator(_PerColumnCalibrator):
    """Isotonic (monotone, higher-DOF) calibration; clipped off 0/1 (paper Apx D)."""

    def _base(self) -> Any:
        return IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0, increasing=True)


def resolve_calibrator(method: str, *, n_calib: int | None = None) -> CalibratorFactory:
    """Map a ``calibrate`` method to a zero-arg ``Calibrator`` factory (ADR-0030 §1).

    ``auto`` picks isotonic only when ``n_calib`` is large enough, else sigmoid — isotonic
    overfits small calibration sets (paper Apx D); ``n_calib=None`` (the selection path) keeps
    it deterministic at sigmoid (ADR-0031 §5).
    """
    if method == "sigmoid":
        return SigmoidCalibrator
    if method == "isotonic":
        return IsotonicCalibrator
    if method == "auto":
        if n_calib is not None and n_calib >= _AUTO_ISOTONIC_MIN:
            return IsotonicCalibrator
        return SigmoidCalibrator
    raise ConfigError(f"unknown calibrate method {method!r}; use 'sigmoid'/'isotonic'/'auto'")
