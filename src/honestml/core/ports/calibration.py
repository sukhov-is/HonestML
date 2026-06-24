"""The ``Calibrator`` port.

A 1-D probability recalibration map fitted on leakage-free OOF probabilities, shared by
the production calibrator and refinement-based selection. The port
stays numpy-only — the sklearn calibrator lives in the adapter — so the cross-fit use-case
``crossfit_calibrate`` is testable on a fake.

Label convention (the caller codes it, the calibrator stays semantics-free): ``proba`` is
``P(positive)`` ``(n,)`` for binary with ``y`` in ``{0, 1}`` (1 = positive), or ``(n, K)``
for multiclass with ``y`` the true column index in ``{0..K-1}``. ``transform`` returns the
same shape, strictly inside ``(0, 1)`` and row-normalized for multiclass, so a proper loss
never hits ``log(0)`` (isotonic can emit exactly 0/1 on small sets).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Calibrator(Protocol):
    """A fit/transform probability recalibration map (sklearn-shaped, numpy-only port)."""

    def fit(
        self, proba: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None
    ) -> None: ...

    def transform(self, proba: np.ndarray) -> np.ndarray: ...


CalibratorFactory = Callable[[], Calibrator]
