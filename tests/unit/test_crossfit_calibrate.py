"""M4d-1: ``crossfit_calibrate`` (ADR-0030 §3 / ADR-0031 §1) — out-of-fold, pure, no sklearn."""

from __future__ import annotations

import numpy as np
import pytest

from honestml.application import crossfit_calibrate, viable_blocks

pytestmark = pytest.mark.unit


def test_viable_blocks_multiclass_requires_every_class_per_block() -> None:
    # F023: for multiclass the per-class OvR calibrator needs EVERY class present in each block's train
    # side; ">= 2 distinct overall" is not enough — an absent class fits its OvR map on an all-zero
    # target and zeroes that class's calibrated mass. n_classes=K enforces the stronger check.
    blocks = np.array([0] * 60 + [1] * 60)
    y = np.array(
        [0, 1, 2] * 20 + [0, 1] * 30
    )  # block0 train side has 0/1/2; block1 train side only 0/1
    assert viable_blocks(blocks, y) is True  # legacy >=2 check: each train side has >= 2 classes
    assert (
        viable_blocks(blocks, y, n_classes=3) is False
    )  # block0's train side (block1) lacks class 2


def test_crossfit_is_out_of_fold() -> None:
    """Block ``b`` is calibrated by a map fitted on the OTHER blocks only (no in-sample)."""
    proba = np.array([0.0, 0.0, 1.0, 1.0, 2.0, 2.0])  # row value encodes its block
    blocks = np.array([0, 0, 1, 1, 2, 2])
    y = np.array([0, 1, 0, 1, 0, 1])
    seen_train: list[set[float]] = []

    class _Rec:
        def fit(self, p, yy, sample_weight=None):  # type: ignore[no-untyped-def]
            seen_train.append(set(np.round(p, 3).tolist()))

        def transform(self, p):  # type: ignore[no-untyped-def]
            return np.full_like(p, 0.5)

    out = crossfit_calibrate(proba, y, blocks, lambda: _Rec())
    assert seen_train[0] == {1.0, 2.0}  # block 0 trained on blocks 1 and 2, never 0
    assert seen_train[1] == {0.0, 2.0}
    assert np.allclose(out, 0.5)


def test_crossfit_identity_returns_input() -> None:
    """An identity calibrator leaves probabilities unchanged (shape/dtype preserved)."""

    class _Id:
        def fit(self, p, y, sample_weight=None):  # type: ignore[no-untyped-def]
            pass

        def transform(self, p):  # type: ignore[no-untyped-def]
            return p

    proba = np.array([0.1, 0.7, 0.3, 0.9])
    blocks = np.array([0, 0, 1, 1])
    out = crossfit_calibrate(proba, np.array([0, 1, 0, 1]), blocks, lambda: _Id())
    assert np.allclose(out, proba)


def test_crossfit_forwards_sample_weight() -> None:
    """sample_weight is sliced to each block's train side and forwarded into ``fit``."""
    received: list[np.ndarray | None] = []

    class _W:
        def fit(self, p, y, sample_weight=None):  # type: ignore[no-untyped-def]
            received.append(sample_weight)

        def transform(self, p):  # type: ignore[no-untyped-def]
            return p

    proba = np.linspace(0.0, 1.0, 6)
    blocks = np.array([0, 0, 1, 1, 2, 2])
    w = np.arange(6, dtype=float) + 1.0
    crossfit_calibrate(proba, np.zeros(6, int), blocks, lambda: _W(), sample_weight=w)
    assert all(sw is not None for sw in received)
    assert received[0].shape == (4,)  # block 0 held out -> 4 train rows
