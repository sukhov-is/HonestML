"""Cross-fitted probability calibration (ADR-0030 §3 / ADR-0031 §1) — pure use-case.

Out-of-fold calibration: for each block, fit a fresh ``Calibrator`` on the OTHER blocks and
transform this block, so the calibrated probabilities carry no in-sample optimism — the key
to an honest refinement-based selection and to the calibration improvement gate. Pure numpy
over the ``Calibrator`` port (no sklearn here, NFR-M4-6) — testable on a fake.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from honestml.core import Calibrator, CalibratorFactory, Metric

# per-block calibration viability floor (ADR-0030 §3 / ADR-0031 §4a): fewer training rows than
# this in any cross-fit block and the calibrator is under-fit -> the whole run falls back to raw.
# Single floor in M4d for both sigmoid and isotonic; a method-dependent floor (isotonic higher,
# paper Apx D) is deferred (ADR-0030 §3 / ADR-0031 §4a impl-note).
MIN_CALIB_N = 50


def viable_blocks(
    blocks: np.ndarray,
    y_code: np.ndarray,
    min_n: int = MIN_CALIB_N,
    *,
    n_classes: int | None = None,
) -> bool:
    """Every cross-fit block's train side has >= ``min_n`` rows and all needed classes (ADR-0031 §4a).

    The shared precondition of :func:`crossfit_calibrate`, enforced by both callers — refinement
    selection (per-run all-or-nothing) and the production calibration gate. ``n_classes`` (multiclass)
    requires EVERY class present in each block's train side: a class absent there fits its per-class OvR
    map on an all-zero target and zeroes that class's calibrated mass (F023). Binary leaves it ``None``
    -> the >= 2 check.
    """
    need = n_classes if n_classes is not None else 2
    for b in np.unique(blocks):
        train = blocks != b
        if int(train.sum()) < min_n or np.unique(y_code[train]).size < need:
            return False
    return True


def crossfit_calibrate(
    proba: np.ndarray,
    y: np.ndarray,
    blocks: np.ndarray,
    factory: CalibratorFactory,
    *,
    sample_weight: np.ndarray | None = None,
) -> np.ndarray:
    """Out-of-fold calibrated probabilities (same shape as ``proba``).

    ``y`` is the coded target the calibrator expects (binary ``{0, 1}``, 1 = positive;
    multiclass the true column index). ``blocks`` assigns each row to a cross-fit block (the
    CV fold id); block ``b`` is calibrated by a map fitted on all rows with ``blocks != b``.
    Assumes the caller validated viability (every block's train side is non-degenerate,
    ADR-0031 §4a).
    """
    cal = np.empty_like(proba, dtype=np.float64)
    for b in np.unique(blocks):
        test = blocks == b
        train = ~test
        calibrator = factory()
        calibrator.fit(
            proba[train], y[train], sample_weight[train] if sample_weight is not None else None
        )
        cal[test] = calibrator.transform(proba[test])
    return cal


def calibrate_winner(
    proba: np.ndarray,
    y_true: np.ndarray,
    y_code: np.ndarray,
    blocks: np.ndarray,
    factory: CalibratorFactory,
    *,
    brier: Metric,
    ece: Metric,
    method: str,
    sample_weight: np.ndarray | None = None,
    n_bins: int = 10,
) -> tuple[Calibrator | None, dict[str, Any]]:
    """Cross-fit gate then (if it improves Brier) the production calibrator on the FULL OOF.

    The improvement gate compares out-of-fold calibrated vs raw Brier (proper; the lead metric,
    ADR-0030 §3): no in-sample optimism. If calibrated Brier ≤ raw, a fresh calibrator is fit on
    ALL valid OOF rows (no data lost) and returned; otherwise ``None`` (not attached). The report
    carries raw/calibrated Brier & ECE and a reliability curve (FR-M4-11). All arrays are the
    winner's already-masked valid OOF; ``y_code`` is the calibrator's coded target, ``y_true`` the
    raw labels the Brier/ECE metrics score on.
    """
    cal_oof = crossfit_calibrate(proba, y_code, blocks, factory, sample_weight=sample_weight)
    brier_raw = brier.score(y_true, proba, sample_weight)
    brier_cal = brier.score(y_true, cal_oof, sample_weight)
    ece_raw = ece.score(y_true, proba, sample_weight)
    ece_cal = ece.score(y_true, cal_oof, sample_weight)
    conf, correct = _confidence_correct(cal_oof, y_code)
    prob_true, prob_pred = _reliability_curve(conf, correct, n_bins)
    applied = brier_cal <= brier_raw
    calibrator: Calibrator | None = None
    if applied:
        calibrator = factory()
        calibrator.fit(proba, y_code, sample_weight)
    report = {
        "method": method,
        "applied": applied,
        "brier_raw": brier_raw,
        "brier_calibrated": brier_cal,
        "ece_raw": ece_raw,
        "ece_calibrated": ece_cal,
        "reliability": {"prob_true": prob_true, "prob_pred": prob_pred},
    }
    return calibrator, report


def _confidence_correct(proba: np.ndarray, y_code: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Top-label confidence and correctness (binary P(pos) or multiclass argmax)."""
    if proba.ndim == 2:
        return proba.max(axis=1), (proba.argmax(axis=1) == y_code).astype(np.float64)
    pred_pos = proba >= 0.5
    conf = np.where(pred_pos, proba, 1.0 - proba)
    return conf, (pred_pos == (y_code == 1)).astype(np.float64)


def _reliability_curve(
    conf: np.ndarray, correct: np.ndarray, n_bins: int
) -> tuple[list[float], list[float]]:
    """Per-bin (fraction correct, mean confidence) over populated uniform bins (JSON-ready)."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_id = np.clip(np.digitize(conf, edges[1:-1]), 0, n_bins - 1)
    prob_true: list[float] = []
    prob_pred: list[float] = []
    for m in range(n_bins):
        sel = bin_id == m
        if sel.any():
            prob_true.append(float(correct[sel].mean()))
            prob_pred.append(float(conf[sel].mean()))
    return prob_true, prob_pred
