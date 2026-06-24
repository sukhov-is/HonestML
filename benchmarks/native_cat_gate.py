"""Native-categorical cardinality-gate calibration (ADR-0093, FR-1).

Empirically pins ``Task.native_cat_max_unique``: the category count above which routing a column
NATIVELY into CatBoost/LightGBM stops paying off and starts costing, so the gate demotes it to the
ordinal-codes path. Offline, deterministic, no network (a seeded synthetic, like ``corpus.py``) — so
the recommended cap is reproducible and the default is not an arbitrary constant.

For a sweep of category counts (rows fixed), one categorical predictor of that cardinality is
synthesized with a FIXED per-level signal-to-noise; for each boosting backend the runner measures, on
the UNTUNED native path (the unprotected default — the overfit knobs ``one_hot_max_size``/``cat_smooth``/
``min_data_per_group`` are tuned only under ``preset="best"``, ADR-0090 §A):

  * ``native_oof``  — honest k-fold OOF AUC (per-fold ordered target statistics, no leak)
  * ``codes_oof``   — the same model on the ordinal-codes path (the gate's fallback)
  * ``overfit_gap`` — train AUC minus native OOF AUC: the downside-risk signal the gate targets
  * ``fit_seconds`` — native fit wall-clock (diagnostic only; NOT in the recommendation, not reproducible)

:func:`recommend_cap` reads the overfit-gap-vs-cardinality curve and returns the largest cardinality
whose gap stays within a margin of the low-card baseline — the knee past which native ordered-TS
overfits the thinly-populated levels. The value justifies the pinned default; the default itself is a
sound default + opt-out (``native_cat_max_unique=None`` disables the gate).

The cap is a calibrated heuristic, not a universal optimum (ADR-0093, R-1): it is corroborated by the
cardinality ranges of genuinely useful categoricals in real data (district/profession/product — tens),
which sit well below id-like / a__b-intersection cardinalities (hundreds+).
"""
# ruff: noqa: T201  (a CLI tool: stdout IS the interface)

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

_HERE = Path(__file__).resolve().parent
RESULTS = _HERE / "native_cat_gate_results.json"

SEED = 20260623
N_ROWS = 1500
N_SPLITS = 3
CARDINALITIES: tuple[int, ...] = (4, 8, 16, 32, 64, 96, 128, 192, 256)
# the low-card baseline gap is the healthy reference; the cap is the last cardinality whose overfit
# gap stays within this margin of it (the knee). Tuned for a clear, reproducible separation.
GAP_MARGIN = 0.04


@dataclass(frozen=True)
class GatePoint:
    """One aggregated point of the calibration curve: cardinality -> mean overfit gap across backends."""

    cardinality: int
    overfit_gap: float


def recommend_cap(curve: Sequence[GatePoint], *, gap_margin: float = GAP_MARGIN) -> int:
    """The knee: the last cardinality before the overfit gap first climbs past the low-card baseline.

    The smallest-cardinality point is the healthy reference (native handling is safe there); the cap is
    the last cardinality before the gap **first** exceeds ``baseline + gap_margin`` — where native ordered
    target statistics start overfitting the thinly-populated levels. Walks the curve in cardinality order
    and stops at the first crossing (not ``max`` of all points within the margin), so a later noisy dip
    back under the threshold cannot pull the cap past the knee. Pure and deterministic (unit-tested).
    """
    ordered = sorted(curve, key=lambda p: p.cardinality)
    threshold = ordered[0].overfit_gap + gap_margin
    knee = ordered[0].cardinality
    for point in ordered:
        if point.overfit_gap > threshold:
            break
        knee = point.cardinality
    return knee


def _synth(cardinality: int, *, n: int = N_ROWS, seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    """One categorical predictor of ``cardinality`` levels with a fixed per-level signal + 3 numerics.

    The per-level effect is genuine signal (not pure noise), so a low-card category is exploitable;
    as cardinality grows toward ``n`` the levels thin out and native ordered-TS starts memorizing them
    — exactly the overfit the gap measures. The categorical sits at column 3.
    """
    rng = np.random.default_rng(seed)
    cat = rng.integers(0, cardinality, size=n)
    num = rng.normal(size=(n, 3))
    level_effect = rng.normal(size=cardinality) * 0.8
    signal = num[:, 0] + level_effect[cat] + 0.7 * rng.normal(size=n)
    y = (signal > np.median(signal)).astype(int)
    x = np.hstack([num, cat.astype(np.float64).reshape(-1, 1)])
    return x, y


def _folds(n: int, k: int = N_SPLITS) -> list[np.ndarray]:
    return [np.arange(i, n, k) for i in range(k)]


def _oof_auc(x: np.ndarray, y: np.ndarray, backend, *, native: bool) -> float:
    from honestml.adapters.boosting import build_boosting
    from honestml.core import Task

    n = y.shape[0]
    oof = np.full(n, np.nan)
    for test in _folds(n):
        train = np.setdiff1d(np.arange(n), test)
        est = build_boosting(backend, task=Task(kind="binary"), random_state=0)
        est.feature_names = ["n0", "n1", "n2", "c0"]
        if native:
            est.categorical_indices = [3]  # the categorical column routed natively
        est.fit(x[train], y[train])
        oof[test] = est.predict_proba(x[test])[:, 1]
    return float(roc_auc_score(y, oof))


def _train_auc_and_cost(x: np.ndarray, y: np.ndarray, backend) -> tuple[float, float]:
    """Native full-train fit: in-sample AUC (for the overfit gap) + fit wall-clock (diagnostic)."""
    from honestml.adapters.boosting import build_boosting
    from honestml.core import Task

    est = build_boosting(backend, task=Task(kind="binary"), random_state=0)
    est.feature_names = ["n0", "n1", "n2", "c0"]
    est.categorical_indices = [3]
    t0 = time.perf_counter()
    est.fit(x, y)
    cost = time.perf_counter() - t0
    return float(roc_auc_score(y, est.predict_proba(x)[:, 1])), cost


def run_sweep(cardinalities: Sequence[int] = CARDINALITIES) -> dict:
    """Fit every (cardinality, backend) point and return the records + aggregated calibration curve."""
    from honestml.adapters.boosting import CATBOOST, LIGHTGBM

    backends = {"catboost": CATBOOST, "lightgbm": LIGHTGBM}
    records: dict[str, list[dict]] = {name: [] for name in backends}
    curve: list[GatePoint] = []
    for cardinality in cardinalities:
        x, y = _synth(cardinality)
        gaps: list[float] = []
        for name, backend in backends.items():
            native_oof = _oof_auc(x, y, backend, native=True)
            codes_oof = _oof_auc(x, y, backend, native=False)
            train_auc, cost = _train_auc_and_cost(x, y, backend)
            gap = train_auc - native_oof
            gaps.append(gap)
            records[name].append(
                {
                    "cardinality": cardinality,
                    "native_oof": round(native_oof, 4),
                    "codes_oof": round(codes_oof, 4),
                    "train_auc": round(train_auc, 4),
                    "overfit_gap": round(gap, 4),
                    "fit_seconds": round(cost, 3),
                }
            )
            print(
                f"[card={cardinality:>4} {name:>9}] native_oof {native_oof:.3f} "
                f"codes_oof {codes_oof:.3f} gap {gap:+.3f} ({cost:.2f}s)",
                flush=True,
            )
        curve.append(GatePoint(cardinality, float(np.mean(gaps))))
    cap = recommend_cap(curve)
    return {
        "seed": SEED,
        "n_rows": N_ROWS,
        "n_splits": N_SPLITS,
        "gap_margin": GAP_MARGIN,
        "records": records,
        "curve": [{"cardinality": p.cardinality, "overfit_gap": round(p.overfit_gap, 4)} for p in curve],
        "recommended_cap": cap,
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    result = run_sweep()
    RESULTS.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nresults written to {RESULTS}")
    print(f"recommended native_cat_max_unique cap (knee of the overfit-gap curve): {result['recommended_cap']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
