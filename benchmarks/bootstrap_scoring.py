"""Compare bootstrap scoring on fixed predictions without fitting estimators."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import sklearn

from honestml.adapters.metrics import Accuracy, LogLoss, Mae, Rmse
from honestml.adapters.significance import BootstrapSignificanceTest
from honestml.core import Metric


class _ReferenceMetric:
    def __init__(self, metric: Metric) -> None:
        self.metric = metric
        self.name = metric.name
        self.needs = metric.needs
        self.greater_is_better = metric.greater_is_better
        self.optimum = metric.optimum
        self.average = metric.average

    def score(
        self, y_true: np.ndarray, y_pred: np.ndarray, sample_weight: np.ndarray | None = None
    ) -> float:
        return self.metric.score(y_true, y_pred, sample_weight)


def compare(rows: int, repeats: int, n_boot: int) -> dict[str, Any]:
    rng = np.random.default_rng(42)
    labels = np.arange(9)
    y = np.resize(labels, rows)
    a = rng.dirichlet(np.ones(9), rows)
    b = rng.dirichlet(np.arange(1, 10), rows)
    weights = rng.uniform(0.1, 2, rows)
    results = []
    for metric in [LogLoss(classes=labels), Accuracy(), Mae(), Rmse()]:
        target, first, second = y, a, b
        if isinstance(metric, Accuracy):
            first, second = a.argmax(axis=1), b.argmax(axis=1)
        elif isinstance(metric, (Mae, Rmse)):
            target, first, second = y.astype(float), a[:, 0], b[:, 0]
        tests = {
            "reference": BootstrapSignificanceTest(_ReferenceMetric(metric), seed=7, n_boot=n_boot),
            "prepared": BootstrapSignificanceTest(metric, seed=7, n_boot=n_boot),
        }
        timings: dict[str, list[float]] = {"reference": [], "prepared": []}
        arrays = {}
        for repeat in range(repeats):
            order = ("reference", "prepared") if repeat % 2 == 0 else ("prepared", "reference")
            for name in order:
                start = time.perf_counter()
                arrays[name] = tests[name]._delta_distribution(first, second, target, weights, None)
                timings[name].append(time.perf_counter() - start)
        np.testing.assert_array_equal(arrays["reference"], arrays["prepared"])
        medians = {key: statistics.median(value) for key, value in timings.items()}
        results.append(
            {
                "metric": metric.name,
                "seconds": timings,
                "median_seconds": medians,
                "reference_over_prepared": medians["reference"] / medians["prepared"],
                "distribution_byte_equal": arrays["reference"].tobytes()
                == arrays["prepared"].tobytes(),
                "max_abs_delta_difference": float(
                    np.max(np.abs(arrays["reference"] - arrays["prepared"]))
                ),
            }
        )
    return {
        "kind": "fixed_predictions_bootstrap_only",
        "rows": rows,
        "n_boot": n_boot,
        "repeats": repeats,
        "seed": 42,
        "bootstrap_seed": 7,
        "weighted": True,
        "dtype": "float64",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=10000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare(args.rows, args.repeats, args.n_boot)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
