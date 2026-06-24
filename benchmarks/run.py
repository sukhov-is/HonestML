"""Honesty benchmark runner (ADR-0076).

Verifies the project's main claim — the score reported at selection is not
optimistic against an untouched outer holdout — and keeps releases comparable:

    optimism_d = (selection_score - holdout_score) * sign(greater_is_better)

The metric orientation is taken from the LIBRARY (``resolve_metric``) and recorded
per dataset — the checker orients both the optimism and the holdout comparison.
The gate is NO REGRESS against the committed ``baseline.json`` (per-dataset
tolerances); the absolute mean/max optimism per metric family is reported as a
diagnostic, not gated (an initial run cannot certify itself, ADR-0076 §3).
``results.json`` carries no timings — two runs in one environment are
byte-identical; the baseline is generated/updated ONLY by the CI job
(pinned environment via ``uv sync --frozen``), see README.md.
"""
# ruff: noqa: T201  (a CLI tool: stdout IS the interface)

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from corpus import CORPUS, DatasetSpec  # noqa: E402

SEED = 20260612
HOLDOUT = 0.2
DEFAULT_ATOL = 0.02
RESULTS = _HERE / "results.json"
BASELINE = _HERE / "baseline.json"


def optimism(selection: float, holdout: float, greater_is_better: bool) -> float:
    """Signed optimism in higher-is-better orientation (ADR-0076 §3)."""
    return (selection - holdout) * (1.0 if greater_is_better else -1.0)


def _versions() -> dict[str, str]:
    """The compute stack behind the numbers (ADR-0076 §3) — constant within one env."""
    from importlib.metadata import PackageNotFoundError, version

    import honestml

    out = {"honestml": honestml.__version__}
    for pkg in ("scikit-learn", "numpy", "lightgbm", "xgboost", "catboost"):
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            pass
    return out


def run_corpus(specs: tuple[DatasetSpec, ...] = CORPUS) -> dict:
    from honestml import AutoML, CVConfig
    from honestml.adapters import resolve_metric

    records: dict[str, dict] = {}
    for spec in specs:
        print(f"[{spec.name}] fitting ...", flush=True)
        X, y = spec.load()
        model = AutoML(
            task=spec.task,
            random_state=SEED,
            cv=CVConfig(outer_holdout=HOLDOUT),
        ).fit(X, y)
        report = model.run_report_
        greater = bool(resolve_metric(report["metric"]).greater_is_better)
        selection = next(
            entry["score"]
            for entry in report["leaderboard"]
            if entry["model_id"] == report["winner"]
        )
        records[spec.name] = {
            "task": spec.task,
            "metric": report["metric"],
            "greater_is_better": greater,
            "winner": report["winner"],
            "models": sorted(entry["model_id"] for entry in report["leaderboard"]),
            "band_members": list(report["band"]["member_ids"]),
            "selection_score": round(float(selection), 6),
            "holdout_score": round(float(report["holdout_score"]), 6),
            "optimism": round(
                optimism(float(selection), float(report["holdout_score"]), greater), 6
            ),
            "atol": DEFAULT_ATOL,
        }
    return {
        "benchmark_version": 1,
        "seed": SEED,
        "outer_holdout": HOLDOUT,
        "versions": _versions(),
        "datasets": records,
    }


def summarize(results: dict) -> dict[str, dict[str, float]]:
    """Diagnostic mean/max optimism per metric family — families are never mixed."""
    families: dict[str, list[float]] = {}
    for record in results["datasets"].values():
        families.setdefault(record["metric"], []).append(record["optimism"])
    return {
        metric: {"mean": sum(vals) / len(vals), "max": max(vals), "n": len(vals)}
        for metric, vals in families.items()
    }


def check_results(results: dict, baseline: dict) -> list[str]:
    """No-regress gate (ADR-0076 §3): per-dataset, metric-ORIENTED, within baseline atol."""
    failures: list[str] = []
    for name, base in baseline["datasets"].items():
        record = results["datasets"].get(name)
        if record is None:
            failures.append(f"{name}: dataset missing from the run")
            continue
        atol = float(base.get("atol", DEFAULT_ATOL))
        if record["optimism"] > base["optimism"] + atol:
            failures.append(
                f"{name}: optimism regressed {base['optimism']:+.4f} -> "
                f"{record['optimism']:+.4f} (atol {atol})"
            )
        # holdout quality regress in the METRIC's own orientation: for lower-is-better
        # metrics (rmse/log_loss) a GROWN holdout score is the degradation
        sign = 1.0 if record.get("greater_is_better", True) else -1.0
        if sign * (base["holdout_score"] - record["holdout_score"]) > atol:
            failures.append(
                f"{name}: holdout quality regressed {base['holdout_score']:.4f} -> "
                f"{record['holdout_score']:.4f} (atol {atol})"
            )
    new = sorted(set(results["datasets"]) - set(baseline["datasets"]))
    if new:
        failures.append(
            f"datasets {new} are not in the baseline — refresh it consciously "
            "(--update-baseline via the CI job + CHANGELOG)"
        )
    return failures


def _merged_baseline(results: dict) -> dict:
    """--update-baseline keeps hand-tuned per-dataset atol values (README policy)."""
    merged = json.loads(json.dumps(results))  # deep copy of plain JSON
    if BASELINE.exists():
        old = json.loads(BASELINE.read_text(encoding="utf-8"))
        for name, record in merged["datasets"].items():
            previous = old.get("datasets", {}).get(name)
            if previous is not None and "atol" in previous:
                record["atol"] = previous["atol"]
    return merged


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="compare against baseline.json")
    mode.add_argument(
        "--update-baseline",
        action="store_true",
        help="rewrite baseline.json (CI only, + CHANGELOG)",
    )
    args = parser.parse_args(argv)

    results = run_corpus()
    RESULTS.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    print(f"results written to {RESULTS}")
    for metric, stats in summarize(results).items():
        print(
            f"optimism[{metric}]: mean {stats['mean']:+.4f}, max {stats['max']:+.4f} "
            f"over {stats['n']} dataset(s)"
        )

    if args.update_baseline:
        BASELINE.write_text(
            json.dumps(_merged_baseline(results), indent=2, sort_keys=True), encoding="utf-8"
        )
        print(f"baseline updated at {BASELINE} — commit it together with a CHANGELOG line")
        return 0
    if args.check:
        if not BASELINE.exists():
            print("FAIL: no baseline.json — bootstrap it via the CI job (--update-baseline)")
            return 1
        failures = check_results(results, json.loads(BASELINE.read_text(encoding="utf-8")))
        if failures:
            print("FAIL: honesty regression detected:")
            for line in failures:
                print(f"  - {line}")
            return 1
        print("OK: no regress against baseline")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
