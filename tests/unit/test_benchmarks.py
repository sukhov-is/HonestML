"""M9-4: the honesty benchmark contour (ADR-0076, FR-DLV-4, NFR-DLV-3).

Slow-marked: lives in one CI job, not the unit matrix. The full corpus runs in
`benchmark.yml`; here a tiny corpus pins same-env determinism, the orientation
formula and the no-regress check logic.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow

_BENCH = Path(__file__).resolve().parents[2] / "benchmarks"


def _load(module: str):
    spec = importlib.util.spec_from_file_location(module, _BENCH / f"{module}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module] = mod  # register before exec so a dataclass can resolve cls.__module__
    spec.loader.exec_module(mod)
    return mod


def test_optimism_orientation() -> None:
    """ADR-0076 §3: the runner applies the metric orientation, scores stay raw."""
    run = _load("run")
    assert run.optimism(0.95, 0.90, True) == pytest.approx(0.05)  # AUC: higher better
    assert run.optimism(10.0, 12.0, False) == pytest.approx(2.0)  # RMSE: lower better
    assert run.optimism(0.90, 0.95, True) == pytest.approx(-0.05)  # pessimism is negative


def test_tiny_corpus_double_run_is_byte_identical() -> None:
    """NFR-DLV-3 (runner determinism): two runs with a fixed deterministic model set produce an
    identical results JSON. Boosting backends are not bit-stable across runs on multi-core (the
    no-regress gate absorbs that via atol), so the runner's own determinism uses baseline+linear."""
    run = _load("run")
    corpus = _load("corpus")

    def _tiny_load():
        from sklearn.datasets import make_classification

        return make_classification(
            n_samples=80, n_features=6, n_informative=4, n_redundant=0, random_state=7
        )

    tiny = (corpus.DatasetSpec("tiny", "binary", _tiny_load),)
    models = ("baseline", "linear")
    first = json.dumps(run.run_corpus(tiny, models=models), indent=2, sort_keys=True)
    second = json.dumps(run.run_corpus(tiny, models=models), indent=2, sort_keys=True)
    assert first == second
    record = json.loads(first)["datasets"]["tiny"]
    assert {"selection_score", "holdout_score", "optimism", "winner", "atol"} <= set(record)


def test_no_regress_check_logic() -> None:
    run = _load("run")
    baseline = {
        "datasets": {
            "a": {"optimism": 0.01, "holdout_score": 0.90, "atol": 0.02},
            "b": {"optimism": 0.00, "holdout_score": 0.80, "atol": 0.02},
        }
    }
    ok = {
        "datasets": {
            "a": {"optimism": 0.02, "holdout_score": 0.89, "greater_is_better": True},
            "b": {"optimism": -0.01, "holdout_score": 0.81, "greater_is_better": True},
        }
    }
    assert run.check_results(ok, baseline) == []
    bad = {
        "datasets": {
            # optimism regressed; "b" missing from the run entirely
            "a": {"optimism": 0.08, "holdout_score": 0.90, "greater_is_better": True},
        }
    }
    failures = run.check_results(bad, baseline)
    assert len(failures) == 2
    assert any("optimism regressed" in f for f in failures)
    assert any("missing" in f for f in failures)


def test_check_orients_lower_is_better_holdout() -> None:
    """Review M9-4 #3: for rmse/log_loss a GROWN holdout score is the degradation."""
    run = _load("run")
    baseline = {"datasets": {"reg": {"optimism": 0.0, "holdout_score": 50.0, "atol": 1.0}}}
    improved = {
        "datasets": {"reg": {"optimism": 0.0, "holdout_score": 45.0, "greater_is_better": False}}
    }
    assert run.check_results(improved, baseline) == []  # lower rmse = better, no failure
    degraded = {
        "datasets": {"reg": {"optimism": 0.0, "holdout_score": 80.0, "greater_is_better": False}}
    }
    failures = run.check_results(degraded, baseline)
    assert len(failures) == 1 and "holdout quality regressed" in failures[0]


def test_new_dataset_requires_conscious_baseline_refresh() -> None:
    run = _load("run")
    baseline = {"datasets": {}}
    results = {"datasets": {"fresh": {"optimism": 0.0, "holdout_score": 0.9}}}
    failures = run.check_results(results, baseline)
    assert len(failures) == 1 and "not in the baseline" in failures[0]


def test_runner_orientation_comes_from_the_library() -> None:
    """Review M9-4 #2 root cause: orientation is derived, never hand-declared. The default
    multiclass metric is lower-is-better (log_loss) — exactly the case the hand-written
    corpus flag got wrong."""
    from honestml.adapters import resolve_metric
    from honestml.core import Task

    assert resolve_metric(Task(kind="multiclass").target_metric).greater_is_better is False
    assert resolve_metric(Task(kind="regression").target_metric).greater_is_better is False
    assert resolve_metric(Task(kind="binary").target_metric).greater_is_better is True
    corpus = _load("corpus")
    assert not hasattr(corpus.DatasetSpec("x", "binary", lambda: None), "greater_is_better")


def test_corpus_is_offline_and_seeded() -> None:
    """NFR-DLV-3: every dataset loads without network and is deterministic."""
    corpus = _load("corpus")
    for spec in corpus.CORPUS:
        X, y = spec.load()
        X2, y2 = spec.load()
        assert (X == X2).all() and (y == y2).all(), spec.name
        assert spec.task in ("binary", "multiclass", "regression")


# --- native-categorical cardinality-gate calibration (ADR-0093, FR-1) ---


def test_native_cat_gate_recommend_cap_finds_the_knee() -> None:
    """recommend_cap = the last cardinality whose overfit gap stays within the margin of the baseline.

    Locks the computation that pins Task.native_cat_max_unique (so the default is derived, not arbitrary):
    a flat low-card region followed by a knee resolves to the cardinality just before the climb.
    """
    gate = _load("native_cat_gate")
    pt = gate.GatePoint
    curve = [pt(4, 0.02), pt(16, 0.03), pt(32, 0.04), pt(64, 0.05), pt(128, 0.18), pt(256, 0.30)]
    # baseline gap 0.02; margin 0.04 -> threshold 0.06; 64@0.05 passes, 128@0.18 is the knee
    assert gate.recommend_cap(curve, gap_margin=0.04) == 64
    # a flat curve (gate barely bites) recommends the largest cardinality, not a spurious low cap
    flat = [pt(4, 0.02), pt(64, 0.03), pt(256, 0.035)]
    assert gate.recommend_cap(flat, gap_margin=0.04) == 256
    # robustness: a noisy DIP back under the threshold after the knee must NOT pull the cap past it
    # (the knee is the FIRST crossing, not max of all points within the margin) -> still 32, not 256
    noisy = [pt(4, 0.02), pt(32, 0.05), pt(64, 0.18), pt(128, 0.20), pt(256, 0.04)]
    assert gate.recommend_cap(noisy, gap_margin=0.04) == 32
