"""Measured costs remain outside the deterministic honesty baseline."""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.unit


def _runner() -> Any:
    path = Path(__file__).resolve().parents[2] / "benchmarks" / "training_cost.py"
    spec = importlib.util.spec_from_file_location("training_cost", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _manifest(runner: Any) -> Any:
    recipe = runner.WorkRecipe("split-hash", "features-hash", "tree", {"depth": 6}, 100, 10, 1, 0)
    return runner.TrainingManifest(
        "data-hash", "oot-hash", "fixed_recipe", 42, "cold", recipe, "fit"
    )


def test_measurement_regions_and_no_prediction_values(monkeypatch) -> None:
    runner = _runner()
    now = [0.0]
    monkeypatch.setattr(runner.time, "perf_counter", lambda: now[0])
    prepared: list[int] = []

    def prepare(repeat: int) -> None:
        prepared.append(repeat)
        now[0] += 100.0

    def train() -> object:
        now[0] += 2.0
        return object()

    def save(model: object) -> None:
        now[0] += 0.5

    def predict(model: object) -> list[str]:
        now[0] += 0.25
        return ["private row value"]

    def evaluate(model: object, predictions: list[str]) -> dict[str, float]:
        assert predictions == ["private row value"]
        now[0] += 50.0
        return {"average_precision": 0.8}

    report = runner.measure_training(
        train,
        manifest=_manifest(runner),
        repeats=3,
        prepare=prepare,
        save=save,
        predict=predict,
        evaluate=evaluate,
    )
    assert prepared == [0, 1, 2]
    assert report["median_wall_s"] == 2.75
    assert report["median_regions_s"] == {"training": 2.0, "save": 0.5, "inference": 0.25}
    assert report["median_quality"] == {"average_precision": 0.8}
    assert "private row value" not in json.dumps(report)
    assert report["manifest"]["cache_mode"] == "cold"


def _measurement(runner: Any) -> dict[str, Any]:
    return runner.measure_training(
        object,
        manifest=_manifest(runner),
        repeats=1,
        prepare=lambda repeat: None,
        evaluate=lambda model, pred: {"average_precision": 0.8},
    )


def test_comparison_requires_thresholds_and_work_review() -> None:
    runner = _runner()
    reference = _measurement(runner)
    candidate = json.loads(json.dumps(reference))
    reference["median_wall_s"], candidate["median_wall_s"] = 10.0, 11.0
    pending = runner.compare_measurements(reference, candidate, metric="average_precision")
    assert pending["status"] == "pending_thresholds"
    assert pending["time_ratio"] == 1.1
    kwargs = {"metric": "average_precision", "quality_tolerance": 0.0, "max_time_ratio": 1.2}
    assert (
        runner.compare_measurements(reference, candidate, **kwargs)["status"]
        == "pending_work_review"
    )
    assert (
        runner.compare_measurements(reference, candidate, executed_work_verified=True, **kwargs)[
            "status"
        ]
        == "passed"
    )
    candidate["manifest"]["work"]["refit_count"] = 1
    assert runner.compare_measurements(reference, candidate, **kwargs)["status"] == "not_comparable"
    assert (
        runner.compare_measurements(reference, candidate, require_equivalent_work=False, **kwargs)[
            "status"
        ]
        == "passed"
    )
    candidate["manifest"]["cache_mode"] = "warm"
    assert (
        runner.compare_measurements(reference, candidate, require_equivalent_work=False, **kwargs)[
            "status"
        ]
        == "not_comparable"
    )


def test_comparison_uses_lower_is_better_orientation() -> None:
    runner = _runner()
    reference = _measurement(runner)
    candidate = json.loads(json.dumps(reference))
    reference["median_quality"], candidate["median_quality"] = {"loss": 0.5}, {"loss": 0.6}
    result = runner.compare_measurements(
        reference, candidate, metric="loss", higher_is_better=False
    )
    assert result["quality_loss"] == pytest.approx(0.1)


def test_average_precision_and_trapezoid_are_distinct() -> None:
    runner = _runner()
    metrics = runner.binary_quality([0, 1, 1], [0.8, 0.6, 0.3])
    assert metrics["average_precision"] != metrics["pr_auc"]
    assert set(metrics) == {"roc_auc", "average_precision", "pr_auc"}


def test_scenario_is_separate_from_declared_work() -> None:
    runner = _runner()
    manifest = _manifest(runner)
    for scenario in ("manual_reference", "fixed_recipe", "full_search", "fast_search"):
        for cache_mode in ("cold", "warm"):
            selected = replace(manifest, scenario=scenario, cache_mode=cache_mode)
            report = runner.measure_training(
                object, manifest=selected, repeats=1, prepare=lambda i: None
            )
            assert report["manifest"]["scenario"] == scenario
            assert report["manifest"]["cache_mode"] == cache_mode


def _label_partition(path: Path, target: Any, row_ids: Any) -> str:
    import hashlib

    import numpy as np

    np.savez(path, y_true=np.asarray(target), row_ids=np.asarray(row_ids))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _quality_inputs(runner: Any, tmp_path: Path, *, metric: str = "roc_auc") -> tuple[Any, ...]:
    import numpy as np

    row_ids = tuple(f"external-{i}" for i in range(40))
    regression = metric in ("rmse", "mae")
    target = np.arange(40, dtype=float) if regression else np.tile([0, 1], 20)
    labels_path = tmp_path / "labels.npz"
    contract = runner.QualityContract(
        task="regression" if regression else "binary",
        metric=metric,
        classes=() if regression else (0, 1),
        positive=None if regression else 0,
        margin=0.01,
        alpha=0.25,
        seed=42,
        n_bootstrap=200,
        baseline_artifact_sha256="a" * 64,
        candidate_artifact_sha256="b" * 64,
        external_partition_sha256=_label_partition(labels_path, target, row_ids),
        row_id_namespace="fixture-v1",
    )
    predictions = target + 1.0 if regression else np.where(target == 0, 0.9, 0.1)
    baseline = runner.PredictionBatch(
        artifact_sha256="a" * 64,
        row_ids=row_ids,
        values=predictions,
        classes=contract.classes,
        positive=contract.positive,
    )
    candidate = replace(baseline, artifact_sha256="b" * 64)
    return contract, baseline, candidate, labels_path, target


def _freeze(
    runner: Any, tmp_path: Path, contract: Any, baseline: Any, candidate: Any
) -> tuple[Path, str]:
    path = tmp_path / "quality-contract.json"
    digest = runner.freeze_quality_contract(
        path,
        contract,
        baseline=baseline,
        candidate=candidate,
        baseline_training_ids=("train-1",),
        baseline_tuning_ids=(),
        candidate_training_ids=("train-1",),
        candidate_tuning_ids=(),
        provenance={"source_manifest": "d" * 64},
    )
    return path, digest


def test_quality_freeze_receipt_and_independence_are_separate(tmp_path: Path) -> None:
    runner = _runner()
    contract, baseline, candidate, labels_path, _ = _quality_inputs(runner, tmp_path)
    path, digest = _freeze(runner, tmp_path, contract, baseline, candidate)
    receipt = json.loads(path.read_text())
    assert receipt["freeze_scope"] == "contract_created_without_labels"
    assert receipt["scorer"]["metric_semantics"] == "roc_auc"
    assert receipt["scorer"]["versions"]["scikit-learn"]
    assert receipt["weights_sha256"] is None and receipt["blocks_sha256"] is None
    assert set(receipt["prediction_sha256"]) == {"baseline", "candidate"}
    assert receipt["lineage"]["baseline_training"]["disjoint"] is True
    assert "external-0" not in path.read_text()
    with pytest.raises(FileExistsError):
        _freeze(runner, tmp_path, contract, baseline, candidate)
    report = runner.evaluate_paired_quality(
        path,
        frozen_sha256=digest,
        baseline=baseline,
        candidate=candidate,
        labels_path=labels_path,
    )
    assert report["statistical_verdict"] == "passed"
    assert report["quality_acceptance"] == "pending_review"
    assert report["independence_review"]["status"] == "pending"
    assert report["row_alignment_verified"] is True
    assert report["effective_rows"] == 40
    assert report["labels_file_sha256"] == contract.external_partition_sha256
    assert "external-0" not in json.dumps(report)


def test_quality_pr_auc_uses_average_precision_and_explicit_positive(tmp_path: Path) -> None:
    import numpy as np
    from sklearn.metrics import average_precision_score

    runner = _runner()
    contract, baseline, candidate, labels_path, target = _quality_inputs(
        runner, tmp_path, metric="pr_auc"
    )
    values = np.linspace(0.01, 0.99, len(target))
    baseline, candidate = replace(baseline, values=values), replace(candidate, values=values)
    path, digest = _freeze(runner, tmp_path, contract, baseline, candidate)
    result = runner.evaluate_paired_quality(
        path,
        frozen_sha256=digest,
        baseline=baseline,
        candidate=candidate,
        labels_path=labels_path,
    )
    assert result["metric_semantics"] == "average_precision"
    assert result["baseline_score"] == pytest.approx(average_precision_score(target == 0, values))


def test_quality_lower_is_better_ni_does_not_require_time_threshold(tmp_path: Path) -> None:
    runner = _runner()
    contract, baseline, candidate, labels_path, target = _quality_inputs(
        runner, tmp_path, metric="rmse"
    )
    candidate = replace(candidate, values=target + 3.0)
    path, digest = _freeze(runner, tmp_path, contract, baseline, candidate)
    result = runner.evaluate_paired_quality(
        path,
        frozen_sha256=digest,
        baseline=baseline,
        candidate=candidate,
        labels_path=labels_path,
    )
    assert result["statistical_verdict"] == "failed"
    assert result["oriented_improvement"] == pytest.approx(-2.0)
    assert "time_ratio" not in result


@pytest.mark.parametrize(
    "fault",
    [
        "row_order",
        "row_duplicate",
        "artifact",
        "classes",
        "positive",
        "shape",
        "nan",
        "range",
        "complex",
        "weights",
        "complex_weights",
        "blocks",
    ],
)
def test_quality_rejects_invalid_prediction_before_reading_labels(
    tmp_path: Path, fault: str
) -> None:
    import numpy as np

    runner = _runner()
    contract, baseline, candidate, labels_path, target = _quality_inputs(runner, tmp_path)
    path, digest = _freeze(runner, tmp_path, contract, baseline, candidate)
    if fault == "row_order":
        candidate = replace(candidate, row_ids=candidate.row_ids[::-1])
    elif fault == "row_duplicate":
        candidate = replace(candidate, row_ids=(candidate.row_ids[0],) * len(target))
    elif fault == "artifact":
        candidate = replace(candidate, artifact_sha256="e" * 64)
    elif fault == "classes":
        candidate = replace(candidate, classes=(1, 0))
    elif fault == "positive":
        candidate = replace(candidate, positive=1)
    elif fault == "shape":
        candidate = replace(candidate, values=np.column_stack([candidate.values] * 2))
    elif fault == "nan":
        candidate = replace(candidate, values=np.full(len(target), np.nan))
    elif fault == "range":
        candidate = replace(candidate, values=np.full(len(target), 1.1))
    elif fault == "complex":
        candidate = replace(candidate, values=candidate.values.astype(complex) + 0.2j)
    elif fault == "weights":
        candidate = replace(candidate, sample_weight=np.ones(len(target)))
    elif fault == "complex_weights":
        candidate = replace(candidate, sample_weight=np.ones(len(target), dtype=complex) + 0.1j)
    elif fault == "blocks":
        candidate = replace(candidate, block_index=np.arange(len(target)))
    with pytest.raises(ValueError):
        runner.freeze_quality_contract(
            tmp_path / "invalid.json", contract, baseline=baseline, candidate=candidate
        )
    labels_path.unlink()
    with pytest.raises(ValueError):
        runner.evaluate_paired_quality(
            path,
            frozen_sha256=digest,
            baseline=baseline,
            candidate=candidate,
            labels_path=labels_path,
        )


@pytest.mark.parametrize(
    "fault",
    ["order", "duplicate", "shape", "nan", "complex", "single_class", "unknown_class", "pickle"],
)
def test_quality_rejects_invalid_custodian_label_payload(tmp_path: Path, fault: str) -> None:
    import numpy as np

    runner = _runner()
    contract, baseline, candidate, labels_path, target = _quality_inputs(runner, tmp_path)
    ids = np.asarray(baseline.row_ids)
    if fault == "order":
        ids = ids[::-1]
    elif fault == "duplicate":
        ids[:] = ids[0]
    elif fault == "shape":
        target = target[:, None]
    elif fault == "nan":
        target = np.full(40, np.nan)
    elif fault == "complex":
        target = target.astype(complex) + 0.2j
    elif fault == "single_class":
        target = np.zeros(40)
    elif fault == "unknown_class":
        target = np.full(40, 2)
    elif fault == "pickle":
        ids = ids.astype(object)
    contract = replace(
        contract, external_partition_sha256=_label_partition(labels_path, target, ids)
    )
    path, digest = _freeze(runner, tmp_path, contract, baseline, candidate)
    with pytest.raises(ValueError):
        runner.evaluate_paired_quality(
            path,
            frozen_sha256=digest,
            baseline=baseline,
            candidate=candidate,
            labels_path=labels_path,
        )


@pytest.mark.parametrize("payload", ["receipt", "predictions", "labels"])
def test_quality_rejects_replacement_after_freeze(tmp_path: Path, payload: str) -> None:
    runner = _runner()
    contract, baseline, candidate, labels_path, _ = _quality_inputs(runner, tmp_path)
    path, digest = _freeze(runner, tmp_path, contract, baseline, candidate)
    if payload == "receipt":
        receipt = json.loads(path.read_text())
        receipt["contract"]["margin"] = 1.0
        path.write_text(json.dumps(receipt))
    elif payload == "predictions":
        candidate = replace(candidate, values=candidate.values * 0.9)
    else:
        labels_path.write_bytes(b"not even a valid NPZ")
    with pytest.raises(
        ValueError,
        match={
            "receipt": "receipt",
            "predictions": "prediction payload",
            "labels": "labels file digest",
        }[payload],
    ):
        runner.evaluate_paired_quality(
            path,
            frozen_sha256=digest,
            baseline=baseline,
            candidate=candidate,
            labels_path=labels_path,
        )


def test_quality_freeze_does_not_open_labels_and_unknown_lineage_is_not_empty(
    tmp_path: Path,
) -> None:
    runner = _runner()
    contract, baseline, candidate, labels_path, _ = _quality_inputs(runner, tmp_path)
    labels_path.unlink()
    path = tmp_path / "unknown.json"
    runner.freeze_quality_contract(path, contract, baseline=baseline, candidate=candidate)
    receipt = json.loads(path.read_text())
    assert receipt["lineage"]["baseline_training"]["status"] == "unknown"
    assert receipt["lineage"]["baseline_training"]["row_count"] is None
    with pytest.raises(ValueError, match="overlap"):
        runner.freeze_quality_contract(
            tmp_path / "overlap.json",
            contract,
            baseline=baseline,
            candidate=candidate,
            candidate_tuning_ids=(baseline.row_ids[0],),
        )
    assert not (tmp_path / "overlap.json").exists()


@pytest.mark.parametrize("metric", ["rmse", "mae"])
def test_quality_zero_weights_are_excluded_from_scoring_and_bootstrap(
    tmp_path: Path, metric: str
) -> None:
    import numpy as np

    runner = _runner()
    contract, baseline, candidate, labels_path, target = _quality_inputs(
        runner, tmp_path, metric=metric
    )
    weights, blocks = np.ones(len(target)), np.arange(len(target)) // 2
    weights[2:] = 0
    blocks[1] = 1
    values = baseline.values.copy()
    values[2:] = 1e6
    baseline = replace(baseline, values=values, sample_weight=weights, block_index=blocks)
    candidate = replace(candidate, values=values, sample_weight=weights, block_index=blocks)
    path, digest = _freeze(runner, tmp_path, contract, baseline, candidate)
    report = runner.evaluate_paired_quality(
        path,
        frozen_sha256=digest,
        baseline=baseline,
        candidate=candidate,
        labels_path=labels_path,
    )
    assert report["statistical_verdict"] == "passed"
    assert report["effective_rows"] == 2 and report["excluded_zero_weight_rows"] == 38
    assert report["effective_blocks"] == 2
    assert report["baseline_score"] == pytest.approx(1.0)
    changed = weights.copy()
    changed[2] = 1
    with pytest.raises(ValueError, match="weights"):
        runner.evaluate_paired_quality(
            path,
            frozen_sha256=digest,
            baseline=replace(baseline, sample_weight=changed),
            candidate=replace(candidate, sample_weight=changed),
            labels_path=labels_path,
        )


@pytest.mark.parametrize(
    "fault", ["one_row", "one_block", "no_class_mass", "negative", "nan", "zero_mass"]
)
def test_quality_rejects_insufficient_effective_population(tmp_path: Path, fault: str) -> None:
    import numpy as np

    runner = _runner()
    contract, baseline, candidate, labels_path, _ = _quality_inputs(runner, tmp_path)
    weights, blocks = np.ones(40), np.arange(40) // 2
    expected = "weights"
    if fault == "one_row":
        weights[1:] = 0
        expected = "insufficient_effective_rows"
    elif fault == "one_block":
        weights[2:] = 0
        expected = "insufficient_effective_blocks"
    elif fault == "no_class_mass":
        weights[1::2] = 0
        expected = "positive mass"
    elif fault == "negative":
        weights[0] = -1
    elif fault == "nan":
        weights[0] = np.nan
    elif fault == "zero_mass":
        weights[:] = 0
    baseline = replace(baseline, sample_weight=weights, block_index=blocks)
    candidate = replace(candidate, sample_weight=weights, block_index=blocks)
    with pytest.raises(ValueError, match=expected):
        path, digest = _freeze(runner, tmp_path, contract, baseline, candidate)
        runner.evaluate_paired_quality(
            path,
            frozen_sha256=digest,
            baseline=baseline,
            candidate=candidate,
            labels_path=labels_path,
        )


def test_quality_multiclass_shape_and_column_order(tmp_path: Path) -> None:
    import numpy as np

    runner = _runner()
    contract, baseline, candidate, labels_path, _ = _quality_inputs(runner, tmp_path)
    target = np.arange(40) % 3
    contract = replace(
        contract,
        task="multiclass",
        metric="log_loss",
        classes=(0, 1, 2),
        positive=None,
        external_partition_sha256=_label_partition(labels_path, target, baseline.row_ids),
    )
    values = np.full((40, 3), 0.1)
    values[np.arange(40), target] = 0.8
    baseline = replace(baseline, values=values, classes=contract.classes, positive=None)
    candidate = replace(candidate, values=values, classes=contract.classes, positive=None)
    path, digest = _freeze(runner, tmp_path, contract, baseline, candidate)
    result = runner.evaluate_paired_quality(
        path,
        frozen_sha256=digest,
        baseline=baseline,
        candidate=candidate,
        labels_path=labels_path,
    )
    assert result["statistical_verdict"] == "passed"
    with pytest.raises(ValueError, match="probabilities"):
        runner.evaluate_paired_quality(
            path,
            frozen_sha256=digest,
            baseline=baseline,
            candidate=replace(candidate, values=values * 0.5),
            labels_path=labels_path,
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("margin", -0.01),
        ("margin", float("nan")),
        ("alpha", 0.0),
        ("alpha", 1.0),
        ("n_bootstrap", 1),
        ("seed", -1),
        ("metric", "average_precision"),
        ("positive", None),
        ("classes", (1, 0)),
        ("baseline_artifact_sha256", "not-a-sha"),
    ],
)
def test_quality_contract_validates_before_creating_receipt(
    tmp_path: Path, field: str, value: Any
) -> None:
    runner = _runner()
    contract, baseline, candidate, _, _ = _quality_inputs(runner, tmp_path)
    path = tmp_path / "invalid-contract.json"
    with pytest.raises(ValueError):
        runner.freeze_quality_contract(
            path, replace(contract, **{field: value}), baseline=baseline, candidate=candidate
        )
    assert not path.exists()


def test_quality_rejects_complex_regression_target_before_float_cast(tmp_path: Path) -> None:
    runner = _runner()
    contract, baseline, candidate, labels_path, target = _quality_inputs(
        runner, tmp_path, metric="rmse"
    )
    contract = replace(
        contract,
        external_partition_sha256=_label_partition(
            labels_path,
            target.astype(complex) + 0.2j,
            baseline.row_ids,
        ),
    )
    path, digest = _freeze(runner, tmp_path, contract, baseline, candidate)
    with pytest.raises(ValueError, match="target cannot contain complex"):
        runner.evaluate_paired_quality(
            path,
            frozen_sha256=digest,
            baseline=baseline,
            candidate=candidate,
            labels_path=labels_path,
        )


def test_quality_two_row_weights_one_zero_rejected_explicitly(tmp_path: Path) -> None:
    import numpy as np

    runner = _runner()
    contract, baseline, candidate, _, _ = _quality_inputs(runner, tmp_path, metric="mae")
    baseline = replace(
        baseline,
        row_ids=baseline.row_ids[:2],
        values=baseline.values[:2],
        sample_weight=np.array([1, 0]),
    )
    candidate = replace(
        candidate,
        row_ids=candidate.row_ids[:2],
        values=candidate.values[:2],
        sample_weight=np.array([1, 0]),
    )
    with pytest.raises(ValueError, match="insufficient_effective_rows"):
        _freeze(runner, tmp_path, contract, baseline, candidate)
