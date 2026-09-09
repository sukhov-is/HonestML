"""Standalone wall-time measurements; the deterministic honesty benchmark is independent."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import statistics
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Literal

import numpy as np

from honestml.core import Metric

Scalar = str | int | float | bool | None
Scenario = Literal["manual_reference", "fixed_recipe", "full_search", "fast_search"]


@dataclass(frozen=True)
class WorkRecipe:
    """Declared fixed work; identifiers are digests or opaque labels, never row values."""

    split_id: str
    feature_set_id: str
    estimator: str
    parameters: dict[str, Scalar]
    tree_budget: int | None
    early_stopping_rounds: int | None
    fit_count: int
    refit_count: int
    fs_recipe_id: str | None = None


@dataclass(frozen=True)
class TrainingManifest:
    """Local comparison provenance, with separate evaluation and training partitions."""

    dataset_id: str
    evaluation_id: str
    scenario: Scenario
    seed: int
    cache_mode: Literal["cold", "warm"]
    work: WorkRecipe
    training_boundary: str


def binary_quality(y_true: Any, scores: Any) -> dict[str, float]:
    """Separate average precision from trapezoidal PR area on the same untouched rows."""
    from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_auc_score

    precision, recall, _ = precision_recall_curve(y_true, scores)
    return {
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "average_precision": float(average_precision_score(y_true, scores)),
        "pr_auc": float(auc(recall, precision)),
    }


def measure_training(
    train: Callable[[], Any],
    *,
    manifest: TrainingManifest,
    repeats: int,
    prepare: Callable[[int], None],
    save: Callable[[Any], None] | None = None,
    predict: Callable[[Any], Any] | None = None,
    evaluate: Callable[[Any, Any], Mapping[str, float]] | None = None,
) -> dict[str, Any]:
    """Measure training, serialization and inference with explicit cold/warm preparation.

    ``prepare(repeat)`` establishes the declared cache state outside the measured
    interval. ``train`` returns a fresh fitted model; its callback boundary states
    whether loading, FE and FS are included. Evaluation runs after timing on the
    same untouched partition for all compared scenarios. Predictions and source
    rows never enter the returned report.
    """
    if repeats < 1:
        raise ValueError("repeats must be positive")
    if manifest.cache_mode not in ("cold", "warm"):
        raise ValueError("cache_mode must be cold or warm")
    from honestml.composition.run_report import configure_run_cost
    from honestml.core import RunContext

    context = RunContext()
    configure_run_cost(context)
    runs: list[dict[str, Any]] = []
    regions = ["training"] + (["save"] if save is not None else [])
    if predict is not None:
        regions.append("inference")
    for repeat in range(repeats):
        prepare(repeat)
        memory = {"before_training": context.sample_memory()}
        started = time.perf_counter()
        model = train()
        trained = time.perf_counter()
        durations = {"training": trained - started}
        memory["after_training"] = context.sample_memory()
        if save is not None:
            save_start = time.perf_counter()
            save(model)
            durations["save"] = time.perf_counter() - save_start
            memory["after_save"] = context.sample_memory()
        predictions = None
        if predict is not None:
            infer_start = time.perf_counter()
            predictions = predict(model)
            durations["inference"] = time.perf_counter() - infer_start
            memory["after_inference"] = context.sample_memory()
        total = time.perf_counter() - started
        quality = dict(evaluate(model, predictions)) if evaluate is not None else {}
        if any(not math.isfinite(float(value)) for value in quality.values()):
            raise ValueError("quality metrics must be finite scalar values")
        if runs and set(quality) != set(runs[0]["quality"]):
            raise ValueError("each repeat must report the same quality metrics")
        run_report = getattr(model, "run_report_", None)
        cost = run_report.get("cost") if isinstance(run_report, Mapping) else None
        runs.append(
            {
                "repeat": repeat,
                "total_wall_s": total,
                "regions_s": durations,
                "overhead_s": total - sum(durations.values()),
                "quality": {key: float(value) for key, value in quality.items()},
                "cost": cost,
                "boundary_rss_mb": memory,
                "sampled_peak_rss_mb": max(
                    (value for value in memory.values() if value is not None), default=None
                ),
            }
        )
        del model, predictions
    result = {
        "performance_report_version": 1,
        "manifest": asdict(manifest),
        "timed_regions": regions,
        "excluded_regions": ["cache_preparation", "quality_evaluation"],
        "environment": context.environment,
        "repeats": repeats,
        "runs": runs,
        "median_wall_s": statistics.median(run["total_wall_s"] for run in runs),
        "median_regions_s": {
            region: statistics.median(run["regions_s"][region] for run in runs)
            for region in regions
        },
        "median_quality": {
            metric: statistics.median(run["quality"][metric] for run in runs)
            for metric in runs[0]["quality"]
        },
    }
    return json.loads(json.dumps(result, allow_nan=False))


def compare_measurements(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    metric: str,
    higher_is_better: bool = True,
    quality_tolerance: float | None = None,
    max_time_ratio: float | None = None,
    require_equivalent_work: bool = True,
    executed_work_verified: bool = False,
) -> dict[str, Any]:
    """Compare local medians; acceptance remains pending until thresholds are supplied.

    Fixed-recipe overhead requires identical declared work. Search comparisons may
    use different recipes, while data, evaluation, environment and timed boundaries
    remain equal. Matching declarations alone do not prove identical executed work.
    """
    if quality_tolerance is not None and quality_tolerance < 0:
        raise ValueError("quality_tolerance must be nonnegative")
    if max_time_ratio is not None and max_time_ratio <= 0:
        raise ValueError("max_time_ratio must be positive")
    ref_manifest, cand_manifest = reference["manifest"], candidate["manifest"]
    mismatches = [
        key
        for key in ("dataset_id", "evaluation_id", "seed", "cache_mode", "training_boundary")
        if ref_manifest[key] != cand_manifest[key]
    ]
    mismatches += [
        key
        for key in ("environment", "timed_regions", "excluded_regions")
        if reference[key] != candidate[key]
    ]
    ref_work, cand_work = ref_manifest["work"], cand_manifest["work"]
    work_differences = sorted(
        key for key in set(ref_work) | set(cand_work) if ref_work.get(key) != cand_work.get(key)
    )
    if require_equivalent_work and work_differences:
        mismatches.append("work")
    ref_seconds = float(reference["median_wall_s"])
    candidate_seconds = float(candidate["median_wall_s"])
    if not math.isfinite(ref_seconds) or ref_seconds <= 0:
        raise ValueError("reference median wall time must be finite and positive")
    if not math.isfinite(candidate_seconds) or candidate_seconds < 0:
        raise ValueError("candidate median wall time must be finite and nonnegative")
    ratio = candidate_seconds / ref_seconds
    ref_quality = float(reference["median_quality"][metric])
    cand_quality = float(candidate["median_quality"][metric])
    if not math.isfinite(ref_quality) or not math.isfinite(cand_quality):
        raise ValueError("compared quality must be finite")
    loss = (ref_quality - cand_quality) * (1.0 if higher_is_better else -1.0)
    quality_ok = None if quality_tolerance is None else loss <= quality_tolerance
    time_ok = None if max_time_ratio is None else ratio <= max_time_ratio
    if mismatches:
        status = "not_comparable"
    elif quality_ok is None or time_ok is None:
        status = "pending_thresholds"
    elif require_equivalent_work and quality_ok and time_ok and not executed_work_verified:
        status = "pending_work_review"
    else:
        status = "passed" if quality_ok and time_ok else "failed"
    return {
        "status": status,
        "comparison": "fixed_recipe" if require_equivalent_work else "search",
        "mismatches": mismatches,
        "declared_work_matches": not work_differences,
        "work_differences": work_differences,
        "executed_work_equivalence": "verified" if executed_work_verified else "requires_review",
        "time_ratio": ratio,
        "metric": metric,
        "quality_loss": loss,
        "quality_tolerance": quality_tolerance,
        "max_time_ratio": max_time_ratio,
        "quality_passed": quality_ok,
        "time_passed": time_ok,
    }


@dataclass(frozen=True)
class QualityContract:
    """Predeclared paired test; external_partition_sha256 pins the custodian NPZ label file."""

    task: Literal["binary", "multiclass", "regression"]
    metric: str
    classes: tuple[str | int | float, ...]
    positive: str | int | float | None
    margin: float
    alpha: float
    seed: int
    n_bootstrap: int
    baseline_artifact_sha256: str
    candidate_artifact_sha256: str
    external_partition_sha256: str
    row_id_namespace: str


@dataclass(frozen=True)
class PredictionBatch:
    """Predictions in declared row/class order; binary values are P(explicit positive)."""

    artifact_sha256: str
    row_ids: tuple[str, ...]
    values: np.ndarray
    classes: tuple[str | int | float, ...] = ()
    positive: str | int | float | None = None
    sample_weight: np.ndarray | None = None
    block_index: np.ndarray | None = None


def _json_digest(value: object) -> str:
    content = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _check_sha256(value: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError("expected a lowercase SHA-256 digest")


def _row_ids(values: Sequence[str], *, allow_empty: bool = False) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError("row IDs must be a sequence")
    ids = tuple(values)
    if (not ids and not allow_empty) or any(not isinstance(v, str) or not v for v in ids):
        raise ValueError("row IDs must be nonempty strings")
    if len(set(ids)) != len(ids):
        raise ValueError("row IDs must be unique")
    return ids


def _quality_scorer(contract: QualityContract) -> Metric:
    from honestml.adapters.metrics import resolve_metric

    allowed = {
        "binary": {"roc_auc", "pr_auc", "log_loss", "brier"},
        "multiclass": {"log_loss", "brier"},
        "regression": {"rmse", "mae"},
    }
    if contract.task not in allowed or contract.metric not in allowed[contract.task]:
        raise ValueError("unsupported task/metric ID for paired quality")
    if not math.isfinite(contract.margin) or contract.margin < 0:
        raise ValueError("margin must be finite and nonnegative")
    if not 0 < contract.alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if type(contract.seed) is not int or contract.seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if type(contract.n_bootstrap) is not int or contract.n_bootstrap * contract.alpha < 50:
        raise ValueError("n_bootstrap * alpha must be at least 50")
    for digest in (
        contract.baseline_artifact_sha256,
        contract.candidate_artifact_sha256,
        contract.external_partition_sha256,
    ):
        _check_sha256(digest)
    if not isinstance(contract.row_id_namespace, str) or not contract.row_id_namespace.strip():
        raise ValueError("row ID namespace is required")
    labels = tuple(contract.classes)
    if contract.task == "regression":
        if labels or contract.positive is not None:
            raise ValueError("regression has no classes or positive label")
    else:
        numeric = all(
            isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
            for v in labels
        )
        strings = all(isinstance(v, str) and v for v in labels)
        minimum = 2 if contract.task == "binary" else 3
        if not (numeric or strings) or len(labels) < minimum:
            raise ValueError("classes must be homogeneous finite labels")
        if tuple(sorted(set(labels))) != labels:
            raise ValueError("classes must be unique and sorted in probability-column order")
        if contract.task == "binary":
            if len(labels) != 2 or contract.positive not in labels:
                raise ValueError("binary requires two classes and an explicit positive label")
        elif contract.positive is not None:
            raise ValueError("multiclass has no binary positive label")
    return resolve_metric(
        contract.metric,
        classes=np.asarray(labels) if labels else None,
        positive=contract.positive,
    )


def _scorer_identity(metric: Metric) -> dict[str, object]:
    from importlib.metadata import version

    from honestml.adapters.significance import BootstrapSignificanceTest

    return {
        "metric_id": metric.name,
        "metric_semantics": "average_precision" if metric.name == "pr_auc" else metric.name,
        "greater_is_better": metric.greater_is_better,
        "versions": {
            name: version(name) for name in ("honestml", "numpy", "scikit-learn", "scipy")
        },
        "scorer_source_sha256": hashlib.sha256(
            Path(inspect.getfile(type(metric))).read_bytes()
        ).hexdigest(),
        "bootstrap_source_sha256": hashlib.sha256(
            Path(inspect.getfile(BootstrapSignificanceTest)).read_bytes()
        ).hexdigest(),
    }


def _quality_weights_blocks(
    sample_weight: np.ndarray | None,
    block_index: np.ndarray | None,
    rows: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    weights = None if sample_weight is None else _real_float_array(sample_weight, "weights")
    blocks = None if block_index is None else np.asarray(block_index)
    if weights is not None and (
        weights.shape != (rows,)
        or not np.isfinite(weights).all()
        or np.any(weights < 0)
        or not np.isfinite(weights.sum())
        or weights.sum() <= 0
    ):
        raise ValueError("weights must be finite nonnegative row weights with positive mass")
    if blocks is not None:
        if (
            blocks.shape != (rows,)
            or blocks.dtype.kind not in "iu"
            or np.any(blocks < 0)
            or np.any(blocks > np.iinfo(np.int64).max)
            or len(np.unique(blocks)) < 2
        ):
            raise ValueError("blocks must be nonnegative integer row IDs with at least two blocks")
        blocks = blocks.astype(np.int64, copy=True)
    return weights, blocks


def _array_digest(array: np.ndarray | None) -> str | None:
    if array is None:
        return None
    return _json_digest({"shape": list(array.shape), "values": array.tolist()})


def _real_float_array(values: np.ndarray, name: str) -> np.ndarray:
    raw = np.asarray(values)
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} cannot contain complex values")
    if raw.dtype.kind not in "biuf":
        raise ValueError(f"{name} must contain real numeric values")
    return np.array(raw, dtype=np.float64, copy=True)


def _effective_mask(weights: np.ndarray | None, blocks: np.ndarray | None, rows: int) -> np.ndarray:
    keep = np.ones(rows, dtype=bool) if weights is None else weights > 0
    if np.count_nonzero(keep) < 2:
        raise ValueError("insufficient_effective_rows: need at least two positive-weight rows")
    if blocks is not None and len(np.unique(blocks[keep])) < 2:
        raise ValueError("insufficient_effective_blocks: need at least two positive-weight blocks")
    return keep


def _quality_batch(
    contract: QualityContract,
    role: str,
    batch: PredictionBatch,
    ids: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    if batch.artifact_sha256 != getattr(contract, f"{role}_artifact_sha256"):
        raise ValueError("prediction artifact differs from frozen contract")
    if _row_ids(batch.row_ids) != ids:
        raise ValueError("prediction row alignment differs; realignment is not automatic")
    if tuple(batch.classes) != tuple(contract.classes) or batch.positive != contract.positive:
        raise ValueError("prediction classes/positive differ from frozen contract")
    rows = len(ids)
    weights, blocks = _quality_weights_blocks(batch.sample_weight, batch.block_index, rows)
    values = _real_float_array(batch.values, "predictions")
    shape = (rows, len(contract.classes)) if contract.task == "multiclass" else (rows,)
    if values.shape != shape or not np.isfinite(values).all():
        raise ValueError("prediction shape/finite values invalid")
    if contract.task != "regression":
        if np.any(values < 0) or np.any(values > 1):
            raise ValueError("probabilities must lie in [0, 1]")
        if contract.task == "multiclass" and not np.allclose(
            values.sum(axis=1),
            1.0,
            rtol=0,
            atol=1e-8,
        ):
            raise ValueError("multiclass probabilities must sum to one")
    return values, weights, blocks


def freeze_quality_contract(
    path: Path,
    contract: QualityContract,
    *,
    baseline: PredictionBatch,
    candidate: PredictionBatch,
    baseline_training_ids: Sequence[str] | None = None,
    baseline_tuning_ids: Sequence[str] | None = None,
    candidate_training_ids: Sequence[str] | None = None,
    candidate_tuning_ids: Sequence[str] | None = None,
    provenance: Mapping[str, str] | None = None,
) -> str:
    """Create an exclusive receipt without labels; chronology and lineage still require review.

    All IDs must belong to the same stable namespace across external/train/tune partitions.
    None means unknown lineage, whereas an empty sequence declares no rows used in that role.
    The receipt binds predictions/weights/blocks without reading labels. The expected NPZ label
    digest comes from a custodian; this function cannot prove absence of prior human access.
    Training/tuning inventories must include all FS, ES, calibration and model-selection use.
    """
    from datetime import datetime, timezone

    metric = _quality_scorer(contract)
    row_ids = _row_ids(baseline.row_ids)
    if len(row_ids) < 2:
        raise ValueError("paired evaluation needs at least two rows")
    base_values, weights, blocks = _quality_batch(contract, "baseline", baseline, row_ids)
    cand_values, cand_weights, cand_blocks = _quality_batch(
        contract, "candidate", candidate, row_ids
    )
    if _array_digest(weights) != _array_digest(cand_weights):
        raise ValueError("baseline/candidate weights differ")
    if _array_digest(blocks) != _array_digest(cand_blocks):
        raise ValueError("baseline/candidate blocks differ")
    _effective_mask(weights, blocks, len(row_ids))
    external = set(row_ids)
    lineage: dict[str, object] = {}
    for name, values in (
        ("baseline_training", baseline_training_ids),
        ("baseline_tuning", baseline_tuning_ids),
        ("candidate_training", candidate_training_ids),
        ("candidate_tuning", candidate_tuning_ids),
    ):
        ids = None if values is None else _row_ids(values, allow_empty=True)
        if ids is not None and external.intersection(ids):
            raise ValueError(f"external rows overlap with {name}")
        lineage[name] = {
            "status": "unknown" if ids is None else "declared_disjoint",
            "row_count": None if ids is None else len(ids),
            "row_ids_sha256": None if ids is None else _json_digest(sorted(ids)),
            "disjoint": None if ids is None else True,
        }
    evidence = dict(provenance or {})
    if any(
        not isinstance(k, str) or not isinstance(v, str) or not k or not v
        for k, v in evidence.items()
    ):
        raise ValueError("provenance must contain nonempty string references")
    receipt = {
        "schema_version": 2,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "freeze_scope": "contract_created_without_labels",
        "contract": asdict(contract),
        "scorer": _scorer_identity(metric),
        "external_row_count": len(row_ids),
        "external_row_ids_sha256": _json_digest(row_ids),
        "weights_sha256": _array_digest(weights),
        "blocks_sha256": _array_digest(blocks),
        "prediction_sha256": {
            "baseline": _array_digest(base_values),
            "candidate": _array_digest(cand_values),
        },
        "lineage": lineage,
        "provenance": evidence,
        "independence_review": "pending_external_review_of_access_history_and_lineage_completeness",
    }
    content = (json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    with path.open("xb") as stream:
        stream.write(content)
    return hashlib.sha256(content).hexdigest()


def evaluate_paired_quality(
    path: Path,
    *,
    frozen_sha256: str,
    baseline: PredictionBatch,
    candidate: PredictionBatch,
    labels_path: Path,
) -> dict[str, Any]:
    """Verify frozen predictions, then load only the custodian-hashed NPZ label partition.

    The NPZ contains y_true and row_ids without pickle. Zero-weight rows are removed from
    scoring/bootstrap after validating full payloads. API order does not prove prior human
    access, completeness of lineage, or the origin of predictions from the declared artifacts.
    """
    from honestml.adapters.significance import BootstrapSignificanceTest

    _check_sha256(frozen_sha256)
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != frozen_sha256:
        raise ValueError("frozen receipt digest mismatch")
    receipt = json.loads(content)
    if receipt["schema_version"] != 2:
        raise ValueError("unsupported quality receipt version")
    config = dict(receipt["contract"])
    config["classes"] = tuple(config["classes"])
    contract = QualityContract(**config)
    metric = _quality_scorer(contract)
    if _scorer_identity(metric) != receipt["scorer"]:
        raise ValueError("scorer/library differs from frozen receipt")
    ids = _row_ids(baseline.row_ids)
    if (
        len(ids) != receipt["external_row_count"]
        or _json_digest(ids) != receipt["external_row_ids_sha256"]
    ):
        raise ValueError("prediction row alignment differs from frozen receipt")
    predictions: list[np.ndarray] = []
    weights = None
    blocks = None
    for role, batch in (("baseline", baseline), ("candidate", candidate)):
        values, weights, blocks = _quality_batch(contract, role, batch, ids)
        if _array_digest(weights) != receipt["weights_sha256"]:
            raise ValueError("prediction weights differ from frozen receipt")
        if _array_digest(blocks) != receipt["blocks_sha256"]:
            raise ValueError("prediction blocks differ from frozen receipt")
        if _array_digest(values) != receipt["prediction_sha256"][role]:
            raise ValueError("prediction payload differs from frozen receipt")
        predictions.append(values)
    label_bytes = labels_path.read_bytes()
    if hashlib.sha256(label_bytes).hexdigest() != contract.external_partition_sha256:
        raise ValueError("labels file digest differs from external partition contract")
    with np.load(BytesIO(label_bytes), allow_pickle=False) as labels:
        if set(labels.files) != {"y_true", "row_ids"}:
            raise ValueError("labels NPZ must contain exactly y_true and row_ids")
        target = labels["y_true"].copy()
        label_ids = labels["row_ids"].copy()
    if label_ids.shape != (len(ids),) or _row_ids(label_ids.tolist()) != ids:
        raise ValueError("label row alignment differs from frozen receipt")
    rows = len(ids)
    if target.shape != (rows,):
        raise ValueError("target must be a row-aligned vector")
    if np.iscomplexobj(target):
        raise ValueError("target cannot contain complex values")
    if contract.task == "regression":
        target = _real_float_array(target, "target")
        if not np.isfinite(target).all():
            raise ValueError("target must be finite")
    elif not np.array_equal(np.unique(target), np.asarray(contract.classes)):
        raise ValueError("target must contain every declared class and no unknown classes")
    target_digest = _array_digest(target)
    keep = _effective_mask(weights, blocks, rows)
    effective_target = target[keep]
    if contract.task != "regression" and not np.array_equal(
        np.unique(effective_target),
        np.asarray(contract.classes),
    ):
        raise ValueError("weights must give positive mass to every class")
    target = effective_target
    predictions = [values[keep] for values in predictions]
    weights = None if weights is None else weights[keep]
    blocks = None if blocks is None else blocks[keep]
    reference_score = float(metric.score(target, predictions[0], weights))
    candidate_score = float(metric.score(target, predictions[1], weights))
    if not math.isfinite(reference_score) or not math.isfinite(candidate_score):
        raise ValueError("quality scores must be finite")
    test = BootstrapSignificanceTest(metric, seed=contract.seed, n_boot=contract.n_bootstrap)
    noninferior = test.noninferior(
        predictions[1],
        predictions[0],
        target,
        alpha=contract.alpha,
        margin=contract.margin,
        sample_weight=weights,
        block_index=blocks,
    )
    missing = [name for name, record in receipt["lineage"].items() if record["status"] == "unknown"]
    reasons = [
        "prior_label_access_not_proven",
        "lineage_completeness_requires_external_review",
        "prediction_artifact_origin_requires_external_review",
    ]
    if missing:
        reasons.append("missing_used_row_inventories")
    return {
        "schema_version": 2,
        "scope": "fixed_prediction_pair_on_declared_external_partition",
        "frozen_receipt_sha256": frozen_sha256,
        "contract": receipt["contract"],
        "scorer": receipt["scorer"],
        "metric_semantics": receipt["scorer"]["metric_semantics"],
        "row_alignment_verified": True,
        "external_row_count": rows,
        "effective_rows": int(np.count_nonzero(keep)),
        "excluded_zero_weight_rows": int(np.count_nonzero(~keep)),
        "effective_blocks": None if blocks is None else len(np.unique(blocks)),
        "evaluation_population": "positive_weight_rows_only",
        "external_row_ids_sha256": receipt["external_row_ids_sha256"],
        "labels_file_sha256": contract.external_partition_sha256,
        "weights_sha256": receipt["weights_sha256"],
        "blocks_sha256": receipt["blocks_sha256"],
        "prediction_sha256": receipt["prediction_sha256"],
        "target_sha256": target_digest,
        "baseline_score": reference_score,
        "candidate_score": candidate_score,
        "oriented_improvement": (candidate_score - reference_score)
        * (1 if metric.greater_is_better else -1),
        "statistical_verdict": "passed" if noninferior else "failed",
        "statistical_scope": "paired_bootstrap_noninferiority; not joint quality/time acceptance",
        "lineage": receipt["lineage"],
        "provenance": receipt["provenance"],
        "independence_review": {
            "status": "pending",
            "reasons": reasons,
            "missing_inventories": missing,
        },
        "quality_acceptance": "pending_review",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metric", required=True)
    parser.add_argument("--lower-is-better", action="store_true")
    parser.add_argument("--quality-tolerance", type=float)
    parser.add_argument("--max-time-ratio", type=float)
    parser.add_argument("--search-comparison", action="store_true")
    parser.add_argument("--work-verified", action="store_true")
    args = parser.parse_args()
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    comparison = compare_measurements(
        reference,
        candidate,
        metric=args.metric,
        higher_is_better=not args.lower_is_better,
        quality_tolerance=args.quality_tolerance,
        max_time_ratio=args.max_time_ratio,
        require_equivalent_work=not args.search_comparison,
        executed_work_verified=args.work_verified,
    )
    args.output.write_text(json.dumps(comparison, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
