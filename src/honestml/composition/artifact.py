"""Minimal versioned model artifact + standalone prediction.

The artifact is a directory: ``manifest.json`` (versioned metadata),
``schema.json`` (the serialized ``FeatureSchema`` — the single preprocessing
source of truth, so train==inference), a model body in the serializer's
format (``model_type``/``model_file`` via the registry: joblib by default,
a native boosting file under ``model_format="native"``) and
``leaderboard.json``. :class:`FittedModel` is the one inference path shared
by the facade and the standalone load:
``Reader.read(X, schema)`` → ``design_matrix`` → estimator.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import numpy as np

from honestml.adapters import Reader
from honestml.adapters.serializers import (
    CatBoostSerializer,
    JoblibSerializer,
    LightGbmSerializer,
    XGBoostSerializer,
)
from honestml.application import (
    LeaderboardEntry,
    align_proba,
    design_matrix,
    project_for_metric,
    resolve_positive,
)
from honestml.core import (
    ArtifactIntegrityError,
    Calibrator,
    Dataset,
    Estimator,
    FeatureSchema,
    Metric,
    ModelSerializer,
    ProbabilisticEstimator,
    SchemaValidationError,
    SupportsNativeModel,
    Task,
    get_logger,
)

ARTIFACT_VERSION = 1
_CALIBRATOR_FILE = "calibrator.joblib"
_SIGNATURE_FILE = "signature"
_CHECKSUM_ALGO = "sha256"

_JOBLIB = JoblibSerializer()
# ordered serializer registry: native formats first, the joblib catch-all last (ADR-0069 §2);
# a new model_type is a new adapter plus an entry here -- the save/load cores stay unchanged
_SERIALIZERS: list[ModelSerializer] = [
    XGBoostSerializer(),
    CatBoostSerializer(),
    LightGbmSerializer(),
    _JOBLIB,
]


@dataclass
class FittedModel:
    """A fitted model with its preprocessing schema — the unified inference path.

    ``classes`` is the global class order for classification and ``None`` for regression,
    so the inference path is kind-aware: multiclass proba is aligned to it and
    a regression model has no probabilities.
    """

    estimator: Estimator
    schema: FeatureSchema
    task: Task
    # the scoring metric is held by NAME + averaging mode and resolved lazily (see ``metric`` below), so the
    # inference path (predict/predict_proba) never resolves the AutoML metric — no ``adapters.metrics``/
    # ``resolve_metric`` import (ADR-0066 §2, NFR-SRV-1).
    metric_name: str
    classes: np.ndarray | None
    leaderboard: list[LeaderboardEntry]
    best_model_id: str
    metric_average: str | None = None
    # honesty-band metadata (ADR-0026 §6) — additive manifest keys, NOT in the frozen
    # LeaderboardEntry (so an older build still reads leaderboard.json; NFR-M4-5). Defaults
    # describe a lone-anchor band, so a legacy artifact without these keys loads unchanged.
    band_member_ids: tuple[str, ...] = ()
    band_unstable: bool = False
    band_width: int = 1
    winner_by_tiebreak: bool = False
    # probability calibration (ADR-0030 §1) + refinement-selection observability (ADR-0031 §6):
    # additive, default-off so a legacy artifact loads unchanged. ``calibrator`` recalibrates
    # predict_proba; ``calibration`` is the report (Brier/ECE before/after + reliability).
    calibrator: object | None = None
    calibration: dict[str, Any] | None = None
    selection_mode: str = "raw"
    score_space: str = "raw_oof"
    # honest-regime holdout (ADR-0029 §3): the winner's unbiased score on the once-touched outer
    # holdout (raw metric, comparable to leaderboard_); None when outer_holdout is off. Additive key.
    holdout_score: float | None = None
    # ensemble provenance (ADR-0064 §3): the recipe block (applied/method/member_ids/weights/gate_reason)
    # when a BlendedEstimator was shipped, else None (single model). Additive manifest key, `.get` on load;
    # the inference path stays opaque (the BlendedEstimator IS the estimator), so nothing branches on it.
    ensemble: dict[str, Any] | None = None
    # finalize provenance (ADR-0068 §2): "all" when the shipped model was refit on DEV+holdout, else "dev"
    # (the default — outer_holdout off, or finalize=False). Additive manifest key, `.get` on load; the
    # honest score (holdout_score/leaderboard) stays the DEV-model estimate regardless.
    shipped_on: str = "dev"
    # early stopping (ADR-0080): True when a boosting early-stopped on a carved es tail this run, so the
    # leaderboard comparison is not the old overfit-favoring one. Additive manifest key, default off.
    early_stopping: bool = False
    _metric: Metric | None = field(default=None, init=False, repr=False)

    @property
    def metric(self) -> Metric:
        """The scoring metric, resolved lazily by name.

        ``predict``/``predict_proba`` never touch it, so a standalone load that only predicts never resolves
        the AutoML metric (no ``adapters.metrics``/``resolve_metric`` import; sklearn.metrics may still arrive
        via the estimator's own unpickle, which is outside our control). ``score`` triggers the one-time
        resolution.
        """
        if self._metric is None:
            from honestml.adapters import resolve_metric

            # binary proba metrics score P(positive); orient them so a non-greatest positive_label
            # does not invert a standalone score() (F111)
            positive = (
                resolve_positive(self.task, self.classes)
                if self.task.kind == "binary" and self.classes is not None
                else None
            )
            self._metric = resolve_metric(
                self.metric_name,
                classes=self.classes,
                average=self.metric_average,
                positive=positive,
            )
        return self._metric

    def predict(self, X: object) -> np.ndarray:
        # raw argmax: calibration changes confidence, not the decision (ADR-0030 §1)
        return self.estimator.predict(design_matrix(self._read(X)))

    def predict_proba(self, X: object) -> np.ndarray:
        if not isinstance(self.estimator, ProbabilisticEstimator):
            raise SchemaValidationError("regression model has no probabilities")
        return self._calibrate(self._aligned_proba(design_matrix(self._read(X))))

    def score(self, X: object, y: object, sample_weight: object | None = None) -> float:
        # sklearn convention: higher is better, so flip a lower-is-better metric
        s = self._score_dataset(self._read(X, y, sample_weight))
        return s if self.metric.greater_is_better else -s

    def _score_dataset(self, ds: Dataset) -> float:
        """Raw metric value (the metric's own orientation) on an already-built dataset — scoring core.

        Shared by the public ``score`` (which flips to sklearn convention) and the honest-regime
        outer-holdout scoring, which scores the dev-trained winner once on the untouched
        holdout ``Dataset`` and keeps the raw value so it is directly comparable to ``leaderboard_``.
        """
        target = ds.target()
        if target is None:
            raise SchemaValidationError("score requires a target column")
        x_mat = design_matrix(ds)
        needs_proba = self.metric.needs in ("proba", "threshold")
        if needs_proba and isinstance(self.estimator, ProbabilisticEstimator):
            cal = self._calibrate(self._aligned_proba(x_mat))
            proba = cal if self.task.kind == "multiclass" else cal[:, self._positive_index()]
            pred = np.empty(0)
        else:
            proba = None
            pred = self.estimator.predict(x_mat)
        proj = project_for_metric(self.metric, proba=proba, pred=pred, kind=self.task.kind)
        return self.metric.score(target, proj, ds.sample_weight())

    def _aligned_proba(self, x_mat: np.ndarray) -> np.ndarray:
        """Raw proba aligned to the global class order (binary 2-col; multiclass (n, K))."""
        raw = cast(ProbabilisticEstimator, self.estimator).predict_proba(x_mat)
        if self.task.kind == "multiclass" and self.classes is not None:
            return align_proba(raw, self.estimator.classes_, self.classes)  # type: ignore[attr-defined]
        return raw

    def _calibrate(self, proba: np.ndarray) -> np.ndarray:
        """Apply the calibrator if attached: multiclass per-class, binary on P(pos)."""
        if self.calibrator is None:
            return proba
        transform = cast(Calibrator, self.calibrator).transform
        if self.task.kind == "multiclass":
            return transform(proba)
        pos = self._positive_index()
        p_pos = transform(proba[:, pos])
        out = np.empty_like(proba, dtype=np.float64)
        out[:, pos] = p_pos
        out[:, 1 - pos] = 1.0 - p_pos
        return out

    def _read(
        self, X: object, y: object | None = None, sample_weight: object | None = None
    ) -> Dataset:
        return Reader(self.task).read(X, y, schema=self.schema, sample_weight=sample_weight)

    def _positive_index(self) -> int:
        if self.classes is None:
            raise SchemaValidationError(
                "a positive class is undefined without classes (regression?)"
            )
        positive = resolve_positive(self.task, self.classes)
        return int(np.where(self.estimator.classes_ == positive)[0][0])  # type: ignore[attr-defined]


def save_artifact(
    model: FittedModel,
    directory: str | Path,
    *,
    honestml_version: str | None = None,
    sign: Callable[[str], str] | None = None,
    model_format: str = "joblib",
) -> None:
    """Serialize *model* to a versioned artifact directory.

    Writes the data files first, then a ``checksums`` block (sha256 of every file plus a digest of
    the manifest payload) so ``load_artifact`` can verify integrity before deserializing the model
    body. ``sign`` is an optional hook: it receives the manifest digest (hex) and returns a
    signature string written to ``signature`` for an authenticated ``verify=`` on load.

    ``model_format`` picks the body serializer: ``"joblib"`` (the default) or ``"native"`` —
    a boosting body goes through the library's stable format (xgb ubj / cat cbm / lgbm text)
    instead of pickle; anything without a native format (sklearn models, a shipped ensemble)
    transparently stays joblib.
    """
    import joblib

    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    serializer = _serializer_for_format(model_format, model)
    body = serializer.save(model.estimator, path)
    manifest = {
        "artifact_version": ARTIFACT_VERSION,
        "honestml_version": honestml_version or _installed_version(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task": model.task.model_dump(mode="json"),
        "metric": model.metric_name,
        # additive keys (ADR-0024 §4): classes for classification (null for regression),
        # the metric averaging mode, and the honest "no early stopping in M3" flag (ADR-0020 §2).
        "classes": model.classes.tolist() if model.classes is not None else None,
        "metric_average": model.metric_average,
        "early_stopping": model.early_stopping,
        # a shipped ensemble records the synthetic id "ensemble" (the real member ids stay in the
        # ensemble block + leaderboard); a single model keeps its real winner id (ADR-0064 §3).
        "best_model_id": (
            "ensemble"
            if (model.ensemble and model.ensemble.get("applied"))
            else model.best_model_id
        ),
        # honesty band (ADR-0026 §6) — additive, forward/backward symmetric via `.get` on load
        "band_member_ids": list(model.band_member_ids),
        "band_unstable": model.band_unstable,
        "band_width": model.band_width,
        "winner_by_tiebreak": model.winner_by_tiebreak,
        # calibration + refinement (ADR-0030 §4 / ADR-0031 §6) — additive plain keys, `.get` on load
        "calibration": model.calibration,
        "selection_mode": model.selection_mode,
        "score_space": model.score_space,
        # honest-regime holdout (ADR-0029 §3) — additive plain key, `.get` on load
        "holdout_score": model.holdout_score,
        # ensemble provenance (ADR-0064 §3) — additive, null for a single model; `.get` on load
        "ensemble": model.ensemble,
        # finalize provenance (ADR-0068 §2) — additive; "dev" for a legacy/no-holdout artifact
        "shipped_on": model.shipped_on,
        "calibrator_file": _CALIBRATOR_FILE if model.calibrator is not None else None,
        "model_type": serializer.model_type,
        "model_file": body.files[0],
    }
    if body.required_extra is not None:
        # additive runtime self-description (ADR-0070 §6) -- absent on the joblib default
        manifest["required_extra"] = body.required_extra
    # native categorical routing (ADR-0091): persist the indices so a native (.cbm) reload can int-cast on
    # predict (joblib also keeps them inside the pickled wrapper); native_categorical records the routing
    # in the artifact (NFR-6). Additive keys, NO artifact_version bump — absent ⇒ codes path on load (NFR-7).
    cat_idx = getattr(model.estimator, "categorical_indices", None)
    if cat_idx and isinstance(model.estimator, SupportsNativeModel):
        manifest["categorical_indices"] = list(cat_idx)
        manifest["native_categorical"] = {
            "backend": model.estimator.native_format,
            "n_cat": len(cat_idx),
        }
    # data files first, then checksum them; the manifest (carrying the checksums) is written last
    (path / "schema.json").write_text(model.schema.model_dump_json(indent=2), encoding="utf-8")
    (path / "leaderboard.json").write_text(
        json.dumps([e.model_dump() for e in model.leaderboard], indent=2), encoding="utf-8"
    )
    files = ["schema.json", "leaderboard.json", *body.files]
    if model.calibrator is not None:
        joblib.dump(model.calibrator, path / _CALIBRATOR_FILE)
        files.append(_CALIBRATOR_FILE)
    # integrity block (ADR-0067 §1): sha256 of every present file + a digest of the manifest payload.
    # `manifest` carries no `checksums` yet, so `_manifest_digest` roots on exactly the payload (`.files`
    # reflects only the files actually written — no calibrator entry when calibrator is None).
    checksums: dict[str, Any] = {
        "algo": _CHECKSUM_ALGO,
        "files": {name: _sha256_file(path / name) for name in files},
    }
    manifest["checksums"] = checksums
    # the digest covers the payload + checksums.files (so a signature over it authenticates every file,
    # ADR-0067 §1/§5); it is stored under "manifest", which `_manifest_digest` itself excludes.
    digest = _manifest_digest(manifest)
    checksums["manifest"] = digest
    (path / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if sign is not None:
        (path / _SIGNATURE_FILE).write_text(sign(digest), encoding="utf-8")


def load_artifact(
    directory: str | Path,
    *,
    require_integrity: bool = False,
    verify: Callable[[str | None, str], bool] | None = None,
) -> FittedModel:
    """Load an artifact directory into a :class:`FittedModel`.

    Order: read manifest -> version-gate -> verify integrity -> ``model_type`` dispatch +
    deserialize. ``require_integrity`` makes a missing ``checksums`` block an error (older
    artifacts warn by default); ``verify`` is an optional signature hook
    ``(signature, manifest_digest) -> bool``.

    SECURITY: a ``joblib`` body and ``calibrator.joblib`` are deserialized via joblib/pickle (a native
    boosting body is a structural file instead). The sha256 integrity check detects corruption
    and naive substitution, NOT authenticity — a malicious author can embed code with a matching
    digest; use ``verify`` (a signature) and load only from a trusted source. The version-gate is
    compatibility-only, not a trust check.
    """
    import joblib

    path = Path(directory)
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    found = manifest.get("artifact_version")
    if found != ARTIFACT_VERSION:
        raise SchemaValidationError(
            f"unsupported artifact_version {found!r} (this build reads {ARTIFACT_VERSION})"
        )
    _verify_integrity(path, manifest, require_integrity=require_integrity, verify=verify)
    task_raw = _require(manifest, "task")
    metric_name = _require(manifest, "metric")
    best_model_id = _require(manifest, "best_model_id")
    # model_type dispatch through the serializer registry (ADR-0065 §3 / ADR-0069 §2); the
    # serializer confines its file to the artifact directory and raises MissingDependencyError
    # when the native runtime is absent (ADR-0070 §6)
    serializer = _serializer_for_type(manifest.get("model_type", "joblib"))

    schema = FeatureSchema.model_validate_json((path / "schema.json").read_text(encoding="utf-8"))
    leaderboard = [
        LeaderboardEntry.model_validate(e)
        for e in json.loads((path / "leaderboard.json").read_text(encoding="utf-8"))
    ]
    estimator = serializer.load(path, manifest)
    calibrator_file = manifest.get("calibrator_file")
    # .name is the ADR-0067 §2 anti-traversal guard (coincides with the already-verified basename)
    calibrator = joblib.load(path / Path(calibrator_file).name) if calibrator_file else None
    # classes from the manifest if present; else fall back to a classifier's classes_
    # (legacy binary artifact); a regression model has neither -> None (ADR-0024 §4).
    classes_raw = manifest.get("classes")
    estimator_classes = getattr(estimator, "classes_", None)
    if classes_raw is not None:
        classes = np.asarray(classes_raw)
    elif estimator_classes is not None:
        classes = np.asarray(estimator_classes)
    else:
        classes = None
    return FittedModel(
        estimator=estimator,
        schema=schema,
        task=Task.model_validate(task_raw),
        # the metric is resolved lazily on first `.score()` (ADR-0066 §2) — not here, so a standalone
        # load that only predicts never imports the metric machinery
        metric_name=metric_name,
        metric_average=manifest.get("metric_average"),
        classes=classes,
        leaderboard=leaderboard,
        best_model_id=best_model_id,
        # additive band keys; absent in pre-M4a2 manifests -> lone-anchor defaults (NFR-M4-5)
        band_member_ids=tuple(manifest.get("band_member_ids", ())),
        band_unstable=bool(manifest.get("band_unstable", False)),
        band_width=int(manifest.get("band_width", 1)),
        winner_by_tiebreak=bool(manifest.get("winner_by_tiebreak", False)),
        # additive calibration/refinement keys; absent in pre-M4d manifests -> off (NFR-M4-5)
        calibrator=calibrator,
        calibration=manifest.get("calibration"),
        selection_mode=manifest.get("selection_mode", "raw"),
        score_space=manifest.get("score_space", "raw_oof"),
        # additive holdout key; absent in pre-M4c manifests -> off (NFR-M4-5)
        holdout_score=manifest.get("holdout_score"),
        # additive ensemble block; absent in pre-M7b manifests -> single model (NFR-M7-4)
        ensemble=manifest.get("ensemble"),
        # additive finalize provenance; absent in pre-M8-3 manifests -> "dev" (ADR-0068 §2)
        shipped_on=manifest.get("shipped_on", "dev"),
    )


def _require(manifest: dict[str, Any], key: str) -> Any:
    """Read a required manifest key, surfacing a clear boundary error if absent."""
    if key not in manifest:
        raise SchemaValidationError(f"artifact manifest missing required key {key!r}")
    return manifest[key]


def _serializer_for_type(model_type: str) -> ModelSerializer:
    """Dispatch a manifest ``model_type`` to its registered serializer.

    An unknown type is an explicit error, never a silent joblib fallback.
    """
    for serializer in _SERIALIZERS:
        if serializer.model_type == model_type:
            return serializer
    known = sorted(s.model_type for s in _SERIALIZERS)
    raise SchemaValidationError(f"unsupported model_type {model_type!r} (this build loads {known})")


def _serializer_for_format(model_format: str, model: FittedModel) -> ModelSerializer:
    """Pick the body serializer for ``save_artifact``.

    ``"native"`` takes the first serializer matching the estimator; the joblib catch-all is
    last, so anything without a native format falls through to it. A shipped ensemble has no
    native path yet -- that fallback is disclosed, not silent.
    """
    if model_format == "joblib":
        return _JOBLIB
    if model_format != "native":
        raise SchemaValidationError(
            f"unsupported model_format {model_format!r} (this build writes 'joblib' or 'native')"
        )
    chosen = next(s for s in _SERIALIZERS if s.can_serialize(model.estimator))
    if chosen is _JOBLIB and model.ensemble and model.ensemble.get("applied"):
        get_logger().warning(
            "model_format='native': a shipped ensemble has no native format; "
            "the blend ships as a joblib (pickle) body (per-member native is Day-2)"
        )
    return chosen


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest_digest(manifest: dict[str, Any]) -> str:
    """sha256 of the manifest payload, covering the per-file checksums but **excluding** the manifest
    digest itself (the integrity/authenticity root).

    Including ``checksums.files`` here makes the optional signature over this digest *transitively*
    authenticate every file: a substituted file whose ``checksums.files`` entry was recomputed shifts
    this digest, so the original signature no longer matches. Only ``checksums.manifest``
    (the digest itself) is excluded. Canonicalized (sorted keys, compact) so it is reproducible across the
    round-trip regardless of the manifest file's formatting.
    """
    payload = {k: v for k, v in manifest.items() if k != "checksums"}
    checks = manifest.get("checksums")
    if isinstance(checks, dict):
        payload["checksums"] = {k: v for k, v in checks.items() if k != "manifest"}
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _verify_integrity(
    path: Path,
    manifest: dict[str, Any],
    *,
    require_integrity: bool,
    verify: Callable[[str | None, str], bool] | None,
) -> None:
    """Verify checksums before any deserialization. Raises :class:`ArtifactIntegrityError`.

    An older artifact without a ``checksums`` block warns by default and proceeds (``require_integrity``
    makes it an error). File names are confined to the artifact directory by basename (anti-traversal on
    read). The optional ``verify`` hook authenticates the manifest digest against a detached ``signature``.
    """
    checksums = manifest.get("checksums")
    if checksums is None:
        if require_integrity:
            raise ArtifactIntegrityError("missing_checksums")
        get_logger().warning(
            "artifact %s has no checksums block; integrity not verified "
            "(load only from a trusted source)",
            path.name,
        )
        return
    expected = checksums.get("manifest")
    if _manifest_digest(manifest) != expected:
        raise ArtifactIntegrityError("digest_mismatch", file="manifest.json")
    for name, digest in checksums.get("files", {}).items():
        # anti-traversal: basenames only — reject empty/'.'/'..'/separators/drive-or-ADS colon (ADR-0067 §2)
        if (
            not name
            or name in (".", "..")
            or "/" in name
            or "\\" in name
            or ":" in name
            or Path(name).name != name
        ):
            raise ArtifactIntegrityError("missing_file", file=name)
        target = path / name
        if not target.is_file():
            raise ArtifactIntegrityError("missing_file", file=name)
        if _sha256_file(target) != digest:
            raise ArtifactIntegrityError("digest_mismatch", file=name)
    if verify is not None:
        sig_path = path / _SIGNATURE_FILE
        signature = sig_path.read_text(encoding="utf-8") if sig_path.is_file() else None
        if not verify(signature, expected):
            raise ArtifactIntegrityError("signature_mismatch", file=_SIGNATURE_FILE)


def _installed_version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("honestml")
    except PackageNotFoundError:
        return "0+unknown"
