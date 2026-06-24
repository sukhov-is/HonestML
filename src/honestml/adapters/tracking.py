"""MLflow experiment-tracking adapter (ADR-0073).

Library discipline (DM-C4): the adapter NEVER mutates the user's global mlflow
state — no ``set_tracking_uri``/``set_experiment``/fluent ``start_run``; everything
goes through an ``MlflowClient`` bound to an explicit ``run_id``. mlflow itself is
imported only inside ``log_run`` (the module stays import-light); the constructor
gates on ``find_spec`` so a missing extra fails BEFORE the expensive fit (FR-TRK-4).
MLflow client-side limits are respected here rather than surfacing as
``MlflowException`` after training (NFR-TRK-4): values are pre-truncated, keys
sanitized to the MLflow alphabet, batches chunked.
"""

from __future__ import annotations

import math
import posixpath
import re
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import suppress
from importlib.util import find_spec
from typing import TYPE_CHECKING, Any

from honestml.core import ConfigError, MissingDependencyError, get_logger

if TYPE_CHECKING:
    from mlflow.tracking import MlflowClient

logger = get_logger("adapters.tracking")

_MAX_KEY = 250
_MAX_PARAM_VALUE = 6000
_MAX_TAG_VALUE = 8000
_BATCH = 100  # log_batch client-side cap for params and tags; metrics allow 1000
_METRIC_BATCH = 1000
# anything outside the MLflow key alphabet (e.g. a plugin model_id) is replaced by "_"
_BAD_KEY_CHARS = re.compile(r"[^/\w.\- ]")
# basic-auth userinfo in a tracking URI must not leak into the INFO log
_URI_CREDENTIALS = re.compile(r"//[^/@]+@")


def _sanitize_key(key: str) -> str:
    text = _BAD_KEY_CHARS.sub("_", key)[:_MAX_KEY]
    # mlflow additionally rejects path-like keys (leading "/", "a//b", "a/../b" — the normpath
    # rule): losing the "/" semantics beats losing the whole record to a post-fit MlflowException
    if text.startswith("/") or posixpath.normpath(text) != text:
        text = text.replace("/", "_")
    if text in (".", ".."):
        text = text.replace(".", "_")
    return text


def _truncate(value: Any, limit: int, kind: str, key: str) -> str:
    text = str(value)
    if len(text) > limit:
        logger.warning("tracking %s %r truncated to %d chars", kind, key, limit)
        text = text[:limit]
    return text


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    """Dot-join a nested mapping (``cv.n_splits``); leaves are returned as-is. A NESTED empty
    mapping is a leaf (a future additive ``{}``-default config block stays visible in params)."""
    if isinstance(value, Mapping) and (value or not prefix):
        out: dict[str, Any] = {}
        for key, item in value.items():
            out.update(_flatten(item, f"{prefix}.{key}" if prefix else str(key)))
        return out
    return {prefix: value}


def _chunks(seq: Sequence[Any], size: int) -> Iterator[Sequence[Any]]:
    for start in range(0, len(seq), size):
        yield seq[start : start + size]


def _report_metrics(report: Mapping[str, Any]) -> dict[str, float]:
    """The numeric view of the report (ADR-0073 §3); non-finite values are skipped."""
    out: dict[str, float] = {}
    for entry in report.get("leaderboard") or []:
        out[f"score.{entry['model_id']}"] = float(entry["score"])
    winner = report.get("winner")
    if f"score.{winner}" in out:
        out["winner_score"] = out[f"score.{winner}"]
    if report.get("holdout_score") is not None:
        out["holdout_score"] = float(report["holdout_score"])
    for group, stages in (report.get("timings") or {}).items():
        for stage, elapsed in stages.items():
            out[f"time.{group}.{stage}"] = float(elapsed)
    finite = {key: value for key, value in out.items() if math.isfinite(value)}
    for key in out.keys() - finite.keys():
        logger.debug("tracking metric %r skipped (non-finite)", key)
    return finite


class MlflowTracker:
    """``ExperimentTracker`` over the MLflow client API (ADR-0073).

    Constructor arguments mirror :class:`honestml.core.TrackerConfig` minus ``backend``
    (the dispatch key); the defaults are duplicated deliberately so the instance form
    works standalone. ``tracking_uri=None`` defers to mlflow's own resolution
    (env ``MLFLOW_TRACKING_URI`` -> ``file:./mlruns``).
    """

    def __init__(
        self,
        experiment: str = "honestml",
        tracking_uri: str | None = None,
        run_name: str | None = None,
        tags: Mapping[str, str] | None = None,
    ) -> None:
        if find_spec("mlflow") is None:
            raise MissingDependencyError("mlflow")
        # bad user input must fail BEFORE the expensive fit, not lose tracking after (ADR-0073 §1)
        if not isinstance(experiment, str) or not experiment.strip():
            raise ConfigError(f"tracker experiment must be a non-empty str, got {experiment!r}")
        if run_name is not None and not isinstance(run_name, str):
            raise ConfigError(
                f"tracker run_name must be a str or None, got {type(run_name).__name__}"
            )
        self._experiment = experiment
        self._tracking_uri = tracking_uri
        self._run_name = run_name
        self._tags = {self._tag_key(key): str(value) for key, value in (tags or {}).items()}
        if len(self._tags) != len(tags or {}):
            raise ConfigError("tracker tag keys collide after sanitization to the MLflow alphabet")

    @staticmethod
    def _tag_key(key: object) -> str:
        if not isinstance(key, str) or not key or len(key) > _MAX_KEY:
            raise ConfigError(
                f"tracker tag key must be a non-empty str of <={_MAX_KEY} chars, got {key!r}"
            )
        if key.startswith("honestml."):
            raise ConfigError(
                f"tracker tag key {key!r} shadows the reserved provenance namespace 'honestml.'"
            )
        return _sanitize_key(key)

    def log_run(self, report: Mapping[str, Any]) -> None:
        from mlflow.tracking import MlflowClient

        client = MlflowClient(tracking_uri=self._tracking_uri)
        run = client.create_run(self._experiment_id(client), run_name=self._run_name)
        run_id = run.info.run_id
        try:
            self._log_payload(client, run_id, report)
            client.log_dict(run_id, dict(report), "run_report.json")
            client.set_terminated(run_id)
        except BaseException:
            # mark FAILED without masking the original cause (its own failure is suppressed);
            # KeyboardInterrupt is re-raised through the facade (ADR-0072 §2 / ADR-0073 §2)
            with suppress(Exception):
                client.set_terminated(run_id, "FAILED")
            raise
        logger.info(
            "experiment tracking: run %s logged to experiment %r (uri=%s)",
            run_id,
            self._experiment,
            "<default>"
            if self._tracking_uri is None
            else _URI_CREDENTIALS.sub("//***@", self._tracking_uri),
        )

    def _experiment_id(self, client: MlflowClient) -> str:
        """Get-or-create by name with the create-then-get idiom (ADR-0073 §2).

        Two parallel fits can race the create; the loser re-reads instead of losing
        the whole run record to the fail-soft WARNING.
        """
        from mlflow.exceptions import MlflowException

        experiment = client.get_experiment_by_name(self._experiment)
        if experiment is not None:
            return str(experiment.experiment_id)
        try:
            return str(client.create_experiment(self._experiment))
        except MlflowException:
            experiment = client.get_experiment_by_name(self._experiment)
            if experiment is None:
                raise
            return str(experiment.experiment_id)

    def _log_payload(self, client: MlflowClient, run_id: str, report: Mapping[str, Any]) -> None:
        from mlflow.entities import Metric, Param, RunTag

        tags = {
            "honestml.version": str(report.get("honestml_version")),
            "honestml.fingerprint": str(report.get("run_fingerprint")),
            "honestml.winner": str(report.get("winner")),
            "honestml.run_manifest_version": str(report.get("run_manifest_version")),
            **self._tags,
        }
        tag_entities = [
            RunTag(key, _truncate(value, _MAX_TAG_VALUE, "tag", key)) for key, value in tags.items()
        ]
        params = [
            Param(_sanitize_key(key), _truncate(value, _MAX_PARAM_VALUE, "param", key))
            for key, value in _flatten(report.get("config") or {}).items()
        ]
        now_ms = int(time.time() * 1000)
        metrics = [
            Metric(_sanitize_key(key), value, now_ms, 0)
            for key, value in _report_metrics(report).items()
        ]
        for chunk in _chunks(tag_entities, _BATCH):
            client.log_batch(run_id, tags=chunk)
        for chunk in _chunks(params, _BATCH):
            client.log_batch(run_id, params=chunk)
        for chunk in _chunks(metrics, _METRIC_BATCH):
            client.log_batch(run_id, metrics=chunk)
