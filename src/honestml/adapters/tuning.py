"""The Optuna ``Tuner`` adapter (ADR-0061 §3).

Translates a backend-neutral ``SearchSpace`` (``ParamSpec``) to ``optuna.Trial.suggest_*`` and runs
a single-thread TPE search. ``optuna`` is imported **lazily inside** :meth:`OptunaTuner.tune` (the
heavy ``hpo`` extra, ADR-0061 §3): ``import honestml`` never pulls it; composition gates availability
via ``find_spec`` (ADR-0062). Determinism: ``TPESampler(seed)`` + ``n_jobs=1`` give identical
``best_params`` across runs for a fixed seed when ``timeout_s`` is ``None`` (SPIKE-M7-hpo Q1, NFR-M7-2).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import uuid
from collections.abc import Callable, Mapping
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from honestml.core import TuneOutcome, get_logger
from honestml.core.exceptions import BudgetExhaustedError
from honestml.core.ports.tuner import CategoricalParam, FloatParam, IntParam, ParamSpec


def _native(value: Any) -> Any:
    """Normalize a suggested value to a python-native scalar (ADR-0061 §2; byte-stable report/hash)."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def _suggest(trial: Any, name: str, spec: ParamSpec) -> Any:
    if isinstance(spec, IntParam):
        return trial.suggest_int(name, spec.low, spec.high, step=spec.step)
    if isinstance(spec, FloatParam):
        return trial.suggest_float(name, spec.low, spec.high, log=spec.log)
    if isinstance(spec, CategoricalParam):
        return trial.suggest_categorical(name, list(spec.choices))
    raise TypeError(f"unsupported ParamSpec {type(spec).__name__}")  # pragma: no cover


class _IncompatibleReplay(Exception):
    """The saved trial prefix does not reproduce under the active sampler."""


class OptunaTuner:
    """Seeded Optuna TPE with optional atomic trial checkpoints in a trusted cache directory.

    A fixed trial budget replays saved observations through the seeded sampler, validating every
    suggestion before reusing its score. This reconstructs sampler state without private attributes.
    Wall-clock budgets include replay overhead and do not promise an identical trial sequence.
    """

    name = "optuna"

    def __init__(self) -> None:
        self._cache_root: Path | None = None
        self._current_trial_number: int | None = None
        self._search_context: tuple[str, tuple[str, ...]] | None = None

    @property
    def current_trial_number(self) -> int | None:
        return self._current_trial_number

    def configure_cache(self, cache_dir: str, fingerprint: str) -> None:
        self._cache_root = Path(cache_dir) / Path(fingerprint).name / "hpo"

    def set_search_context(self, model_id: str, features: tuple[str, ...]) -> None:
        self._search_context = (model_id, features)

    def _checkpoint_path(
        self,
        search_space: Mapping[str, ParamSpec],
        max_trials: int,
        timeout_s: float | None,
        greater_is_better: bool,
        random_state: int,
    ) -> Path | None:
        if self._cache_root is None or self._search_context is None:
            return None
        versions: dict[str, str | None] = {}
        for package in ("optuna", "numpy", "catboost", "lightgbm", "xgboost", "honestml"):
            try:
                versions[package] = version(package)
            except PackageNotFoundError:
                versions[package] = None
        context = {
            "format": 2,
            "iteration_contract": 2,
            "search": self._search_context,
            "space": [(name, spec.model_dump(mode="json")) for name, spec in search_space.items()],
            "max_trials": max_trials,
            "timeout_s": timeout_s,
            "greater_is_better": greater_is_better,
            "random_state": random_state,
            "versions": versions,
        }
        digest = hashlib.sha256(json.dumps(context, sort_keys=True).encode()).hexdigest()
        return self._cache_root / (digest + ".json")

    @staticmethod
    def _load_checkpoint(path: Path | None, max_trials: int) -> list[dict[str, Any]]:
        if path is None or not path.exists():
            return []
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows = payload["trials"]
            if payload["format"] != 2 or not isinstance(rows, list) or len(rows) > max_trials:
                return []
            for row in rows:
                if not isinstance(row, dict) or not isinstance(row["params"], dict):
                    return []
                value = float.fromhex(row["score"])
                if row["status"] not in ("complete", "failed"):
                    return []
                if (row["status"] == "complete") != math.isfinite(value):
                    return []
            return rows
        except (OSError, ValueError, TypeError, KeyError, OverflowError) as exc:
            get_logger("adapters.tuning").warning(
                "HPO checkpoint unreadable (%s); recomputing", exc
            )
            return []

    @staticmethod
    def _save_checkpoint(path: Path, rows: list[dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
        temporary.write_text(
            json.dumps({"format": 2, "trials": rows}, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )
        os.replace(temporary, path)

    def tune(
        self,
        search_space: Mapping[str, ParamSpec],
        score: Callable[[Mapping[str, Any]], float],
        *,
        max_trials: int,
        timeout_s: float | None,
        greater_is_better: bool,
        random_state: int,
    ) -> TuneOutcome:
        import optuna
        from optuna.samplers import TPESampler

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        path = self._checkpoint_path(
            search_space, max_trials, timeout_s, greater_is_better, random_state
        )
        started = perf_counter()
        saved = self._load_checkpoint(path, max_trials)
        reused = 0

        def run(prefix: list[dict[str, Any]]) -> tuple[Any, int, int]:
            rows = list(prefix)
            study = optuna.create_study(
                direction="maximize" if greater_is_better else "minimize",
                sampler=TPESampler(seed=random_state),
            )

            def objective(trial: Any) -> float:
                nonlocal reused
                params = {
                    name: _native(_suggest(trial, name, spec))
                    for name, spec in search_space.items()
                }
                if trial.number < len(prefix):
                    row = prefix[trial.number]
                    if row["params"] != params:
                        raise _IncompatibleReplay(
                            "saved parameters differ from the seeded suggestion"
                        )
                    reused += 1
                    return (
                        float.fromhex(row["score"]) if row["status"] == "complete" else float("nan")
                    )
                self._current_trial_number = int(trial.number)
                try:
                    value = float(score(params))
                finally:
                    self._current_trial_number = None
                status = "complete" if math.isfinite(value) else "failed"
                if status == "failed":
                    value = float("nan")
                rows.append({"params": params, "score": value.hex(), "status": status})
                if path is not None:
                    self._save_checkpoint(path, rows)
                return value

            # replay has no model fits; it rebuilds the seeded sampler before any new objective
            if prefix:
                study.optimize(objective, n_trials=len(prefix), n_jobs=1)
            remaining = max_trials - len(prefix)
            if remaining:
                time_left = (
                    None if timeout_s is None else max(0.0, timeout_s - (perf_counter() - started))
                )
                try:
                    study.optimize(objective, n_trials=remaining, timeout=time_left, n_jobs=1)
                except BudgetExhaustedError:
                    get_logger("adapters.tuning").warning(
                        "HPO budget exhausted; retaining completed trials"
                    )
            return study, len(rows), sum(row["status"] == "failed" for row in rows)

        try:
            study, n_trials_run, failed_trials = run(saved)
        except _IncompatibleReplay:
            get_logger("adapters.tuning").warning("HPO checkpoint prefix incompatible; recomputing")
            reused = 0
            study, n_trials_run, failed_trials = run([])
        try:
            best_params = {k: _native(v) for k, v in study.best_params.items()}
            best_score = float(study.best_value)
        except ValueError:
            best_params, best_score = {}, float("nan")
        return TuneOutcome(
            best_params=best_params,
            n_trials_run=n_trials_run,
            best_score=best_score,
            reused_trials=reused,
            completed=n_trials_run == max_trials and failed_trials < n_trials_run,
            failed_trials=failed_trials,
        )
