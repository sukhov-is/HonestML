"""The Optuna ``Tuner`` adapter (ADR-0061 §3).

Translates a backend-neutral ``SearchSpace`` (``ParamSpec``) to ``optuna.Trial.suggest_*`` and runs
a single-thread TPE search. ``optuna`` is imported **lazily inside** :meth:`OptunaTuner.tune` (the
heavy ``hpo`` extra, ADR-0061 §3): ``import honestml`` never pulls it; composition gates availability
via ``find_spec`` (ADR-0062). Determinism: ``TPESampler(seed)`` + ``n_jobs=1`` give identical
``best_params`` across runs for a fixed seed when ``timeout_s`` is ``None`` (SPIKE-M7-hpo Q1, NFR-M7-2).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from honestml.core import TuneOutcome
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


class OptunaTuner:
    """A :class:`~honestml.core.Tuner` backed by Optuna TPE (single-thread, seeded)."""

    name = "optuna"

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

        def objective(trial: Any) -> float:
            params = {name: _suggest(trial, name, spec) for name, spec in search_space.items()}
            return score(params)

        study = optuna.create_study(
            direction="maximize" if greater_is_better else "minimize",
            sampler=TPESampler(seed=random_state),
        )
        # n_jobs=1 is mandatory for reproducibility: parallel trials finish in a nondeterministic
        # order, breaking the seed->best_params guarantee (SPIKE-M7-hpo Q1, NFR-M7-2).
        study.optimize(objective, n_trials=max_trials, timeout=timeout_s, n_jobs=1)
        try:
            best_params = {k: _native(v) for k, v in study.best_params.items()}
            best_score = float(study.best_value)
        except ValueError:
            # no completed trial (budget cut before any finished): empty params -> baseline factory
            best_params, best_score = {}, float("nan")
        return TuneOutcome(
            best_params=best_params, n_trials_run=len(study.trials), best_score=best_score
        )
