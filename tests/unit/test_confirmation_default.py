"""A larger confirmation default changes evidence without bypassing fit budgets."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np
import pytest

from honestml.adapters.run_budget import RunBudget
from honestml.application.search import probe_models
from honestml.application.slice import refit_best, run_slice
from honestml.core import (
    BudgetConfig,
    Candidate,
    Fold,
    RunConfig,
    RunContext,
    SelectionPolicy,
    Task,
)
from honestml.core.config import SearchConfig
from honestml.core.ports.splitter import validate_fold

from .test_fast_search_integration import _Dataset, _Estimator, _Fit, _Metric, _Splitter
from .test_run_fingerprint import _fp
from .test_search_confirmation import Metric

pytestmark = pytest.mark.unit


def test_confirmation_default_resolves_and_invalidates_smaller_resource_fingerprint() -> None:
    default = RunConfig(search=SearchConfig())
    explicit = RunConfig(search=SearchConfig(confirmation_rows=65536))
    smaller = RunConfig(search=SearchConfig(confirmation_rows=16384))
    restored = RunConfig.model_validate_json(default.model_dump_json())
    assert _fp(run_config=default) == _fp(run_config=explicit)
    assert _fp(run_config=default) != _fp(run_config=smaller)
    assert _fp(run_config=restored) == _fp(run_config=explicit)


@pytest.mark.parametrize("rows", [100, 100000])
def test_confirmation_total_cap_and_partition_saturation(rows: int) -> None:
    fit_end, es_end = rows * 3 // 5, rows * 4 // 5
    original = Fold(np.arange(fit_end), np.arange(fit_end, es_end), np.arange(es_end, rows))
    originals = [original, original]
    y = np.arange(rows, dtype=np.float64)
    groups, times = np.arange(rows) // 10, np.arange(rows)
    observations: list[tuple[str, int, tuple[Fold, ...]]] = []

    def evaluate(name: str, parts: Sequence[Fold], iterations: int) -> Candidate:
        observations.append((name, iterations, tuple(parts)))
        return Candidate(name, 0.9 if name == "a" else 0.8, train_time=1.0)

    result = probe_models(
        ["a", "b", "discarded"],
        originals,
        y=y,
        groups=groups,
        times=times,
        task=Task(kind="regression"),
        metric=Metric(),
        config=SearchConfig(),
        seed=42,
        evaluate=evaluate,
    )
    confirmations = [item for item in observations if item[1] == 256]
    assert [name for name, _, _ in confirmations] == ["a", "b"]
    assert result.reason == "confirmation_score"
    for _, _, parts in confirmations:
        assert len(parts) == 2
        total = sum(
            len(part) for fold in parts for part in (fold.fit_idx, fold.es_idx, fold.test_idx)
        )
        assert total == min(65536, rows * 2)
        for fold in parts:
            validate_fold(fold, groups=groups, times=times, time_ordered=True)
            for actual, source in zip(
                (fold.fit_idx, fold.es_idx, fold.test_idx),
                (original.fit_idx, original.es_idx, original.test_idx),
                strict=True,
            ):
                assert np.isin(actual, source).all()
                if rows == 100:
                    np.testing.assert_array_equal(actual, source)
    for _, iterations, parts in observations:
        if iterations == 64:
            assert sum(
                len(part) for fold in parts for part in (fold.fit_idx, fold.es_idx, fold.test_idx)
            ) == min(4096, rows)


@pytest.mark.parametrize("stop", ["phase_deadline", "fit_limit"])
def test_confirmation_checkpoint_preserves_completed_control_and_finish_reserve(
    monkeypatch: pytest.MonkeyPatch, stop: str
) -> None:
    import honestml.application.slice as slice_module

    clock = [0.0]
    monkeypatch.setattr(slice_module.time, "perf_counter", lambda: clock[0])
    search = SearchConfig(max_probe_fits=3 if stop == "fit_limit" else 64)
    budget = RunBudget(BudgetConfig(mode="time", time_budget_s=100), clock=lambda: clock[0])
    budget.start()
    budget.reserve(100 * search.reserve_fraction)
    assert budget.time_left() == 70
    fits: list[_Fit] = []

    class TimedEstimator(_Estimator):
        def fit(
            self,
            X: np.ndarray,
            y: np.ndarray,
            X_val: np.ndarray | None = None,
            y_val: np.ndarray | None = None,
            sample_weight: np.ndarray | None = None,
        ) -> TimedEstimator:
            super().fit(X, y, X_val, y_val, sample_weight)
            if self.iteration_budget == 64:
                clock[0] += 1
            elif self.iteration_budget == 256:
                clock[0] += 2 if stop == "fit_limit" else 13
            else:
                clock[0] += 20 if len(X) == 120 else 27
            return self

    class TwoFolds(_Splitter):
        def split(self, dataset: _Dataset) -> Iterator[Fold]:
            yield from super().split(dataset)
            yield from super().split(dataset)

    dataset = _Dataset()
    ctx = RunContext()
    result = run_slice(
        dataset,
        Task(kind="regression"),
        estimators={name: lambda name=name: TimedEstimator(name, fits) for name in ("good", "bad")},
        splitter=TwoFolds(),
        metric=_Metric(),
        policy=SelectionPolicy(),
        search=search,
        budget=budget,
        ctx=ctx,
    )
    assert result.search is not None
    assert result.search["model_reason"] == "incomplete_confirmation_budget"
    assert result.search["confirmation_probes"] == []
    assert result.search["probe_fit_count"] == 3
    assert result.best_model_id == "good"
    assert sum(fit.iterations == 64 for fit in fits) == 2
    assert sum(fit.iterations == 256 for fit in fits) == 1
    assert sum(fit.iterations == 100 for fit in fits) == 2
    work = ctx.cost_report()["work"]
    assert sum(item["stage"] == "scouting" for item in work) == 3
    assert sum(item["stage"] == "wide_control" for item in work) == 2
    assert all(item["status"] == "completed" for item in work)
    before_finish = 58 if stop == "fit_limit" else 69
    assert clock[0] == before_finish
    assert budget.time_left() == 70 - before_finish

    refit_best(
        dataset,
        Task(kind="regression"),
        factory=lambda: TimedEstimator("good", fits),
        ctx=ctx,
        model_id="good",
    )
    assert fits[-1].rows == dataset.n_rows
    assert clock[0] == before_finish + 20 < 100
    assert budget.exhausted
    assert ctx.cost_report()["work"][-1]["stage"] == "refit"
    assert ctx.cost_report()["work"][-1]["status"] == "completed"
