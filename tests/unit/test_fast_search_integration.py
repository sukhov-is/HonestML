"""Fast search integrates bounded probes, chosen-recipe execution and HPO filtering."""

from __future__ import annotations

from collections.abc import Iterator
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import pytest

from honestml.application.slice import FeatureSelectionBundle, TuningBundle, run_slice
from honestml.core import (
    BudgetExhaustedError,
    Candidate,
    FeatureSelectionConfig,
    FEConfig,
    Fold,
    NoSignificanceTest,
    RunContext,
    SelectionPolicy,
    Task,
    TuneOutcome,
)
from honestml.core.config import SearchConfig
from honestml.core.schema import CategoryTable, ColumnRole, FeatureSchema, TargetEncodingSpec

pytestmark = pytest.mark.unit


class _Dataset:
    def __init__(self, rows: int = 120) -> None:
        self.x = np.column_stack((np.arange(rows, dtype=float), np.zeros((rows, 3))))
        self.schema = FeatureSchema(roles={f"f{i}": ColumnRole.NUMERIC for i in range(4)})
        self.n_rows = rows

    def target(self) -> np.ndarray:
        return self.x[:, 0].copy()

    def to_numpy(self) -> np.ndarray:
        return self.x

    def categorical_codes(self) -> np.ndarray:
        return np.empty((self.n_rows, 0), dtype=int)

    def sample_weight(self) -> None:
        return None


class _Splitter:
    def split(self, dataset: _Dataset) -> Iterator[Fold]:
        yield Fold(np.arange(72), np.arange(72, 96), np.arange(96, 120))


class _Metric:
    name = "negative_mae"
    greater_is_better = True
    needs = "value"
    optimum = 0.0
    average = None
    proper_proba = False

    def score(
        self, y_true: np.ndarray, y_pred: np.ndarray, sample_weight: np.ndarray | None = None
    ) -> float:
        return -float(np.mean(np.abs(y_true - y_pred)))


@dataclass(frozen=True)
class _Fit:
    model: str
    rows: int
    columns: int
    iterations: int
    threads: int


class _Estimator:
    supports_early_stopping = True

    def __init__(self, name: str, fits: list[_Fit]) -> None:
        self.name = name
        self.fits = fits
        self.feature_names: list[str] = []
        self.iteration_budget = 100
        self.fitted_iterations: int | None = None
        self.threads = 4

    def set_refit_iterations(self, count: int) -> None:
        self.iteration_budget = count

    def set_threads(self, threads: int) -> None:
        self.threads = threads

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> _Estimator:
        self.fits.append(_Fit(self.name, len(X), X.shape[1], self.iteration_budget, self.threads))
        self.fitted_iterations = self.iteration_budget
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.name == "good" and "f0" in self.feature_names:
            return X[:, self.feature_names.index("f0")]
        return np.zeros(len(X))


class _Ranker:
    def __init__(self, name: str, column: int) -> None:
        self.name = name
        self.column = column
        self.rows: list[int] = []
        self.widths: list[int] = []

    def rank(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        categorical: np.ndarray,
        random_state: int,
        sample_weight: np.ndarray | None = None,
        groups: np.ndarray | None = None,
    ) -> np.ndarray:
        self.rows.append(len(x))
        self.widths.append(x.shape[1])
        scores = np.zeros(x.shape[1])
        scores[self.column] = 1.0
        return scores

    def auto_threshold(self, n_features: int) -> float:
        return 0.0


def _cheap_fit(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_te: np.ndarray,
    sample_weight: np.ndarray | None,
    seed: int,
) -> tuple[None, np.ndarray, None]:
    return None, np.zeros(len(x_te)), None


def _carve(dataset: _Dataset, fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    pytest.fail("chosen single recipe must not invoke multi-strategy carve")


def test_discarded_model_stays_on_probe_rows_iterations_and_threads() -> None:
    fits: list[_Fit] = []
    result = run_slice(
        _Dataset(),
        Task(kind="regression"),
        estimators={name: lambda name=name: _Estimator(name, fits) for name in ("bad", "good")},
        splitter=_Splitter(),
        metric=_Metric(),
        policy=SelectionPolicy(),
        search=SearchConfig(
            max_rows=40,
            model_iterations=7,
            threads=1,
            confirmation_rows=80,
            confirmation_iterations=14,
        ),
    )
    assert result.best_model_id == "good"
    assert result.search is not None and result.search["selected_model"] == "good"
    assert [_fit for _fit in fits if _fit.model == "bad"] == [
        _Fit("bad", 24, 4, 7, 1),
        _Fit("bad", 48, 4, 14, 1),
    ]
    assert [_fit for _fit in fits if _fit.model == "good"] == [
        _Fit("good", 24, 4, 7, 1),
        _Fit("good", 48, 4, 14, 1),
        _Fit("good", 72, 4, 100, 4),
    ]
    assert len(result.leaderboard) == 1


def test_only_chosen_recipe_runs_full_and_hpo_receives_chosen_family() -> None:
    fits: list[_Fit] = []
    chosen, discarded = _Ranker("importance", 0), _Ranker("random_probe", 1)
    probe_chosen, probe_discarded = _Ranker("importance", 0), _Ranker("random_probe", 1)
    selected_models: list[tuple[str, ...]] = []
    tuned_features: list[tuple[str, ...] | None] = []

    def tune(dataset: _Dataset, features: tuple[str, ...] | None) -> dict:
        assert selected_models == [("good",)]
        tuned_features.append(features)
        return {}

    features = FeatureSelectionBundle(
        config=FeatureSelectionConfig(
            compare=("importance", "random_probe"), cutoff="top_k", top_k=1, refine=False
        ),
        strategies=(("importance", chosen), ("random_probe", discarded)),
        probe_strategies=(("importance", probe_chosen), ("random_probe", probe_discarded)),
        prefilter=_Ranker("importance", 0),
        fit_predict=_cheap_fit,
        carve=_carve,
    )
    result = run_slice(
        _Dataset(),
        Task(kind="regression"),
        estimators={name: lambda name=name: _Estimator(name, fits) for name in ("bad", "good")},
        splitter=_Splitter(),
        metric=_Metric(),
        policy=SelectionPolicy(),
        significance_test=NoSignificanceTest(),
        search=SearchConfig(
            max_rows=40,
            model_iterations=7,
            threads=1,
            confirmation_rows=80,
            confirmation_iterations=14,
        ),
        features=features,
        tuning=TuningBundle(
            tune=tune, tuned_factories=lambda outcomes: {}, select_models=selected_models.append
        ),
    )
    assert result.search is not None and result.search["selected_recipe"] == "importance"
    assert result.feature_selection is not None
    assert result.feature_selection.selected_features == ("f0",)
    assert chosen.rows == [40]
    assert discarded.rows == []
    assert discarded.widths == []
    assert chosen.widths == [4]
    assert probe_chosen.rows == probe_discarded.rows == [24]
    assert tuned_features == [("f0",)]
    assert all(fit.model == "good" for fit in fits if fit.iterations == 100)
    assert any(fit.rows == 72 and fit.columns == 1 for fit in fits)


def test_probe_excludes_target_encodings_and_preserves_native_categories() -> None:
    class _TEDataset(_Dataset):
        def __init__(self) -> None:
            super().__init__()
            self.x[:, 0] = np.arange(self.n_rows) % 2
            self.schema = FeatureSchema(
                roles={
                    **self.schema.roles,
                    "cat": ColumnRole.CATEGORICAL,
                    "cat_te": ColumnRole.NUMERIC,
                },
                categories={"cat": CategoryTable.fit(["a", "b"])},
                target_encoding=TargetEncodingSpec(
                    encodings={"cat": {"0": 0.2, "1": 0.8}}, global_mean=0.5, smoothing=1.0
                ),
            )

        def to_numpy(self) -> np.ndarray:
            return np.column_stack((self.x, np.full(self.n_rows, 0.5)))

        def categorical_codes(self) -> np.ndarray:
            return (np.arange(self.n_rows) % 2).reshape(-1, 1)

    observations: list[tuple[int, tuple[str, ...], tuple[int, ...]]] = []

    class _NativeEstimator(_Estimator):
        supports_native_categorical = True

        def __init__(self, name: str, fits: list[_Fit]) -> None:
            super().__init__(name, fits)
            self.categorical_indices: list[int] = []

        def fit(
            self,
            X: np.ndarray,
            y: np.ndarray,
            X_val: np.ndarray | None = None,
            y_val: np.ndarray | None = None,
            sample_weight: np.ndarray | None = None,
        ) -> _NativeEstimator:
            super().fit(X, y, X_val, y_val, sample_weight)
            observations.append(
                (self.iteration_budget, tuple(self.feature_names), tuple(self.categorical_indices))
            )
            return self

    fits: list[_Fit] = []
    chosen, discarded = _Ranker("importance", 0), _Ranker("random_probe", 1)
    probe_chosen, probe_discarded = _Ranker("importance", 0), _Ranker("random_probe", 1)
    features = FeatureSelectionBundle(
        config=FeatureSelectionConfig(
            compare=("importance", "random_probe"), cutoff="top_k", top_k=1, refine=False
        ),
        strategies=(("importance", chosen), ("random_probe", discarded)),
        probe_strategies=(("importance", probe_chosen), ("random_probe", probe_discarded)),
        prefilter=_Ranker("importance", 0),
        fit_predict=_cheap_fit,
        carve=_carve,
    )
    cache = _StageCache({})
    result = run_slice(
        _TEDataset(),
        Task(kind="binary"),
        estimators={
            name: lambda name=name: _NativeEstimator(name, fits) for name in ("bad", "good")
        },
        splitter=_Splitter(),
        metric=_Metric(),
        policy=SelectionPolicy(),
        significance_test=NoSignificanceTest(),
        search=SearchConfig(
            max_rows=40,
            model_iterations=7,
            threads=1,
            confirmation_rows=80,
            confirmation_iterations=14,
        ),
        features=features,
        fe=FEConfig(target_encoding=True),
        stage_cache=cache,
    )
    assert (
        result.search is not None and result.search["probe_target_encoding"] == "raw_features_only"
    )
    probe_observations = [row for row in observations if row[0] == 7]
    assert all("cat_te" not in names for _, names, _ in probe_observations)
    assert any(names[-1] == "cat" and indices == (4,) for _, names, indices in probe_observations)
    assert any(
        "cat_te" in names and indices == (5,)
        for iterations, names, indices in observations
        if iterations == 100
    )
    assert probe_chosen.widths == probe_discarded.widths == [5]
    assert chosen.widths == [6]
    assert not discarded.widths

    resumed = run_slice(
        _TEDataset(),
        Task(kind="binary"),
        estimators={
            name: lambda name=name: _NativeEstimator(name, fits) for name in ("bad", "good")
        },
        splitter=_Splitter(),
        metric=_Metric(),
        policy=SelectionPolicy(),
        significance_test=NoSignificanceTest(),
        search=SearchConfig(
            max_rows=40,
            model_iterations=7,
            threads=1,
            confirmation_rows=80,
            confirmation_iterations=14,
        ),
        features=features,
        fe=FEConfig(target_encoding=True),
        stage_cache=cache,
    )
    assert resumed.feature_selection == result.feature_selection
    assert resumed.native_routing == result.native_routing
    assert chosen.widths == [6]
    assert probe_chosen.widths == probe_discarded.widths == [5]
    assert not discarded.widths


class _StageCache:
    def __init__(self, values: dict[tuple[str, str], object], scope: str = "run") -> None:
        self.values = values
        self.scope = scope

    def get_stage(self, stage: str) -> object | None:
        return deepcopy(self.values.get((self.scope, stage)))

    def put_stage(self, stage: str, value: object) -> None:
        self.values[(self.scope, stage)] = deepcopy(value)


def test_interrupted_hpo_restores_completed_preparation_without_repeat_fits() -> None:
    fits: list[_Fit] = []
    chosen, discarded = _Ranker("importance", 0), _Ranker("random_probe", 1)
    probe_chosen, probe_discarded = _Ranker("importance", 0), _Ranker("random_probe", 1)
    features = FeatureSelectionBundle(
        config=FeatureSelectionConfig(
            compare=("importance", "random_probe"), cutoff="top_k", top_k=1, refine=False
        ),
        strategies=(("importance", chosen), ("random_probe", discarded)),
        probe_strategies=(("importance", probe_chosen), ("random_probe", probe_discarded)),
        prefilter=_Ranker("importance", 0),
        fit_predict=_cheap_fit,
        carve=_carve,
    )
    selected_models: list[tuple[str, ...]] = []
    calls = 0

    def tune(dataset: _Dataset, selected: tuple[str, ...] | None) -> dict:
        nonlocal calls
        calls += 1
        assert selected == ("f0",)
        if calls == 1:
            raise RuntimeError("interrupted HPO")
        return {}

    stage_cache = _StageCache({})

    def run(cache: _StageCache):
        return run_slice(
            _Dataset(),
            Task(kind="regression"),
            estimators={name: lambda name=name: _Estimator(name, fits) for name in ("bad", "good")},
            splitter=_Splitter(),
            metric=_Metric(),
            policy=SelectionPolicy(),
            significance_test=NoSignificanceTest(),
            search=SearchConfig(
                max_rows=40,
                model_iterations=7,
                threads=1,
                confirmation_rows=80,
                confirmation_iterations=14,
            ),
            features=features,
            tuning=TuningBundle(
                tune=tune, tuned_factories=lambda outcomes: {}, select_models=selected_models.append
            ),
            stage_cache=cache,
        )

    with pytest.raises(RuntimeError, match="interrupted HPO"):
        run(stage_cache)
    assert len(stage_cache.values) == 1
    before = len(fits)
    result = run(stage_cache)
    assert fits[before:] == [_Fit("good", 72, 1, 100, 4)]
    assert chosen.rows == [40]
    assert discarded.rows == []
    assert probe_chosen.rows == probe_discarded.rows == [24]
    assert selected_models == [("good",), ("good",)]
    assert result.feature_selection is not None
    stored = next(iter(stage_cache.values.values()))
    assert result.feature_selection == stored.report
    assert result.search is not None and result.search["preparation_reused"] is True
    before = len(fits)
    run(_StageCache(stage_cache.values, "changed_context"))
    assert len(fits) > before + 1
    assert chosen.rows == [40, 40]


def test_actual_probe_fit_cap_preserves_full_control_and_does_not_cache_partial_work() -> None:
    fits: list[_Fit] = []
    cache = _StageCache({})
    ctx = RunContext()
    result = run_slice(
        _Dataset(),
        Task(kind="regression"),
        estimators={name: lambda name=name: _Estimator(name, fits) for name in ("good", "bad")},
        splitter=_Splitter(),
        metric=_Metric(),
        policy=SelectionPolicy(),
        search=SearchConfig(max_rows=40, model_iterations=7, max_probe_fits=1),
        stage_cache=cache,
        ctx=ctx,
    )
    assert fits == [_Fit("good", 24, 4, 7, 1), _Fit("good", 72, 4, 100, 4)]
    assert result.search is not None
    assert result.search["model_reason"] == "incomplete_probe_budget"
    assert result.search["probe_fit_count"] == result.search["probe_fit_limit"] == 1
    assert result.search["skipped_models"] == ["bad"]
    assert not cache.values


def test_unknown_preparation_payload_is_a_miss() -> None:
    fits: list[_Fit] = []
    cache = _StageCache({("run", "prepared_features_v1"): {"incomplete": True}})
    result = run_slice(
        _Dataset(),
        Task(kind="regression"),
        estimators={"good": lambda: _Estimator("good", fits)},
        splitter=_Splitter(),
        metric=_Metric(),
        policy=SelectionPolicy(),
        search=SearchConfig(),
        stage_cache=cache,
    )
    assert fits == [_Fit("good", 72, 4, 100, 4)]
    assert result.search is not None and not result.search.get("preparation_reused")


def test_resumed_preparation_installs_hpo_budget_checkpoint() -> None:
    class _Budget:
        stopped = False

        @property
        def exhausted(self) -> bool:
            return self.stopped

        @property
        def exhausted_reason(self) -> str | None:
            return "time" if self.stopped else None

        def time_left(self) -> float:
            return float("inf")

        def memory_left(self) -> None:
            return None

        def consume(self, seconds: float) -> None:
            return None

    fits: list[_Fit] = []
    cache = _StageCache({})
    run_slice(
        _Dataset(),
        Task(kind="regression"),
        estimators={"good": lambda: _Estimator("good", fits)},
        splitter=_Splitter(),
        metric=_Metric(),
        policy=SelectionPolicy(),
        search=SearchConfig(),
        stage_cache=cache,
    )
    before = len(fits)
    ctx, budget = RunContext(), _Budget()

    def tune(dataset: _Dataset, selected: tuple[str, ...] | None) -> dict:
        budget.stopped = True
        with ctx.timed_fit("hpo", model_id="good", rows=10, columns=4):
            pytest.fail("exhausted resumed HPO must stop before training")

    with pytest.raises(BudgetExhaustedError):
        run_slice(
            _Dataset(),
            Task(kind="regression"),
            estimators={"good": lambda: _Estimator("good", fits)},
            splitter=_Splitter(),
            metric=_Metric(),
            policy=SelectionPolicy(),
            search=SearchConfig(),
            stage_cache=cache,
            ctx=ctx,
            budget=budget,
            tuning=TuningBundle(tune=tune, tuned_factories=lambda outcomes: {}),
        )
    assert len(fits) == before


@pytest.mark.parametrize("total_seconds", [10.0, 100.0])
def test_actual_probe_folds_stop_after_global_or_phase_deadline(
    monkeypatch: pytest.MonkeyPatch, total_seconds: float
) -> None:
    import honestml.application.slice as slice_module

    clock = [0.0]
    monkeypatch.setattr(slice_module.time, "perf_counter", lambda: clock[0])

    class _TimedEstimator(_Estimator):
        def fit(
            self,
            X: np.ndarray,
            y: np.ndarray,
            X_val: np.ndarray | None = None,
            y_val: np.ndarray | None = None,
            sample_weight: np.ndarray | None = None,
        ) -> _TimedEstimator:
            super().fit(X, y, X_val, y_val, sample_weight)
            clock[0] += 30.0
            return self

    class _TwoFolds(_Splitter):
        def split(self, dataset: _Dataset) -> Iterator[Fold]:
            yield from super().split(dataset)
            yield from super().split(dataset)

    class _Budget:
        @property
        def exhausted(self) -> bool:
            return clock[0] >= total_seconds

        @property
        def exhausted_reason(self) -> str | None:
            return "time" if self.exhausted else None

        def time_left(self) -> float:
            return max(0.0, total_seconds - clock[0])

        def memory_left(self) -> None:
            return None

        def consume(self, seconds: float) -> None:
            return None

    fits: list[_Fit] = []

    def run():
        return run_slice(
            _Dataset(),
            Task(kind="regression"),
            estimators={
                name: lambda name=name: _TimedEstimator(name, fits) for name in ("good", "bad")
            },
            splitter=_TwoFolds(),
            metric=_Metric(),
            policy=SelectionPolicy(),
            search=SearchConfig(max_rows=80, max_folds=2, model_iterations=7),
            ctx=RunContext(),
            budget=_Budget(),
        )

    if total_seconds == 10.0:
        with pytest.raises(BudgetExhaustedError):
            run()
        assert len(fits) == 1
    else:
        result = run()
        assert (
            result.search is not None and result.search["model_reason"] == "incomplete_probe_budget"
        )
        assert len(fits) == 3
    assert sum(fit.iterations == 7 for fit in fits) == 1


def test_partial_hpo_candidate_is_recomputed_after_search_completes() -> None:
    class _Candidates:
        def __init__(self) -> None:
            self.values: dict[str, Candidate] = {}
            self.reads = 0

        def get(self, name: str) -> Candidate | None:
            self.reads += 1
            return self.values.get(name)

        def put(self, name: str, candidate: Candidate) -> None:
            self.values[name] = candidate

    fits: list[_Fit] = []
    cache = _Candidates()
    complete = [False]

    def tune(dataset: _Dataset, selected: tuple[str, ...] | None) -> dict[str, TuneOutcome]:
        return {
            "good": TuneOutcome(
                best_params={"quality": int(complete[0])},
                n_trials_run=2 if complete[0] else 1,
                best_score=1.0,
                completed=complete[0],
            )
        }

    def run():
        return run_slice(
            _Dataset(),
            Task(kind="regression"),
            estimators={"good": lambda: _Estimator("good", fits)},
            splitter=_Splitter(),
            metric=_Metric(),
            policy=SelectionPolicy(),
            tuning=TuningBundle(
                tune=tune,
                tuned_factories=lambda outcomes: {
                    "good": lambda: _Estimator("good" if complete[0] else "bad", fits)
                },
            ),
            cache=cache,
        )

    partial = run()
    assert not cache.values
    assert cache.reads == 0
    complete[0] = True
    final = run()
    assert len(fits) == 2
    assert final.candidates[0].score > partial.candidates[0].score
    assert final.reused == ()
    assert "good" in cache.values
    resumed = run()
    assert len(fits) == 2
    assert resumed.reused == ("good",)


def test_memory_limit_stops_wide_control_before_next_fold() -> None:
    from honestml.adapters import RunBudget
    from honestml.core import BudgetConfig

    rss = [1.0]
    fits: list[_Fit] = []

    class _MemoryEstimator(_Estimator):
        def fit(
            self,
            X: np.ndarray,
            y: np.ndarray,
            X_val: np.ndarray | None = None,
            y_val: np.ndarray | None = None,
            sample_weight: np.ndarray | None = None,
        ) -> _MemoryEstimator:
            super().fit(X, y, X_val, y_val, sample_weight)
            rss[0] = 100.0
            return self

    class _TwoFolds:
        def split(self, dataset: _Dataset) -> Iterator[Fold]:
            yield Fold(np.arange(36), np.arange(36, 48), np.arange(48, 60))
            yield Fold(np.arange(72), np.arange(72, 96), np.arange(96, 120))

    ctx = RunContext()
    budget = RunBudget(BudgetConfig(memory_limit_mb=50), mem_probe=lambda: rss[0])
    with pytest.raises(BudgetExhaustedError, match="memory"):
        run_slice(
            _Dataset(),
            Task(kind="regression"),
            estimators={"good": lambda: _MemoryEstimator("good", fits)},
            splitter=_TwoFolds(),
            metric=_Metric(),
            policy=SelectionPolicy(),
            search=SearchConfig(max_rows=80, max_folds=2),
            budget=budget,
            ctx=ctx,
        )
    assert len(fits) == 1
    assert ctx.cost_report()["fit_counts"] == {"attempted": 1, "completed": 1, "failed": 0}
    with pytest.raises(BudgetExhaustedError, match="memory"):
        with ctx.timed_fit("refit", model_id="good", rows=120, columns=4):
            pytest.fail("full-data refit must honor the memory checkpoint")
    assert ctx.cost_report()["fit_counts"]["attempted"] == 1
