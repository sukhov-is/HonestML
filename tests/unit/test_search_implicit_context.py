"""Direct search calls enforce actual fit limits without an explicit context."""

import numpy as np
import pytest

from honestml.adapters import Reader, make_ranker_fit_predict
from honestml.application.slice import FeatureSelectionBundle, run_slice
from honestml.composition import build_default_components
from honestml.core import (
    BudgetExhaustedError,
    FeatureSelectionConfig,
    RunConfig,
    RunContext,
    SelectionPolicy,
    Task,
)
from honestml.core.config import SearchConfig
from honestml.core.ports.estimator import SupportsFitContext, SupportsRankerBudget

from .test_fast_search_integration import _Dataset, _Estimator, _Metric, _Splitter

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("provided_context", [False, True])
def test_context_enforces_actual_model_probe_fit_limit(provided_context: bool) -> None:
    fits = []
    ctx = RunContext(run_config=RunConfig(seed=19)) if provided_context else None
    result = run_slice(
        _Dataset(),
        Task(kind="regression"),
        estimators={name: (lambda name=name: _Estimator(name, fits)) for name in ("good", "bad")},
        splitter=_Splitter(),
        metric=_Metric(),
        policy=SelectionPolicy(greater_is_better=True),
        search=SearchConfig(max_rows=40, confirmation_rows=80, max_probe_fits=1),
        ctx=ctx,
    )
    assert sum(fit.rows < 72 for fit in fits) == 1
    assert result.search["model_reason"] == "incomplete_probe_budget"
    if ctx is not None:
        assert ctx.run_config.seed == 19
        assert sum(work["stage"] == "scouting" for work in ctx.cost_report()["work"]) == 1


@pytest.mark.parametrize("strategy", ["null_importance", "importance"])
def test_omitted_context_enforces_native_fs_fit_limit(strategy: str) -> None:
    task = Task(kind="regression")
    x = np.column_stack((np.arange(120.0), np.arange(120.0) % 3))
    dataset = Reader(task).read(x, x[:, 0])
    config = FeatureSelectionConfig(strategy=strategy, n_runs=3, refine=True)
    components = build_default_components(
        task, models=("linear",), cv=2, random_state=0, feature_selection=config
    )
    for _, ranker in components.feature_strategies or ():
        if isinstance(ranker, SupportsRankerBudget):
            ranker.set_ranker_iterations(1)
    if isinstance(components.feature_ranker, SupportsRankerBudget):
        components.feature_ranker.set_ranker_iterations(1)
    result = run_slice(
        dataset,
        task,
        estimators=components.estimators,
        splitter=components.splitter,
        metric=components.metric,
        policy=components.policy,
        features=FeatureSelectionBundle(
            config=config,
            ranker=components.feature_ranker,
            strategies=components.feature_strategies,
            carve=components.feature_carve,
            fit_predict=components.feature_fit_predict,
            arbitration_splitter=components.feature_arbitration_splitter,
        ),
        search=SearchConfig(max_rows=40, max_fs_fits=1),
    )
    assert result.search["stop_reason"] == "fs_fit_limit"
    assert result.search["fs_fit_count"] == 1
    execution = result.search["fs_execution"]
    assert execution["proxy_resource_source"] == "native_component"
    assert execution["n_runs"] is execution["n_probes"] is execution["ranker_iterations"] is None
    assert execution["rows_per_fit"] == 40 and execution["fit_limit"] == 1
    assert result.feature_selection is None
    assert result.best_model_id == "linear"


def test_native_fit_predict_accepts_late_context_and_checks_each_real_fit() -> None:
    predictor = make_ranker_fit_predict(Task(kind="regression"), threads=1, n_estimators=1)
    assert isinstance(predictor, SupportsFitContext)
    ctx = RunContext()

    def guard(stage: str) -> None:
        if ctx.cost_report()["work"]:
            raise BudgetExhaustedError("trials", completed=1, skipped=1, failed=0)

    ctx.before_fit = guard
    predictor.set_run_context(ctx)
    x = np.arange(20.0)[:, None]
    predictor(x[:12], x[:12, 0], x[12:], None, 0)
    with pytest.raises(BudgetExhaustedError):
        predictor(x[:12], x[:12, 0], x[12:], None, 0)
    assert len(ctx.cost_report()["work"]) == 1
