"""Backend FS fits retain application recipe, fold and null-trial attribution."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from honestml.adapters.feature_rankers import (
    ImportanceRanker,
    NullImportanceRanker,
    RandomProbeRanker,
    make_ranker_fit_predict,
)
from honestml.adapters.feature_selectors import SequentialSelector
from honestml.application.feature_compare import _select_one
from honestml.application.feature_selection import aggregate_scores
from honestml.application.search import scout_feature_recipes
from honestml.core import Candidate, FeatureSelectionConfig, Fold, RunContext, Task
from honestml.core.config import SearchConfig

pytestmark = pytest.mark.unit


class _Mean:
    name = "mean"
    greater_is_better = True
    needs = "value"
    optimum = float("inf")
    average = None
    proper_proba = False

    def score(
        self, y_true: np.ndarray, y_pred: np.ndarray, sample_weight: np.ndarray | None = None
    ) -> float:
        return float(np.mean(y_pred))


def _folds() -> list[Fold]:
    return [
        Fold(np.arange(15), np.empty(0, dtype=int), np.arange(15, 30)),
        Fold(np.arange(15, 30), np.empty(0, dtype=int), np.arange(15)),
    ]


def test_real_ranker_fits_record_recipe_fold_and_null_trial() -> None:
    ctx, task = RunContext(), Task(kind="regression")
    x = np.random.default_rng(1).normal(size=(30, 3))
    y = x[:, 0]
    for ranker in (
        ImportanceRanker(task),
        RandomProbeRanker(task),
        NullImportanceRanker(task, n_runs=1),
    ):
        ranker.set_run_context(ctx)
        ranker.set_ranker_iterations(2)
        ranker.set_threads(1)
        aggregate_scores(
            x,
            y,
            _folds(),
            ranker=ranker,
            categorical=np.zeros(3, dtype=bool),
            config=FeatureSelectionConfig(),
            ctx=ctx,
        )
    work = ctx.cost_report()["work"]
    assert [(row["recipe"], row["fold"], row["trial"]) for row in work] == [
        ("importance", 0, None),
        ("importance", 1, None),
        ("random_probe", 0, None),
        ("random_probe", 1, None),
        ("null_importance", 0, 0),
        ("null_importance", 0, 1),
        ("null_importance", 1, 0),
        ("null_importance", 1, 1),
    ]
    assert all(row["model_id"] == "proxy" and row["tree_budget"] == 2 for row in work)


def test_sequential_oof_fits_keep_recipe_and_actual_fold() -> None:
    ctx, task = RunContext(), Task(kind="regression")
    x = np.random.default_rng(2).normal(size=(30, 3))
    _select_one(
        SequentialSelector(full_descent=True),
        x,
        x[:, 0],
        _folds(),
        categorical=np.zeros(3, dtype=bool),
        config=FeatureSelectionConfig(),
        seed=2,
        sample_weight=None,
        fit_predict=make_ranker_fit_predict(task, ctx=ctx, n_estimators=2, threads=1),
        metric=_Mean(),
        task=task,
        global_classes=None,
        run_context=ctx,
    )
    work = ctx.cost_report()["work"]
    assert work
    assert {row["recipe"] for row in work} == {"sequential"}
    assert [row["fold"] for row in work] == [0, 1] * (len(work) // 2)


def test_scouting_distinguishes_prefilter_recipe_and_chosen_model_confirmation() -> None:
    ctx, task = RunContext(), Task(kind="regression")
    x = np.random.default_rng(3).normal(size=(30, 4))
    ranker, prefilter = ImportanceRanker(task), ImportanceRanker(task)
    for adapter in (ranker, prefilter):
        adapter.set_run_context(ctx)
        adapter.set_ranker_iterations(2)
        adapter.set_threads(1)

    def evaluate(indices: tuple[int, ...], folds: Sequence[Fold]) -> Candidate:
        with ctx.timed_fit(
            "scouting", model_id="chosen", rows=len(folds[0].fit_idx), columns=len(indices), fold=0
        ):
            return Candidate("chosen", 1.0, n_features=len(indices))

    scout_feature_recipes(
        [("importance", ranker)],
        x,
        x[:, 0],
        [Fold(np.arange(15), np.arange(15, 22), np.arange(22, 30))],
        categorical=np.zeros(4, dtype=bool),
        fs_config=FeatureSelectionConfig(refine=False),
        search_config=SearchConfig(max_features=2),
        metric=_Mean(),
        task=task,
        fit_predict=make_ranker_fit_predict(task, ctx=ctx, n_estimators=2, threads=1),
        prefilter=prefilter,
        seed=1,
        evaluate=evaluate,
        ctx=ctx,
    )
    work = ctx.cost_report()["work"]
    assert [(row["stage"], row["recipe"], row["fold"]) for row in work] == [
        ("scouting", "no_selection", 0),
        ("fs", "prefilter", 0),
        ("fs", "importance", 0),
        ("scouting", "importance", 0),
    ]
