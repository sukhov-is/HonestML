"""Full completion cost must be known before a quality-equivalent choice."""

from collections.abc import Sequence

import numpy as np
import pytest

from honestml.application.search import probe_models
from honestml.core import Fold, SearchConfig, Task
from honestml.core.selection_policy import Candidate

from .test_search_confirmation import Metric, folds

pytestmark = pytest.mark.unit


class Noninferior:
    def noninferior(self, *args: object, **kwargs: object) -> bool:
        return True


@pytest.mark.parametrize("known", [True, False])
def test_full_cost_precedes_family_choice_and_unknown_keeps_quality(known: bool) -> None:
    calls: list[str] = []

    def evaluate(name: str, parts: Sequence[Fold], iterations: int) -> Candidate:
        calls.append(name)
        return Candidate(
            name,
            0.9 if name == "anchor" else 0.895,
            train_time=2.0 if name == "anchor" else 1.0,
            oof_pred=np.arange(1400.0),
            oof_mask=np.ones(1400, dtype=bool),
        )

    def profile(names: tuple[str, ...]) -> dict[str, dict[str, object]]:
        assert len(calls) == 4
        assert set(names) == {"anchor", "cheap_cv"}
        return {
            "anchor": {
                "fs": {"status": "disabled", "estimated_s": 0.0},
                "hpo": {"status": "conditional", "estimated_s": 2.0},
                "additional_cv_count": 1,
            },
            "cheap_cv": {
                "fs": {"status": "disabled", "estimated_s": 0.0},
                "hpo": {
                    "status": "conditional" if known else "unavailable",
                    "estimated_s": 1000.0 if known else None,
                },
                "additional_cv_count": 1,
            },
        }

    result = probe_models(
        ("anchor", "cheap_cv"),
        folds(),
        y=np.arange(1400.0),
        groups=None,
        task=Task(kind="regression"),
        metric=Metric(),
        config=SearchConfig(max_rows=100, confirmation_rows=600, model_margin=0.01),
        seed=1,
        evaluate=evaluate,
        significance_test=Noninferior(),
        completion_refit_rows=(1400,),
        profile_completion=profile,
    )
    assert result.winner == "anchor"
    cost = result.completion_costs["cheap_cv"]
    assert cost["estimated_s"] is not None if known else cost["estimated_s"] is None
    assert result.reason == ("confirmation_score" if known else "incomplete_completion_cost")
    anchor = result.completion_costs["anchor"]
    assert anchor["estimated_s"] == pytest.approx(
        result.cost_estimates["anchor"] + anchor["additional_cv_s"] + 2.0
    )


def test_single_family_does_not_add_cost_probe_fits() -> None:
    def unexpected(*args: object) -> object:
        raise AssertionError("fixed family must not run selection probes")

    result = probe_models(
        ("fixed",),
        folds(),
        y=np.arange(1400.0),
        groups=None,
        task=Task(kind="regression"),
        metric=Metric(),
        config=SearchConfig(),
        seed=1,
        evaluate=unexpected,
        profile_completion=unexpected,
    )
    assert result.reason == "single_model"
    assert not result.completion_costs


class RecordingRanker:
    name = "importance"

    def __init__(self, ctx: object) -> None:
        self.ctx = ctx
        self.rows: list[np.ndarray] = []
        self.seeds: list[object] = []

    def rank(self, x: np.ndarray, y: np.ndarray, **kwargs: object) -> np.ndarray:
        self.rows.append(x[:, 0].copy())
        self.seeds.append(kwargs["random_state"])
        self.ctx.record_fit(
            "fs",
            model_id="proxy",
            rows=len(x),
            columns=x.shape[1],
            elapsed_s=1.0,
            tree_budget=8,
            iterations=8,
        )
        return np.arange(1, x.shape[1] + 1, dtype=float)

    def auto_threshold(self, n_features: int) -> float:
        return 0.0


def test_fs_profile_limits_only_original_train_and_has_separate_fit_ceiling() -> None:
    from honestml.application.search import profile_fs_cost
    from honestml.core import FeatureSelectionConfig, RunContext

    ctx = RunContext()
    ranker = RecordingRanker(ctx)
    x = np.column_stack((np.arange(200.0), np.arange(200.0) % 3))
    parts = (
        Fold(np.arange(100), np.arange(100, 120), np.arange(120, 160)),
        Fold(np.arange(140), np.arange(140, 160), np.arange(160, 200)),
    )
    result = profile_fs_cost(
        ranker,
        x,
        x[:, 0],
        parts,
        categorical=np.zeros(2, dtype=bool),
        feature_names=("row", "b"),
        config=FeatureSelectionConfig(refine=False),
        search=SearchConfig(max_rows=80),
        fit_predict=None,
        metric=Metric(),
        task=Task(kind="regression"),
        seed=42,
        ctx=ctx,
    )
    assert [len(rows) for rows in ranker.rows] == [40, 80]
    assert ranker.seeds == [42, 42]
    assert result["seed"] == 42
    assert all(np.all(rows < 120) for rows in ranker.rows)
    assert result["status"] == "conditional"
    assert result["fit_count_upper_bound"] == 2
    assert result["planned_fit_count"] is None
    assert result["observed_path_projected_fit_count"] == 2
    assert result["estimated_s"] == pytest.approx(result["profiles"][1]["elapsed_s"] * 2)
    assert result["validation"]["same_observed_fit_path"] is True


def test_fs_profile_reports_budget_exhaustion_without_training() -> None:
    from honestml.application.search import profile_fs_cost
    from honestml.core import FeatureSelectionConfig, RunContext

    ctx = RunContext()
    ranker = RecordingRanker(ctx)
    x = np.column_stack((np.arange(200.0), np.arange(200.0) % 3))
    result = profile_fs_cost(
        ranker,
        x,
        x[:, 0],
        (Fold(np.arange(120), np.array([], dtype=int), np.arange(120, 200)),),
        categorical=np.zeros(2, dtype=bool),
        feature_names=("a", "b"),
        config=FeatureSelectionConfig(refine=False),
        search=SearchConfig(max_rows=80),
        fit_predict=None,
        metric=Metric(),
        task=Task(kind="regression"),
        seed=42,
        ctx=ctx,
        can_start=lambda: False,
    )
    assert result["estimated_s"] is None
    assert result["reason"] == "probe_budget"
    assert not ranker.rows


@pytest.mark.parametrize("with_fs", [False, True])
@pytest.mark.slow
def test_facade_profiles_before_hpo_with_shared_budget_and_full_candidate_plan(
    with_fs: bool,
) -> None:
    pytest.importorskip("lightgbm")
    from honestml import AutoML, CVConfig, FeatureSelectionConfig, HPOConfig

    rng = np.random.default_rng(91)
    x = rng.normal(size=(640, 6))
    y = (x[:, 0] + 0.5 * x[:, 1] > 0).astype(int)
    model = AutoML(
        models=("baseline", "lightgbm"),
        task="binary",
        metric="roc_auc",
        cv=CVConfig(n_splits=2, outer_holdout=0, calibrate="off"),
        search=SearchConfig(
            max_rows=128,
            confirmation_rows=512,
            model_iterations=8,
            confirmation_iterations=16,
            min_class_count=2,
        ),
        feature_selection=FeatureSelectionConfig(strategy="importance", refine=False)
        if with_fs
        else None,
        hpo=HPOConfig(n_trials=2, inner_cv=2, keep_baseline=True),
        significance="off",
        run_mode="selection",
        random_state=42,
    ).fit(x, y)
    search = model.run_report_["search"]
    forecasts = search["completion_cost_forecast"]
    assert forecasts["baseline"]["hpo"]["status"] == "disabled"
    profile = forecasts["lightgbm"]
    assert profile["hpo"]["status"] == "conditional"
    assert profile["hpo"]["planned_fit_count"] == 4
    assert profile["additional_cv_count"] == 2
    assert profile["fs"]["status"] == ("conditional" if with_fs else "disabled")
    assert profile["estimated_s"] == pytest.approx(
        profile["cv_refit_s"]
        + profile["additional_cv_s"]
        + profile["fs"]["estimated_s"]
        + profile["hpo"]["estimated_s"]
    )
    work = model.run_report_["cost"]["work"]
    first_hpo = next(i for i, w in enumerate(work) if w["stage"] == "hpo")
    profiling = [i for i, w in enumerate(work) if w["model_id"].endswith("__cost_hpo")]
    assert profiling and max(profiling) < first_hpo
    assert search["probe_fit_count"] <= search["probe_fit_limit"]
    assert all(w["status"] == "completed" for w in work)
