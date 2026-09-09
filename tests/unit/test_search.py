"""Bounded search preserves partition isolation and reports incomplete decisions."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from honestml.application.search import bounded_probe_folds, probe_models, scout_feature_recipes
from honestml.core import FeatureSelectionConfig, Fold, Task
from honestml.core.config import SearchConfig
from honestml.core.exceptions import FitFailedError
from honestml.core.ports.splitter import validate_fold
from honestml.core.selection_policy import Candidate

pytestmark = pytest.mark.unit


class _Metric:
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


class _Ranker:
    name = "importance"

    def __init__(self) -> None:
        self.rows: list[np.ndarray] = []
        self.targets: list[np.ndarray] = []

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
        self.rows.append(x.copy())
        self.targets.append(y.copy())
        return np.arange(x.shape[1], 0, -1, dtype=float)

    def auto_threshold(self, n_features: int) -> float:
        return 0.0


def _fit(
    x_tr: np.ndarray, y_tr: np.ndarray, x_te: np.ndarray, sw: np.ndarray | None, seed: int
) -> tuple[None, np.ndarray, None]:
    return None, np.ones(len(x_te)), None


def _fold() -> Fold:
    return Fold(np.arange(60), np.arange(60, 80), np.arange(80, 100))


def test_probe_partitions_keep_classes_groups_and_time_isolation() -> None:
    fold = _fold()
    y = np.zeros(100)
    y[[0, 60, 80]] = 1
    config = SearchConfig(max_rows=40)
    probe = bounded_probe_folds([fold], y=y, task=Task(kind="binary"), config=config, seed=2)[0]
    assert sum(len(part) for part in (probe.fit_idx, probe.es_idx, probe.test_idx)) <= 40
    for original, part in zip(
        (fold.fit_idx, fold.es_idx, fold.test_idx),
        (probe.fit_idx, probe.es_idx, probe.test_idx),
        strict=True,
    ):
        assert np.isin(part, original).all()
        assert np.array_equal(np.unique(y[part]), [0, 1])
    validate_fold(probe, groups=np.repeat(np.arange(10), 10), times=np.arange(100))
    again = bounded_probe_folds([fold], y=y, task=Task(kind="binary"), config=config, seed=2)[0]
    np.testing.assert_array_equal(probe.fit_idx, again.fit_idx)


def test_multiple_probe_folds_share_total_row_cap() -> None:
    probes = bounded_probe_folds(
        [_fold(), _fold()],
        y=np.arange(100.0),
        task=Task(kind="regression"),
        config=SearchConfig(max_rows=40, max_folds=2),
        seed=0,
    )
    assert len(probes) == 2
    assert sum(len(fold.fit_idx) + len(fold.es_idx) + len(fold.test_idx) for fold in probes) <= 40


def test_single_model_has_bounded_fs_folds_and_no_model_fit() -> None:
    def evaluate(name: str, folds: Sequence[Fold], iterations: int) -> Candidate:
        pytest.fail("explicit single model must skip model probes")

    out = probe_models(
        ["one"],
        [_fold()],
        y=np.arange(100.0),
        groups=None,
        task=Task(kind="regression"),
        metric=_Metric(),
        config=SearchConfig(max_rows=40),
        seed=0,
        evaluate=evaluate,
    )
    assert out.winner == "one"
    assert out.reason == "single_model"
    assert out.folds and not out.candidates


def test_model_ties_use_id_and_all_candidates_share_folds() -> None:
    seen: list[Sequence[Fold]] = []

    def evaluate(name: str, folds: Sequence[Fold], iterations: int) -> Candidate:
        seen.append(folds)
        return Candidate(name, 1.0)

    out = probe_models(
        ["z", "a"],
        [_fold()],
        y=np.arange(100.0),
        groups=None,
        task=Task(kind="regression"),
        metric=_Metric(),
        config=SearchConfig(max_rows=40),
        seed=0,
        evaluate=evaluate,
    )
    assert out.winner == "a"
    assert seen[0] is seen[1]


def test_budget_interruption_reports_fallback_and_skipped() -> None:
    allowance = iter([True, False])
    out = probe_models(
        ["z", "a"],
        [_fold()],
        y=np.arange(100.0),
        groups=None,
        task=Task(kind="regression"),
        metric=_Metric(),
        config=SearchConfig(max_rows=40),
        seed=0,
        evaluate=lambda name, folds, iterations: Candidate(name, 1.0),
        can_start=lambda: next(allowance),
    )
    assert out.winner == "z"
    assert out.reason == "incomplete_probe_budget"
    assert out.skipped == ("a",)


def test_model_failure_is_recorded_and_all_failures_raise() -> None:
    def evaluate(name: str, folds: Sequence[Fold], iterations: int) -> Candidate:
        if name.startswith("bad"):
            raise RuntimeError("probe failed")
        return Candidate(name, 2.0)

    out = probe_models(
        ["bad", "ok"],
        [_fold()],
        y=np.arange(100.0),
        groups=None,
        task=Task(kind="regression"),
        metric=_Metric(),
        config=SearchConfig(),
        seed=0,
        evaluate=evaluate,
    )
    assert out.winner == "ok"
    assert out.failures == (("bad", "probe failed"),)
    with pytest.raises(FitFailedError):
        probe_models(
            ["bad", "bad2"],
            [_fold()],
            y=np.arange(100.0),
            groups=None,
            task=Task(kind="regression"),
            metric=_Metric(),
            config=SearchConfig(),
            seed=0,
            evaluate=evaluate,
        )


def test_recipe_prefilter_and_rankers_never_see_confirmation_rows() -> None:
    x = np.repeat(np.arange(100.0)[:, None], 8, axis=1)
    y = np.arange(100.0)
    rankers = [_Ranker(), _Ranker(), _Ranker()]
    prefilter = _Ranker()
    evaluated: list[tuple[int, ...]] = []

    def evaluate(indices: tuple[int, ...], folds: Sequence[Fold]) -> Candidate:
        evaluated.append(indices)
        np.testing.assert_array_equal(folds[0].test_idx, np.arange(80, 100))
        return Candidate("model", 1.0, n_features=len(indices))

    out = scout_feature_recipes(
        [(str(i), ranker) for i, ranker in enumerate(rankers)],
        x,
        y,
        [_fold()],
        categorical=np.zeros(8, dtype=bool),
        fs_config=FeatureSelectionConfig(refine=False),
        search_config=SearchConfig(max_features=4),
        metric=_Metric(),
        task=Task(kind="regression"),
        fit_predict=_fit,
        prefilter=prefilter,
        seed=0,
        evaluate=evaluate,
    )
    assert out.winner == "0"
    assert out.subset == (0, 1)
    assert len(out.candidates) == 4
    assert len(evaluated) == 2
    assert prefilter.rows[0].shape == (60, 8)
    for ranker in rankers:
        assert ranker.rows[0].shape == (60, 4)
        assert ranker.targets[0].max() < 60


def test_recipe_regression_falls_back_to_wide_control() -> None:
    out = scout_feature_recipes(
        [("rank", _Ranker())],
        np.ones((100, 4)),
        np.ones(100),
        [_fold()],
        categorical=np.zeros(4, dtype=bool),
        fs_config=FeatureSelectionConfig(refine=False),
        search_config=SearchConfig(),
        metric=_Metric(),
        task=Task(kind="regression"),
        fit_predict=_fit,
        prefilter=_Ranker(),
        seed=0,
        evaluate=lambda indices, folds: Candidate("model", float(len(indices))),
    )
    assert out.winner is None
    assert out.subset == (0, 1, 2, 3)
    assert out.reason == "wide_control"


def test_recipe_without_es_discloses_no_inner_validation() -> None:
    fold = Fold(np.arange(80), np.empty(0, dtype=int), np.arange(80, 100))

    def evaluate(indices: tuple[int, ...], folds: Sequence[Fold]) -> Candidate:
        pytest.fail("recipe without independent inner validation must skip probes")

    out = scout_feature_recipes(
        [("rank", _Ranker())],
        np.ones((100, 4)),
        np.ones(100),
        [fold],
        categorical=np.zeros(4, dtype=bool),
        fs_config=FeatureSelectionConfig(refine=False),
        search_config=SearchConfig(),
        metric=_Metric(),
        task=Task(kind="regression"),
        fit_predict=_fit,
        prefilter=_Ranker(),
        seed=0,
        evaluate=evaluate,
    )
    assert out.winner is None
    assert out.reason == "no_inner_validation"


def test_loss_metric_prefers_lower_probe_score() -> None:
    metric = _Metric()
    metric.greater_is_better = False
    out = probe_models(
        ["high", "low"],
        [_fold()],
        y=np.arange(100.0),
        groups=None,
        task=Task(kind="regression"),
        metric=metric,
        config=SearchConfig(),
        seed=0,
        evaluate=lambda name, folds, iterations: Candidate(name, 0.1 if name == "low" else 0.9),
    )
    assert out.winner == "low"


def test_budget_after_control_prevents_prefilter_and_reports_skips() -> None:
    prefilter = _Ranker()
    allowance = iter([True, False])
    out = scout_feature_recipes(
        [("rank", _Ranker())],
        np.ones((100, 8)),
        np.ones(100),
        [_fold()],
        categorical=np.zeros(8, dtype=bool),
        fs_config=FeatureSelectionConfig(refine=False),
        search_config=SearchConfig(max_features=4),
        metric=_Metric(),
        task=Task(kind="regression"),
        fit_predict=_fit,
        prefilter=prefilter,
        seed=0,
        evaluate=lambda indices, folds: Candidate("model", 1.0),
        can_start=lambda: next(allowance),
    )
    assert out.winner is None
    assert out.reason == "probe_budget"
    assert out.skipped == ("rank",)
    assert not prefilter.rows


def test_class_cap_infeasibility_avoids_partial_class_probe() -> None:
    y = np.tile(np.arange(20), 5)
    bounded = bounded_probe_folds(
        [_fold()], y=y, task=Task(kind="multiclass"), config=SearchConfig(max_rows=32), seed=0
    )
    assert not bounded


def test_sequential_probe_stops_at_oof_fit_limit_without_recording_failure() -> None:
    from honestml.adapters.feature_selectors import SequentialSelector

    calls = 0

    def fit(
        x_tr: np.ndarray, y_tr: np.ndarray, x_te: np.ndarray, sw: np.ndarray | None, seed: int
    ) -> tuple[None, np.ndarray, None]:
        nonlocal calls
        calls += 1
        return None, np.ones(len(x_te)), None

    out = scout_feature_recipes(
        [("sequential", SequentialSelector(full_descent=True))],
        np.ones((100, 8)),
        np.ones(100),
        [_fold()],
        categorical=np.zeros(8, dtype=bool),
        fs_config=FeatureSelectionConfig(),
        search_config=SearchConfig(max_probe_fits=3),
        metric=_Metric(),
        task=Task(kind="regression"),
        fit_predict=fit,
        prefilter=_Ranker(),
        seed=0,
        evaluate=lambda indices, folds: Candidate("model", 1.0),
    )
    assert calls == 3
    assert out.winner is None
    assert out.reason == "incomplete_recipe_budget"
    assert out.skipped == ("sequential",)
    assert not out.failures
