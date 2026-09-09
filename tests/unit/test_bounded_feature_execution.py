"""Bounded FS samples only fold training rows and preserves aligned metadata."""

from __future__ import annotations

import numpy as np
import pytest

from honestml.application.feature_selection import (
    BoundedFeatureRanker,
    aggregate_scores,
    bounded_fit_predict,
    sample_training_rows,
)
from honestml.core import FeatureSelectionConfig, Fold, Task

pytestmark = pytest.mark.unit


def test_training_sample_preserves_proportions_and_rare_classes_without_prefix_bias() -> None:
    y = np.array(["common"] * 90 + ["rare"] * 10)
    idx = sample_training_rows(y, max_rows=20, task=Task(kind="binary"), random_state=6)
    assert len(idx) == len(np.unique(idx)) == 20
    np.testing.assert_array_equal(idx, np.sort(idx))
    assert np.count_nonzero(y[idx] == "rare") == 2
    assert idx[y[idx] == "common"].max() > 50
    assert idx[y[idx] == "rare"][0] != 90
    np.testing.assert_array_equal(
        idx, sample_training_rows(y, max_rows=20, task=Task(kind="binary"), random_state=6)
    )
    sparse = np.array([0] * 96 + [1, 2, 3, 4])
    sampled = sample_training_rows(sparse, max_rows=5, task=Task(kind="multiclass"), random_state=1)
    np.testing.assert_array_equal(np.unique(sparse[sampled]), np.arange(5))


def test_infeasible_class_cap_fails_before_training() -> None:
    with pytest.raises(ValueError, match="classes"):
        sample_training_rows(np.arange(5), max_rows=4, task=Task(kind="multiclass"), random_state=0)


def test_unbounded_training_sample_keeps_every_row() -> None:
    np.testing.assert_array_equal(
        sample_training_rows(
            np.arange(5), max_rows=8, task=Task(kind="regression"), random_state=9
        ),
        np.arange(5),
    )


class _AlignedRanker:
    name = "aligned"

    def __init__(self) -> None:
        self.rows: list[np.ndarray] = []

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
        rows = x[:, 0].astype(int)
        self.rows.append(rows)
        np.testing.assert_array_equal(y, rows % 2)
        np.testing.assert_array_equal(sample_weight, rows + 1)
        np.testing.assert_array_equal(groups, rows // 4)
        return np.ones(x.shape[1])

    def auto_threshold(self, n_features: int) -> float:
        return 1 / n_features


def test_bounded_ranker_keeps_each_original_fold_and_aligned_training_metadata() -> None:
    rows = np.arange(60)
    folds = [Fold(rows[:24], rows[24:28], rows[32:40]), Fold(rows[:40], rows[40:44], rows[48:56])]
    spy = _AlignedRanker()
    wrapped = BoundedFeatureRanker(spy, task=Task(kind="binary"), max_rows=12)
    assert wrapped.name == spy.name and wrapped.auto_threshold(2) == 0.5
    aggregate_scores(
        rows[:, None].astype(float),
        rows % 2,
        folds,
        ranker=wrapped,
        categorical=np.array([False]),
        config=FeatureSelectionConfig(),
        sample_weight=rows + 1,
        groups=rows // 4,
    )
    assert len(spy.rows) == len(folds)
    for seen, fold in zip(spy.rows, folds, strict=True):
        assert len(seen) == 12 and len(np.unique(seen)) == 12
        assert set(seen) <= set(np.union1d(fold.fit_idx, fold.es_idx))
        assert not set(seen) & set(fold.test_idx)
        assert seen.max() < fold.test_idx.min()
        assert not set(seen // 4) & set(fold.test_idx // 4)


def test_bounded_refinement_samples_only_train_and_keeps_all_test_rows() -> None:
    rows = np.arange(60)
    seen: list[np.ndarray] = []

    def predict(
        x: np.ndarray,
        y: np.ndarray,
        test: np.ndarray,
        weights: np.ndarray | None,
        random_state: int,
    ) -> tuple[None, np.ndarray, None]:
        ids = x[:, 0].astype(int)
        seen.append(ids)
        np.testing.assert_array_equal(y, ids % 2)
        np.testing.assert_array_equal(weights, ids + 1)
        np.testing.assert_array_equal(test[:, 0], rows[40:])
        return None, test[:, 0], None

    bounded = bounded_fit_predict(predict, task=Task(kind="binary"), max_rows=12)
    _, pred, _ = bounded(rows[:40, None], rows[:40] % 2, rows[40:, None], rows[:40] + 1, 3)
    assert len(seen[0]) == 12 and seen[0].max() < 40
    np.testing.assert_array_equal(pred, rows[40:])


def test_bounded_null_ranking_and_refinement_record_actual_resource_caps() -> None:
    from honestml.adapters import NullImportanceRanker, make_ranker_fit_predict
    from honestml.core import RunContext

    task = Task(kind="binary")
    ctx = RunContext()
    native = NullImportanceRanker(task, n_runs=3)
    native.set_threads(1)
    native.set_ranker_iterations(3)
    native.set_run_context(ctx)
    ranker = BoundedFeatureRanker(native, task=task, max_rows=12)
    rows = np.arange(60)
    x = np.column_stack((rows, np.sin(rows)))
    folds = [Fold(rows[:24], rows[24:28], rows[32:40]), Fold(rows[:40], rows[40:44], rows[48:56])]
    aggregate_scores(
        x,
        rows % 2,
        folds,
        ranker=ranker,
        categorical=np.zeros(2, dtype=bool),
        config=FeatureSelectionConfig(refine=False),
        ctx=ctx,
    )
    predict = bounded_fit_predict(
        make_ranker_fit_predict(task, threads=1, ctx=ctx, n_estimators=3), task=task, max_rows=12
    )
    proba, pred, _ = predict(x[:40], rows[:40] % 2, x[40:], None, 0)
    assert proba is not None and proba.shape == (20, 2) and len(pred) == 20
    fits = ctx.cost_report()["work"]
    assert len(fits) == 2 * (1 + 3) + 1
    assert all(w["rows"] == 12 and w["tree_budget"] == w["iterations"] == 3 for w in fits)
    assert all(w["status"] == "completed" for w in fits)
