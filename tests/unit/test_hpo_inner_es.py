"""Inner HPO validation preserves the evaluation and outer holdout partitions."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pytest

from honestml.adapters import Reader
from honestml.application.slice import EstimatorFactory
from honestml.composition import build as build_module
from honestml.composition import build_default_components
from honestml.core import CVConfig, Estimator, HPOConfig, Task

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("kind", "scheme"),
    [
        ("binary", "stratified"),
        ("regression", "kfold"),
        ("binary", "group"),
        ("regression", "group"),
        ("regression", "timeseries"),
        ("regression", "timeseries_period"),
    ],
)
def test_inner_es_stays_inside_original_training_partition(
    monkeypatch: pytest.MonkeyPatch, kind: str, scheme: str
) -> None:
    task = Task(kind=kind)
    rows = np.arange(120)
    y = rows % 2 if task.is_classification else rows.astype(float)
    groups = np.repeat(np.arange(30), 4) if scheme == "group" else None
    times = rows if scheme.startswith("timeseries") else None
    full = Reader(task).read(rows[:, None].astype(float), y, groups=groups, time=times)
    dev = full.take(rows[:96])
    cfg = CVConfig(
        scheme=scheme,
        n_splits=3,
        n_test=1 if scheme == "timeseries_period" else 8,
        n_es=5,
        purge=2 if times is not None else 0,
        embargo=1 if times is not None else 0,
        period="delta" if scheme == "timeseries_period" else None,
        period_size=4 if scheme == "timeseries_period" else None,
    )
    options = dict(
        random_state=7,
        models=("baseline",),
        cv=cfg,
        hpo=HPOConfig(n_trials=1, inner_cv=3),
        has_group=groups is not None,
        has_time=times is not None,
    )
    monkeypatch.setattr(build_module, "_any_early_stopping", lambda estimators: False)
    plain = build_default_components(task, **options)
    monkeypatch.setattr(build_module, "_any_early_stopping", lambda estimators: True)
    enabled = build_default_components(task, **options)
    assert plain.inner_splitter is not None and enabled.inner_splitter is not None
    plain_folds = list(plain.inner_splitter.split(dev))
    es_folds = list(enabled.inner_splitter.split(dev))
    assert len(plain_folds) == len(es_folds) == 3
    for original, carved in zip(plain_folds, es_folds, strict=True):
        assert carved.es_idx.size > 0
        np.testing.assert_array_equal(carved.test_idx, original.test_idx)
        np.testing.assert_array_equal(
            np.union1d(carved.fit_idx, carved.es_idx),
            np.union1d(original.fit_idx, original.es_idx),
        )
        for a, b in (
            (carved.fit_idx, carved.es_idx),
            (carved.fit_idx, carved.test_idx),
            (carved.es_idx, carved.test_idx),
        ):
            assert np.intersect1d(a, b).size == 0
            assert not set(a) & set(rows[96:])
            if groups is not None:
                assert not set(groups[a]) & set(groups[b])
        if times is not None:
            np.testing.assert_array_equal(carved.fit_idx, original.fit_idx)
            np.testing.assert_array_equal(carved.es_idx, original.es_idx)
            assert times[carved.fit_idx].max() < times[carved.es_idx].min()
            assert times[carved.es_idx].max() < times[carved.test_idx].min()


def test_non_es_hpo_keeps_iid_training_rows() -> None:
    task = Task(kind="regression")
    ds = Reader(task).read(np.arange(60)[:, None].astype(float), np.arange(60).astype(float))
    components = build_default_components(
        task, random_state=3, models=("linear",), hpo=HPOConfig(n_trials=1, inner_cv=3)
    )
    assert components.inner_splitter is not None
    folds = list(components.inner_splitter.split(ds))
    assert len(folds) == 3
    assert all(not f.es_idx.size and len(f.fit_idx) == 40 for f in folds)


def test_real_hpo_stops_on_inner_es_and_refits_all_dev_rows() -> None:
    pytest.importorskip("lightgbm")
    from honestml.application.slice import refit_best
    from honestml.application.tuning import tune_estimators
    from honestml.core import RunContext
    from honestml.core.ports.estimator import SupportsIterationBudget, SupportsThreadLimit

    task = Task(kind="regression")
    rng = np.random.default_rng(8)
    ds = Reader(task).read(rng.normal(size=(120, 3)), np.zeros(120))
    components = build_default_components(
        task, random_state=8, models=("lightgbm",), hpo=HPOConfig(n_trials=1, inner_cv=3)
    )
    assert components.make_factory is not None and components.tuner is not None
    assert components.inner_splitter is not None
    original = components.make_factory

    def make_factory(name: str, params: Mapping[str, Any]) -> EstimatorFactory:
        base = original(name, params)

        def make() -> Estimator:
            est = base()
            assert isinstance(est, SupportsThreadLimit)
            est.set_threads(1)
            return est

        return make

    ctx = RunContext()
    outcome = tune_estimators(
        ds,
        task,
        tunable={"lightgbm": {"n_estimators": {"type": "categorical", "choices": [100]}}},
        make_factory=make_factory,
        tuner=components.tuner,
        metric=components.metric,
        policy=components.policy,
        inner_splitter=components.inner_splitter,
        n_trials=1,
        timeout_s=None,
        random_state=8,
        ctx=ctx,
    )["lightgbm"]
    fits = ctx.cost_report()["work"]
    assert outcome.successful_trials == 1 and len(fits) == 3
    assert all(row["stage"] == "hpo" and row["rows"] == 72 for row in fits)
    assert all(0 < row["iterations"] < row["tree_budget"] == 100 for row in fits)
    rounds = int(np.median([row["iterations"] for row in fits]))
    refitted = refit_best(
        ds, task, factory=make_factory("lightgbm", outcome.best_params), iterations=rounds, ctx=ctx
    )
    assert isinstance(refitted, SupportsIterationBudget)
    assert refitted.fitted_iterations == rounds
    assert ctx.cost_report()["work"][-1]["rows"] == 120
