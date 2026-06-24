"""M2-5: the sklearn-compatible AutoML facade (ADR-0011)."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.datasets import make_classification, make_regression
from sklearn.pipeline import Pipeline

from honestml import AutoML
from honestml.composition.artifact import load_artifact, save_artifact
from honestml.core import (
    BudgetConfig,
    ConfigError,
    CVConfig,
    FeatureSelectionConfig,
    FEConfig,
    NotFittedError,
    SchemaValidationError,
    Task,
)

pytestmark = pytest.mark.unit


def _data_with_cats(n: int = 160, seed: int = 0, *, classes: int = 2):
    rng = np.random.default_rng(seed)
    cat1 = rng.choice(["a", "b", "c"], size=n)
    df = pd.DataFrame(
        {
            "num1": rng.normal(size=n),
            "num2": rng.normal(size=n),
            "cat1": cat1,
            "cat2": rng.choice(["x", "y"], size=n),
        }
    )
    signal = df["num1"].to_numpy() + (cat1 == "a").astype(float)
    if classes == 2:
        y = (signal > signal.mean()).astype(int)
    else:
        y = np.digitize(signal, np.quantile(signal, [1 / 3, 2 / 3]))
    return df, y


def _data(n: int = 120, seed: int = 0):
    return make_classification(
        n_samples=n, n_features=6, n_informative=4, n_redundant=0, random_state=seed
    )


def _data_multiclass(n: int = 150, seed: int = 0):
    return make_classification(
        n_samples=n,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        n_classes=3,
        random_state=seed,
    )


def _data_regression(n: int = 120, seed: int = 0):
    return make_regression(n_samples=n, n_features=6, n_informative=4, random_state=seed)


def test_get_set_params_round_trip() -> None:
    m = AutoML(task="binary", random_state=7, cv=3)
    assert m.get_params()["random_state"] == 7
    m.set_params(random_state=9)
    assert m.random_state == 9


def test_clone_preserves_params() -> None:
    m = AutoML(task="binary", cv=3, models=("linear",), random_state=5)
    c = clone(m)
    assert c.get_params() == m.get_params()


def test_init_does_not_compute() -> None:
    # sklearn invariant: __init__ stores params verbatim, no fitted attributes
    m = AutoML(task="binary")
    assert not hasattr(m, "classes_")
    assert not hasattr(m, "fitted_")


def test_fit_predict_surface() -> None:
    X, y = _data()
    m = AutoML(task="binary", random_state=0).fit(X, y)
    assert m.predict(X).shape == (len(y),)
    proba = m.predict_proba(X)
    assert proba.shape == (len(y), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert set(m.classes_.tolist()) == {0, 1}
    assert m.n_features_in_ == 6
    # baseline+linear always; boosting adds rows when its extra is installed
    assert len(m.leaderboard_) >= 2
    assert {"baseline", "linear"} <= {e.model_id for e in m.leaderboard_}
    assert m.leaderboard_[0].rank == 1


def test_works_in_pipeline() -> None:
    X, y = _data()
    pipe = Pipeline([("clf", AutoML(task="binary", random_state=0))])
    pipe.fit(X, y)
    assert pipe.predict(X).shape == (len(y),)


def test_predict_before_fit_raises() -> None:
    X, _ = _data()
    with pytest.raises(NotFittedError):
        AutoML(task="binary").predict(X)


def test_score_returns_float() -> None:
    X, y = _data()
    m = AutoML(task="binary", random_state=0).fit(X, y)
    assert isinstance(m.score(X, y), float)


def test_metric_param_drives_leaderboard() -> None:
    X, y = _data()
    m = AutoML(task="binary", metric="accuracy", random_state=0).fit(X, y)
    assert m.leaderboard_[0].metric == "accuracy"


def test_non_greatest_positive_label_does_not_invert_score() -> None:
    """F111 e2e: positive_label set to the non-greatest class must not invert the OOF roc_auc.

    The use-case feeds P(positive); without orientation the metric would read it as P(greatest)
    and a separable problem would collapse to ~0 instead of ~1.
    """
    X, y = make_classification(n_samples=200, n_features=6, n_informative=4, random_state=0)
    task = Task(kind="binary", positive_label=0)  # 0 is NOT the greatest label in {0, 1}
    m = AutoML(task=task, metric="roc_auc", random_state=0).fit(X, y)
    # non-inversion is the point: a flipped P(positive) would collapse to ~0.2; the exact value
    # drifts with numpy/BLAS across versions, so assert clearly-not-inverted, not a tight bound
    assert m.leaderboard_[0].score > 0.7


def test_period_carve_empty_dev_raises_configerror() -> None:
    """F102: when the holdout + purge consume every earlier period, the empty dev fails with a clear
    ConfigError instead of a raw numpy ValueError deep in the period splitter."""
    rng = np.random.default_rng(0)
    n = 10
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    y = rng.normal(size=n)
    times = pd.to_datetime(
        [
            "2021-01-15",
            "2021-01-20",
            "2021-02-15",
            "2021-02-20",
            "2021-03-15",
            "2021-03-20",
            "2021-04-15",
            "2021-04-20",
            "2021-05-15",
            "2021-05-20",
        ]
    ).to_numpy()
    cv = CVConfig(
        scheme="timeseries_period", period="month", outer_holdout=0.2, purge=4, n_splits=2, n_test=1
    )
    with pytest.raises(
        ConfigError, match="no dev rows remain after the period outer_holdout carve"
    ):
        AutoML(task="regression", cv=cv, models=("linear",), random_state=0).fit(X, y, time=times)


def test_deterministic_leaderboard_across_fits() -> None:
    X, y = _data()
    a = AutoML(task="binary", random_state=0).fit(X, y).leaderboard_
    b = AutoML(task="binary", random_state=0).fit(X, y).leaderboard_
    assert [(e.model_id, e.score) for e in a] == [(e.model_id, e.score) for e in b]


# --- M5a-wire: public budget API + graceful degradation (FR-M5-1/2/3, ADR-0032) ---


def test_budget_param_caps_run() -> None:
    X, y = _data()
    m = AutoML(task="binary", random_state=0, budget=BudgetConfig(mode="trials", n_trials=1)).fit(
        X, y
    )
    assert len(m.leaderboard_) == 1  # exactly one completed candidate
    assert m.predict(X).shape == (len(y),)  # graceful degradation: a working model is shipped


def test_no_budget_unbounded_mode_none() -> None:
    X, y = _data()
    m = AutoML(task="binary", random_state=0).fit(X, y)  # budget=None -> unbounded
    assert len(m.leaderboard_) >= 2  # all default candidates run (M3/M4 behavior unchanged)


def test_clone_preserves_budget() -> None:
    m = AutoML(task="binary", budget=BudgetConfig(mode="trials", n_trials=3), random_state=5)
    c = clone(m)
    assert c.get_params()["budget"] == BudgetConfig(mode="trials", n_trials=3)


def test_degraded_refit_runs_predict_works() -> None:
    """A budget-degraded run still refits the winner (not gated) so predict works."""
    X, y = _data()
    m = AutoML(task="binary", random_state=0, budget=BudgetConfig(mode="trials", n_trials=1)).fit(
        X, y
    )
    assert m.predict_proba(X).shape == (len(y), 2)
    assert isinstance(m.score(X, y), float)


def test_float_budget_is_time_seconds() -> None:
    X, y = _data()
    # a generous time budget does not cap a fast run -> all candidates complete
    m = AutoML(task="binary", random_state=0, budget=1000.0).fit(X, y)
    assert len(m.leaderboard_) >= 2


# --- M5c: public significance toggle (FR-M5-5, ADR-0034) ---


def test_significance_off_pure_argmax() -> None:
    X, y = _data()
    m = AutoML(task="binary", random_state=0, significance="off").fit(X, y)
    # off -> degenerate lone-anchor band: just the winner, no tie-break, no instability
    assert m.band_member_ids_ == (m.best_model_id_,)
    assert m.band_width_ == 1
    assert m.winner_by_tiebreak_ is False


def test_significance_default_band_unchanged() -> None:
    X, y = _data()
    # default honest-on must not change the M4 surface: a winner and a band are exposed
    m = AutoML(task="binary", random_state=0).fit(X, y)
    assert m.best_model_id_ in {e.model_id for e in m.leaderboard_}
    assert m.best_model_id_ in m.band_member_ids_


def test_clone_preserves_significance() -> None:
    m = AutoML(task="binary", significance="off", random_state=5)
    c = clone(m)
    assert c.get_params()["significance"] == "off"
    c.set_params(significance="bootstrap")
    assert c.significance == "bootstrap"


# --- M5b: tracker-independent run report (FR-M5-4/6, NFR-M5-6, ADR-0033) ---


def test_run_report_attr_after_fit() -> None:
    import json

    X, y = _data()
    m = AutoML(task="binary", random_state=0).fit(X, y)
    report = m.run_report_
    assert report["winner"] == m.best_model_id_
    assert set(report["timings"]["run"]) == {"selection", "refit"}  # both stages timed
    assert report["budget"]["mode"] == "none"  # unbounded by default
    assert report["significance"] == "bootstrap"
    json.dumps(report)  # serializable without a tracker (FR-M5-4)


def test_run_report_significance_mode() -> None:
    X, y = _data()
    m = AutoML(task="binary", random_state=0, significance="off").fit(X, y)
    assert m.run_report_["significance"] == "off"  # truthful resolved mode (NFR-M5-6)


def test_run_report_budget_outcome_truthful() -> None:
    X, y = _data()
    m = AutoML(task="binary", random_state=0, budget=BudgetConfig(mode="trials", n_trials=1)).fit(
        X, y
    )
    budget = m.run_report_["budget"]
    assert budget["mode"] == "trials"
    assert budget["exhausted"] is True  # degraded run is visible, not implied
    assert len(budget["skipped"]) >= 1


def test_save_run_report_from_fit(tmp_path) -> None:
    import json

    from honestml import save_run_report

    X, y = _data()
    m = AutoML(task="binary", random_state=0).fit(X, y)
    out = save_run_report(m.run_report_, tmp_path)
    assert json.loads(out.read_text(encoding="utf-8"))["winner"] == m.best_model_id_


def test_budget_off_refinement_combined() -> None:
    """FR-M5-2(4): a budget-degraded run with significance='off' AND selection='refinement'
    returns best-so-far; the run report is truthful and the resolved selection mode reflects the
    completed subset (refinement falls back to raw with <2 completed candidates)."""
    X, y = _data()
    m = AutoML(
        task="binary",
        random_state=0,
        significance="off",
        cv=CVConfig(selection="refinement", scheme="stratified", n_splits=3),
        budget=BudgetConfig(mode="trials", n_trials=1),
    ).fit(X, y)
    assert m.predict(X).shape == (len(y),)  # best-so-far winner refit (not budget-gated) works
    report = m.run_report_
    assert report["budget"] == {
        "mode": "trials",
        "exhausted": True,
        "skipped": report["budget"]["skipped"],
        "exhausted_by": "trials",
    }
    assert len(report["budget"]["skipped"]) >= 1
    assert report["significance"] == "off"
    assert (
        report["config"]["cv"]["selection"] == "refinement"
    )  # requested config recorded truthfully
    assert m.selection_mode_ == "raw"  # resolved over the completed subset (<2 -> raw fallback)


# --- M5-resume RC-d1: public cache/resume API + fingerprint scoping (FR-RC-4, ADR-0037 §1) ---


def test_cache_none_default_unchanged() -> None:
    X, y = _data()
    m = AutoML(task="binary", random_state=0).fit(X, y)  # cache=None (default)
    assert m.predict(X).shape == (len(y),)
    assert m.get_params()["cache"] is None  # default is off (M5 behavior)


def test_cache_dir_persists_and_reuse_is_identical(tmp_path) -> None:
    X, y = _data()
    cdir = tmp_path / "cache"
    m1 = AutoML(task="binary", random_state=0, cache=str(cdir)).fit(X, y)
    assert any(cdir.rglob("entry.joblib"))  # candidates persisted durably
    m2 = AutoML(task="binary", random_state=0, cache=cdir).fit(
        X, y
    )  # same dir + fp -> reuse/resume
    np.testing.assert_array_equal(m1.predict(X), m2.predict(X))
    assert [(e.model_id, e.score) for e in m1.leaderboard_] == [
        (e.model_id, e.score) for e in m2.leaderboard_
    ]


def test_cache_fingerprint_invalidates_on_config_change(tmp_path, caplog) -> None:
    """A changed config -> a different fingerprint subdir -> no stale reuse (FR-RC-5);
    the cold run next to the old fingerprint is named in the log (F4.7)."""
    X, y = _data()
    cdir = tmp_path / "cache"
    AutoML(task="binary", random_state=0, cache=str(cdir)).fit(X, y)
    with caplog.at_level("INFO", logger="honestml"):
        AutoML(task="binary", random_state=1, cache=str(cdir)).fit(X, y)  # different seed
    subdirs = [p for p in cdir.iterdir() if p.is_dir()]
    assert len(subdirs) == 2  # two distinct fingerprints -> two subdirs (auto-invalidation)
    assert any("other fingerprint" in r.message for r in caplog.records)  # F4.7 diagnosis


def test_clone_preserves_cache() -> None:
    m = AutoML(task="binary", cache="some/dir", random_state=5)
    c = clone(m)
    assert c.get_params()["cache"] == "some/dir"
    c.set_params(cache=None)
    assert c.cache is None


def test_cache_works_in_pipeline(tmp_path) -> None:
    X, y = _data()
    pipe = Pipeline([("clf", AutoML(task="binary", random_state=0, cache=str(tmp_path / "c")))])
    pipe.fit(X, y)
    assert pipe.predict(X).shape == (len(y),)


def test_cache_accepts_str_and_path(tmp_path) -> None:
    X, y = _data()
    AutoML(task="binary", random_state=0, cache=str(tmp_path / "s")).fit(X, y)
    AutoML(task="binary", random_state=0, cache=tmp_path / "p").fit(X, y)  # both coerce via Path


def test_run_report_cache_outcome(tmp_path) -> None:
    """run_report carries the fingerprint + truthful cache outcome across a fresh run then a reuse."""
    X, y = _data()
    cdir = tmp_path / "c"
    m1 = AutoML(task="binary", random_state=0, cache=str(cdir)).fit(X, y)
    r1 = m1.run_report_
    assert isinstance(r1["run_fingerprint"], str) and len(r1["run_fingerprint"]) == 64
    ids = {e.model_id for e in m1.leaderboard_}
    assert r1["cache"]["enabled"] is True
    assert (
        set(r1["cache"]["computed"]) == ids and r1["cache"]["reused"] == []
    )  # first run: all fresh
    m2 = AutoML(task="binary", random_state=0, cache=str(cdir)).fit(X, y)
    r2 = m2.run_report_
    assert (
        set(r2["cache"]["reused"]) == ids and r2["cache"]["computed"] == []
    )  # second run: all reused
    assert (
        r2["run_fingerprint"] == r1["run_fingerprint"]
    )  # identical inputs -> identical fingerprint


def test_run_report_cache_disabled_default() -> None:
    X, y = _data()
    m = AutoML(task="binary", random_state=0).fit(X, y)  # cache=None
    assert m.run_report_["cache"] == {"enabled": False, "reused": [], "computed": []}
    assert len(m.run_report_["run_fingerprint"]) == 64  # computed even when the cache is unused


# --- M5 run-modes: public run_mode stage-gate (selection/full) (ADR-0038, FR-RM-1/2/3) ---


def test_run_mode_full_default_unchanged() -> None:
    X, y = _data()
    m = AutoML(task="binary", random_state=0).fit(X, y)  # run_mode defaults to "full"
    assert m.get_params()["run_mode"] == "full"
    assert hasattr(m, "fitted_") and m.predict(X).shape == (len(y),)  # ships a model (M5 unchanged)


def test_clone_preserves_run_mode() -> None:
    m = AutoML(task="binary", run_mode="selection", random_state=5)
    c = clone(m)
    assert c.get_params()["run_mode"] == "selection"
    c.set_params(run_mode="full")
    assert c.run_mode == "full"


def test_invalid_run_mode_raises_configerror() -> None:
    X, y = _data()
    with pytest.raises(ConfigError, match="run_mode"):
        AutoML(task="binary", run_mode="evaluation", random_state=0).fit(X, y)  # type: ignore[arg-type]


def test_selection_no_refit(monkeypatch: pytest.MonkeyPatch) -> None:
    """selection runs only run_slice: refit_best / _calibrate_winner are never called (FR-RM-2)."""
    import honestml.composition.facade as facade_mod

    refit_calls: list[int] = []
    calib_calls: list[int] = []
    real_refit = facade_mod.refit_best
    monkeypatch.setattr(
        facade_mod, "refit_best", lambda *a, **k: refit_calls.append(1) or real_refit(*a, **k)
    )
    monkeypatch.setattr(
        AutoML, "_calibrate_winner", lambda *a, **k: (calib_calls.append(1), (None, None))[1]
    )
    X, y = _data()
    m = AutoML(task="binary", run_mode="selection", random_state=0).fit(X, y)
    assert refit_calls == [] and calib_calls == []  # no post-selection stage ran
    assert hasattr(m, "leaderboard_") and hasattr(m, "best_model_id_")  # leaderboard built
    assert not hasattr(m, "fitted_") and not hasattr(m, "best_estimator_")  # no model shipped


def test_selection_sets_describing_attrs_not_model() -> None:
    """selection sets describing-input sklearn attrs + band observability, not the fitted model."""
    import pandas as pd

    X, y = _data()
    Xdf = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    m = AutoML(task="binary", run_mode="selection", random_state=0).fit(Xdf, y)
    assert m.n_features_in_ == 6 and set(m.classes_.tolist()) == {0, 1}
    assert list(m.feature_names_in_) == list(Xdf.columns)
    assert m.best_model_id_ in set(m.band_member_ids_)  # band observability present
    for absent in (
        "fitted_",
        "best_estimator_",
        "calibration_",
        "reliability_curve_",
        "holdout_score_",
    ):
        assert not hasattr(m, absent)


def test_selection_predict_raises_with_hint() -> None:
    X, y = _data()
    m = AutoML(task="binary", run_mode="selection", random_state=0).fit(X, y)
    with pytest.raises(NotFittedError, match="selection"):
        m.predict(X)
    with pytest.raises(NotFittedError):
        m.score(X, y)


def test_selection_report_truthful() -> None:
    X, y = _data()
    m = AutoML(task="binary", run_mode="selection", random_state=0).fit(X, y)
    assert m.run_report_["winner"] == m.best_model_id_
    assert (
        m.run_report_["config"]["run_mode"] == "selection"
    )  # facade threaded run_mode into RunConfig


def test_selection_deterministic() -> None:
    X, y = _data()
    a = AutoML(task="binary", run_mode="selection", random_state=0).fit(X, y).leaderboard_
    b = AutoML(task="binary", run_mode="selection", random_state=0).fit(X, y).leaderboard_
    assert [(e.model_id, e.score) for e in a] == [(e.model_id, e.score) for e in b]


def test_full_run_mode_in_report() -> None:
    X, y = _data()
    m = AutoML(task="binary", random_state=0).fit(X, y)
    assert m.run_report_["config"]["run_mode"] == "full"  # default mode is truthful in the manifest


# --- M5 memory-enforce at the facade boundary (ADR-0039 §3, NFR-RM-4) ---


def test_memory_limit_requires_psutil(monkeypatch: pytest.MonkeyPatch) -> None:
    """A memory limit with the default probe imports psutil when building the budget; absent -> raise."""
    import builtins

    from honestml.core import MissingDependencyError

    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        if name == "psutil":
            raise ImportError("No module named 'psutil'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    X, y = _data()
    with pytest.raises(MissingDependencyError, match="memory"):
        AutoML(task="binary", random_state=0, budget=BudgetConfig(memory_limit_mb=100)).fit(X, y)


def test_default_run_report_memory_outcome() -> None:
    X, y = _data()
    m = AutoML(task="binary", random_state=0).fit(X, y)  # no memory limit
    assert m.run_report_["config"]["budget"]["memory_limit_mb"] is None  # input in config dump
    assert m.run_report_["budget"]["exhausted_by"] is None  # no degradation


# --- M3b: multiclass + regression end-to-end via the facade (ADR-0020/0021/0024) ---


def test_multiclass_fit_predict_surface() -> None:
    X, y = _data_multiclass()
    m = AutoML(task="multiclass", random_state=0).fit(X, y)
    assert m.predict(X).shape == (len(y),)
    proba = m.predict_proba(X)
    assert proba.shape == (len(y), 3)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert set(m.classes_.tolist()) == {0, 1, 2}
    assert isinstance(m.score(X, y), float)
    assert len(m.leaderboard_) >= 2  # baseline + linear (+ boosting when installed)


def test_regression_no_classes_no_proba() -> None:
    X, y = _data_regression()
    # regression default CV is kfold (M3c); use holdout for the M3b end-to-end path
    m = AutoML(task="regression", cv=CVConfig(scheme="holdout"), random_state=0).fit(X, y)
    assert not hasattr(m, "classes_")  # ADR-0020 §4: no classification attributes
    assert m.predict(X).shape == (len(y),)
    assert isinstance(m.score(X, y), float)
    with pytest.raises(SchemaValidationError, match="no probabilities"):
        m.predict_proba(X)


def test_regression_default_kfold_fits() -> None:
    # M3c: regression default CV (kfold) is now implemented -> the default path fits
    X, y = _data_regression()
    m = AutoML(task="regression", random_state=0).fit(X, y)
    assert m.predict(X).shape == (len(y),)
    assert isinstance(m.score(X, y), float)


# --- C4 honest CV-selection via the facade (ADR-0016, FR-3) -----------------


def test_cv_config_holdout_fits() -> None:
    X, y = _data()
    m = AutoML(task="binary", cv=CVConfig(scheme="holdout"), random_state=0).fit(X, y)
    assert m.predict(X).shape == (len(y),)
    assert len(m.leaderboard_) >= 2


def test_cv_config_timeseries_without_time_fails_fast() -> None:
    X, y = _data()
    with pytest.raises(ConfigError, match="requires a time column"):
        AutoML(task="binary", cv=CVConfig(scheme="timeseries"), random_state=0).fit(X, y)


def test_timeseries_end_to_end_from_fit() -> None:
    # M4b (FR-M4-6): declare the time axis via fit(time=) -> timeseries CV runs end-to-end
    X, y = _data(n=200)
    t = np.arange(len(y))  # row order is the time order here
    m = AutoML(
        task="binary", cv=CVConfig(scheme="timeseries", n_splits=3, n_test=20), random_state=0
    ).fit(X, y, time=t)
    assert m.predict(X).shape == (len(y),)  # inference needs no time
    assert len(m.leaderboard_) >= 2


def test_timeseries_period_end_to_end_from_fit() -> None:
    # ADR-0096: a datetime axis + scheme='timeseries_period' runs calendar-period CV end-to-end;
    # the manifest carries the truthful cv block (densified period counts), inference needs no time.
    X, y = _data(n=360)
    t = np.arange("2021-01-01", "2022-01-05", dtype="datetime64[D]")[: len(y)]  # ~12 months daily
    m = AutoML(
        task="binary",
        cv=CVConfig(scheme="timeseries_period", period="month", n_splits=3, n_test=2),
        random_state=0,
    ).fit(X, y, time=t)
    assert m.predict(X).shape == (len(y),)
    assert len(m.leaderboard_) >= 2
    cv = m.run_report_["cv"]
    assert cv is not None
    assert cv["period"] == "month" and cv["n_folds"] == 3
    assert cv["n_periods"] >= 7  # >= n_splits * n_test + 1
    assert m.run_report_["config"]["cv"]["scheme"] == "timeseries_period"


def test_timeseries_period_weighting_reports_periods_used() -> None:
    # ADR-0098 §4 (G7): weighting='period' runs end-to-end (leaderboard macro-by-period + period-block
    # significance) and the manifest cv block surfaces the mode + how many periods the average used.
    X, y = _data(n=360)
    t = np.arange("2021-01-01", "2022-01-05", dtype="datetime64[D]")[: len(y)]  # ~12 months daily
    m = AutoML(
        task="binary",
        cv=CVConfig(
            scheme="timeseries_period", period="month", n_splits=8, n_test=1, weighting="period"
        ),
        random_state=0,
    ).fit(X, y, time=t)
    assert m.predict(X).shape == (len(y),)
    cv = m.run_report_["cv"]
    assert cv["weighting"] == "period"
    assert isinstance(cv["n_periods_used"], int) and cv["n_periods_used"] >= 4
    assert m.run_report_["config"]["cv"]["weighting"] == "period"


def test_timeseries_period_without_time_fails_fast() -> None:
    X, y = _data()
    with pytest.raises(ConfigError, match="requires a time column"):
        AutoML(
            task="binary",
            cv=CVConfig(scheme="timeseries_period", period="month"),
            random_state=0,
        ).fit(X, y)


def test_clone_preserves_cvconfig_param() -> None:
    m = AutoML(task="binary", cv=CVConfig(scheme="holdout"), random_state=5)
    c = clone(m)
    assert c.get_params() == m.get_params()
    assert c.get_params()["cv"] == CVConfig(scheme="holdout")


# --- follow-up: public group-column API, groups= in fit (ADR-0025, FR-4b) ----


def _groups_for(y: np.ndarray, size: int = 4) -> np.ndarray:
    """Contiguous group labels (size rows each), row-aligned with X."""
    return np.arange(len(y)) // size


def test_group_cv_end_to_end_from_fit() -> None:
    X, y = _data()
    m = AutoML(task="binary", cv=CVConfig(scheme="group"), random_state=0).fit(
        X, y, groups=_groups_for(y)
    )
    assert m.predict(X).shape == (len(y),)  # inference needs no groups
    assert len(m.leaderboard_) >= 2


def test_group_cv_save_load_predict(tmp_path) -> None:
    # inference-safety holds across serialization: schema carries the GROUP role, predict needs none
    X, y = _data()
    m = AutoML(task="binary", cv=CVConfig(scheme="group"), random_state=0).fit(
        X, y, groups=_groups_for(y)
    )
    save_artifact(m.fitted_, tmp_path / "art")
    loaded = load_artifact(tmp_path / "art")
    np.testing.assert_array_equal(loaded.predict(X), m.predict(X))


def test_group_scheme_without_groups_raises_configerror() -> None:
    X, y = _data()
    with pytest.raises(ConfigError, match="group"):
        AutoML(task="binary", cv=CVConfig(scheme="group"), random_state=0).fit(X, y)


def test_groups_with_non_group_scheme_warns(caplog: pytest.LogCaptureFixture) -> None:
    X, y = _data()
    with caplog.at_level(logging.WARNING, logger="honestml"):
        AutoML(task="binary", cv=CVConfig(scheme="stratified"), random_state=0).fit(
            X, y, groups=_groups_for(y)
        )
    assert any("group" in r.getMessage().lower() for r in caplog.records)


def test_numpy_X_with_groups() -> None:
    X, y = _data(n=80)
    m = AutoML(task="binary", cv=CVConfig(scheme="group"), random_state=0).fit(
        X,
        y,
        groups=_groups_for(y).tolist(),  # list groups + numpy X
    )
    assert m.predict(X).shape == (len(y),)


def test_groups_does_not_change_n_features_in() -> None:
    X, y = _data()
    base = AutoML(task="binary", random_state=0).fit(X, y).n_features_in_
    grouped = (
        AutoML(task="binary", cv=CVConfig(scheme="group"), random_state=0)
        .fit(X, y, groups=_groups_for(y))
        .n_features_in_
    )
    assert grouped == base == 6


def test_fit_backward_compatible_without_groups() -> None:
    X, y = _data()
    m = AutoML(task="binary", random_state=0).fit(X, y)  # no groups kwarg
    assert m.predict(X).shape == (len(y),)
    m2 = AutoML(task="binary", random_state=0).fit(X, y, sample_weight=np.ones(len(y)))
    assert m2.predict(X).shape == (len(y),)


# --- M4a2: significance band ON by default + honesty observability (FR-M4-3/4, NFR-M4-7) ---


def test_facade_publishes_honesty_attrs() -> None:
    """After fit the band observability attributes are public (sklearn `*_` convention)."""
    X, y = _data()
    m = AutoML(task="binary", random_state=0).fit(X, y)
    assert m.best_model_id_ in set(m.band_member_ids_)  # winner is a band member
    assert isinstance(m.band_unstable_, bool)
    assert m.band_width_ == len(m.band_member_ids_) >= 1
    assert isinstance(m.winner_by_tiebreak_, bool)


def test_band_default_unambiguous_winner_unchanged() -> None:
    """A clear-signal dataset: the default band leaves the winner = the absolute argmax (rank 1)."""
    X, y = _data(n=200)
    m = AutoML(task="binary", random_state=0).fit(X, y)
    assert m.leaderboard_[0].rank == 1
    # an unambiguous winner is the rank-1 model and is not flagged unstable
    assert m.best_model_id_ == m.leaderboard_[0].model_id
    assert m.winner_by_tiebreak_ is False
    assert m.band_unstable_ is False


# --- M4c: honest-regime outer holdout (ADR-0029, FR-M4-9, NFR-M4-3/7) --------


def test_outer_holdout_off_default() -> None:
    """Default (outer_holdout=0.0): no holdout score, behavior unchanged (M3/M4a)."""
    X, y = _data()
    m = AutoML(task="binary", random_state=0).fit(X, y)
    assert m.holdout_score_ is None and m.fitted_.holdout_score is None


def test_outer_holdout_touched_once() -> None:
    """The untouched holdout is scored exactly once, on the holdout rows only (NFR-M4-3 one-touch)."""
    from honestml.composition.artifact import FittedModel

    X, y = _data(n=200)
    calls: list[int] = []
    original = FittedModel._score_dataset

    def spy(self, ds):  # type: ignore[no-untyped-def]
        calls.append(ds.n_rows)
        return original(self, ds)

    FittedModel._score_dataset = spy  # type: ignore[method-assign]
    try:
        m = AutoML(task="binary", cv=CVConfig(outer_holdout=0.25), random_state=0).fit(X, y)
    finally:
        FittedModel._score_dataset = original  # type: ignore[method-assign]

    assert calls == [50]  # scored once, on the 50-row holdout (0.25*200), never the 150-row dev
    assert isinstance(m.holdout_score_, float)
    assert m.fitted_.holdout_score == m.holdout_score_


def test_outer_holdout_too_small_raises() -> None:
    """A holdout too small to hold >= 2 rows per class fails fast (ADR-0029 §1)."""
    X, y = _data(n=120)
    with pytest.raises(ConfigError, match="outer_holdout"):
        AutoML(task="binary", cv=CVConfig(outer_holdout=0.02), random_state=0).fit(X, y)


def test_holdout_unbiased_vs_cv_optimism() -> None:
    """Selection optimism (the best-looking CV candidate) does not carry to the holdout (FR-M4-9)."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((300, 60))  # pure noise: no real signal to generalize
    y = rng.integers(0, 2, size=300)
    # multiple candidates -> the rank-1 model is selected by fold noise (argmax over candidates); its
    # CV-leaderboard advantage is inflated (Cawley-Talbot max-of-noisy-estimates), the once-touched
    # holdout is honest (~0.5). roc_auc amplifies the overfit gap.
    m = AutoML(task="binary", metric="roc_auc", cv=CVConfig(outer_holdout=0.3), random_state=0).fit(
        X, y
    )
    assert len(m.leaderboard_) >= 2  # a real argmax-over-candidates selection occurred
    assert m.leaderboard_[0].score > m.holdout_score_


def test_outer_holdout_single_class_window_raises() -> None:
    """An unstratified (timeseries) holdout window that is single-class fails fast (ADR-0029 §1)."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 6))
    y = np.concatenate(
        [np.tile([0, 1], 75), np.ones(50, dtype=int)]
    )  # dev mixed, late window all 1s
    t = np.arange(200)
    with pytest.raises(ConfigError, match="single-class"):
        AutoML(
            task="binary",
            cv=CVConfig(scheme="timeseries", n_splits=3, n_test=20, outer_holdout=0.25),
            random_state=0,
        ).fit(X, y, time=t)


# --- M6a feature engineering (ADR-0040/0041/0042) -------------------------


def test_fe_off_default_schema_like_m5() -> None:
    df, y = _data_with_cats()
    m = AutoML(task="binary", random_state=0).fit(df, y)
    assert m.schema_.target_encoding is None and m.schema_.frequency_encoding is None
    assert m.schema_.intersections is None and m.schema_.datetime_spec is None
    assert m.n_features_in_ == 4  # num1, num2, cat1, cat2 (no FE columns)
    assert m.predict(df).shape == (len(y),)


def test_clone_preserves_feature_engineering() -> None:
    fe = FEConfig(target_encoding=True, intersections=True)
    m = AutoML(task="binary", feature_engineering=fe, random_state=0)
    assert clone(m).get_params()["feature_engineering"] == fe


def test_invalid_feature_engineering_raises_configerror() -> None:
    df, y = _data_with_cats()
    with pytest.raises(ConfigError, match="feature_engineering"):
        AutoML(task="binary", feature_engineering="boost").fit(df, y)  # type: ignore[arg-type]


def test_fe_on_adds_features_and_predicts() -> None:
    df, y = _data_with_cats()
    fe = FEConfig(target_encoding=True, frequency_encoding=True, intersections=True)
    m = AutoML(task="binary", feature_engineering=fe, random_state=0).fit(df, y)
    # base 4 + cat{1,2}_freq + cat{1,2}_te + cat1__cat2 = 9
    assert m.n_features_in_ == 9
    assert m.schema_.target_encoding is not None and m.schema_.frequency_encoding is not None
    assert m.schema_.intersections is not None
    assert {"cat1_te", "cat1_freq", "cat1__cat2"} <= set(m.schema_.features)
    assert m.predict(df).shape == (len(y),)
    assert m.predict_proba(df).shape == (len(y), 2)


def test_te_multiclass_skips_with_warning(caplog: pytest.LogCaptureFixture) -> None:
    df, y = _data_with_cats(classes=3)
    fe = FEConfig(target_encoding=True, frequency_encoding=True)
    with caplog.at_level(logging.WARNING):
        m = AutoML(task="multiclass", feature_engineering=fe, random_state=0).fit(df, y)
    assert any("target encoding skipped" in r.message for r in caplog.records)
    assert m.schema_.target_encoding is None  # TE gracefully skipped (not a ConfigError)
    assert m.schema_.frequency_encoding is not None  # target-independent FE still applies
    # the resolved manifest reflects the effective (TE-off) config (ADR-0041 §4)
    assert m.run_report_["config"]["fe"]["target_encoding"] is False


def test_te_regression_skips_with_warning(caplog: pytest.LogCaptureFixture) -> None:
    # FR-FE-3 #4: regression + target_encoding -> graceful skip + WARNING (not ConfigError)
    df, y = _data_with_cats()
    rng = np.random.default_rng(1)
    yf = df["num1"].to_numpy() + (df["cat1"].to_numpy() == "a") + rng.normal(0, 0.1, len(df))
    fe = FEConfig(target_encoding=True, frequency_encoding=True)
    with caplog.at_level(logging.WARNING):
        m = AutoML(task="regression", feature_engineering=fe, random_state=0).fit(df, yf)
    assert any("target encoding skipped" in r.message for r in caplog.records)
    assert m.schema_.target_encoding is None  # TE skipped for regression
    assert m.schema_.frequency_encoding is not None  # target-independent FE still applies


def test_te_active_under_timeseries_expanding(caplog: pytest.LogCaptureFixture) -> None:
    # ADR-0082: under time-series CV the honest expanding-window OOF encoder runs (each fold from strictly
    # earlier folds), so TE is NOT skipped — the full-train spec ships and *_te columns exist.
    df, y = _data_with_cats(n=200)
    t = np.arange(len(y))
    fe = FEConfig(target_encoding=True, frequency_encoding=True)
    with caplog.at_level(logging.WARNING):
        m = AutoML(
            task="binary",
            cv=CVConfig(scheme="timeseries", n_splits=3, n_test=20),
            feature_engineering=fe,
            random_state=0,
        ).fit(df, y, time=t)
    assert not any(
        "target encoding skipped" in r.message and "time-series" in r.message
        for r in caplog.records
    )
    assert m.schema_.target_encoding is not None  # full-train TE spec ships (refit/inference)
    assert m.schema_.frequency_encoding is not None
    assert m.run_report_["config"]["fe"]["target_encoding"] is True
    assert any(f.endswith("_te") for f in m.schema_.features)  # TE columns present


def test_fe_changes_run_fingerprint() -> None:
    df, y = _data_with_cats()
    base = AutoML(task="binary", random_state=0).fit(df, y)
    with_fe = AutoML(
        task="binary", feature_engineering=FEConfig(frequency_encoding=True), random_state=0
    ).fit(df, y)
    assert base.run_report_["run_fingerprint"] != with_fe.run_report_["run_fingerprint"]
    assert with_fe.run_report_["config"]["fe"]["frequency_encoding"] is True


def test_fe_artifact_roundtrip_predicts_identically(tmp_path) -> None:  # noqa: ANN001
    df, y = _data_with_cats()
    fe = FEConfig(target_encoding=True, frequency_encoding=True, intersections=True)
    m = AutoML(task="binary", feature_engineering=fe, random_state=0).fit(df, y)
    save_artifact(m.fitted_, tmp_path)
    loaded = load_artifact(tmp_path)
    assert loaded.schema.target_encoding is not None  # FE specs travel in schema.json
    assert np.allclose(loaded.predict_proba(df), m.predict_proba(df))


# --- M6b feature selection (ADR-0043/0044/0045, FR-FS-1/4) -------------------


def test_fs_off_default_content_identical_to_m6a() -> None:
    # FR-FS-1: default (None) keeps the leaderboard content; no subset on the schema
    X, y = _data()
    base = AutoML(task="binary", models=("linear",), random_state=0).fit(X, y)
    off = AutoML(task="binary", models=("linear",), random_state=0, feature_selection=None).fit(
        X, y
    )
    assert off.schema_.selected_features is None
    assert [e.score for e in base.leaderboard_] == [e.score for e in off.leaderboard_]


def test_clone_preserves_feature_selection() -> None:
    cfg = FeatureSelectionConfig(strategy="random_probe", cutoff="auto")
    m = AutoML(task="binary", feature_selection=cfg, random_state=0)
    assert clone(m).get_params()["feature_selection"] == cfg


def test_invalid_feature_selection_raises_configerror() -> None:
    X, y = _data()
    with pytest.raises(ConfigError, match="feature_selection"):
        AutoML(task="binary", feature_selection="importance").fit(X, y)  # type: ignore[arg-type]


def test_fs_on_reduces_features_and_predicts() -> None:
    # FR-FS-1/4: selection trims the model-facing features; the winner refits on the subset
    X, y = _data()  # 6 numeric features
    cfg = FeatureSelectionConfig(cutoff="top_k", top_k=3)
    m = AutoML(task="binary", models=("linear",), random_state=0, feature_selection=cfg).fit(X, y)
    assert m.schema_.selected_features is not None and len(m.schema_.selected_features) == 3
    assert m.leaderboard_[0].n_features == 3
    assert m.predict(X).shape[0] == len(y)
    assert m.run_report_["feature_selection"]["n_selected"] == 3
    # finding #10: the no-selection honest gate ran and its verdict is surfaced (never silent)
    assert m.run_report_["feature_selection"]["no_selection_gate"] in {
        "selection_kept",
        "no_selection_better",
        "all_features_selected",
    }


def test_fs_changes_run_fingerprint() -> None:
    X, y = _data()
    base = AutoML(task="binary", models=("linear",), random_state=0).fit(X, y)
    with_fs = AutoML(
        task="binary",
        models=("linear",),
        random_state=0,
        feature_selection=FeatureSelectionConfig(cutoff="top_k", top_k=3),
    ).fit(X, y)
    assert base.run_report_["run_fingerprint"] != with_fs.run_report_["run_fingerprint"]
    assert with_fs.run_report_["config"]["fs"]["cutoff"] == "top_k"


def test_fs_artifact_roundtrip_predicts_identically(tmp_path) -> None:  # noqa: ANN001
    X, y = _data()
    cfg = FeatureSelectionConfig(cutoff="top_k", top_k=3)
    m = AutoML(task="binary", models=("linear",), random_state=0, feature_selection=cfg).fit(X, y)
    save_artifact(m.fitted_, tmp_path)
    loaded = load_artifact(tmp_path)
    assert loaded.schema.selected_features is not None  # subset travels in schema.json
    assert np.allclose(loaded.predict_proba(X), m.predict_proba(X))


def test_fe_on_fs_on_roundtrip() -> None:
    # FR-FS-7: selection works over the FE-augmented feature set
    df, y = _data_with_cats()
    m = AutoML(
        task="binary",
        random_state=0,
        feature_engineering=FEConfig(frequency_encoding=True),
        feature_selection=FeatureSelectionConfig(cutoff="top_frac", top_frac=0.6),
    ).fit(df, y)
    assert m.schema_.selected_features is not None
    assert m.predict(df).shape[0] == len(y)


# --- M6c honest-compare (ADR-0046/0048/0049) ---


def test_fs_compare_reduces_features_and_reports_winner() -> None:
    X, y = _data()  # 6 numeric features
    cfg = FeatureSelectionConfig(compare=("importance", "sequential"), selection_holdout=0.3)
    m = AutoML(task="binary", models=("linear",), random_state=0, feature_selection=cfg).fit(X, y)
    fs = m.run_report_["feature_selection"]
    assert m.schema_.selected_features is not None
    assert fs["winner"] in ("importance", "sequential")
    assert set(fs["strategies_evaluated"]) == {"importance", "sequential"}
    assert set(fs["per_strategy"]) == {"importance", "sequential"}
    assert m.predict(X).shape[0] == len(y)


def test_fs_single_via_compare_matches_m6b_form() -> None:
    # FR-FSC-1: compare=(X,) selects on full DEV with the run seed -> same subset as strategy=X (M6b)
    X, y = _data()
    m6b = AutoML(
        task="binary",
        models=("linear",),
        random_state=0,
        feature_selection=FeatureSelectionConfig(strategy="importance"),
    ).fit(X, y)
    one = AutoML(
        task="binary",
        models=("linear",),
        random_state=0,
        feature_selection=FeatureSelectionConfig(compare=("importance",)),
    ).fit(X, y)
    assert one.schema_.selected_features == m6b.schema_.selected_features


def test_fs_auto_arbitration_resolves_and_writes_back() -> None:
    # M6f (ADR-0057): n_rows=120 < 2000 -> arbitration="auto" resolves to nested_per_fold; the effective
    # value is written back into the manifest config, and the provenance surfaces in fs_resolution.
    X, y = _data(n=120)
    cfg = FeatureSelectionConfig(compare=("importance", "random_probe"), arbitration="auto")
    m = AutoML(task="binary", models=("linear",), random_state=0, feature_selection=cfg).fit(X, y)
    fs = m.run_report_["feature_selection"]
    assert fs["fs_resolution"]["arbitration_resolved_from"] == "auto"
    assert (
        m.run_report_["config"]["fs"]["arbitration"] == "nested_per_fold"
    )  # write-back, not the "auto" sentinel


def test_fs_per_fold_block_stats_in_null_block_stats() -> None:
    # M6f (ADR-0059 §1a): per-fold degenerate aggregates merge into null_block_stats end-to-end (group scheme
    # supplies the structure blocks; nested_per_fold runs the per-fold re-selection).
    X, y = _data(n=200)
    groups = np.arange(200) // 10  # 20 groups of 10 rows -> structure blocks
    cfg = FeatureSelectionConfig(
        compare=("importance", "random_probe"),
        arbitration="nested_per_fold",
        arbitration_n_splits=2,
    )
    m = AutoML(
        task="binary",
        models=("linear",),
        random_state=0,
        cv=CVConfig(scheme="group", n_splits=2),
        feature_selection=cfg,
    ).fit(X, y, groups=groups)
    nbs = m.run_report_["feature_selection"]["null_block_stats"]
    assert (
        "n_blocks" in nbs and "per_fold_degenerate_mean" in nbs
    )  # full-DEV + merged per-fold aggregate


def test_fs_compare_artifact_roundtrip_predicts_identically(tmp_path) -> None:  # noqa: ANN001
    X, y = _data()
    cfg = FeatureSelectionConfig(compare=("importance", "sequential"), selection_holdout=0.3)
    m = AutoML(task="binary", models=("linear",), random_state=0, feature_selection=cfg).fit(X, y)
    save_artifact(m.fitted_, tmp_path)
    loaded = load_artifact(tmp_path)
    assert loaded.schema.selected_features == m.schema_.selected_features  # only the winner travels
    assert np.allclose(loaded.predict_proba(X), m.predict_proba(X))


def test_fs_compare_deterministic_winner() -> None:
    X, y = _data()
    cfg = FeatureSelectionConfig(compare=("importance", "random_probe"), selection_holdout=0.3)
    a = AutoML(task="binary", models=("linear",), random_state=0, feature_selection=cfg).fit(X, y)
    b = AutoML(task="binary", models=("linear",), random_state=0, feature_selection=cfg).fit(X, y)
    assert a.schema_.selected_features == b.schema_.selected_features
    assert (
        a.run_report_["feature_selection"]["winner"] == b.run_report_["feature_selection"]["winner"]
    )


def test_fs_compare_multiclass_predicts() -> None:
    # the arbiter aligns proba to the whole-DEV class order, so multiclass compare scores correctly
    df, y = _data_with_cats(n=210, classes=3)
    cfg = FeatureSelectionConfig(compare=("importance", "sequential"), selection_holdout=0.3)
    m = AutoML(task="multiclass", models=("linear",), random_state=0, feature_selection=cfg).fit(
        df, y
    )
    assert m.schema_.selected_features is not None
    assert m.predict(df).shape[0] == len(y)


def test_fs_compare_nested_arbitration_ships_winner() -> None:
    # M6d (ADR-0052/0053): nested arbitration + honest significance winner runs end-to-end and ships a subset
    X, y = _data(n=200)
    cfg = FeatureSelectionConfig(
        compare=("importance", "random_probe"), arbitration="nested", arbitration_n_splits=3
    )
    m = AutoML(task="binary", models=("linear",), random_state=0, feature_selection=cfg).fit(X, y)
    assert m.schema_.selected_features is not None
    assert m.predict(X).shape[0] == len(y)


def test_fs_compare_per_fold_arbitration_ships_winner() -> None:
    # M6e (ADR-0054): per-fold re-selection runs end-to-end, ships a subset, and reports the honest procedure
    X, y = _data(n=200)
    cfg = FeatureSelectionConfig(
        compare=("importance", "random_probe"),
        arbitration="nested_per_fold",
        arbitration_n_splits=3,
    )
    m = AutoML(task="binary", models=("linear",), random_state=0, feature_selection=cfg).fit(X, y)
    assert m.schema_.selected_features is not None
    assert m.predict(X).shape[0] == len(y)
    assert m.run_report_["feature_selection"]["arbitration_effective"] == "nested_per_fold"
    assert m.run_report_["feature_selection"]["per_fold_reselection"] is True


def test_fs_null_importance_timeseries_structure_aware() -> None:
    # M6d (ADR-0050): null_importance on timeseries no longer raises — it permutes the target WITHIN
    # time blocks (structure-aware), so the run completes and ships a winner subset.
    X, y = _data(n=120)
    cfg = FeatureSelectionConfig(compare=("importance", "null_importance"))
    m = AutoML(
        task="binary",
        cv=CVConfig(scheme="timeseries", n_splits=3, n_test=20),
        random_state=0,
        feature_selection=cfg,
    ).fit(X, y, time=np.arange(120))
    assert m.schema_.selected_features is not None
    assert m.predict(X).shape[0] == len(y)


def test_datetime_deltas_via_task_report_date() -> None:
    df, y = _data_with_cats(n=120)
    df["event_dt"] = pd.to_datetime("2021-06-01") - pd.to_timedelta(np.arange(120) % 30, unit="D")
    df["report_dt"] = pd.to_datetime(["2021-06-01"] * 120)  # consistent [ns] resolution
    m = AutoML(task=Task(kind="binary", report_date="report_dt"), random_state=0).fit(df, y)
    assert m.schema_.datetime_spec is not None
    assert "event_dt__days_to_report" in m.schema_.features
    assert m.predict(df).shape == (len(y),)


def test_fit_with_nan_data_keeps_the_simple_zoo(caplog: pytest.LogCaptureFixture) -> None:
    # ADR-0078: the imputer keeps linear/baseline in the comparison on NaN data — they are no longer
    # evicted, and the "skipping models" advisory does not fire.
    X, y = _data(n=120)
    X[::7, 0] = np.nan
    with caplog.at_level(logging.WARNING, logger="honestml"):
        m = AutoML(task="binary", cv=3, random_state=0).fit(X, y)
    ids = {e.model_id for e in m.leaderboard_}
    assert {"linear", "baseline"} <= ids  # the simple candidates survive NaN
    assert m.predict(X).shape == (len(y),)
    assert not any("skipping models" in r.getMessage() for r in caplog.records)


def test_stratified_cv_fails_fast_on_rare_class() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 5))
    y = np.zeros(100, dtype=int)
    y[:2] = 1  # least populated class: 2 rows < default n_splits=5
    with pytest.raises(ConfigError, match="least populated class"):
        AutoML(task="binary", models=("baseline", "linear"), random_state=0).fit(X, y)


def test_kfold_cv_fails_fast_on_fewer_rows_than_folds() -> None:
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(3, 4)), rng.normal(size=3)
    with pytest.raises(ConfigError, match="one row per fold"):
        AutoML(task="regression", models=("baseline", "linear"), cv=5, random_state=0).fit(X, y)


def test_tiny_outer_holdout_warns_high_variance(caplog: pytest.LogCaptureFixture) -> None:
    X, y = _data(n=100)
    with caplog.at_level(logging.WARNING, logger="honestml"):
        m = AutoML(
            task="binary",
            models=("baseline", "linear"),
            cv=CVConfig(outer_holdout=0.2),
            random_state=0,
        ).fit(X, y)
    assert m.holdout_score_ is not None
    assert any("high-variance" in r.getMessage() for r in caplog.records)
