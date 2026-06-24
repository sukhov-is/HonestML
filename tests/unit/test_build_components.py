"""M2-6: composition root wires defaults, syncs policy, filters by capability (ADR-0009)."""

from __future__ import annotations

import importlib.util
import logging

import numpy as np
import pytest

from honestml.adapters import BaselineRegressor, HoldoutSplitter, StratifiedKFoldSplitter
from honestml.composition import build_default_components
from honestml.composition import registry as regmod
from honestml.composition.build import resolve_fs_defaults
from honestml.composition.registry import ComponentDescriptor
from honestml.core import (
    Capabilities,
    ConfigError,
    CVConfig,
    FeatureSelectionConfig,
    MissingDependencyError,
    ModelSpec,
    Task,
)

pytestmark = pytest.mark.unit


class _FakeEP:
    def __init__(self, descriptor: ComponentDescriptor) -> None:
        self._descriptor = descriptor

    def load(self) -> ComponentDescriptor:
        return self._descriptor


def test_feature_ranker_off_by_default() -> None:
    c = build_default_components(Task(kind="binary"), random_state=0)
    assert c.feature_selection is None and c.feature_ranker is None


def test_feature_ranker_resolved_by_strategy() -> None:
    from honestml.adapters import ImportanceRanker, RandomProbeRanker
    from honestml.core import FeatureSelectionConfig

    imp = build_default_components(
        Task(kind="binary"),
        random_state=0,
        feature_selection=FeatureSelectionConfig(strategy="importance"),
    )
    assert isinstance(imp.feature_ranker, ImportanceRanker)
    probe = build_default_components(
        Task(kind="binary"),
        random_state=0,
        feature_selection=FeatureSelectionConfig(strategy="random_probe"),
    )
    assert isinstance(probe.feature_ranker, RandomProbeRanker)


def test_null_importance_single_uses_m6b_ranker_path() -> None:
    # null_importance is a ranker -> single-strategy stays the M6b feature_ranker path (no compare wiring)
    from honestml.adapters import NullImportanceRanker
    from honestml.core import FeatureSelectionConfig

    c = build_default_components(
        Task(kind="binary"),
        random_state=0,
        feature_selection=FeatureSelectionConfig(strategy="null_importance", n_runs=10),
    )
    assert isinstance(c.feature_ranker, NullImportanceRanker)
    assert c.feature_strategies is None


def test_compare_and_sequential_wire_the_m6c_path() -> None:
    from honestml.core import FeatureSelectionConfig

    comp = build_default_components(
        Task(kind="binary"),
        random_state=0,
        feature_selection=FeatureSelectionConfig(compare=("importance", "sequential")),
    )
    assert comp.feature_ranker is None
    assert comp.feature_strategies is not None
    assert [n for n, _ in comp.feature_strategies] == ["importance", "sequential"]
    assert comp.feature_carve is not None and comp.feature_fit_predict is not None

    seq = build_default_components(
        Task(kind="binary"),
        random_state=0,
        feature_selection=FeatureSelectionConfig(strategy="sequential"),
    )
    assert seq.feature_strategies is not None and seq.feature_ranker is None


def test_null_importance_resolves_for_timeseries_and_group() -> None:
    # M6d (ADR-0050): structure-aware permutation replaces the M6c ConfigError; null_importance now
    # resolves on every scheme (the spine threads the per-row structure label).
    from honestml.core import FeatureSelectionConfig

    for scheme, kw in (("timeseries", {"has_time": True}), ("group", {"has_group": True})):
        comp = build_default_components(
            Task(kind="binary"),
            random_state=0,
            cv=CVConfig(scheme=scheme),
            feature_selection=FeatureSelectionConfig(compare=("importance", "null_importance")),
            **kw,
        )
        assert comp.feature_strategies is not None
        assert "null_importance" in {name for name, _ in comp.feature_strategies}


def test_nested_arbitration_resolves_splitter_with_compare() -> None:
    # M6d (ADR-0052): arbitration="nested" with a compare list builds a K-fold DEV arbitration splitter
    from honestml.core import FeatureSelectionConfig

    comp = build_default_components(
        Task(kind="binary"),
        random_state=0,
        feature_selection=FeatureSelectionConfig(
            compare=("importance", "random_probe"), arbitration="nested", arbitration_n_splits=3
        ),
    )
    assert comp.feature_arbitration_splitter is not None


def test_fs_cost_warning_for_expensive_strategies(caplog) -> None:
    # NFR-FSH-2: enabling null_importance / shap logs a cost WARNING at build
    pytest.importorskip("shap")  # the "shap" strategy instantiates ShapRanker (needs shap) at build
    from honestml.core import FeatureSelectionConfig

    with caplog.at_level("WARNING"):
        build_default_components(
            Task(kind="binary"),
            random_state=0,
            feature_selection=FeatureSelectionConfig(
                compare=("importance", "null_importance", "shap")
            ),
        )
    msgs = " ".join(r.message for r in caplog.records)
    assert "null_importance" in msgs and "shap" in msgs


def test_nested_arbitration_without_compare_warns(caplog) -> None:
    # nested has no effect on the single-strategy path -> WARNING, no arbitration splitter (ADR-0052 §1)
    from honestml.core import FeatureSelectionConfig

    with caplog.at_level("WARNING"):
        comp = build_default_components(
            Task(kind="binary"),
            random_state=0,
            feature_selection=FeatureSelectionConfig(strategy="importance", arbitration="nested"),
        )
    assert comp.feature_arbitration_splitter is None
    assert any("nested" in r.message for r in caplog.records)


def test_per_fold_arbitration_without_compare_warns() -> None:
    # FR-FSE-1: nested_per_fold on the single-strategy path is dead config -> WARNING, no arbitration splitter
    from honestml.core import FeatureSelectionConfig

    comp = build_default_components(
        Task(kind="binary"),
        random_state=0,
        feature_selection=FeatureSelectionConfig(
            strategy="importance", arbitration="nested_per_fold"
        ),
    )
    assert comp.feature_arbitration_splitter is None


def test_per_fold_arbitration_resolves_splitter_and_warns_cost(caplog) -> None:
    # M6e (ADR-0054): nested_per_fold with compare builds the outer splitter AND logs the per-fold cost WARNING
    from honestml.core import FeatureSelectionConfig

    with caplog.at_level("WARNING"):
        comp = build_default_components(
            Task(kind="binary"),
            random_state=0,
            feature_selection=FeatureSelectionConfig(
                compare=("importance", "random_probe"),
                arbitration="nested_per_fold",
                arbitration_n_splits=3,
            ),
        )
    assert comp.feature_arbitration_splitter is not None
    assert any(
        "nested_per_fold" in r.message and "SELECTION cost" in r.message for r in caplog.records
    )


def test_block_window_at_rank_mode_warns_dead_config(caplog) -> None:
    # ADR-0055 §1: a null_block_window under the default rank mode is dead config -> WARNING (not error)
    from honestml.core import FeatureSelectionConfig

    with caplog.at_level("WARNING"):
        build_default_components(
            Task(kind="binary"),
            random_state=0,
            feature_selection=FeatureSelectionConfig(
                strategy="null_importance", null_block_window=5.0
            ),
        )
    assert any(
        "null_block_window is set but null_block_mode='rank'" in r.message for r in caplog.records
    )


def test_shap_background_at_tpd_warns_dead_config(caplog) -> None:
    # ADR-0056 §1: shap_background_samples only acts under interventional -> WARNING under tree_path_dependent
    from honestml.core import FeatureSelectionConfig

    if importlib.util.find_spec("shap") is None:
        pytest.skip("shap not installed; ShapRanker construction would fail before the warning")
    with caplog.at_level("WARNING"):
        build_default_components(
            Task(kind="binary"),
            random_state=0,
            feature_selection=FeatureSelectionConfig(strategy="shap", shap_background_samples=32),
        )
    assert any("shap_background_samples is set but" in r.message for r in caplog.records)


def test_shap_background_kmeans_at_tpd_warns_dead_config(caplog) -> None:
    # ADR-0060 §3: shap_background='kmeans' only acts under interventional -> WARNING under tree_path_dependent
    from honestml.core import FeatureSelectionConfig

    if importlib.util.find_spec("shap") is None:
        pytest.skip("shap not installed; ShapRanker construction would fail before the warning")
    with caplog.at_level("WARNING"):
        build_default_components(
            Task(kind="binary"),
            random_state=0,
            feature_selection=FeatureSelectionConfig(strategy="shap", shap_background="kmeans"),
        )
    assert any("shap_background='kmeans' is set but" in r.message for r in caplog.records)


# --- M6f resolve_fs_defaults: data-shape auto-defaults + hard cost-budget (ADR-0057/0058) ---

_CMP = ("importance", "null_importance")


def _resolve(
    fs, *, n_rows=5000, n_features=20, inner=5, times=None, scheme="auto", purge=0, purge_delta=None
):
    return resolve_fs_defaults(
        fs,
        n_rows=n_rows,
        n_features=n_features,
        inner_n_splits=inner,
        times=times,
        scheme=scheme,
        purge=purge,
        purge_delta=purge_delta,
    )


def test_arbitration_auto_resolves_by_n_rows() -> None:
    # ADR-0057 §1 ladder with boundary points (1999/2000/19999/20000)
    for n, expected in {
        1999: "nested_per_fold",
        2000: "nested",
        19999: "nested",
        20000: "holdout",
    }.items():
        eff, rec = _resolve(FeatureSelectionConfig(compare=_CMP, arbitration="auto"), n_rows=n)
        assert eff.arbitration == expected
        assert rec["arbitration_resolved_from"] == "auto" and rec["arbitration_requested"] == "auto"


def test_auto_single_strategy_resolves_holdout() -> None:
    # single strategy (compare=None) and sequential-without-compare go the compare path but arbitration is moot
    assert (
        _resolve(FeatureSelectionConfig(strategy="importance", arbitration="auto"), n_rows=500)[
            0
        ].arbitration
        == "holdout"
    )
    assert (
        _resolve(FeatureSelectionConfig(strategy="sequential", arbitration="auto"), n_rows=500)[
            0
        ].arbitration
        == "holdout"
    )


def test_auto_timeseries_purge0_resolves_holdout() -> None:
    fs = FeatureSelectionConfig(compare=_CMP, arbitration="auto")
    assert (
        _resolve(fs, n_rows=500, scheme="timeseries", purge=0)[0].arbitration == "holdout"
    )  # anti-leakage
    assert (
        _resolve(fs, n_rows=500, scheme="timeseries", purge=1)[0].arbitration == "nested_per_fold"
    )  # purged -> ok


def test_auto_timeseries_period_purge0_resolves_holdout() -> None:
    # ADR-0096 §3: the period scheme inherits the same anti-leakage downgrade as 'timeseries'
    fs = FeatureSelectionConfig(compare=_CMP, arbitration="auto")
    assert _resolve(fs, n_rows=500, scheme="timeseries_period", purge=0)[0].arbitration == "holdout"
    assert (
        _resolve(fs, n_rows=500, scheme="timeseries_period", purge=1)[0].arbitration
        == "nested_per_fold"
    )  # purged -> per-fold ok


def test_auto_purge_delta_counts_as_purged() -> None:
    # ADR-0097/ADR-0096 §3: a Δt purge separates the inner/outer boundary too, so it is NOT downgraded
    # to leak-safe holdout (unlike an unpurged purge=0 boundary)
    fs = FeatureSelectionConfig(compare=_CMP, arbitration="auto")
    assert (
        _resolve(fs, n_rows=500, scheme="timeseries", purge=0, purge_delta=2.0)[0].arbitration
        == "nested_per_fold"
    )
    assert (
        _resolve(fs, n_rows=500, scheme="timeseries_period", purge=0, purge_delta=2.0)[
            0
        ].arbitration
        == "nested_per_fold"
    )


def test_auto_does_not_override_explicit_arbitration() -> None:
    eff, rec = _resolve(
        FeatureSelectionConfig(compare=_CMP, arbitration="nested_per_fold"), n_rows=10**6
    )
    assert eff.arbitration == "nested_per_fold" and rec == {}  # explicit untouched, no record


def test_block_mode_auto_irregular_to_time_window() -> None:
    times = np.concatenate(
        [np.arange(50.0), np.array([500.0, 1000.0])]
    )  # dense then big gaps -> irregular Δt
    eff, rec = _resolve(FeatureSelectionConfig(compare=_CMP, null_block_mode="auto"), times=times)
    assert eff.null_block_mode == "time_window" and (eff.null_block_window or 0) > 0
    assert rec["block_mode_resolved_from"] == "auto"


def test_block_mode_auto_regular_to_rank() -> None:
    eff, _ = _resolve(
        FeatureSelectionConfig(null_block_mode="auto"), times=np.arange(100.0)
    )  # CV(Δt)=0
    assert eff.null_block_mode == "rank" and eff.null_block_window is None


def test_block_mode_auto_no_time_to_rank() -> None:
    assert (
        _resolve(FeatureSelectionConfig(null_block_mode="auto"), times=None)[0].null_block_mode
        == "rank"
    )


def test_block_auto_zero_dt_falls_back_to_rank() -> None:
    eff, _ = _resolve(
        FeatureSelectionConfig(null_block_mode="auto"), times=np.zeros(10)
    )  # dup timestamps, median Δt=0
    assert (
        eff.null_block_mode == "rank" and eff.null_block_window is None
    )  # no 0-window (gt=0 safe, ADR-0057 §2)


def test_block_auto_derives_window() -> None:
    times = np.concatenate(
        [np.arange(0.0, 30.0), np.array([200.0, 400.0])]
    )  # median Δt=1, irregular
    eff, _ = _resolve(
        FeatureSelectionConfig(null_block_mode="auto", null_block_size=10), times=times
    )
    assert eff.null_block_mode == "time_window" and eff.null_block_window == pytest.approx(
        10.0
    )  # median*block_size


def test_cost_budget_downgrades_arbitration() -> None:
    # per_fold = 2×K_outer(5)×inner(5)×(1+30)=1550 > 400; nested/holdout = 2×(31)×5=310 <= 400 -> downgrade to nested
    fs = FeatureSelectionConfig(
        compare=_CMP, arbitration="nested_per_fold", n_runs=30, cost_budget_refits=400
    )
    eff, rec = _resolve(fs, n_rows=500)
    assert eff.arbitration == "nested"
    assert (
        rec["arbitration_resolved_from"] == "cost_budget"
        and rec["arbitration_requested"] == "nested_per_fold"
    )


def test_cost_budget_floor_exceeded_raises() -> None:
    fs = FeatureSelectionConfig(
        compare=_CMP, arbitration="nested_per_fold", n_runs=100, cost_budget_refits=10
    )
    with pytest.raises(ConfigError, match="below the holdout floor"):
        _resolve(fs, n_rows=500)


def test_cost_budget_none_no_gate() -> None:
    fs = FeatureSelectionConfig(compare=_CMP, arbitration="nested_per_fold", n_runs=100)
    eff, rec = _resolve(fs, n_rows=500)
    assert eff is fs and rec == {}  # None budget -> no change, no record (M6e behavior unchanged)


def test_cost_budget_downgrade_warns_loudly(caplog) -> None:
    # NFR-FSF-6: a budget downgrade of an explicit arbitration is logged loudly (explicit -> effective)
    fs = FeatureSelectionConfig(
        compare=_CMP, arbitration="nested_per_fold", n_runs=30, cost_budget_refits=400
    )
    with caplog.at_level("WARNING"):
        _resolve(fs, n_rows=500)
    assert any(
        "exceeds budget" in r.message and "downgraded to" in r.message for r in caplog.records
    )


def test_auto_arbitration_with_cost_budget_downgrades() -> None:
    # auto resolves to nested_per_fold (n<2000), then the budget downgrades it; requested stays "auto"
    fs = FeatureSelectionConfig(compare=_CMP, arbitration="auto", n_runs=30, cost_budget_refits=400)
    eff, rec = _resolve(fs, n_rows=500)
    assert eff.arbitration == "nested"
    assert (
        rec["arbitration_requested"] == "auto" and rec["arbitration_resolved_from"] == "cost_budget"
    )


def test_per_fold_timeseries_without_purge_warns_boundary(caplog) -> None:
    # ADR-0054 §4 (fix R2): per-fold on a timeseries scheme with purge=0 surfaces the unpurged-boundary WARNING
    from honestml.core import FeatureSelectionConfig

    with caplog.at_level(logging.WARNING, logger="honestml"):
        build_default_components(
            Task(kind="binary"),
            random_state=0,
            cv=CVConfig(scheme="timeseries", n_splits=3),
            has_time=True,
            feature_selection=FeatureSelectionConfig(
                compare=("importance", "random_probe"), arbitration="nested_per_fold"
            ),
        )
    assert any("does not purge the inner/outer boundary" in r.message for r in caplog.records)


def test_defaults_for_binary() -> None:
    c = build_default_components(Task(kind="binary"), random_state=0)
    assert c.metric.name == "roc_auc"
    # baseline+linear are always present; boosting appears only when its extra is installed
    # (find_spec gate, ADR-0020 §5), so the default set is env-dependent -> assert the subset.
    assert {"baseline", "linear"} <= set(c.estimators)
    assert isinstance(c.splitter, StratifiedKFoldSplitter)


def test_uninstalled_boosting_excluded_from_defaults(monkeypatch) -> None:
    # force "nothing installed": is_available -> False for any requires -> boosting dropped
    monkeypatch.setattr(regmod, "_module_present", lambda module: False)
    c = build_default_components(Task(kind="binary"), random_state=0)
    assert set(c.estimators) == {"baseline", "linear"}


def test_policy_direction_synced_to_metric() -> None:
    auc = build_default_components(Task(kind="binary"), random_state=0, metric="roc_auc")
    ll = build_default_components(Task(kind="binary"), random_state=0, metric="log_loss")
    assert auc.policy.greater_is_better is True
    assert ll.policy.greater_is_better is False


def test_seed_propagates_to_splitter_and_linear() -> None:
    c = build_default_components(Task(kind="binary"), random_state=123)
    assert c.splitter.random_state == 123
    assert c.estimators["linear"]().random_state == 123


def test_build_resolves_significance_toggle() -> None:
    # M5c (ADR-0034): default -> honest bootstrap band; "off" -> inert NoSignificanceTest
    from honestml.adapters import BootstrapSignificanceTest
    from honestml.core import NoSignificanceTest

    on = build_default_components(Task(kind="binary"), random_state=0)
    off = build_default_components(Task(kind="binary"), random_state=0, significance="off")
    assert isinstance(on.significance, BootstrapSignificanceTest)
    assert isinstance(off.significance, NoSignificanceTest)


def test_models_subset_and_unknown() -> None:
    c = build_default_components(Task(kind="binary"), random_state=0, models=("linear",))
    assert set(c.estimators) == {"linear"}
    with pytest.raises(ConfigError, match="unknown models"):
        build_default_components(Task(kind="binary"), random_state=0, models=("ghost",))


def test_regression_with_proba_metric_raises() -> None:
    # ADR-0021 §4 / R1-4b: a proba metric on a regression task fails fast at metric
    # resolution (before the estimator filter / CV), not deep inside sklearn.
    with pytest.raises(ConfigError, match="cannot score a regression task"):
        build_default_components(Task(kind="regression", metric="roc_auc"), random_state=0)


def test_classification_with_value_metric_raises() -> None:
    with pytest.raises(ConfigError, match="cannot score"):
        build_default_components(Task(kind="binary", metric="rmse"), random_state=0)


def test_pr_auc_multiclass_raises() -> None:
    import numpy as np

    with pytest.raises(ConfigError, match="pr_auc"):
        build_default_components(
            Task(kind="multiclass", metric="pr_auc"),
            random_state=0,
            classes=np.array([0, 1, 2]),
        )


def test_no_estimator_for_task_raises_configerror(monkeypatch) -> None:
    # a regression-only plugin selected for a binary task -> capability filter empties
    # the set -> "no estimator supports" (the real ADR-0009 §F3 path).
    reg_only = ComponentDescriptor(
        name="ext_reg",
        spec=ModelSpec(name="ext_reg", capabilities=Capabilities(tasks=("regression",))),
        build=lambda **kw: BaselineRegressor(),
    )
    monkeypatch.setattr(regmod, "entry_points", lambda group=None: [_FakeEP(reg_only)])
    with pytest.raises(ConfigError, match="no estimator supports"):
        build_default_components(Task(kind="binary"), random_state=0, models=("ext_reg",))


def test_explicit_missing_extra_raises_missing_dependency(monkeypatch) -> None:
    # an explicitly requested model whose extra is absent fails fast (ADR-0020 §5),
    # regardless of which boosting libs happen to be installed in this env.
    plug = ComponentDescriptor(
        name="ext_needs",
        spec=ModelSpec(
            name="ext_needs", capabilities=Capabilities(tasks=("binary",), probabilistic=True)
        ),
        build=lambda **kw: BaselineRegressor(),
        requires=("definitely_absent_pkg_xyz",),
    )
    monkeypatch.setattr(regmod, "entry_points", lambda group=None: [_FakeEP(plug)])
    with pytest.raises(MissingDependencyError, match="ext_needs"):
        build_default_components(Task(kind="binary"), random_state=0, models=("ext_needs",))


# --- C4 honest CV-selection (ADR-0016) -------------------------------------


def test_auto_scheme_binary_resolves_to_stratified() -> None:
    c = build_default_components(Task(kind="binary"), random_state=0, cv=CVConfig(scheme="auto"))
    assert isinstance(c.splitter, StratifiedKFoldSplitter)
    assert c.cv.scheme == "stratified"  # resolved scheme written back (truthful manifest)


def test_default_cv_is_five_folds() -> None:
    c = build_default_components(Task(kind="binary"), random_state=0)
    assert isinstance(c.splitter, StratifiedKFoldSplitter)
    assert c.splitter.n_splits == 5


def test_holdout_scheme_selects_holdout_splitter() -> None:
    c = build_default_components(Task(kind="binary"), random_state=0, cv=CVConfig(scheme="holdout"))
    assert isinstance(c.splitter, HoldoutSplitter)
    assert c.cv.scheme == "holdout"


def test_timeseries_requires_time_column() -> None:
    # M4b: timeseries is implemented but needs a declared time axis (pass time= to fit)
    with pytest.raises(ConfigError, match="requires a time column"):
        build_default_components(
            Task(kind="binary"), random_state=0, cv=CVConfig(scheme="timeseries")
        )


def test_timeseries_scheme_resolves_with_time() -> None:
    from honestml.adapters import TimeSeriesSplitter

    c = build_default_components(
        Task(kind="binary"),
        random_state=0,
        cv=CVConfig(scheme="timeseries", n_splits=3, purge=2, embargo=1),
        has_time=True,
    )
    assert isinstance(c.splitter, TimeSeriesSplitter)
    assert c.splitter.purge == 2 and c.splitter.embargo == 1 and c.splitter.n_splits == 3
    assert c.cv.scheme == "timeseries"


def test_timeseries_period_resolves_with_time() -> None:
    from honestml.adapters import PeriodTimeSeriesSplitter

    c = build_default_components(
        Task(kind="binary"),
        random_state=0,
        cv=CVConfig(scheme="timeseries_period", period="month", n_test=2, purge=1, step_periods=1),
        has_time=True,
    )
    assert isinstance(c.splitter, PeriodTimeSeriesSplitter)
    assert c.splitter.period == "month" and c.splitter.n_test == 2 and c.splitter.purge == 1
    assert c.splitter.step_periods == 1
    assert c.cv.scheme == "timeseries_period"


def test_timeseries_period_requires_time_column() -> None:
    with pytest.raises(ConfigError, match="requires a time column"):
        build_default_components(
            Task(kind="binary"),
            random_state=0,
            cv=CVConfig(scheme="timeseries_period", period="month"),
        )


def test_timeseries_period_requires_period_unit() -> None:
    with pytest.raises(ConfigError, match="requires a period unit"):
        build_default_components(
            Task(kind="binary"),
            random_state=0,
            cv=CVConfig(scheme="timeseries_period"),
            has_time=True,
        )


@pytest.mark.parametrize("cfg", [CVConfig(period="month"), CVConfig(step_periods=2)])
def test_period_knobs_require_period_scheme(cfg: CVConfig) -> None:
    # a period knob under a non-period scheme would be silently dead -> ConfigError (FR-9)
    with pytest.raises(ConfigError, match="require scheme='timeseries_period'"):
        build_default_components(Task(kind="binary"), random_state=0, cv=cfg, has_time=True)


def test_period_purge_embargo_allowed() -> None:
    # purge/embargo are valid under the period scheme too (unit = periods), not only 'timeseries'
    c = build_default_components(
        Task(kind="binary"),
        random_state=0,
        cv=CVConfig(scheme="timeseries_period", period="week", purge=1, embargo=1),
        has_time=True,
    )
    assert c.splitter.purge == 1 and c.splitter.embargo == 1


# --- Etap2: Δt gaps + rolling caps wiring/gates (ADR-0097/0099, FR-4/5/9) ---


def test_timeseries_delta_and_rolling_wire_into_splitter() -> None:
    from honestml.adapters import TimeSeriesSplitter

    c = build_default_components(
        Task(kind="binary"),
        random_state=0,
        cv=CVConfig(scheme="timeseries", purge_delta=2.0, embargo_delta=1.5, max_train_size=500),
        has_time=True,
    )
    assert isinstance(c.splitter, TimeSeriesSplitter)
    assert c.splitter.purge_delta == 2.0 and c.splitter.embargo_delta == 1.5
    assert c.splitter.max_train_size == 500


def test_timeseries_period_delta_and_rolling_wire_into_splitter() -> None:
    from honestml.adapters import PeriodTimeSeriesSplitter

    c = build_default_components(
        Task(kind="binary"),
        random_state=0,
        cv=CVConfig(
            scheme="timeseries_period", period="month", purge_delta=2.0, max_train_periods=12
        ),
        has_time=True,
    )
    assert isinstance(c.splitter, PeriodTimeSeriesSplitter)
    assert c.splitter.purge_delta == 2.0 and c.splitter.max_train_periods == 12


@pytest.mark.parametrize(
    "cfg",
    [CVConfig(scheme="stratified", purge_delta=2.0), CVConfig(scheme="kfold", embargo_delta=2.0)],
)
def test_delta_gaps_require_time_scheme(cfg: CVConfig) -> None:
    # a Δt gap under a non-time scheme would be silently dead -> ConfigError (FR-9)
    with pytest.raises(ConfigError, match="require a time-series scheme"):
        build_default_components(Task(kind="binary"), random_state=0, cv=cfg, has_time=True)


def test_max_train_periods_requires_period_scheme() -> None:
    with pytest.raises(ConfigError, match="max_train_periods requires scheme='timeseries_period'"):
        build_default_components(
            Task(kind="binary"),
            random_state=0,
            cv=CVConfig(scheme="timeseries", max_train_periods=12),
            has_time=True,
        )


def test_max_train_size_requires_timeseries_scheme() -> None:
    with pytest.raises(ConfigError, match="max_train_size requires scheme='timeseries'"):
        build_default_components(
            Task(kind="binary"),
            random_state=0,
            cv=CVConfig(scheme="timeseries_period", period="month", max_train_size=500),
            has_time=True,
        )


# --- Etap3: period weighting wiring/gate (ADR-0098, FR-6/9) ---


def test_weighting_period_requires_time_ordered_scheme() -> None:
    # FR-9: macro-by-period needs a per-row block index -> only a time-ordered scheme; else ConfigError
    with pytest.raises(ConfigError, match="weighting='period' requires a time-ordered"):
        build_default_components(
            Task(kind="binary"),
            random_state=0,
            cv=CVConfig(scheme="stratified", weighting="period"),
        )


def test_weighting_period_sets_aggregate_and_wires() -> None:
    from honestml.adapters import BootstrapSignificanceTest

    c = build_default_components(
        Task(kind="binary"),
        random_state=0,
        cv=CVConfig(scheme="timeseries_period", period="month", weighting="period"),
        has_time=True,
    )
    assert c.weighting == "period"
    assert isinstance(c.significance, BootstrapSignificanceTest)
    assert c.significance.aggregate == "period"  # significance mirrors the leaderboard weighting


def test_weighting_default_pooled_keeps_aggregate_pooled() -> None:
    from honestml.adapters import BootstrapSignificanceTest

    c = build_default_components(
        Task(kind="binary"),
        random_state=0,
        cv=CVConfig(scheme="timeseries", n_test=2),
        has_time=True,
    )
    assert c.weighting == "pooled"
    assert isinstance(c.significance, BootstrapSignificanceTest)
    assert c.significance.aggregate == "pooled"


def test_time_column_with_shuffling_scheme_warns(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="honestml"):
        build_default_components(
            Task(kind="binary"), random_state=0, cv=CVConfig(scheme="stratified"), has_time=True
        )
    assert any("look-ahead" in r.getMessage() for r in caplog.records)


# --- M3c CV set (ADR-0023) -------------------------------------------------


def test_kfold_scheme_resolves_to_plain_kfold() -> None:
    from honestml.adapters import KFoldSplitter

    c = build_default_components(Task(kind="binary"), random_state=0, cv=CVConfig(scheme="kfold"))
    assert isinstance(c.splitter, KFoldSplitter)
    assert c.cv.scheme == "kfold"


def test_regression_default_resolves_to_kfold() -> None:
    from honestml.adapters import KFoldSplitter

    # regression default_cv_scheme == "kfold"; rmse is a value metric so the filter keeps regressors
    c = build_default_components(Task(kind="regression"), random_state=0)
    assert isinstance(c.splitter, KFoldSplitter)
    assert c.cv.scheme == "kfold"


def test_group_scheme_requires_group_column() -> None:
    with pytest.raises(ConfigError, match="requires a group column"):
        build_default_components(
            Task(kind="binary"), random_state=0, cv=CVConfig(scheme="group"), has_group=False
        )


def test_group_scheme_resolves_by_kind() -> None:
    from honestml.adapters import GroupKFoldSplitter, StratifiedGroupKFoldSplitter

    clf = build_default_components(
        Task(kind="binary"), random_state=0, cv=CVConfig(scheme="group"), has_group=True
    )
    assert isinstance(clf.splitter, StratifiedGroupKFoldSplitter)
    reg = build_default_components(
        Task(kind="regression"), random_state=0, cv=CVConfig(scheme="group"), has_group=True
    )
    assert isinstance(reg.splitter, GroupKFoldSplitter)


def test_group_column_with_shuffling_scheme_warns(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="honestml"):
        build_default_components(
            Task(kind="binary"), random_state=0, cv=CVConfig(scheme="stratified"), has_group=True
        )
    assert any("not group-aware" in r.getMessage() for r in caplog.records)


def test_kfold_with_datetime_warns_look_ahead(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="honestml"):
        build_default_components(
            Task(kind="binary"), random_state=0, cv=CVConfig(scheme="kfold"), has_datetime=True
        )
    assert any("look-ahead" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize("field", ["purge", "embargo"])
def test_purge_embargo_require_timeseries(field: str) -> None:
    cfg = CVConfig(scheme="stratified", **{field: 1})
    with pytest.raises(ConfigError, match="require a time-series scheme"):
        build_default_components(Task(kind="binary"), random_state=0, cv=cfg)


def test_timeseries_without_time_is_config_error() -> None:
    with pytest.raises(ConfigError, match="requires a time column"):
        build_default_components(
            Task(kind="binary"), random_state=0, cv=CVConfig(scheme="timeseries")
        )


def test_stratified_requires_two_folds() -> None:
    # a plain ConfigError about n_splits, NOT an UnsupportedSchemeError
    with pytest.raises(ConfigError, match="n_splits >= 2"):
        build_default_components(
            Task(kind="binary"), random_state=0, cv=CVConfig(scheme="stratified", n_splits=1)
        )


def test_holdout_ignores_n_splits() -> None:
    c = build_default_components(
        Task(kind="binary"), random_state=0, cv=CVConfig(scheme="holdout", n_splits=1)
    )
    assert isinstance(c.splitter, HoldoutSplitter)


def test_int_cv_below_two_raises() -> None:
    with pytest.raises(ConfigError, match="cv must be >= 2"):
        build_default_components(Task(kind="binary"), random_state=0, cv=1)


def test_timeseries_snapshot_loads_then_requires_time() -> None:
    # NFR-3: an old snapshot deserializes fine; without a time axis it fails fast (M4b)
    cfg = CVConfig.model_validate({"scheme": "timeseries", "n_splits": 2})
    with pytest.raises(ConfigError, match="requires a time column"):
        build_default_components(Task(kind="binary"), random_state=0, cv=cfg)


@pytest.mark.parametrize("cv", [None, CVConfig(scheme="stratified"), CVConfig(scheme="holdout")])
def test_datetime_with_shuffling_scheme_warns(
    cv: CVConfig | None, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="honestml"):
        build_default_components(Task(kind="binary"), random_state=0, cv=cv, has_datetime=True)
    assert any("look-ahead" in r.getMessage() for r in caplog.records)


def test_no_datetime_no_warning(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="honestml"):
        build_default_components(Task(kind="binary"), random_state=0, has_datetime=False)
    assert not [r for r in caplog.records if "look-ahead" in r.getMessage()]


def test_nan_data_keeps_linear_baseline_via_imputer(caplog: pytest.LogCaptureFixture) -> None:
    # ADR-0078: linear/baseline now impute NaN inside their Pipeline, so the gate keeps them.
    with caplog.at_level(logging.WARNING, logger="honestml"):
        c = build_default_components(Task(kind="binary"), random_state=0, has_missing=True)
    assert {"linear", "baseline"} <= set(c.estimators)
    assert not [r for r in caplog.records if "NaN" in r.getMessage()]


def test_nan_gate_without_nan_keeps_default_set(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="honestml"):
        c = build_default_components(Task(kind="binary"), random_state=0, has_missing=False)
    assert {"linear", "baseline"} <= set(c.estimators)
    assert not [r for r in caplog.records if "NaN" in r.getMessage()]


def test_explicit_nan_capable_models_build_on_missing_data() -> None:
    # the imputer makes the simple zoo NaN-safe: explicitly selecting linear on NaN data no longer fails.
    c = build_default_components(
        Task(kind="binary"), random_state=0, models=("linear",), has_missing=True
    )
    assert set(c.estimators) == {"linear"}


def test_early_stopping_active_only_with_a_boosting() -> None:
    # ADR-0080: the es tail (and the manifest flag) is on only when an ES-capable model is selected.
    pytest.importorskip("lightgbm")
    with_boost = build_default_components(
        Task(kind="binary"), random_state=0, models=("lightgbm", "linear")
    )
    assert with_boost.early_stopping is True
    no_boost = build_default_components(Task(kind="binary"), random_state=0, models=("linear",))
    assert no_boost.early_stopping is False


def test_early_stopping_active_under_group_scheme() -> None:
    # the amendment closes the group hole: ES is active under scheme="group" too (group-disjoint es).
    pytest.importorskip("lightgbm")
    c = build_default_components(
        Task(kind="binary"),
        random_state=0,
        models=("lightgbm", "linear"),
        cv=CVConfig(scheme="group"),
        has_group=True,
    )
    assert c.early_stopping is True
