"""M0-2: typed config — JSON round-trip (manifest basis), validation, immutability."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from honestml.core import (
    BudgetConfig,
    ConfigError,
    CVConfig,
    EnsembleConfig,
    FeatureSelectionConfig,
    FEConfig,
    HPOConfig,
    RunConfig,
)

pytestmark = pytest.mark.unit


def test_runconfig_json_round_trip_default() -> None:
    rc = RunConfig()
    assert RunConfig.model_validate_json(rc.model_dump_json()) == rc


def test_runconfig_json_round_trip_nested() -> None:
    rc = RunConfig(
        seed=7,
        cv=CVConfig(scheme="kfold", n_splits=5, purge=2, embargo=1),
        budget=BudgetConfig(mode="time", time_budget_s=300.0, memory_limit_mb=4096),
        model_types=("catboost", "xgboost"),
    )
    assert RunConfig.model_validate_json(rc.model_dump_json()) == rc


@given(seed=st.integers(min_value=0, max_value=2**31 - 1), n_trials=st.integers(1, 1000))
@pytest.mark.property
def test_runconfig_round_trip_property(seed: int, n_trials: int) -> None:
    rc = RunConfig(seed=seed, budget=BudgetConfig(mode="trials", n_trials=n_trials))
    assert RunConfig.model_validate_json(rc.model_dump_json()) == rc


def test_cvconfig_round_trip() -> None:
    for cfg in (CVConfig(), CVConfig(scheme="holdout"), CVConfig(scheme="stratified", n_splits=3)):
        assert CVConfig.model_validate_json(cfg.model_dump_json()) == cfg


def test_hpo_config_defaults_inert() -> None:
    # default RunConfig keeps hpo off; HPOConfig defaults are the documented values (ADR-0062 §1)
    assert RunConfig().hpo is None
    h = HPOConfig()
    assert (h.backend, h.n_trials, h.inner_cv, h.models, h.keep_baseline, h.random_state) == (
        "optuna",
        50,
        3,
        None,
        False,
        None,
    )


def test_hpo_config_round_trip() -> None:
    rc = RunConfig(hpo=HPOConfig(n_trials=20, inner_cv=4, models=("catboost",), random_state=7))
    assert RunConfig.model_validate_json(rc.model_dump_json()) == rc


@pytest.mark.parametrize(
    "kwargs", [{"n_trials": 0}, {"inner_cv": 1}, {"timeout_s": 0.0}, {"backend": "hyperopt"}]
)
def test_hpo_config_validation(kwargs: dict) -> None:
    with pytest.raises(Exception):  # pydantic ValidationError
        HPOConfig(**kwargs)


def test_ensemble_config_defaults_inert() -> None:
    # default RunConfig keeps ensemble off; EnsembleConfig defaults are the documented values (ADR-0063 §4)
    assert RunConfig().ensemble is None
    e = EnsembleConfig()
    assert (e.method, e.size, e.n_bags, e.metric, e.random_state) == ("caruana", 50, 20, None, None)


def test_ensemble_config_round_trip() -> None:
    rc = RunConfig(ensemble=EnsembleConfig(method="weighted", size=10, n_bags=1, random_state=3))
    assert RunConfig.model_validate_json(rc.model_dump_json()) == rc


@pytest.mark.parametrize("kwargs", [{"method": "stacking"}, {"size": 0}, {"n_bags": 0}])
def test_ensemble_config_validation(kwargs: dict) -> None:
    with pytest.raises(Exception):  # pydantic ValidationError (stacking is M7-future)
        EnsembleConfig(**kwargs)


def test_legacy_timeseries_scheme_still_deserializes() -> None:
    # NFR-3.1: old snapshots remain loadable; the behavior change (fail-fast) is at build time
    assert CVConfig.model_validate({"scheme": "timeseries"}).scheme == "timeseries"


# --- Etap1: calendar/Δt period CV config (ADR-0096 §1, FR-1/2/3/9) ---


def test_period_cv_fields_default_off() -> None:
    # default CVConfig is unchanged: no period scheme, no period knobs (NFR-5)
    cv = CVConfig()
    assert cv.scheme == "auto"
    assert (cv.period, cv.period_size, cv.step_periods) == (None, None, None)


def test_period_cv_round_trips_in_runconfig() -> None:
    rc = RunConfig(
        cv=CVConfig(scheme="timeseries_period", period="month", n_test=2, step_periods=1)
    )
    assert RunConfig.model_validate_json(rc.model_dump_json()) == rc
    # the new fields are in the config dump -> they change the run-fingerprint (FR-8)
    dumped = rc.model_dump(mode="json")["cv"]
    assert dumped["scheme"] == "timeseries_period" and dumped["period"] == "month"
    assert dumped["n_test"] == 2 and dumped["step_periods"] == 1


def test_period_delta_requires_period_size() -> None:
    # field-coherence: a 'delta' window with no width is undefined -> rejected at the model boundary (G2)
    with pytest.raises(ValueError, match="period='delta' requires period_size"):
        CVConfig(scheme="timeseries_period", period="delta")
    assert CVConfig(scheme="timeseries_period", period="delta", period_size=7.0).period_size == 7.0


def test_period_size_without_delta_rejected() -> None:
    # a stray period_size under a calendar unit would be silently dead -> rejected (G2)
    with pytest.raises(ValueError, match="period_size is only used with period='delta'"):
        CVConfig(scheme="timeseries_period", period="month", period_size=3.0)


def test_period_size_and_step_bounds() -> None:
    with pytest.raises(ValueError):
        CVConfig(scheme="timeseries_period", period="delta", period_size=0)  # gt=0
    with pytest.raises(ValueError):
        CVConfig(scheme="timeseries_period", period="day", step_periods=0)  # ge=1


# --- Etap2: Δt purge/embargo + rolling max_train config (ADR-0097/0099, FR-4/5/9) ---


def test_gap_delta_and_max_train_default_off() -> None:
    # default CVConfig is unchanged: no Δt gaps, no rolling caps -> expanding behavior (NFR-5)
    cv = CVConfig()
    assert (cv.purge_delta, cv.embargo_delta) == (None, None)
    assert (cv.max_train_periods, cv.max_train_size) == (None, None)


def test_gap_delta_and_max_train_round_trip_changes_fingerprint() -> None:
    rc = RunConfig(
        cv=CVConfig(scheme="timeseries", purge_delta=2.0, embargo_delta=1.5, max_train_size=500)
    )
    assert RunConfig.model_validate_json(rc.model_dump_json()) == rc
    # the new fields are in the config dump -> they change the run-fingerprint (FR-8)
    dumped = rc.model_dump(mode="json")["cv"]
    assert dumped["purge_delta"] == 2.0 and dumped["embargo_delta"] == 1.5
    assert dumped["max_train_size"] == 500
    assert dumped != CVConfig(scheme="timeseries").model_dump(mode="json")


def test_purge_and_purge_delta_mutually_exclusive() -> None:
    # field-coherence (G2): a gap is one unit per axis — integer OR Δt, never both
    with pytest.raises(ValueError, match="either purge .* or purge_delta"):
        CVConfig(scheme="timeseries", purge=1, purge_delta=2.0)
    with pytest.raises(ValueError, match="either embargo .* or embargo_delta"):
        CVConfig(scheme="timeseries", embargo=1, embargo_delta=2.0)


def test_gap_delta_and_max_train_bounds() -> None:
    with pytest.raises(ValueError):
        CVConfig(scheme="timeseries", purge_delta=0)  # gt=0
    with pytest.raises(ValueError):
        CVConfig(scheme="timeseries", max_train_size=0)  # gt=0
    with pytest.raises(ValueError):
        CVConfig(scheme="timeseries_period", period="month", max_train_periods=0)  # gt=0


def test_weighting_default_and_round_trips() -> None:
    # ADR-0098: default is pooled (current behavior, NFR-5); the field round-trips + enters the fingerprint
    assert CVConfig().weighting == "pooled"
    rc = RunConfig(cv=CVConfig(scheme="timeseries_period", period="month", weighting="period"))
    assert RunConfig.model_validate_json(rc.model_dump_json()) == rc
    assert rc.model_dump(mode="json")["cv"]["weighting"] == "period"


def test_parse_raises_config_error_on_invalid_budget() -> None:
    with pytest.raises(ConfigError):
        RunConfig.parse({"budget": {"mode": "time"}})  # missing time_budget_s


def test_parse_raises_config_error_on_unknown_field() -> None:
    with pytest.raises(ConfigError):
        RunConfig.parse({"unknown_field": 1})


def test_config_is_frozen() -> None:
    rc = RunConfig()
    with pytest.raises(Exception):
        rc.seed = 99  # type: ignore[misc]


# --- M5a-wire: unbounded "none" budget mode (ADR-0032 §5a) ---


def test_budget_none_is_default_and_round_trips() -> None:
    assert BudgetConfig().mode == "none"
    assert RunConfig().budget == BudgetConfig(mode="none")
    bc = BudgetConfig()
    assert BudgetConfig.model_validate_json(bc.model_dump_json()) == bc


@pytest.mark.parametrize("stray", [{"n_trials": 50}, {"time_budget_s": 10.0}])
def test_budget_none_forbids_stray_limit(stray: dict) -> None:
    # a stray limit under "none" would make the manifest contradictory -> rejected (ADR-0032 §5a)
    with pytest.raises(ValueError, match="unbounded"):
        BudgetConfig(mode="none", **stray)


def test_budget_rejects_dead_cross_axis_limit() -> None:
    # F3.5/F3.11 residual: a limit of the non-selected axis would be silently dead -> rejected
    with pytest.raises(ValueError, match="ignores n_trials"):
        BudgetConfig(mode="time", time_budget_s=10.0, n_trials=5)
    with pytest.raises(ValueError, match="ignores time_budget_s"):
        BudgetConfig(mode="trials", n_trials=5, time_budget_s=10.0)
    # memory stays orthogonal to the mode (ADR-0039 §2)
    assert BudgetConfig(mode="time", time_budget_s=10.0, memory_limit_mb=256).mode == "time"


def test_budget_significance_default_round_trips() -> None:
    rc = RunConfig(significance="off")
    assert RunConfig.model_validate_json(rc.model_dump_json()) == rc
    assert RunConfig().significance == "bootstrap"


# --- M5 run-modes: additive RunConfig.run_mode (ADR-0038 §1, FR-RM-1) ---


def test_run_mode_default_full() -> None:
    # default preserves M5: "full" runs selection -> final-fit -> calibration -> holdout
    assert RunConfig().run_mode == "full"


def test_run_mode_round_trips() -> None:
    rc = RunConfig(run_mode="selection")
    assert rc.run_mode == "selection"
    assert RunConfig.model_validate_json(rc.model_dump_json()) == rc


def test_invalid_run_mode_rejected_by_literal() -> None:
    # the Literal rejects an unknown mode at the model boundary (the facade fit guard maps it to
    # ConfigError; the direct constructor raises pydantic's ValidationError) — ADR-0038 §1
    with pytest.raises(ValueError):
        RunConfig(run_mode="evaluation")  # type: ignore[arg-type]


# --- M6a feature engineering: additive RunConfig.fe (FEConfig) (ADR-0040 §4, FR-FE-1) ---


def test_fe_config_default_all_off() -> None:
    # default preserves M5: no FE transformer active
    fe = FEConfig()
    assert (fe.target_encoding, fe.frequency_encoding, fe.intersections) == (False, False, False)
    assert fe.te_smoothing == 10.0 and fe.max_pairs == 50
    assert RunConfig().fe == FEConfig()


def test_fe_config_round_trips_in_runconfig() -> None:
    rc = RunConfig(fe=FEConfig(target_encoding=True, te_smoothing=5.0, intersections=True))
    assert RunConfig.model_validate_json(rc.model_dump_json()) == rc
    # FE is part of the config dump -> changes the run-fingerprint (ADR-0042 §4)
    assert rc.model_dump(mode="json")["fe"]["target_encoding"] is True


def test_fe_config_is_frozen_and_forbids_extra() -> None:
    with pytest.raises(Exception):
        FEConfig(unknown=1)  # type: ignore[call-arg]
    with pytest.raises(ValueError):
        FEConfig(te_smoothing=-1.0)  # smoothing k must be >= 0


# --- M6b feature selection: additive RunConfig.fs (FeatureSelectionConfig) (ADR-0043 §5, FR-FS-1) ---


def test_fs_config_default_off() -> None:
    # default preserves M6a/M5 by content: feature selection disabled (RunConfig.fs is None)
    assert RunConfig().fs is None
    fs = FeatureSelectionConfig()
    assert (fs.strategy, fs.cutoff, fs.top_frac, fs.min_features, fs.n_probes) == (
        "importance",
        "top_frac",
        0.5,
        1,
        3,
    )


def test_fs_config_round_trips_in_runconfig() -> None:
    rc = RunConfig(fs=FeatureSelectionConfig(strategy="random_probe", cutoff="auto", n_probes=5))
    assert RunConfig.model_validate_json(rc.model_dump_json()) == rc
    # FS is part of the config dump -> changes the run-fingerprint (ADR-0045 §5)
    assert rc.model_dump(mode="json")["fs"]["strategy"] == "random_probe"


def test_fs_config_is_frozen_and_forbids_extra() -> None:
    with pytest.raises(Exception):
        FeatureSelectionConfig(unknown=1)  # type: ignore[call-arg]
    with pytest.raises(ValueError):
        FeatureSelectionConfig(top_frac=1.5)  # fraction must be in (0, 1]
    fs = FeatureSelectionConfig()
    with pytest.raises(Exception):
        fs.strategy = "random_probe"  # type: ignore[misc]


def test_fs_config_top_k_requires_top_k() -> None:
    # cutoff='top_k' without top_k is contradictory -> rejected at the model boundary (ADR-0043 §5)
    with pytest.raises(ValueError, match="top_k"):
        FeatureSelectionConfig(cutoff="top_k")
    assert FeatureSelectionConfig(cutoff="top_k", top_k=5).top_k == 5


# --- M6c compare config: additive fields + validation (ADR-0046/0049, FR-FSC-1) ---


def test_fsc_compare_default_off_and_new_fields() -> None:
    fs = FeatureSelectionConfig()
    assert fs.compare is None  # default = single-strategy path (M6b)
    assert (
        fs.selection_holdout,
        fs.n_runs,
        fs.null_percentile,
        fs.seq_min_features,
        fs.seq_patience,
    ) == (
        0.25,
        30,
        95.0,
        1,
        2,
    )


def test_fsc_compare_round_trips_and_accepts_new_strategies() -> None:
    rc = RunConfig(
        fs=FeatureSelectionConfig(compare=("importance", "null_importance", "sequential"))
    )
    assert RunConfig.model_validate_json(rc.model_dump_json()) == rc
    assert rc.model_dump(mode="json")["fs"]["compare"] == [
        "importance",
        "null_importance",
        "sequential",
    ]


def test_fsc_compare_rejects_empty_and_duplicates() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        FeatureSelectionConfig(compare=())
    with pytest.raises(ValueError, match="duplicate"):
        FeatureSelectionConfig(compare=("importance", "importance"))


def test_fsc_compare_unknown_strategy_rejected() -> None:
    with pytest.raises(Exception):
        FeatureSelectionConfig(compare=("importance", "nope"))  # type: ignore[arg-type]


# --- M6d: additive arbitration / structure-aware / shap fields (ADR-0050/0051/0052, FR-FSH-4/9) ---


def test_fsd_config_default_new_fields() -> None:
    # defaults preserve M6c: holdout arbitration, structure block 50, no shap cost-cap
    fs = FeatureSelectionConfig()
    assert (fs.arbitration, fs.arbitration_n_splits, fs.null_block_size, fs.shap_max_samples) == (
        "holdout",
        5,
        50,
        None,
    )


def test_fsd_config_accepts_shap_strategy() -> None:
    assert FeatureSelectionConfig(strategy="shap").strategy == "shap"
    assert FeatureSelectionConfig(compare=("importance", "shap")).compare == ("importance", "shap")


def test_fsd_arbitration_round_trips_in_runconfig() -> None:
    rc = RunConfig(
        fs=FeatureSelectionConfig(
            compare=("importance", "null_importance"), arbitration="nested", arbitration_n_splits=4
        )
    )
    assert RunConfig.model_validate_json(rc.model_dump_json()) == rc
    assert rc.model_dump(mode="json")["fs"]["arbitration"] == "nested"


def test_fsd_arbitration_n_splits_minimum() -> None:
    with pytest.raises(ValueError):
        FeatureSelectionConfig(arbitration_n_splits=1)  # nested needs >= 2 folds
    assert FeatureSelectionConfig(arbitration_n_splits=2).arbitration_n_splits == 2


def test_fsd_null_block_size_and_shap_max_samples_bounds() -> None:
    with pytest.raises(ValueError):
        FeatureSelectionConfig(null_block_size=1)  # block needs >= 2 rows to permute within
    with pytest.raises(ValueError):
        FeatureSelectionConfig(shap_max_samples=0)  # cost-cap must be positive when set
    assert FeatureSelectionConfig(shap_max_samples=1000).shap_max_samples == 1000


def test_fsd_fs_none_keeps_m6b_fingerprint() -> None:
    # new fields live INSIDE FeatureSelectionConfig -> fs=None dumps "fs": null, no new keys (ADR-0049 §4)
    assert RunConfig().model_dump(mode="json")["fs"] is None


# --- M6e: per-fold re-selection arbitration (ADR-0054, FR-FSE-1) ---


def test_fse_arbitration_accepts_nested_per_fold() -> None:
    # third arbitration value; default stays "holdout" (== M6c content)
    assert FeatureSelectionConfig().arbitration == "holdout"
    fs = FeatureSelectionConfig(
        compare=("importance", "null_importance"), arbitration="nested_per_fold"
    )
    assert fs.arbitration == "nested_per_fold"
    rc = RunConfig(fs=fs)
    assert RunConfig.model_validate_json(rc.model_dump_json()) == rc


def test_fse_block_mode_defaults_and_time_window_requires_window() -> None:
    # ADR-0055: default mode is "rank" (== M6d); "time_window" requires null_block_window (the Δt)
    assert (
        FeatureSelectionConfig().null_block_mode,
        FeatureSelectionConfig().null_block_window,
    ) == ("rank", None)
    with pytest.raises(ValueError, match="null_block_window"):
        FeatureSelectionConfig(null_block_mode="time_window")  # no Δt -> ConfigError
    fs = FeatureSelectionConfig(null_block_mode="time_window", null_block_window=7.0)
    assert fs.null_block_window == 7.0
    with pytest.raises(ValueError):
        FeatureSelectionConfig(null_block_mode="time_window", null_block_window=0)  # must be > 0


def test_fse_shap_perturbation_defaults_and_bounds() -> None:
    # ADR-0056: default perturbation is tree_path_dependent (== M6d); interventional + background cap accepted
    assert FeatureSelectionConfig().shap_perturbation == "tree_path_dependent"
    assert FeatureSelectionConfig().shap_background_samples is None
    fs = FeatureSelectionConfig(
        strategy="shap", shap_perturbation="interventional", shap_background_samples=64
    )
    assert fs.shap_perturbation == "interventional" and fs.shap_background_samples == 64
    with pytest.raises(ValueError):
        FeatureSelectionConfig(shap_background_samples=0)  # must be > 0 when set


# --- M6f: data-shape auto-defaults, kmeans background, hard cost-budget (ADR-0057/0058/0060, NFR-FSF-3) ---


def test_m6f_fields_default_inert() -> None:
    # new fields/sentinels are opt-in; defaults stay identical to M6e (auto is NOT the default)
    fs = FeatureSelectionConfig()
    assert fs.shap_background == "linspace"
    assert fs.cost_budget_refits is None
    assert (fs.arbitration, fs.null_block_mode) == ("holdout", "rank")
    # opt-in values accepted and round-trip
    opt = FeatureSelectionConfig(
        compare=("importance", "shap"),
        arbitration="auto",
        null_block_mode="auto",
        shap_background="kmeans",
        cost_budget_refits=100,
    )
    assert (opt.arbitration, opt.null_block_mode, opt.shap_background) == ("auto", "auto", "kmeans")
    assert opt.cost_budget_refits == 100
    rc = RunConfig(fs=opt)
    assert RunConfig.model_validate_json(rc.model_dump_json()) == rc
    with pytest.raises(ValueError):
        FeatureSelectionConfig(cost_budget_refits=0)  # gt=0 when set


def test_fs_config_fields_pinned() -> None:
    # snapshot the config shape: __all__ pins the top-level surface, but new FS fields are attributes ->
    # guard their set explicitly against silent schema drift (NFR-FSF-3, completeness-critic R2)
    assert set(FeatureSelectionConfig.model_fields) == {
        "strategy",
        "compare",
        "selection_holdout",
        "arbitration",
        "arbitration_n_splits",
        "null_block_size",
        "null_block_mode",
        "null_block_window",
        "shap_max_samples",
        "shap_perturbation",
        "shap_background_samples",
        "shap_background",
        "cost_budget_refits",
        "cutoff",
        "top_k",
        "top_frac",
        "min_features",
        "n_probes",
        "n_runs",
        "null_percentile",
        "seq_min_features",
        "seq_patience",
        "random_state",
    }
