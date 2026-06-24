"""M5b: the pure run-report assembler + JSON serialization (ADR-0033, FR-M5-4/6, NFR-M5-3).

``build_run_report`` is a Humble Object: assembled on a hand-built ``SliceResult``/
``RunConfig`` with no training. It must emit JSON primitives only (numpy-carrying
``SliceResult`` fields excluded) and read budget/significance provenance from the resolved
config (truthful).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from honestml.application import (
    RUN_MANIFEST_VERSION,
    BudgetReport,
    FailedCandidate,
    FeatureSelectionReport,
    LeaderboardEntry,
    SliceResult,
    build_run_report,
)
from honestml.composition import save_run_report
from honestml.core import BudgetConfig, CVConfig, FeatureSelectionConfig, FEConfig, RunConfig

pytestmark = pytest.mark.unit

_V1_KEYS = {
    "run_manifest_version",
    "honestml_version",
    "config",
    "timings",
    "winner",
    "leaderboard",
    "band",
    "budget",
    "significance",
}


def _result(**kw) -> SliceResult:
    lb = [
        LeaderboardEntry(
            model_id="a", score=0.9, metric="roc_auc", n_features=3, train_time=0.1, rank=1
        ),
        LeaderboardEntry(
            model_id="b", score=0.8, metric="roc_auc", n_features=3, train_time=0.2, rank=2
        ),
    ]
    # route flat FS/budget kwargs into the nested reports (the test surface stays flat for brevity)
    fs_fields = {
        k: kw.pop(k)
        for k in (
            "selected_features",
            "selection_gate",
            "selected_strategy",
            "per_strategy",
            "winner_rule",
            "band_members",
            "per_strategy_std",
            "null_block_stats",
            "arbitration_effective",
            "fold_subset_jaccard",
            "per_strategy_mean_features",
        )
        if k in kw
    }
    budget = {
        dst: kw.pop(src)
        for src, dst in (
            ("skipped_by_budget", "skipped"),
            ("budget_exhausted", "exhausted"),
            ("exhausted_by", "exhausted_by"),
        )
        if src in kw
    }
    base = dict(leaderboard=lb, best_model_id="a", candidates=[])
    base.update(kw)
    if budget:
        base["budget"] = BudgetReport(**budget)
    if "selected_features" in fs_fields:
        base["feature_selection"] = FeatureSelectionReport(**fs_fields)
    return SliceResult(**base)


def test_report_v1_schema_keys() -> None:
    report = build_run_report(run_config=RunConfig(), timings={}, result=_result())
    # v1 keys are a SUBSET: additive RC keys (run_fingerprint/cache) may extend it (ADR-0037 §3)
    assert _V1_KEYS <= set(report)
    assert report["run_manifest_version"] == RUN_MANIFEST_VERSION
    assert report["winner"] == "a"
    assert report["leaderboard"][0] == {"model_id": "a", "score": 0.9, "rank": 1}
    assert isinstance(report["config"], dict)


def test_preset_block_additive() -> None:
    # M9-1 (ADR-0074 §3): preset provenance is an additive top-level key — None when no
    # preset was requested, the {name, applied} dict otherwise; version not bumped
    report = build_run_report(run_config=RunConfig(), timings={}, result=_result())
    assert report["preset"] is None
    block = {"name": "fast", "applied": ["cv"]}
    with_preset = build_run_report(
        run_config=RunConfig(), timings={}, result=_result(), preset=block
    )
    assert with_preset["preset"] == block
    assert with_preset["run_manifest_version"] == RUN_MANIFEST_VERSION  # additive, not bumped


def test_native_routing_block_additive() -> None:
    # FR-5 (ADR-0095): native-routing verdict is an additive top-level key — None when the gate demoted
    # nothing, else {native, demoted_to_codes:[{column, reason}]}; RUN_MANIFEST_VERSION not bumped.
    report = build_run_report(run_config=RunConfig(), timings={}, result=_result())
    assert report["native_routing"] is None
    routed = build_run_report(
        run_config=RunConfig(),
        timings={},
        result=_result(native_routing={"city": "native", "user_id": "high_cardinality"}),
    )
    assert routed["native_routing"] == {
        "native": ["city"],
        "demoted_to_codes": [{"column": "user_id", "reason": "high_cardinality"}],
    }
    assert routed["run_manifest_version"] == RUN_MANIFEST_VERSION  # additive, not bumped


def test_cv_split_block_additive() -> None:
    # ADR-0096 §4: period-CV split diagnostics are an additive top-level key — None for non-period
    # runs, the densified counts dict otherwise; RUN_MANIFEST_VERSION not bumped.
    report = build_run_report(run_config=RunConfig(), timings={}, result=_result())
    assert report["cv"] is None
    meta = {"period": "month", "n_periods": 12, "n_folds": 4, "n_dropped_empty": 0}
    rep = build_run_report(run_config=RunConfig(), timings={}, result=_result(cv_split=meta))
    assert rep["cv"] == meta
    assert json.dumps(rep["cv"])  # JSON-serializable primitives only
    assert rep["run_manifest_version"] == RUN_MANIFEST_VERSION  # additive, not bumped


def test_failed_candidates_additive() -> None:
    # F4.2: the isolation outcome is visible in the report itself, not only in logging
    report = build_run_report(run_config=RunConfig(), timings={}, result=_result())
    assert report["failed"] == []
    failed = [FailedCandidate(id="xgb", reason="boom")]
    rep = build_run_report(run_config=RunConfig(), timings={}, result=_result(failed=failed))
    assert rep["failed"] == [{"model_id": "xgb", "reason": "boom"}]
    assert rep["run_manifest_version"] == RUN_MANIFEST_VERSION  # additive, not bumped


def test_task_metric_additive() -> None:
    # M9-2 (ADR-0075 §2): task/metric identity keys — None for facade-less callers,
    # the strings when the facade supplies them; version not bumped
    report = build_run_report(run_config=RunConfig(), timings={}, result=_result())
    assert report["task"] is None and report["metric"] is None
    named = build_run_report(
        run_config=RunConfig(), timings={}, result=_result(), task="binary", metric="roc_auc"
    )
    assert named["task"] == "binary" and named["metric"] == "roc_auc"
    assert named["run_manifest_version"] == RUN_MANIFEST_VERSION  # additive, not bumped


def _tight_lb() -> list[LeaderboardEntry]:
    return [
        LeaderboardEntry(
            model_id="a", score=0.81, metric="accuracy", n_features=3, train_time=0.1, rank=1
        ),
        LeaderboardEntry(
            model_id="b", score=0.80, metric="accuracy", n_features=3, train_time=0.1, rank=2
        ),
    ]


def test_holdout_optimism_flags_split_dependence() -> None:
    # #11c: a holdout markedly better than the winner's OOF is flagged (the carve is not independent)
    flagged = build_run_report(
        run_config=RunConfig(),
        timings={},
        result=_result(leaderboard=_tight_lb(), holdout_score=0.96),
    )["holdout_optimism"]
    assert flagged is not None and flagged["relative_optimism"] > 0.10
    assert "finding #11" in flagged["message"]


def test_holdout_optimism_silent_when_benign() -> None:
    # a holdout at/below the OOF (honest) and a holdout-less run both carry no diagnostic
    benign = build_run_report(
        run_config=RunConfig(),
        timings={},
        result=_result(leaderboard=_tight_lb(), holdout_score=0.80),
    )["holdout_optimism"]
    assert benign is None
    no_holdout = build_run_report(
        run_config=RunConfig(), timings={}, result=_result(leaderboard=_tight_lb())
    )["holdout_optimism"]
    assert no_holdout is None


def test_holdout_score_additive() -> None:
    # M8c (ADR-0072 §5): the honest final estimate is an additive top-level key — None in
    # selection mode / outer_holdout=0, the float when the facade scored the holdout
    report = build_run_report(run_config=RunConfig(), timings={}, result=_result())
    assert report["holdout_score"] is None
    scored = build_run_report(
        run_config=RunConfig(), timings={}, result=_result(holdout_score=0.77)
    )
    assert scored["holdout_score"] == 0.77
    assert scored["run_manifest_version"] == RUN_MANIFEST_VERSION  # additive, not bumped


def test_fe_config_in_report_additive() -> None:
    # FR-FE-7: the FE catalog is observable via the resolved config dump; manifest version unchanged
    fe = FEConfig(target_encoding=True, intersections=True)
    report = build_run_report(run_config=RunConfig(fe=fe), timings={}, result=_result())
    assert report["config"]["fe"]["target_encoding"] is True
    assert report["config"]["fe"]["intersections"] is True
    assert report["run_manifest_version"] == RUN_MANIFEST_VERSION  # additive, not bumped


def test_hpo_block_additive() -> None:
    # FR-HPO-6: the hpo block is an additive top-level key — None when off, the dict when supplied
    assert build_run_report(run_config=RunConfig(), timings={}, result=_result())["hpo"] is None
    block = {"backend": "optuna", "tuned": {"x": {"chosen_params": {"C": 1.0}}}}
    report = build_run_report(run_config=RunConfig(), timings={}, result=_result(), hpo=block)
    assert report["hpo"] == block
    assert report["run_manifest_version"] == RUN_MANIFEST_VERSION  # not bumped


def test_ensemble_block_additive() -> None:
    # FR-ENS-6: the ensemble block is an additive top-level key — None when off, the dict when supplied
    assert (
        build_run_report(run_config=RunConfig(), timings={}, result=_result())["ensemble"] is None
    )
    block = {
        "applied": True,
        "method": "caruana",
        "member_ids": ["a", "b"],
        "weights": {"a": 0.6, "b": 0.4},
        "gate_reason": "significant_improvement",
        "oof_delta": 0.01,
    }
    report = build_run_report(run_config=RunConfig(), timings={}, result=_result(), ensemble=block)
    assert report["ensemble"] == block
    assert report["run_manifest_version"] == RUN_MANIFEST_VERSION  # not bumped


def test_fs_observable_in_report_additive() -> None:
    # FR-FS-6: FS config in the config dump; the kept subset is an additive outcome key
    fs = FeatureSelectionConfig(strategy="random_probe", cutoff="auto")
    report = build_run_report(
        run_config=RunConfig(fs=fs), timings={}, result=_result(selected_features=("n1", "c1"))
    )
    assert report["config"]["fs"]["strategy"] == "random_probe"
    assert report["feature_selection"] == {
        "strategy": "random_probe",
        "n_selected": 2,
        "selected": ["n1", "c1"],
    }
    assert report["run_manifest_version"] == RUN_MANIFEST_VERSION  # additive, not bumped


def test_fs_off_report_has_null_outcome() -> None:
    report = build_run_report(run_config=RunConfig(), timings={}, result=_result())
    assert report["config"]["fs"] is None
    assert report["feature_selection"] is None


def test_fs_compare_observable_in_report() -> None:
    # FR-FSC-5: compare adds strategies_evaluated/per_strategy/winner additively (ADR-0049 §3)
    fs = FeatureSelectionConfig(compare=("importance", "sequential"))
    report = build_run_report(
        run_config=RunConfig(fs=fs),
        timings={},
        result=_result(
            selected_features=("n1", "c1"),
            selected_strategy="sequential",
            per_strategy=(("importance", 3, 0.81), ("sequential", 2, 0.86)),
        ),
    )
    block = report["feature_selection"]
    assert block["winner"] == "sequential" and block["strategy"] == "sequential"
    assert block["strategies_evaluated"] == ["importance", "sequential"]
    assert block["per_strategy"]["sequential"] == {"n_selected": 2, "arb_score": 0.86}
    assert report["run_manifest_version"] == RUN_MANIFEST_VERSION  # additive, not bumped


def test_fs_nested_winner_rule_observable_in_report() -> None:
    # M6d (ADR-0052/0053): nested arbitration surfaces winner_rule, band_members and arb_score_std additively
    fs = FeatureSelectionConfig(compare=("importance", "shap"), arbitration="nested")
    report = build_run_report(
        run_config=RunConfig(fs=fs),
        timings={},
        result=_result(
            selected_features=("n1",),
            selected_strategy="importance",
            per_strategy=(("importance", 1, 0.83), ("shap", 4, 0.84)),
            winner_rule="band_tiebreak",
            band_members=("importance", "shap"),
            per_strategy_std=(("importance", 0.01), ("shap", 0.02)),
        ),
    )
    block = report["feature_selection"]
    assert block["winner_rule"] == "band_tiebreak"
    assert block["band_members"] == ["importance", "shap"]
    assert block["per_strategy"]["shap"]["arb_score_std"] == 0.02


def test_fs_per_fold_observable_in_report() -> None:
    # M6e (ADR-0054 §6): per-fold re-selection surfaces arbitration_effective, per_fold_reselection,
    # fold_subset_jaccard and the raw mean per-fold subset size additively.
    fs = FeatureSelectionConfig(compare=("importance", "shap"), arbitration="nested_per_fold")
    report = build_run_report(
        run_config=RunConfig(fs=fs),
        timings={},
        result=_result(
            selected_features=("n1",),
            selected_strategy="importance",
            per_strategy=(("importance", 2, 0.83), ("shap", 4, 0.84)),
            winner_rule="band_tiebreak",
            band_members=("importance", "shap"),
            arbitration_effective="nested_per_fold",
            fold_subset_jaccard=0.66,
            per_strategy_mean_features=(("importance", 2.4), ("shap", 3.8)),
        ),
    )
    block = report["feature_selection"]
    assert block["arbitration_effective"] == "nested_per_fold"
    assert block["per_fold_reselection"] is True
    assert block["fold_subset_jaccard"] == 0.66
    assert block["per_strategy"]["importance"]["mean_n_features"] == 2.4


def test_fs_null_block_stats_observable_in_report() -> None:
    # M6d (ADR-0050 §5): structure-aware null surfaces block diagnostics additively
    fs = FeatureSelectionConfig(compare=("importance", "null_importance"))
    report = build_run_report(
        run_config=RunConfig(fs=fs),
        timings={},
        result=_result(
            selected_features=("n1",),
            selected_strategy="null_importance",
            per_strategy=(("importance", 1, 0.8), ("null_importance", 1, 0.82)),
            null_block_stats={"n_blocks": 24.0, "mean_block_size": 50.0, "degenerate_blocks": 0.0},
        ),
    )
    assert report["feature_selection"]["null_block_stats"]["n_blocks"] == 24.0


def test_per_fold_degenerate_in_report() -> None:
    # M6f (ADR-0059 §4): per-fold degenerate aggregates ride inside null_block_stats (merged in run_slice)
    fs = FeatureSelectionConfig(
        compare=("importance", "null_importance"), arbitration="nested_per_fold"
    )
    report = build_run_report(
        run_config=RunConfig(fs=fs),
        timings={},
        result=_result(
            selected_features=("n1",),
            selected_strategy="importance",
            per_strategy=(("importance", 1, 0.8), ("null_importance", 1, 0.82)),
            arbitration_effective="nested_per_fold",
            null_block_stats={
                "n_blocks": 5.0,
                "mean_block_size": 8.0,
                "degenerate_blocks": 1.0,
                "per_fold_degenerate_mean": 0.6,
                "per_fold_degenerate_max": 0.75,
                "per_fold_n_blocks_mean": 4.0,
            },
        ),
    )
    nbs = report["feature_selection"]["null_block_stats"]
    assert (
        nbs["per_fold_degenerate_mean"] == 0.6 and nbs["per_fold_n_blocks_mean"] == 4.0
    )  # full-DEV + per-fold


def test_resolved_arbitration_observable() -> None:
    # M6f (ADR-0057/0058 §4): fs_resolution provenance (auto -> concrete) surfaces additively
    fs = FeatureSelectionConfig(compare=("importance", "null_importance"), arbitration="auto")
    report = build_run_report(
        run_config=RunConfig(fs=fs),
        timings={},
        result=_result(selected_features=("n1",), selected_strategy="importance"),
        fs_resolution={"arbitration_requested": "auto", "arbitration_resolved_from": "auto"},
    )
    assert report["feature_selection"]["fs_resolution"]["arbitration_resolved_from"] == "auto"


def test_cost_downgrade_observable_in_fs_resolution() -> None:
    # M6f (ADR-0058 §4): cost-downgrade provenance lives in fs_resolution, NOT in arbitration_effective
    fs = FeatureSelectionConfig(
        compare=("importance", "null_importance"),
        arbitration="nested_per_fold",
        cost_budget_refits=400,
    )
    report = build_run_report(
        run_config=RunConfig(fs=fs),
        timings={},
        result=_result(
            selected_features=("n1",),
            selected_strategy="importance",
            per_strategy=(("importance", 1, 0.8), ("null_importance", 1, 0.82)),
            arbitration_effective="nested",
        ),
        fs_resolution={
            "arbitration_requested": "nested_per_fold",
            "arbitration_resolved_from": "cost_budget",
        },
    )
    res = report["feature_selection"]["fs_resolution"]
    assert res == {
        "arbitration_requested": "nested_per_fold",
        "arbitration_resolved_from": "cost_budget",
    }
    # arbitration_effective reflects what actually ran (nested) — no *_cost_downgraded suffix (R2 redesign)
    assert report["feature_selection"]["arbitration_effective"] == "nested"


def test_timings_has_selection_and_refit_keys() -> None:
    timings = {"run": {"selection": 0.1, "refit": 0.0}}
    report = build_run_report(run_config=RunConfig(), timings=timings, result=_result())
    assert set(report["timings"]["run"]) == {"selection", "refit"}


def test_budget_outcome_in_report() -> None:
    report = build_run_report(
        run_config=RunConfig(),
        timings={},
        result=_result(budget_exhausted=True, skipped_by_budget=("c",), exhausted_by="trials"),
    )
    assert report["budget"] == {
        "mode": "none",
        "exhausted": True,
        "skipped": ["c"],
        "exhausted_by": "trials",
    }


def test_budget_block_exhausted_by_memory() -> None:
    report = build_run_report(
        run_config=RunConfig(),
        timings={},
        result=_result(budget_exhausted=True, skipped_by_budget=("c",), exhausted_by="memory"),
    )
    assert report["budget"]["exhausted_by"] == "memory"  # truthful axis (ADR-0039 §5)


def test_memory_limit_in_config_not_duplicated_in_budget_block() -> None:
    cfg = RunConfig(budget=BudgetConfig(memory_limit_mb=512))
    report = build_run_report(run_config=cfg, timings={}, result=_result())
    assert report["config"]["budget"]["memory_limit_mb"] == 512  # input lives in the config dump
    assert "memory_limit_mb" not in report["budget"]  # not duplicated in the outcome block (fix m4)
    assert report["budget"]["exhausted_by"] is None  # within budget


def test_within_budget_outcome_empty() -> None:
    report = build_run_report(run_config=RunConfig(), timings={}, result=_result())
    assert report["budget"]["exhausted"] is False
    assert report["budget"]["skipped"] == []


def test_time_mode_flagged_in_report() -> None:
    cfg = RunConfig(budget=BudgetConfig(mode="time", time_budget_s=10.0))
    report = build_run_report(run_config=cfg, timings={}, result=_result())
    assert report["budget"]["mode"] == "time"


def test_significance_mode_resolved() -> None:
    cfg = RunConfig(significance="off")
    report = build_run_report(run_config=cfg, timings={}, result=_result())
    assert report["significance"] == "off"


def test_report_truthful_resolved_scheme() -> None:
    cfg = RunConfig(cv=CVConfig(scheme="kfold", n_splits=3))
    report = build_run_report(run_config=cfg, timings={}, result=_result())
    assert report["config"]["cv"]["scheme"] == "kfold"
    assert report["config"]["seed"] == cfg.seed


def test_run_report_json_serializable() -> None:
    # a numpy-carrying SliceResult must still yield a JSON-serializable report (numpy excluded)
    result = _result(oof_fold_index=np.array([0, 1, 0, 1], dtype=np.int64))
    report = build_run_report(
        run_config=RunConfig(), timings={"run": {"selection": 0.1}}, result=result
    )
    text = json.dumps(report)  # must not raise
    assert json.loads(text)["winner"] == "a"


def test_save_run_report_writes_json(tmp_path) -> None:
    report = build_run_report(run_config=RunConfig(), timings={}, result=_result())
    out = save_run_report(report, tmp_path)  # directory -> run_report.json
    assert out == tmp_path / "run_report.json"
    assert (
        json.loads(out.read_text(encoding="utf-8"))["run_manifest_version"] == RUN_MANIFEST_VERSION
    )


def test_save_run_report_overwrite_false_raises(tmp_path) -> None:
    report = build_run_report(run_config=RunConfig(), timings={}, result=_result())
    target = tmp_path / "r.json"
    save_run_report(report, target)
    with pytest.raises(FileExistsError):
        save_run_report(report, target, overwrite=False)


# --- M5-resume RC-d2: additive run_fingerprint + cache observability (ADR-0037 §3, FR-RC-6) ---


def test_report_has_fingerprint_and_cache_keys() -> None:
    report = build_run_report(
        run_config=RunConfig(), timings={}, result=_result(), run_fingerprint="abc123"
    )
    assert _V1_KEYS <= set(report)  # additive: v1 is a subset
    assert report["run_fingerprint"] == "abc123"
    assert report["cache"] == {"enabled": False, "reused": [], "computed": []}
    assert report["run_manifest_version"] == RUN_MANIFEST_VERSION  # NOT bumped (additive)


def test_cache_block_truthful_when_enabled() -> None:
    result = _result(reused=("a",), computed=("b",))
    report = build_run_report(
        run_config=RunConfig(), timings={}, result=result, run_fingerprint="fp", cache_enabled=True
    )
    assert report["cache"] == {"enabled": True, "reused": ["a"], "computed": ["b"]}


def test_cache_disabled_ignores_slice_lists() -> None:
    # cache off: even a SliceResult carrying computed ids reports empty lists (truthful enabled=False)
    result = _result(reused=(), computed=("a", "b"))
    report = build_run_report(
        run_config=RunConfig(), timings={}, result=result, run_fingerprint="fp", cache_enabled=False
    )
    assert report["cache"] == {"enabled": False, "reused": [], "computed": []}


def test_report_with_cache_keys_json_serializable() -> None:
    import json as _json

    result = _result(reused=("a",), computed=("b",))
    report = build_run_report(
        run_config=RunConfig(), timings={}, result=result, run_fingerprint="fp", cache_enabled=True
    )
    assert _json.loads(_json.dumps(report))["cache"]["reused"] == ["a"]
