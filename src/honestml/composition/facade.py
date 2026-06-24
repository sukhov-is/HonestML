"""The sklearn-compatible facade ``AutoML``.

``AutoML(BaseEstimator, ClassifierMixin)`` is the single public entry point.
``__init__`` only stores hyperparameters as given (no computation, no data
inference) so ``get_params``/``set_params``/``clone`` and ``Pipeline`` work. All
logic lives in the use-case and ports; the facade adapts the sklearn contract to
``Reader`` + ``run_slice`` + ``refit_best`` and holds a :class:`FittedModel` as the
single inference path (shared with the standalone artifact).
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

from honestml.application import (
    EnsembleOutcome,
    FeatureSelectionBundle,
    build_run_report,
    ensemble_selection,
    refit_best,
    refit_members,
    run_slice,
    tune_estimators,
)
from honestml.core import (
    BudgetConfig,
    ConfigError,
    CVConfig,
    EnsembleConfig,
    ExperimentTracker,
    FeatureSelectionConfig,
    FEConfig,
    HPOConfig,
    NotFittedError,
    RunConfig,
    RunContext,
    Task,
    TimeOrderedSplitter,
    TrackerConfig,
    get_logger,
)
from honestml.core.config import RunMode, SignificanceMode

if TYPE_CHECKING:
    from honestml.adapters import JoblibCandidateCache, RunBudget
    from honestml.application import SliceResult
    from honestml.core import Dataset, Estimator
    from honestml.core.task import TaskKind

    from .build import Components

from .artifact import FittedModel
from .build import _normalize_cv, build_default_components, resolve_fs_defaults

logger = get_logger("composition.facade")

# estimator name -> distribution whose version belongs in the run-fingerprint (ADR-0035 §1); sklearn
# (linear/baseline + all splitters) and numpy are always included by `_packages_for`.
_ESTIMATOR_PACKAGES = {"catboost": "catboost", "lightgbm": "lightgbm", "xgboost": "xgboost"}


def _packages_for(estimators: tuple[str, ...]) -> set[str]:
    """Compute-stack distributions to version-pin in the fingerprint for the resolved estimators."""
    packages = {"scikit-learn", "numpy"}
    for name in estimators:
        pkg = _ESTIMATOR_PACKAGES.get(name)
        if pkg is not None:
            packages.add(pkg)
    return packages


def _hpo_report(
    hpo: HPOConfig, outcomes: dict[str, Any], *, tuned_on_full: bool, time_budget: bool
) -> dict[str, Any]:
    """Assemble the additive run-report ``hpo`` block.

    Surfaces every non-default tuning choice: per-model chosen params + inner score + trials, the cost
    estimate (Σ n_trials × inner_cv), and the honesty disclosures (selection OOF is post-tuning;
    tuned on the full feature space when FS is also on; determinism off under a time budget).

    ``deterministic`` is true only when NO finite Optuna timeout is imposed: neither an explicit
    ``hpo.timeout_s`` nor an active time budget, which forces a fair-share wall-clock cap even when
    ``hpo.timeout_s`` is ``None`` (ADR-0062 §5/§7) — so a budget-capped search is reported honestly.
    """
    block: dict[str, Any] = {
        "backend": hpo.backend,
        "inner_cv": hpo.inner_cv,
        "deterministic": hpo.timeout_s is None and not time_budget,
        "selection_oof_is_post_tuning": True,
        "tuned_on_full_feature_space": tuned_on_full,
        "cost_estimate_fits": len(outcomes) * hpo.n_trials * hpo.inner_cv,
        "tuned": {
            name: {
                "chosen_params": o.best_params,
                "inner_best_score": o.best_score,
                "n_trials_run": o.n_trials_run,
            }
            for name, o in outcomes.items()
        },
    }
    if not outcomes:
        block["note"] = (
            "no tunable models"  # hpo set but nothing to tune (ADR-0062 §5) — not silent
        )
    return block


_W_EPS = 1e-6

# below this many holdout rows the single holdout score is dominated by sampling noise
_MIN_HOLDOUT_ROWS = 30


def _validate_cv_data_floor(cv: CVConfig, ds: Dataset) -> None:
    """Fail fast when the (dev) data cannot support the resolved CV scheme.

    Catches before any fit what sklearn would otherwise raise mid-CV with a cryptic
    message: stratified folds need every class in each fold, k-fold needs at least
    one row per fold. Time-series floors are enforced by the splitter itself.
    """
    if cv.scheme == "stratified":
        target = ds.target()
        if target is not None:
            counts = np.unique(target, return_counts=True)[1]
            smallest = int(counts.min())
            if smallest < cv.n_splits:
                raise ConfigError(
                    f"stratified cv needs every class in each of the {cv.n_splits} folds, but the "
                    f"least populated class has only {smallest} row(s); reduce cv, merge rare "
                    "classes or collect more data"
                )
    elif cv.scheme == "kfold" and ds.n_rows < cv.n_splits:
        raise ConfigError(
            f"kfold cv needs at least one row per fold ({cv.n_splits} folds > {ds.n_rows} rows); "
            "reduce cv or collect more data"
        )


def _ensemble_report(outcome: EnsembleOutcome) -> dict[str, Any]:
    """The additive ``ensemble`` block for the run-report and the artifact manifest."""
    return {
        "applied": outcome.applied,
        "method": outcome.method,
        "member_ids": list(outcome.member_ids),
        "weights": outcome.weights,
        "gate_reason": outcome.gate_reason,
        "oof_delta": outcome.oof_delta,
    }


class AutoML(BaseEstimator, ClassifierMixin):
    """Fit a small leaderboard for a tabular task and expose the winner."""

    def __init__(
        self,
        task: Task | str = "binary",
        metric: str | None = None,
        cv: int | CVConfig | None = None,
        models: tuple[str, ...] | None = None,
        random_state: int = 42,
        budget: float | BudgetConfig | None = None,
        significance: SignificanceMode = "bootstrap",
        cache: str | Path | None = None,
        run_mode: RunMode = "full",
        feature_engineering: FEConfig | None = None,
        feature_selection: FeatureSelectionConfig | None = None,
        hpo: HPOConfig | None = None,
        ensemble: EnsembleConfig | None = None,
        finalize: bool = True,
        tracker: ExperimentTracker | TrackerConfig | str | None = None,
        preset: str | Mapping[str, Any] | None = None,
    ) -> None:
        # store params verbatim — sklearn clone/get_params invariant (no computation here)
        self.task = task
        self.metric = metric
        self.cv = cv
        self.models = models
        self.random_state = random_state
        # run budget (ADR-0032 §5): float -> time seconds, BudgetConfig -> as-is, None -> unbounded
        self.budget = budget
        # honest significance band on by default; "off" returns a pure argmax (ADR-0034)
        self.significance = significance
        # stage-cache/resume dir (ADR-0037 §1): None -> M5 behavior (no persistence); <dir> -> reuse +
        # resume keyed by run-fingerprint. Stored verbatim; resolved in fit (sklearn invariant).
        self.cache = cache
        # pipeline stop-point (ADR-0038): "full" (default) ships a model; "selection" stops at the
        # leaderboard (no refit/calibration/holdout). Stored verbatim; validated/resolved in fit.
        self.run_mode = run_mode
        # feature-engineering catalog (ADR-0040 §4): None -> M5 unchanged; a FEConfig opts transformers
        # in. Stored verbatim (sklearn clone invariant); validated/gated in fit. datetime deltas are a
        # separate axis driven by Task.report_date (ADR-0018), not this param.
        self.feature_engineering = feature_engineering
        # feature-selection catalog (ADR-0043 §5): None -> off (M6a/M5 unchanged by content); a
        # FeatureSelectionConfig opts in. Stored verbatim; validated/resolved in fit.
        self.feature_selection = feature_selection
        # HPO catalog (ADR-0061/0062): None -> off (M6 unchanged); an HPOConfig tunes each model type
        # on an inner-CV of DEV before the honest selection. Stored verbatim; validated/resolved in fit.
        self.hpo = hpo
        # ensembling catalog (ADR-0063/0064): None -> off (single model); an EnsembleConfig blends the
        # leaderboard after selection and ships a BlendedEstimator only if significantly better. Verbatim.
        self.ensemble = ensemble
        # finalize (ADR-0068): when honest outer_holdout is on, refit the shipped winner on DEV+holdout (all
        # data) AFTER scoring the holdout — the reported score stays the DEV estimate (a conservative lower
        # bound). No-op at outer_holdout=0. It is post-selection, so it is NOT in the run-fingerprint (off ==
        # M7). finalize=False keeps the DEV-refit shipping. Stored verbatim (sklearn clone invariant).
        self.finalize = finalize
        # experiment tracking (ADR-0072 §3): None -> off; "mlflow"/TrackerConfig -> MlflowTracker built in
        # fit; an ExperimentTracker instance is used verbatim (custom backends). Post-selection observability,
        # NOT in the run-fingerprint (the finalize precedent). Stored verbatim (sklearn clone invariant).
        self.tracker = tracker
        # named/custom preset (ADR-0074): fills ONLY the None-left surface parameters in fit; an
        # explicit value always wins. Input sugar — the fingerprint carries the RESOLVED parameters,
        # not the preset name. Stored verbatim (a Mapping is snapshotted once per fit).
        self.preset = preset

    def fit(
        self,
        X: Any,
        y: Any,
        sample_weight: Any | None = None,
        groups: Any | None = None,
        time: Any | None = None,
        label_time: Any | None = None,
    ) -> AutoML:
        """Fit the leaderboard and expose the winner.

        ``groups`` (per-row group labels) enables group-aware CV with
        ``cv=CVConfig(scheme="group")``: rows of the same group never span
        train and test. ``time`` declares the CV time axis for ``cv=CVConfig(scheme="timeseries")``
        (purge/embargo, value-based order); ``label_time`` is the optional label-end-time
        ``t1`` for full de Prado purge. All are row-aligned metadata like ``sample_weight`` — not
        features, not needed at ``predict`` time.
        """
        # validate run_mode here (not in __init__, which stores verbatim per the sklearn invariant): the
        # RunConfig is built by the direct constructor, so an invalid value must map to ConfigError, not a
        # raw pydantic ValidationError (ADR-0038 §1, by the cv<2 -> ConfigError precedent).
        if self.run_mode not in ("selection", "full"):
            raise ConfigError(f"run_mode must be 'selection' or 'full', got {self.run_mode!r}")
        task = self._resolve_task()
        # preset fill (ADR-0074 §2): effective values for the None-default surface (cv/models/budget/
        # hpo/ensemble/fs/fe) — a preset fills ONLY parameters left as None; everything below reads
        # the surface from `eff`, self.* stay untouched (sklearn clone invariant).
        eff, preset_block = self._resolve_preset()
        # resolve FE once (sklearn invariant: __init__ stored it verbatim): validate type and gate
        # target-encoding to binary (ADR-0041 §4). The effective config flows to the Reader (boundary
        # FE), run_slice (OOF-TE) and the RunConfig (manifest + fingerprint).
        fe = self._resolve_fe(task, eff["feature_engineering"])
        # resolve FS once (sklearn invariant): validate type and fill the ranker seed from random_state
        # (ADR-0043 §5). None -> off. The effective config flows to build (ranker), run_slice and RunConfig.
        fs = self._resolve_fs(eff["feature_selection"])
        # resolve HPO once (sklearn invariant): validate type, fill random_state from the run seed BEFORE
        # the RunConfig dump so the fingerprint carries the effective tuning seed (ADR-0062 §5). None -> off.
        hpo = self._resolve_hpo(eff["hpo"])
        # resolve ensemble once (sklearn invariant): validate type, fill random_state from the run seed
        # before the RunConfig dump (mirror of hpo/fs). None -> off (single-model M7a behavior).
        ensemble = self._resolve_ensemble(eff["ensemble"])
        # resolve tracker once, BEFORE reading data (ADR-0072 §2): a requested-but-impossible tracking
        # setup (missing mlflow, bad form) must fail fast, not after the expensive training.
        tracker = self._resolve_tracker()
        ds_full = self._reader(task, fe).read(
            X, y, sample_weight=sample_weight, groups=groups, time=time, label_time=label_time
        )
        classes = np.unique(ds_full.target()) if task.is_classification else None
        # M6f (ADR-0057/0058): resolve data-shape "auto" sentinels + apply the hard cost-budget post-read,
        # before build (the effective arbitration drives the arbitration splitter). effective_fs flows to
        # build AND the RunConfig manifest (write-back); the record feeds the run-report fs_resolution block.
        fs_resolution: dict[str, str] = {}
        if fs is not None:
            _cv = _normalize_cv(eff["cv"])
            fs, fs_resolution = resolve_fs_defaults(
                fs,
                n_rows=ds_full.n_rows,
                n_features=len(ds_full.schema.features),
                inner_n_splits=_cv.n_splits,
                times=ds_full.time() if ds_full.schema.time is not None else None,
                scheme=_cv.scheme,
                purge=_cv.purge,
                purge_delta=_cv.purge_delta,
            )
        components = build_default_components(
            task,
            random_state=self.random_state,
            metric=self.metric,
            cv=eff["cv"],
            models=eff["models"],
            has_datetime=bool(ds_full.schema.datetime),
            has_group=ds_full.schema.group is not None,
            has_time=ds_full.schema.time is not None,
            has_missing=bool(np.isnan(ds_full.to_numpy()).any()),
            classes=classes,
            significance=self.significance,
            feature_selection=fs,
            hpo=hpo,
            ensemble=ensemble,
        )
        # honest-regime outer holdout (ADR-0029): carve once scheme-aware; selection/refit/calibration
        # run on dev only, the winner is scored once on the untouched holdout. `ds` is dev (== full
        # when off), so every line below is unchanged for the default outer_holdout=0.0.
        holdout_ds: Dataset | None = None
        if components.cv.outer_holdout > 0.0:
            dev_idx, holdout_idx = self._carve_holdout(ds_full, task, components, classes)
            if holdout_idx.size < _MIN_HOLDOUT_ROWS:
                logger.warning(
                    "outer holdout has only %d row(s); its score is high-variance — treat it "
                    "as indicative, not final",
                    holdout_idx.size,
                )
            ds = ds_full.take(dev_idx)
            holdout_ds = ds_full.take(holdout_idx)
        else:
            ds = ds_full
        _validate_cv_data_floor(components.cv, ds)
        budget_config = self._resolve_budget(eff["budget"])
        # one cooperative budget shared by HPO and selection (ADR-0062 §5): tuning consumes from the same
        # pool, so a tiny budget cuts trials AND candidates; refit below is never budget-gated.
        budget = self._build_budget(budget_config)
        ctx = RunContext(
            run_config=RunConfig(
                seed=self.random_state,
                cv=components.cv,
                budget=budget_config,
                hpo=hpo,
                ensemble=ensemble,
                significance=self.significance,
                run_mode=self.run_mode,
                fe=fe,
                fs=fs,
            )
        )
        # M7a HPO stage (ADR-0062 §2/§2b): tune each tunable type on an inner-CV of DEV BEFORE the outer
        # selection, in BOTH run_modes; the tuned factories are folded into components.estimators.
        hpo_report = self._run_hpo_stage(
            ds, task, components, hpo=hpo, fe=fe, fs=fs, budget=budget, ctx=ctx
        )
        # run-fingerprint over the resolved inputs + the DEV data signature (ADR-0035 §2/§3, post-carve);
        # always computed (for the run-report), used as the cache scope only when cache is enabled.
        run_fingerprint = self._run_fingerprint(ctx.run_config, task, components, ds)
        cache = self._build_cache(run_fingerprint)
        with ctx.timed_stage("run", "selection"):
            result = run_slice(
                ds,
                task,
                estimators=components.estimators,
                splitter=components.splitter,
                metric=components.metric,
                policy=components.policy,
                significance_test=components.significance,
                calibrator_factory=components.refinement_calibrator,
                selection=components.selection,
                refinement_min_oof=components.refinement_min_oof,
                weighting=components.weighting,
                # an ensemble run needs the per-candidate proba channel to blend (classification), like
                # refinement/calibration (ADR-0063 §2); for regression it reuses the band's value OOF.
                capture_proba=(
                    components.selection == "refinement"
                    or components.calibrate != "off"
                    or ensemble is not None
                ),
                fe=fe,
                features=(
                    FeatureSelectionBundle(
                        config=components.feature_selection,
                        ranker=components.feature_ranker,
                        strategies=components.feature_strategies,
                        carve=components.feature_carve,
                        fit_predict=components.feature_fit_predict,
                        arbitration_splitter=components.feature_arbitration_splitter,
                    )
                    if components.feature_selection is not None
                    else None
                ),
                budget=budget,
                cache=cache,
                ctx=ctx,
            )
        # cache observability (F4.7): a cold run next to other fingerprint directories means the
        # resolved config or the data signature changed — name the fingerprint so the user can
        # diff the two run_report configs instead of guessing why everything recomputed.
        cache_dir = self.cache
        if cache is not None and cache_dir is not None and result.computed and not result.reused:
            siblings = [
                p.name
                for p in Path(cache_dir).iterdir()
                if p.is_dir() and p.name != run_fingerprint
            ]
            if siblings:
                logger.info(
                    "cache: no reusable candidates for fingerprint %s; %d other fingerprint(s) "
                    "present in %s — the resolved config or the data signature changed",
                    run_fingerprint,
                    len(siblings),
                    cache_dir,
                )
        # attach the selected subset to the dev (and holdout, and full) schema so refit/inference/holdout
        # AND the finalize refit on ds_full project the model input to it via design_matrix (ADR-0045 §2);
        # a plain run leaves them unchanged. ds_full MUST be projected too, else the all-data finalize refit
        # trains on the full feature set while the shipped schema carries the subset (train≠inference).
        if result.feature_selection is not None:
            selected = result.feature_selection.selected_features
            ds = ds.with_selected_features(selected)
            ds_full = ds_full.with_selected_features(selected)
            if holdout_ds is not None:
                holdout_ds = holdout_ds.with_selected_features(selected)
        # M7b ensemble stage (ADR-0063/0064): blend the candidates' OOF and gate the recipe against the
        # best single. Computed in BOTH run_modes (selection reports the recipe without shipping); OOF-only.
        ensemble_outcome = self._run_ensemble_stage(
            result, ds, task, components, ensemble=ensemble, ctx=ctx
        )
        # run_mode stage-gate (ADR-0038 §2): "selection" stops at the leaderboard; "full" (default) ships
        # a model. The post-selection stages run only for "full".
        ship_model = self.run_mode == "full"
        if ship_model:
            # refit is NOT budget-gated: graceful degradation must ship a working model (ADR-0032 §1). When
            # the ensemble was applied, refit its members on full-DEV and ship a BlendedEstimator instead.
            # _ship_estimator returns the post-refit ensemble provenance block too (C13).
            best, ensemble_outcome, ensemble_block = self._ship_estimator(
                ds, task, result, components, ensemble_outcome, classes, ctx
            )
            calibrator, calibration = self._calibrate_winner(ds, task, result, components, classes)
        else:
            ensemble_block = (
                _ensemble_report(ensemble_outcome) if ensemble_outcome is not None else None
            )
        # serving provenance (ADR-0068 §5): None in selection mode (no model shipped), else filled below
        serving_block: dict[str, Any] | None = None
        self._set_result_attrs(X, result, ds, task, classes)

        # full-only: the shipped model + its post-selection attributes (ADR-0038 §2); selection exposes
        # neither best_estimator_/fitted_ nor calibration_/holdout_score_ -> predict raises NotFittedError.
        if ship_model:
            self.best_estimator_ = best
            # probability calibration (ADR-0030): report (Brier/ECE before/after + reliability) and curve
            self.calibration_ = calibration
            self.reliability_curve_ = calibration.get("reliability") if calibration else None
            self.fitted_ = FittedModel(
                estimator=best,
                schema=ds.schema,
                task=task,
                # the metric is held by name + averaging mode and resolved lazily (ADR-0066 §2)
                metric_name=components.metric.name,
                metric_average=getattr(components.metric, "average", None),
                classes=classes,
                leaderboard=result.leaderboard,
                best_model_id=result.best_model_id,
                band_member_ids=result.band_member_ids,
                band_unstable=result.band_unstable,
                band_width=result.band_width,
                winner_by_tiebreak=result.winner_by_tiebreak,
                calibrator=calibrator,
                calibration=calibration,
                selection_mode=result.selection_mode,
                score_space=result.score_space,
                ensemble=ensemble_block,
                early_stopping=components.early_stopping,
            )
            # honest-regime holdout (ADR-0029 §3, NFR-M4-7): score the shipped dev-trained winner once on
            # the untouched holdout; raw metric, comparable to leaderboard_. None when outer_holdout is off.
            if holdout_ds is not None:
                result.holdout_score = self.fitted_._score_dataset(holdout_ds)
                self.fitted_.holdout_score = result.holdout_score
            self.holdout_score_ = result.holdout_score
            # finalize (ADR-0068): refit the shipped winner on DEV+holdout for production once the honest
            # holdout score is taken. No-op (shipped_on="dev") when outer_holdout is off or finalize=False.
            shipped_on, ensemble_block = self._finalize_ship(
                ds,
                ds_full,
                holdout_ds,
                task,
                result,
                components,
                ensemble_outcome,
                ensemble_block,
                classes,
                ctx,
            )
            self.shipped_on_ = shipped_on
            self.fitted_.shipped_on = shipped_on
            serving_block = {
                "finalize": self.finalize,
                "shipped_on": shipped_on,
                "outer_holdout": components.cv.outer_holdout,
            }
        # tracker-independent run report (ADR-0033, G-O1): resolved config + timings + winner +
        # band + budget/significance provenance, serializable without a tracker. RC adds the
        # run-fingerprint + truthful cache outcome (ADR-0037 §3), additive (manifest version unchanged).
        self.run_report_ = build_run_report(
            run_config=ctx.run_config,
            timings=ctx.timings,
            result=result,
            run_fingerprint=run_fingerprint,
            cache_enabled=self.cache is not None,
            fs_resolution=fs_resolution or None,
            hpo=hpo_report,
            ensemble=ensemble_block,
            serving=serving_block,
            preset=preset_block,
            task=task.kind,
            metric=components.metric.name,
        )
        # surface the split-dependence diagnostic at fit time too (finding #11c): an honest holdout is
        # not markedly better than the OOF — if it is, the carve is suspect, so warn, don't only file it.
        optimism = self.run_report_.get("holdout_optimism")
        if optimism is not None:
            logger.warning("%s", optimism["message"])
        # post-fit one-shot tracking (ADR-0072 §2): the tracker consumes a DEEP COPY (a mutating
        # implementation cannot corrupt run_report_); a tracking failure must not destroy a finished
        # fit — the only place an exception is downgraded to WARNING (external-service boundary).
        # KeyboardInterrupt/SystemExit propagate (not Exception).
        if tracker is not None:
            try:
                tracker.log_run(copy.deepcopy(self.run_report_))
            except Exception:
                logger.warning(
                    "experiment tracking failed; the fit itself is intact", exc_info=True
                )
        return self

    @staticmethod
    def available_models(task: Task | str | None = None) -> dict[str, Any]:
        """Discoverable models (built-in + plugins) and their capabilities.

        Read-only and lazy: reads descriptors without materializing any adapter, so a
        boosting plugin is listed even when its extra is not installed.
        """
        from .registry import available_models

        resolved = Task(kind=cast("TaskKind", task)) if isinstance(task, str) else task
        return dict(available_models(resolved))

    def predict(self, X: Any) -> np.ndarray:
        return self._require_fitted().predict(X)

    def predict_proba(self, X: Any) -> np.ndarray:
        return self._require_fitted().predict_proba(X)

    def score(self, X: Any, y: Any, sample_weight: Any | None = None) -> float:
        """Metric score, sklearn convention (higher is better).

        A lower-is-better metric (e.g. ``log_loss``) is sign-flipped so grid-search
        and ``Pipeline`` maximize it; ``leaderboard_`` carries the raw, unflipped
        value.
        """
        return self._require_fitted().score(X, y, sample_weight=sample_weight)

    # -- internals ----------------------------------------------------------

    def _run_hpo_stage(
        self,
        ds: Dataset,
        task: Task,
        components: Components,
        *,
        hpo: HPOConfig | None,
        fe: FEConfig,
        fs: FeatureSelectionConfig | None,
        budget: RunBudget | None,
        ctx: RunContext,
    ) -> dict[str, Any] | None:
        """Tune each tunable model on an inner-CV of DEV and fold the tuned factories into components.

        Runs in BOTH run_modes (selection's leaderboard must reflect the same tuned candidates the full
        mode would ship, ADR-0038 §2b). Returns the additive hpo run-report block; None when HPO is off.
        """
        if hpo is None or components.tuner is None:
            return None
        assert components.make_factory is not None and components.inner_splitter is not None
        with ctx.timed_stage("run", "hpo"):
            outcomes = tune_estimators(
                ds,
                task,
                tunable=components.tunable or {},
                make_factory=components.make_factory,
                tuner=components.tuner,
                metric=components.metric,
                policy=components.policy,
                inner_splitter=components.inner_splitter,
                n_trials=hpo.n_trials,
                timeout_s=hpo.timeout_s,
                # _resolve_hpo already filled None -> seed; keep an explicit check (not `or`, falsy for 0)
                random_state=hpo.random_state
                if hpo.random_state is not None
                else self.random_state,
                fe=fe,
                sample_weight=ds.sample_weight(),
                budget=budget,
                ctx=ctx,
            )
        for name, outcome in outcomes.items():
            tuned = components.make_factory(name, outcome.best_params)
            components.estimators[f"{name}__tuned" if hpo.keep_baseline else name] = tuned
        # a time budget imposes a finite Optuna timeout (fair-share, _timeout) -> non-deterministic
        # even when hpo.timeout_s is None; mirror that exact condition in the honesty flag (ADR-0062 §7)
        return _hpo_report(
            hpo,
            outcomes,
            tuned_on_full=fs is not None,
            time_budget=budget is not None and budget.mode == "time",
        )

    def _run_ensemble_stage(
        self,
        result: SliceResult,
        ds: Dataset,
        task: Task,
        components: Components,
        *,
        ensemble: EnsembleConfig | None,
        ctx: RunContext,
    ) -> EnsembleOutcome | None:
        """Blend the candidates' OOF and gate the recipe against the best single (ADR-0063/0064).

        Computed in BOTH run_modes (selection reports the recipe without shipping); a blend-stage failure
        degrades to a not-applied outcome with a truthful gate_reason, never killing a valid fit. None when
        ensembling is off.
        """
        if ensemble is None or components.ensembler is None:
            return None
        assert components.ensemble_metric is not None
        with ctx.timed_stage("run", "ensemble"):
            try:
                return ensemble_selection(
                    result.candidates,
                    task,
                    y=cast("np.ndarray", ds.target()),
                    best_model_id=result.best_model_id,
                    ensembler=components.ensembler,
                    metric=components.ensemble_metric,
                    significance_test=components.significance,
                    policy=components.policy,
                    significance_mode=self.significance,
                    block_index=(
                        result.oof_fold_index
                        if isinstance(components.splitter, TimeOrderedSplitter)
                        else None
                    ),
                    sample_weight=ds.sample_weight(),
                    random_state=cast("int", ensemble.random_state),
                )
            except Exception as exc:
                # the ensemble is optional and post-selection: an honest winner already exists, so a
                # blend-stage failure degrades to "not applied" instead of killing a valid fit (ADR-0063
                # §5, like ADR-0022 candidate isolation). The reason is surfaced, never silent.
                logger.warning(
                    "ensemble stage failed (%s); shipping the single honest winner instead", exc
                )
                return EnsembleOutcome(
                    applied=False,
                    method=components.ensembler.name,
                    member_ids=tuple(c.id for c in result.candidates),
                    weights={},
                    gate_reason=f"failed: {exc}",
                    oof_delta=None,
                )

    def _set_result_attrs(
        self, X: Any, result: SliceResult, ds: Dataset, task: Task, classes: np.ndarray | None
    ) -> None:
        """Set the describing-input + honesty-observability attributes — both run_modes (ADR-0038 §2).

        classification-only ``classes_`` (regression has none, ADR-0020 §4); ``n_features_in_``/
        ``feature_names_in_``; the leaderboard/winner; and the band the winner came from (ADR-0026 §6).
        """
        if classes is not None:
            self.classes_ = classes
        self.n_features_in_ = len(ds.schema.features)
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.leaderboard_ = result.leaderboard
        self.best_model_id_ = result.best_model_id
        self.schema_ = ds.schema
        self.task_ = task
        self.band_member_ids_ = result.band_member_ids
        self.band_unstable_ = result.band_unstable
        self.band_width_ = result.band_width
        self.winner_by_tiebreak_ = result.winner_by_tiebreak
        self.selection_mode_ = result.selection_mode

    def _finalize_ship(
        self,
        ds: Dataset,
        ds_full: Dataset,
        holdout_ds: Dataset | None,
        task: Task,
        result: SliceResult,
        components: Components,
        ensemble_outcome: EnsembleOutcome | None,
        ensemble_block: dict[str, Any] | None,
        classes: np.ndarray | None,
        ctx: RunContext,
    ) -> tuple[str, dict[str, Any] | None]:
        """Refit the shipped winner on DEV+holdout for production, once the honest holdout score is taken.

        No-op (returns ``("dev", ensemble_block)``) when outer_holdout is off or ``finalize=False``. Else
        refits on all data (ADR-0068 §1/§4), updates the fitted model in place and detaches a DEV-OOF
        calibrator whose class set is incomplete; returns ``("all", possibly-changed ensemble_block)``.
        """
        if holdout_ds is None or not self.finalize:
            return "dev", ensemble_block
        with ctx.timed_stage("run", "finalize"):
            best, _outcome, ensemble_block = self._ship_estimator(
                ds_full, task, result, components, ensemble_outcome, classes, ctx
            )
        self.best_estimator_ = best
        self.fitted_.estimator = best
        # the shipped recipe (ds_full) may differ from DEV if a member dropped on full data (§4)
        self.fitted_.ensemble = ensemble_block
        self._detach_dev_calibrator_if_unseen_class(ds, classes)
        return "all", ensemble_block

    def _carve_holdout(
        self, ds: Dataset, task: Task, components: Components, classes: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Carve the untouched outer holdout scheme-aware, with boundary guards.

        Fails fast (``ConfigError``) when the holdout is too small to hold >= 2 rows per class (a
        single-class holdout would break a proba metric) or, for time-series, when too little dev is
        left to fit ``n_splits`` folds. The leakage-sensitive carve itself lives behind the splitter
        port (``outer_holdout_carve``); composition only guards and orchestrates.
        """
        from honestml.adapters import outer_holdout_carve

        cfg = components.cv
        n = ds.n_rows
        n_holdout = round(cfg.outer_holdout * n)
        min_holdout = 2 * classes.size if (task.is_classification and classes is not None) else 2
        if n_holdout < min_holdout:
            raise ConfigError(
                f"outer_holdout={cfg.outer_holdout} carves {n_holdout} rows; need >= {min_holdout} "
                "(>= 2 per class) for an unbiased holdout score"
            )
        dev_idx, holdout_idx = outer_holdout_carve(
            ds,
            scheme=cfg.scheme,
            fraction=cfg.outer_holdout,
            stratify=task.is_classification,
            random_state=self.random_state,
            purge=cfg.purge,
            purge_delta=cfg.purge_delta,
            period=cfg.period,
            period_size=cfg.period_size,
        )
        # the group-disjoint carve holds WHOLE groups, so the realized holdout can fall below the floor
        # even when the row estimate (n_holdout) cleared it (S7-corr); validate the ACTUAL size too, so the
        # hard floor and the soft-warning in fit() both judge holdout_idx, not the rounded estimate.
        if holdout_idx.size < min_holdout:
            raise ConfigError(
                f"outer_holdout={cfg.outer_holdout} realizes only {holdout_idx.size} holdout row(s); "
                f"need >= {min_holdout} (>= 2 per class) for an unbiased holdout score"
            )
        # the unstratified carves (timeseries late window, group-disjoint) can be single-class even at
        # >= 2*n_classes rows (a late regime shift, all-positive groups); a proba metric then needs >= 2
        # classes -> fail at the boundary with ConfigError, not a raw sklearn ValueError (ADR-0029 §1).
        target = ds.target()
        if (
            task.is_classification
            and components.metric.needs in ("proba", "threshold")
            and target is not None
            and np.unique(target[holdout_idx]).size < 2
        ):
            raise ConfigError(
                "outer_holdout window is single-class (a probability metric needs >= 2 classes in "
                "the holdout); reduce outer_holdout or use a stratified scheme"
            )
        if cfg.scheme == "timeseries":
            need = cfg.n_splits * cfg.n_test + cfg.purge + cfg.n_es + 1
            if dev_idx.size < need:
                raise ConfigError(
                    f"too few dev rows ({dev_idx.size}) after the outer_holdout carve for "
                    f"{cfg.n_splits} timeseries folds (need >= {need}); reduce outer_holdout or n_splits"
                )
        elif cfg.scheme == "timeseries_period" and dev_idx.size == 0:
            # the period count needed for n_splits folds is only known after materialization, so the
            # splitter validates feasibility; here we only catch an empty dev (holdout + purge consumed
            # every earlier period) before it reaches the splitter as a raw numpy error
            raise ConfigError(
                "no dev rows remain after the period outer_holdout carve (the holdout plus the purge "
                "gap consumed every earlier period); reduce outer_holdout or purge"
            )
        return dev_idx, holdout_idx

    def _calibrate_winner(
        self,
        ds: Any,
        task: Task,
        result: SliceResult,
        components: Components,
        classes: np.ndarray | None,
    ) -> tuple[object | None, dict[str, Any] | None]:
        """Fit the winner's probability calibrator on its OOF, gated by a Brier improvement.

        ``calibrate='off'`` (default) skips it; an unviable winner OOF (no proba channel, too few
        rows, a single class) is not attached but reported with ``applied=False`` + a WARNING.
        Otherwise delegates to the pure ``calibrate_winner`` gate.
        """
        if components.calibrate == "off" or not task.is_classification or classes is None:
            return None, None
        # time-series: the cross-fit gate would look ahead (calibrating a fold from future folds).
        # Disabled in M4 (like refinement, ADR-0031 §3 B2); an expanding TS gate is a future ADR.
        if isinstance(components.splitter, TimeOrderedSplitter):
            logger.warning(
                "calibration skipped: time-series CV (expanding-gate calibration is future)"
            )
            return None, {"method": components.calibrate, "applied": False, "reason": "time-series"}
        from honestml.adapters import resolve_calibrator, resolve_metric
        from honestml.application import calibrate_winner, resolve_positive, viable_blocks

        winner = next(c for c in result.candidates if c.id == result.best_model_id)
        target = ds.target()
        if (
            winner.oof_proba is None
            or winner.oof_mask is None
            or result.oof_fold_index is None
            or target is None
        ):
            logger.warning("calibration skipped: the winner has no probability OOF channel")
            return None, {
                "method": components.calibrate,
                "applied": False,
                "reason": "no proba OOF",
            }
        mask = winner.oof_mask
        proba = winner.oof_proba[mask]
        y = target[mask]
        sw_full = ds.sample_weight()
        sw = sw_full[mask] if sw_full is not None else None
        blocks = result.oof_fold_index[mask]
        if task.kind == "multiclass":
            y_code = np.searchsorted(classes, y)
        else:
            y_code = (y == resolve_positive(task, classes)).astype(np.int64)
        # per-block viability — the shared precondition of crossfit_calibrate (symmetric with the
        # refinement path's all-or-nothing check, fix M4d-review optimality/compliance).
        if not viable_blocks(
            blocks, y_code, n_classes=classes.size if task.kind == "multiclass" else None
        ):
            logger.warning(
                "calibration skipped: too few OOF rows per cross-fit block or a single class"
            )
            return None, {"method": components.calibrate, "applied": False, "reason": "min-n"}
        factory = resolve_calibrator(components.calibrate, n_calib=int(mask.sum()))
        # binary briers/eces score P(positive); orient them on positive so the gate is not inverted (F111)
        positive = resolve_positive(task, classes) if task.kind == "binary" else None
        brier = resolve_metric("brier", classes=classes, positive=positive)
        ece = resolve_metric("ece", classes=classes, positive=positive)
        return calibrate_winner(
            proba,
            y,
            y_code,
            blocks,
            factory,
            brier=brier,
            ece=ece,
            method=components.calibrate,
            sample_weight=sw,
        )

    def _resolve_budget(self, budget: float | BudgetConfig | None) -> BudgetConfig:
        """Coerce the public ``budget`` param to a resolved ``BudgetConfig``."""
        if budget is None:
            return BudgetConfig()  # mode="none" (unbounded)
        if isinstance(budget, BudgetConfig):
            return budget
        return BudgetConfig(mode="time", time_budget_s=budget)

    def _build_budget(self, config: BudgetConfig) -> RunBudget | None:
        """Build the cooperative budget adapter, or ``None`` when the gate would be inert.

        Memory is orthogonal to mode: build the budget when a mode is active **or** a
        memory limit is set, so ``mode="none"`` + ``memory_limit_mb`` is a memory-only run, not inert.
        """
        if config.mode == "none" and config.memory_limit_mb is None:
            return None
        from honestml.adapters import RunBudget

        return RunBudget(config)

    def _run_fingerprint(
        self, run_config: RunConfig, task: Task, components: Components, ds: Dataset
    ) -> str:
        """Assemble the run-fingerprint over the resolved inputs + DEV signature.

        The data-signature is over ``ds`` (DEV, post-carve) — the exact dataset ``run_slice`` trains
        on. ``lib_versions`` pins the resolved compute stack (estimator packages + sklearn + numpy);
        the pure assembler lives in ``application`` (sklearn is only version-read here, not imported).
        """
        from honestml.application import (
            collect_lib_versions,
            compute_run_fingerprint,
            dataset_signature,
        )

        estimators = tuple(components.estimators)
        return compute_run_fingerprint(
            run_config=run_config,
            task=task,
            metric=components.metric,
            data_signature=dataset_signature(ds),
            estimators=estimators,
            lib_versions=collect_lib_versions(_packages_for(estimators)),
        )

    def _build_cache(self, fingerprint: str) -> JoblibCandidateCache | None:
        """Build the fingerprint-scoped cache adapter, or ``None`` when ``cache`` is unset."""
        if self.cache is None:
            return None
        from honestml.adapters import JoblibCandidateCache

        return JoblibCandidateCache(Path(self.cache), fingerprint)

    def _resolve_task(self) -> Task:
        if isinstance(self.task, Task):
            return self.task
        return Task(kind=cast("TaskKind", self.task))

    def _reader(self, task: Task, fe: FEConfig | None = None):
        from honestml.adapters import Reader

        return Reader(task, fe=fe)

    def _resolve_fe(self, task: Task, fe: FEConfig | None) -> FEConfig:
        """Resolve the FE catalog: validate type, gracefully gate TE to binary classification.

        ``None`` -> all-off. A non-binary task with ``target_encoding`` is a graceful skip + WARNING
        (not a ``ConfigError``), like the calibration/refinement gates; frequency/intersections still
        apply (target-independent). Time-series CV keeps TE on — the leaderboard uses the honest
        expanding-window out-of-fold encoder (each fold from strictly earlier folds, ADR-0082) and the
        shipped model the full-train spec.
        """
        if fe is None:
            return FEConfig()
        if not isinstance(fe, FEConfig):
            raise ConfigError(
                f"feature_engineering must be a FEConfig or None, got {type(fe).__name__}"
            )
        if fe.target_encoding and task.kind != "binary":
            logger.warning(
                "target encoding skipped: only binary classification is supported (task=%s); "
                "frequency/intersections still apply",
                task.kind,
            )
            return fe.model_copy(update={"target_encoding": False})
        return fe

    def _resolve_seeded(self, cfg: Any, cls: type[Any], label: str) -> Any:
        """Validate an optional seeded config and fill ``random_state`` from the run seed.

        Shared by ``_resolve_hpo``/``_resolve_ensemble``/``_resolve_fs``: ``None`` -> off; a wrong type ->
        a guard ``ConfigError`` (``label`` names the expected type); ``random_state=None`` inherits the run
        seed and is resolved HERE — before the ``RunConfig`` dump — so the fingerprint carries the effective
        seed (ADR-0062 §5).
        """
        if cfg is None:
            return None
        if not isinstance(cfg, cls):
            raise ConfigError(f"{label} or None, got {type(cfg).__name__}")
        if cfg.random_state is None:
            return cfg.model_copy(update={"random_state": self.random_state})
        return cfg

    def _resolve_hpo(self, hpo: HPOConfig | None) -> HPOConfig | None:
        """Resolve the HPO catalog: validate type, fill random_state from the run seed (``None`` -> off)."""
        return self._resolve_seeded(hpo, HPOConfig, "hpo must be an HPOConfig")

    def _resolve_ensemble(self, ensemble: EnsembleConfig | None) -> EnsembleConfig | None:
        """Resolve the ensemble catalog: validate type, fill random_state (``None`` -> off, single model)."""
        return self._resolve_seeded(ensemble, EnsembleConfig, "ensemble must be an EnsembleConfig")

    def _resolve_preset(self) -> tuple[dict[str, Any], dict[str, Any] | None]:
        """Fill the None-left surface parameters from the preset.

        Returns the effective surface values + the additive run-report block.
        ``self.*`` stay untouched (sklearn clone invariant) — fit reads the surface
        from the returned dict only.
        """
        from .presets import resolve_preset

        return resolve_preset(
            self.preset,
            {
                "cv": self.cv,
                "models": self.models,
                "budget": self.budget,
                "hpo": self.hpo,
                "ensemble": self.ensemble,
                "feature_selection": self.feature_selection,
                "feature_engineering": self.feature_engineering,
            },
        )

    def _resolve_tracker(self) -> ExperimentTracker | None:
        """Resolve the tracking opt-in: validate the form, build the adapter lazily.

        ``None`` -> off. ``"mlflow"`` is sugar for ``TrackerConfig()``; a ``TrackerConfig`` names the
        adapter here (composition), whose constructor gates the missing extra. A port
        instance is used verbatim — ``callable(log_run)`` closes the attribute-not-method hole of
        ``runtime_checkable`` (signature mismatches honestly stay in the fail-soft zone).
        """
        tracker = self.tracker
        if tracker is None:
            return None
        if isinstance(tracker, str):
            if tracker != "mlflow":
                raise ConfigError(
                    f"unknown tracker {tracker!r}; expected 'mlflow', a TrackerConfig "
                    "or an ExperimentTracker instance"
                )
            tracker = TrackerConfig()
        if isinstance(tracker, TrackerConfig):
            from honestml.adapters import MlflowTracker  # local import = lazy (ADR-0073 §1)

            return MlflowTracker(**tracker.model_dump(exclude={"backend"}))
        if isinstance(tracker, ExperimentTracker) and callable(tracker.log_run):
            return tracker
        raise ConfigError(
            f"tracker must be None, 'mlflow', a TrackerConfig or an object with a callable "
            f"log_run(report), got {type(tracker).__name__}"
        )

    def _ship_estimator(
        self,
        ds: Any,
        task: Task,
        result: SliceResult,
        components: Components,
        ensemble_outcome: EnsembleOutcome | None,
        classes: np.ndarray | None,
        ctx: RunContext,
    ) -> tuple[Estimator, EnsembleOutcome | None, dict[str, Any] | None]:
        """Refit the shipped estimator: a ``BlendedEstimator`` when the ensemble applied, else the winner.

        Returns ``(estimator, outcome, ensemble_block)`` — the additive provenance block is built here so
        the dev-ship and finalize-ship call sites do not each repeat ``_ensemble_report`` (C13). Refit is
        never budget-gated (graceful degradation must ship a working model). If a member refit drops the
        ensemble below 2 members, the single winner is shipped and the outcome is marked not-applied with
        a truthful ``gate_reason`` — the gate is never silent.
        """
        if ensemble_outcome is not None and ensemble_outcome.applied:
            blended = self._build_blended(ds, task, components, ensemble_outcome, classes, ctx)
            if blended is not None:
                est, outcome = blended
                return est, outcome, _ensemble_report(outcome)
            ensemble_outcome = replace(
                ensemble_outcome, applied=False, gate_reason="insufficient_members_after_refit"
            )
        with ctx.timed_stage("run", "refit"):
            best = refit_best(
                ds, task, factory=components.estimators[result.best_model_id], ctx=ctx
            )
        block = _ensemble_report(ensemble_outcome) if ensemble_outcome is not None else None
        return best, ensemble_outcome, block

    def _detach_dev_calibrator_if_unseen_class(
        self, dev: Dataset, classes: np.ndarray | None
    ) -> None:
        """Detach the DEV-OOF calibrator if a global class is absent from DEV.

        The calibrator was fit on DEV-OOF; after the all-data finalize refit, a class carved entirely into
        the holdout was never seen by it, so its per-class mapping is invalid — drop it (with a WARNING)
        rather than apply it to an untrained column. Regression has no calibrator (no-op).
        """
        if classes is None or self.fitted_.calibrator is None:
            return
        target = dev.target()
        dev_classes = set(np.unique(target).tolist()) if target is not None else set()
        if set(classes.tolist()) - dev_classes:
            logger.warning("finalize: a class is absent from DEV; detaching the DEV-OOF calibrator")
            self.fitted_.calibrator = None
            # mark the report not-applied (in-memory AND on the shipped model) so the persisted manifest/
            # run-report never claim calibration while predict_proba returns raw probabilities (NFR-SRV-5)
            if self.fitted_.calibration is not None:
                detached = {
                    **self.fitted_.calibration,
                    "applied": False,
                    "reason": "dev_unseen_class",
                }
                self.fitted_.calibration = detached
                self.calibration_ = detached
            else:
                self.calibration_ = None
            self.reliability_curve_ = None

    def _build_blended(
        self,
        ds: Any,
        task: Task,
        components: Components,
        outcome: EnsembleOutcome,
        classes: np.ndarray | None,
        ctx: RunContext,
    ) -> tuple[Estimator, EnsembleOutcome] | None:
        """Refit the active members on full-DEV and wrap them in a ``BlendedEstimator``, or ``None`` if
        fewer than 2 survive (caller ships the single winner). Weights are renormalized over survivors."""
        from honestml.adapters import BlendedEstimator

        active = [mid for mid in outcome.member_ids if outcome.weights[mid] > _W_EPS]
        with ctx.timed_stage("run", "refit"):
            members, kept, _dropped = refit_members(
                ds, task, member_ids=active, factories=components.estimators, ctx=ctx
            )
        if len(kept) < 2:
            return None
        w = np.array([outcome.weights[mid] for mid in kept], dtype=np.float64)
        w = w / w.sum()
        blended = BlendedEstimator(members, w, classes)
        new_outcome = replace(
            outcome,
            member_ids=tuple(kept),
            weights={mid: float(wi) for mid, wi in zip(kept, w)},
        )
        return blended, new_outcome

    def _resolve_fs(self, fs: FeatureSelectionConfig | None) -> FeatureSelectionConfig | None:
        """Resolve the FS catalog: validate type, fill the ranker seed from random_state (``None`` -> off)."""
        return self._resolve_seeded(
            fs, FeatureSelectionConfig, "feature_selection must be a FeatureSelectionConfig"
        )

    def _require_fitted(self) -> FittedModel:
        if not hasattr(self, "fitted_"):
            # a selection run built a leaderboard but shipped no model (ADR-0038 §2): point the user to
            # run_mode="full" instead of the generic "not fitted" message.
            if hasattr(self, "leaderboard_"):
                raise NotFittedError(
                    "run_mode='selection' built a leaderboard but no fitted model; use "
                    "run_mode='full' to ship a model"
                )
            raise NotFittedError("AutoML is not fitted; call fit before predict/score")
        return self.fitted_
