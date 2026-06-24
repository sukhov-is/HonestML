"""Typed run configuration.

Domain-specific column names/roles do **not** live here — they belong to the
user-supplied ``Task``/``FeatureSchema``. These models carry only run parameters,
validate at the boundary, and serialize to JSON as the basis of the run manifest.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .exceptions import ConfigError

CVScheme = Literal[
    "auto", "kfold", "stratified", "group", "holdout", "timeseries", "timeseries_period"
]
# calendar/Δt period unit for scheme='timeseries_period' (ADR-0096 §1); "delta" needs a numeric
# time axis + period_size, the calendar units need a datetime axis (validated in the splitter).
PeriodUnit = Literal["month", "week", "day", "delta"]
# leaderboard/significance weighting of unequal periods (ADR-0098): "pooled" = one metric over all OOF
# rows (default, current behavior); "period" = macro-average over periods (requires a time-ordered scheme).
WeightingMode = Literal["pooled", "period"]
CalibrateMethod = Literal["off", "sigmoid", "isotonic", "auto"]
SelectionMode = Literal["raw", "refinement"]
# "none" encodes an unbounded run explicitly so the manifest is truthful (ADR-0032 §5a)
BudgetMode = Literal["none", "time", "trials"]
SignificanceMode = Literal["bootstrap", "off"]
# pipeline stop-point: "full" ships a model (M5 default); "selection" stops at the leaderboard (ADR-0038)
RunMode = Literal["selection", "full"]
# feature-selection catalog: M6b ranker strategies + M6c null_importance (ranker) / sequential (wrapper)
# + M6d shap (ranker, lazy-extra) (ADR-0043 §5, ADR-0046/0047, ADR-0051). The resolver maps each name to its port.
FSStrategy = Literal["importance", "random_probe", "null_importance", "sequential", "shap"]
FSCutoff = Literal["top_k", "top_frac", "auto"]
# arbitration locus (ADR-0052/0054): "holdout" = M6c single DEV-internal selection-holdout; "nested" = K-fold on
# DEV scoring a FIXED subset; "nested_per_fold" = M6e fully-honest nested (re-select inside each outer fold).
# "auto" (M6f, ADR-0057) resolves to the most-honest affordable locus by data shape at composition time.
FSArbitration = Literal["holdout", "nested", "nested_per_fold", "auto"]


class CVConfig(BaseModel):
    """Cross-validation scheme and its parameters.

    ``scheme="auto"`` resolves to ``Task.default_cv_scheme`` at composition time;
    unimplemented schemes/params fail fast there, never silently.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    scheme: CVScheme = "auto"
    n_splits: int = Field(default=5, ge=1)
    n_test: int = Field(
        default=1,
        ge=1,
        description="test-fold size: rows (timeseries) or periods (timeseries_period)",
    )
    n_es: int = Field(default=1, ge=1, description="early-stopping tail size (always rows)")
    purge: int = Field(
        default=0, ge=0, description="time-series purge gap (rows or periods, per scheme)"
    )
    embargo: int = Field(
        default=0, ge=0, description="time-series embargo gap (rows or periods, per scheme)"
    )
    # calendar/Δt period CV (ADR-0096): the period unit, the Δt width for period='delta' (numeric axis
    # only), and the walk-forward step in periods (None -> =n_test, adjacent non-overlapping tiles). All
    # require scheme='timeseries_period' (gated in composition after the 'auto' resolve).
    period: PeriodUnit | None = None
    period_size: float | None = Field(default=None, gt=0, description="Δt width for period='delta'")
    step_periods: int | None = Field(
        default=None, ge=1, description="walk-forward step in periods; None -> n_test"
    )
    # wall-clock (Δt) gaps (ADR-0097): widths in the time axis' own units (datetime -> the unit dataset.time()
    # stores, numeric -> as-is); value-based cut. Mutually exclusive with the integer purge/embargo on the
    # same axis (validator). Require a time-series scheme (gated in composition).
    purge_delta: float | None = Field(
        default=None, gt=0, description="Δt purge gap before the test (value-based)"
    )
    embargo_delta: float | None = Field(
        default=None, gt=0, description="Δt embargo after earlier test windows"
    )
    # rolling / bounded lookback (ADR-0099): cap train to the last N periods/rows. None -> expanding (ADR-0027).
    max_train_periods: int | None = Field(
        default=None, gt=0, description="rolling train cap in periods (timeseries_period)"
    )
    max_train_size: int | None = Field(
        default=None, gt=0, description="rolling train cap in rows (timeseries)"
    )
    # period weighting of the leaderboard score + significance (ADR-0098): "period" macro-averages over
    # periods so unequal months weigh equally; requires a time-ordered scheme (gated in composition).
    weighting: WeightingMode = "pooled"
    # probability calibration of the winner on OOF (ADR-0030 §1); classification only, opt-in
    calibrate: CalibrateMethod = "off"
    # refinement-based selection: rank by cross-fitted calibrated proper-loss (ADR-0031); opt-in
    selection: SelectionMode = "raw"
    refinement_min_oof: int = Field(
        default=2000, ge=1, description="min valid OOF rows for refinement selection (else raw)"
    )
    # honest-regime outer holdout: a fraction carved once for an unbiased final score (ADR-0029); opt-in
    outer_holdout: float = Field(default=0.0, ge=0.0, lt=1.0)

    @model_validator(mode="after")
    def _check_period(self) -> CVConfig:
        # field-coherence only (no resolved scheme here, ADR-0096 §1): period<->period_size must agree,
        # so a 'delta' window is never undefined and a stray period_size never silently dies. Scheme-
        # dependent gates (period requires timeseries_period) live in composition after the 'auto' resolve.
        if self.period == "delta" and self.period_size is None:
            raise ValueError("period='delta' requires period_size (the Δt window width)")
        if self.period != "delta" and self.period_size is not None:
            raise ValueError("period_size is only used with period='delta'")
        return self

    @model_validator(mode="after")
    def _check_gap_axes(self) -> CVConfig:
        # ADR-0097 §1: a gap is given in ONE unit per axis — integer (rows/periods) OR Δt, never both, else
        # two conflicting semantics for one concept. Field-coherence (no resolved scheme) -> ValueError.
        if self.purge > 0 and self.purge_delta is not None:
            raise ValueError(
                "set either purge (integer rows/periods) or purge_delta (Δt), not both"
            )
        if self.embargo > 0 and self.embargo_delta is not None:
            raise ValueError(
                "set either embargo (integer rows/periods) or embargo_delta (Δt), not both"
            )
        return self


class FEConfig(BaseModel):
    """Feature-engineering catalog toggles; all transformers default off.

    A fixed, configurable catalog (not a plugin port). datetime deltas are a separate per-row
    axis driven by ``Task.report_date``, NOT part of this config. Target-encoding is
    binary-classification-only; multiclass/regression gracefully skip it.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    target_encoding: bool = False
    te_smoothing: float = Field(default=10.0, ge=0.0, description="TE shrink to global mean (k)")
    frequency_encoding: bool = False
    intersections: bool = False
    max_pairs: int = Field(default=50, ge=1, description="cap on categorical intersection pairs")


class FeatureSelectionConfig(BaseModel):
    """Feature-selection catalog; opt-in, default OFF via ``fs=None``.

    Ranker strategies ``importance``/``random_probe``/``null_importance``/``shap`` (lazy ``shap``
    extra) plus the wrapper ``sequential`` (``FeatureSubsetSelector`` port). ``compare`` runs
    several strategies and picks one subset-winner; ``compare=None`` is the single-strategy path.
    ``arbitration`` chooses the locus: ``"holdout"`` (a DEV-internal selection-holdout) or
    ``"nested"`` (K-fold on DEV; timeseries = expanding-window) with an honest significance winner.
    Anti-leakage OOF ranking/scoring lives in the application; the winning subset serializes into
    ``FeatureSchema``. ``cutoff`` applies only to ranker strategies — ``sequential`` returns its own
    subset (``seq_*``). ``null_importance`` works on every scheme: i.i.d. schemes permute uniformly,
    ``timeseries``/``group`` permute the target WITHIN structure blocks of ``null_block_size``
    rows / per group. Per-strategy randomness is isolated via a stable seed hash.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    strategy: FSStrategy = "importance"
    compare: tuple[FSStrategy, ...] | None = Field(
        default=None, description="strategies to compare; None -> single-strategy path"
    )
    selection_holdout: float = Field(
        default=0.25, gt=0.0, lt=1.0, description="DEV fraction carved to arbitrate compare"
    )
    # M6d: arbitration locus (ADR-0052). "nested" averages each subset over K DEV folds (timeseries =
    # expanding-window); ignored without `compare` (>=2 strategies) -> WARNING at resolve, not an error.
    arbitration: FSArbitration = "holdout"
    arbitration_n_splits: int = Field(
        default=5, ge=2, description="K folds for arbitration='nested'"
    )
    # M6d: structure-aware null_importance block size in rows (ADR-0050); group scheme ignores it (block=group)
    null_block_size: int = Field(
        default=50, ge=2, description="timeseries block rows for structure-aware null"
    )
    # M6e (ADR-0055): "rank" = equal-COUNT blocks (M6d); "time_window" = equal-Δt windows over raw times (valid
    # for IRREGULAR series). "auto" (M6f, ADR-0057) picks time_window+derived window on irregular series, else
    # rank. group scheme ignores all. null_block_window is the Δt required by "time_window".
    null_block_mode: Literal["rank", "time_window", "auto"] = "rank"
    null_block_window: float | None = Field(
        default=None, gt=0, description="time-window width for null_block_mode='time_window'"
    )
    # M6d: shap ranker cost-cap on explained rows (ADR-0051); None = explain the whole training part
    shap_max_samples: int | None = Field(
        default=None, gt=0, description="cap on rows explained by shap"
    )
    # M6e (ADR-0056): "tree_path_dependent" (M6d, background-free) or "interventional" (needs a background).
    # shap_background_samples caps the (deterministic, evenly-spaced) background for the interventional mode.
    shap_perturbation: Literal["tree_path_dependent", "interventional"] = "tree_path_dependent"
    shap_background_samples: int | None = Field(
        default=None, gt=0, description="background-size cap for shap_perturbation='interventional'"
    )
    # M6f (ADR-0060): interventional background sampler. "linspace" (M6e, evenly-spaced) or "kmeans"
    # (deterministic cluster centroids, better mode-coverage on small backgrounds). Inert under tpd.
    shap_background: Literal["linspace", "kmeans"] = "linspace"
    # M6f (ADR-0058): hard ceiling on projected selection ranker-refits; None = no gate (M6e behavior). When
    # set, composition downgrades arbitration to fit (or fails loud at the holdout floor).
    cost_budget_refits: int | None = Field(
        default=None, gt=0, description="hard cap on projected FS ranker-refits"
    )
    cutoff: FSCutoff = "top_frac"
    top_k: int | None = Field(default=None, ge=1, description="kept count for cutoff='top_k'")
    top_frac: float = Field(
        default=0.5, gt=0.0, le=1.0, description="kept fraction for cutoff='top_frac'"
    )
    min_features: int = Field(
        default=1, ge=1, description="floor (design_matrix needs >= 1 feature)"
    )
    n_probes: int = Field(default=3, ge=1, description="random probes for strategy='random_probe'")
    n_runs: int = Field(
        default=30, ge=1, description="null_importance target shuffles (~4% noise at p95)"
    )
    null_percentile: float = Field(
        default=95.0, gt=0.0, lt=100.0, description="null_importance background percentile"
    )
    seq_min_features: int = Field(default=1, ge=1, description="sequential floor on kept features")
    seq_patience: int = Field(
        default=2, ge=1, description="sequential plateau patience (no-improve steps)"
    )
    random_state: int | None = Field(default=None, description="None -> inherits RunConfig.seed")

    @model_validator(mode="after")
    def _check_config(self) -> FeatureSelectionConfig:
        if self.cutoff == "top_k" and self.top_k is None:
            raise ValueError("cutoff='top_k' requires top_k to be set")
        if self.compare is not None:
            if not self.compare:
                raise ValueError("compare must be non-empty (or None for the single-strategy path)")
            if len(set(self.compare)) != len(self.compare):
                raise ValueError(f"compare has duplicate strategies: {self.compare}")
        if self.null_block_mode == "time_window" and self.null_block_window is None:
            raise ValueError(
                "null_block_mode='time_window' requires null_block_window (the Δt window width)"
            )
        return self


class BudgetConfig(BaseModel):
    """Run budget: ``"none"`` (unbounded, default), wall-clock ``"time"`` or ``"trials"``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: BudgetMode = "none"
    time_budget_s: float | None = Field(default=None, gt=0)
    n_trials: int | None = Field(default=None, gt=0)
    memory_limit_mb: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _check_mode_params(self) -> BudgetConfig:
        if self.mode == "time" and self.time_budget_s is None:
            raise ValueError("budget mode 'time' requires time_budget_s")
        if self.mode == "trials" and self.n_trials is None:
            raise ValueError("budget mode 'trials' requires n_trials")
        # "none" is unbounded: a stray limit would make the manifest contradictory (ADR-0032 §5a)
        if self.mode == "none" and (self.time_budget_s is not None or self.n_trials is not None):
            raise ValueError("budget mode 'none' is unbounded; do not set time_budget_s/n_trials")
        # the mode names exactly one axis — a limit of the OTHER axis would be silently dead
        # (F3.5/F3.11 residual: never accept config that does nothing). memory is orthogonal.
        if self.mode == "time" and self.n_trials is not None:
            raise ValueError("budget mode 'time' ignores n_trials; drop it or use mode='trials'")
        if self.mode == "trials" and self.time_budget_s is not None:
            raise ValueError(
                "budget mode 'trials' ignores time_budget_s; drop it or use mode='time'"
            )
        return self


class HPOConfig(BaseModel):
    """Hyperparameter-optimization catalog; opt-in, default OFF via ``hpo=None``.

    When set, composition tunes each tunable model type on an inner-CV of DEV (before the outer
    honest selection): the tuned factory replaces (or, with ``keep_baseline``, augments) the
    baseline in the leaderboard. ``n_trials`` is the per-model search budget (distinct from
    ``BudgetConfig.n_trials``, the run candidate-loop); ``inner_cv`` is the inner fold count of the
    tuning objective. ``timeout_s`` (per-model wall-clock cap) makes the search non-deterministic —
    surfaced in the run-report. ``models=None`` tunes every type with a non-empty ``search_space``.
    The whole config is in the run-fingerprint (changed HPO -> new cache key).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    backend: Literal["optuna"] = "optuna"
    n_trials: int = Field(default=50, gt=0, description="per-model HP-search trials")
    timeout_s: float | None = Field(default=None, gt=0, description="opt. per-model wall-clock cap")
    inner_cv: int = Field(default=3, ge=2, description="inner-CV folds for the tuning objective")
    models: tuple[str, ...] | None = Field(
        default=None, description="types to tune; None -> all tunable"
    )
    keep_baseline: bool = Field(
        default=False, description="True -> keep baseline factory alongside tuned"
    )
    random_state: int | None = Field(default=None, description="None -> inherits RunConfig.seed")


class EnsembleConfig(BaseModel):
    """Ensembling catalog; opt-in, default OFF via ``ensemble=None``.

    When set (and ``run_mode='full'``), composition blends the leaderboard candidates after the
    honest selection and ships a :class:`BlendedEstimator` **only if** the blend is *significantly*
    better than the best single (the same ``SignificanceTest`` gate selection uses); otherwise the
    single winner is shipped. ``method`` is the weight search: ``"caruana"`` (default, greedy with
    replacement + seeded bagging) or ``"weighted"`` (SLSQP simplex). ``size`` caps Caruana steps /
    library; ``n_bags`` is the bagging count (1 = no bagging). ``metric=None`` blends on the run
    metric. The whole config is in the run-fingerprint (a changed ensemble config -> a new cache key).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    method: Literal["caruana", "weighted"] = "caruana"
    size: int = Field(default=50, ge=1, description="Caruana steps / library ceiling")
    n_bags: int = Field(default=20, ge=1, description="Caruana bagging subsamples; 1 = no bagging")
    metric: str | None = Field(default=None, description="None -> the run metric")
    random_state: int | None = Field(default=None, description="None -> inherits RunConfig.seed")


class TrackerConfig(BaseModel):
    """Experiment-tracking opt-in; default OFF via ``tracker=None``.

    Post-selection observability: NOT part of :class:`RunConfig` / the run-fingerprint —
    tracking cannot change the model (like ``finalize``).
    ``tracking_uri=None`` defers to the backend's own resolution (e.g. env
    ``MLFLOW_TRACKING_URI`` -> ``file:./mlruns``); ``run_name=None`` lets the backend
    generate a neutral, data-independent name.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    backend: Literal["mlflow"] = "mlflow"
    experiment: str = Field(default="honestml", min_length=1)
    tracking_uri: str | None = None
    run_name: str | None = None
    tags: dict[str, str] = Field(default_factory=dict)


class RunConfig(BaseModel):
    """Top-level run configuration; serializable basis of the run manifest."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    seed: int = 42
    cv: CVConfig = Field(default_factory=CVConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    # feature-engineering catalog (ADR-0040 §4): default all-off -> M5 unchanged; in the run-fingerprint
    # via the config dump (ADR-0042 §4), so a changed FE gives a different cache key (no stale hit).
    fe: FEConfig = Field(default_factory=FEConfig)
    # feature-selection catalog (ADR-0043 §5): default None = off -> M6a/M5 unchanged by content; in the
    # run-fingerprint via the config dump (ADR-0045 §5), so a changed FS gives a different cache key.
    fs: FeatureSelectionConfig | None = None
    # HPO catalog (ADR-0061/0062 §1): default None = off -> M6 unchanged by content; in the run-fingerprint
    # via the config dump, so a changed HPO gives a different cache key. Seed resolved before this dump.
    hpo: HPOConfig | None = None
    # ensembling catalog (ADR-0063/0064): default None = off -> single-model M7a behavior; in the run-
    # fingerprint via the config dump, so a changed ensemble config gives a different cache key. Seed
    # resolved before this dump (mirror of hpo/fs).
    ensemble: EnsembleConfig | None = None
    # honest significance band is the default; "off" is the explicit pure-argmax opt-out (ADR-0034)
    significance: SignificanceMode = "bootstrap"
    # pipeline stop-point (ADR-0038): "full" (default) ships a model; "selection" stops at the leaderboard
    run_mode: RunMode = "full"
    model_types: tuple[str, ...] = ("catboost", "lightgbm")

    @classmethod
    def parse(cls, data: object) -> RunConfig:
        """Validate untrusted input, raising :class:`ConfigError` on failure."""
        from pydantic import ValidationError

        try:
            return cls.model_validate(data)
        except ValidationError as exc:
            raise ConfigError(str(exc)) from exc
