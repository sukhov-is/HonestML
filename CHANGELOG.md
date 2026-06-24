# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/) and the project adheres to
[Semantic Versioning](https://semver.org/) (see `docs/versioning-policy.md`).

## [Unreleased]

## [1.0.0] - 2026-06-24

First public release: a tabular AutoML library for binary/multiclass classification and
regression, built around honest model selection — the score you see is the score you can
expect in production.

The honesty-benchmark baseline (`benchmarks/baseline.json`) is bootstrapped for this release;
subsequent releases gate against it for no regress (see `docs/releasing.md`).

### Added
- **Task and data input contract:** sklearn-compatible `AutoML` facade
  (`fit`/`predict`/`predict_proba`/`score`, works in `Pipeline`) over pandas/polars/numpy
  inputs, with typed schema inference, stable train↔inference categorical encoding,
  boundary validation, and row metadata via `fit(..., sample_weight=, groups=, time=,
  label_time=)`.
- **CV schemes:** stratified / kfold / group / holdout / timeseries (purge+embargo, value-based
  time ordering) / timeseries_period (calendar or Δt periods with wall-clock purge/embargo,
  optional per-period weighting and a rolling max-train window) (`CVConfig(scheme=...)`; `"auto"`
  picks the task default); every fold passes an anti-leakage validation.
- **Honest selection:** out-of-fold scoring on a shared CV split with a seeded
  paired-bootstrap **equivalence band** (`significance="bootstrap"`, default) — the
  simplest model statistically indistinguishable from the best wins and ties are
  disclosed, never hidden; opt-in probability calibration (`CVConfig(calibrate=...)`)
  gated by cross-fitted improvement.
- **Outer holdout + finalize:** `CVConfig(outer_holdout=...)` carves a scheme-aware,
  untouched holdout scored exactly once for an unbiased final estimate; `finalize=True`
  then refits the shipped model on all data after scoring.
- **Presets:** `AutoML(preset="fast" | "balanced" | "best")` — declarative partial configs
  that fill only unset parameters; an explicit argument always wins, and honesty settings
  are not presettable.
- **Budget + resume/cache:** `budget=<seconds>` or `BudgetConfig(...)` (time / trial /
  memory limits) with graceful degradation to the best model so far; `cache="dir/"`
  reuses per-candidate results keyed by a deterministic run fingerprint and resumes
  interrupted runs.
- **Feature engineering + selection:** opt-in `FEConfig` (leakage-controlled out-of-fold
  target encoding, frequency encoding, datetime deltas, categorical intersections) and
  `FeatureSelectionConfig` (importance / random-probe / null-importance / sequential /
  SHAP strategies with honest multi-strategy arbitration on a holdout or nested CV). The
  `sequential` wrapper chooses its feature count honestly: it explores the full backward
  trajectory and the **fewest features statistically indistinguishable from the best**
  (significance band + Occam tie-break, default `significance="bootstrap"`) win, instead of
  the raw out-of-fold argmax; `significance="off"` reproduces the plain argmax. The band is
  scored on the selection folds and is strictly more conservative than argmax (residual
  optimism documented; independent-OOF scoring is a future improvement).
- **Hyperparameter optimization:** `hpo=HPOConfig(...)` — seeded Optuna search over
  per-model spaces on an inner CV of the dev data; tuned candidates then compete in the
  regular honest selection, sharing the run's time budget.
- **Ensembling:** `ensemble=EnsembleConfig(...)` — greedy (Caruana) or weighted blend over
  the out-of-fold predictions, shipped only if significantly better than the best single
  model; the gate decision is always reported.
- **Run report + rendering:** versioned, tracker-independent JSON `run_report_` with full
  provenance (resolved config, run fingerprint, leaderboard, equivalence band, timings,
  budget/FS/HPO/ensemble outcomes); `save_run_report` plus `render_report` to markdown or
  self-contained HTML (charts via the `report` extra).
- **Experiment tracking:** opt-in `tracker="mlflow"` / `TrackerConfig(...)` logs the run
  report after fit — fail-soft and free of global MLflow state; custom backends plug in
  via the `ExperimentTracker` port.
- **Artifacts + serving:** `save_artifact` / `load_artifact` — a self-contained, versioned
  artifact directory with a sha256 integrity manifest; native boosting bodies
  (`model_format="native"`, no pickle); the slim `honestml[inference]` extra serves
  `load_artifact(...).predict(...)` without importing the training stack.
- **ONNX export:** `export_onnx(model, directory, sample=...)` — a parity-gated,
  export-only bundle (linear and boosting models) for external runtimes.
- **Models and plugins:** lightweight built-in zoo (baseline, linear) plus
  catboost/lightgbm/xgboost via the `boosting` extra; third-party estimators via
  `honestml.models` entry points (see `docs/plugin-contract.md`), discovered lazily with
  fail-fast name-conflict detection.
- **Logging, exceptions, typing:** silent-by-default `honestml` logger (`NullHandler`);
  one exception taxonomy rooted at `AutoMLError`; fully typed (`py.typed`);
  `import honestml` stays lightweight — optional extras load lazily and a missing one
  fails fast with the exact `pip install honestml[...]` hint.
