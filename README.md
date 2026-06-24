# honestml

[![PyPI](https://img.shields.io/pypi/v/honestml.svg)](https://pypi.org/project/honestml/)
[![CI](https://github.com/sukhov-is/HonestML/actions/workflows/ci.yml/badge.svg)](https://github.com/sukhov-is/HonestML/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://pypi.org/project/honestml/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Tabular AutoML where the leaderboard doesn't lie.** Most AutoML frameworks ship the
model with the best validation score — but that number is optimistic, because you
selected for it. honestml is built so that the score you see is the score you can
expect in production.

It covers binary / multiclass classification and regression behind a clean, extensible
core. The honesty is in *how it selects*: out-of-fold scoring on a shared CV split; a
bootstrap **equivalence band** that, among the statistically indistinguishable best
candidates, ships the simplest one; leakage-controlled feature engineering and
selection; an optional untouched outer holdout scored exactly once; and reproducible,
fingerprinted runs.

```python
from honestml import AutoML

model = AutoML(task="binary").fit(X, y)
proba = model.predict_proba(X_new)
print(model.best_model_id_, model.leaderboard_)
```

The library is silent by default (a `NullHandler` on the `honestml` logger); enable
progress with `logging.getLogger("honestml").setLevel(logging.INFO)` plus
`logging.basicConfig()`.

## Install

```bash
pip install honestml                 # lightweight core (baseline/linear models)
pip install "honestml[boosting]"     # core + catboost, lightgbm, xgboost
pip install "honestml[all]"          # boosting + optuna (HPO), mlflow (tracking), onnx, shap, report and the rest
pip install "honestml[inference]"    # slim serving runtime (load_artifact + predict only)
```

Requires Python >= 3.10. Heavy dependencies are optional extras and imported
lazily — `import honestml` stays light, and a missing extra fails fast with the
exact `pip install honestml[...]` hint.

## What you get

| Capability | How |
|---|---|
| Honest model selection | OOF scoring on a shared CV split; a seeded bootstrap **equivalence band** (`significance="bootstrap"`, the default) collects candidates statistically indistinguishable from the best, and the simplest member of the band wins — ties are disclosed, not hidden |
| CV schemes | stratified / kfold / group / holdout / **timeseries** (purge+embargo, value-based time order) / **timeseries_period** (calendar or Δt period folds, wall-clock gaps, optional per-period weighting, rolling train window) — `fit(..., time=, label_time=, groups=)` |
| Outer holdout + finalize | `cv=CVConfig(outer_holdout=0.2)`: selection sees only DEV, the holdout is scored once; the shipped model is refit on all data after scoring (`finalize=True`) |
| Presets | `AutoML(preset="fast" / "balanced" / "best")` — declarative, data-driven partial configs; an explicit argument always wins; honesty parameters are not presettable |
| Budget + resume | `budget=600` (seconds) or `BudgetConfig(...)` with graceful degradation; `cache="runs/"` resumes by run fingerprint |
| Feature engineering / selection | OOF-honest target (binary-only) / frequency encoding, datetime deltas, intersections; importance / null-importance / random-probe / sequential / SHAP selection with honest arbitration |
| HPO + ensembling | `hpo=HPOConfig(...)` (Optuna, per-model search before the honest selection); `ensemble=EnsembleConfig()` — a Caruana/weighted blend ships **only if significantly better** |
| Run report | `model.run_report_` (versioned JSON, tracker-independent); `save_run_report` and `render_report` produce markdown or self-contained HTML (charts via the `report` extra) |
| Experiment tracking | `tracker="mlflow"` or `TrackerConfig(...)` — post-fit, fail-soft, no global mlflow state; custom backends via the `ExperimentTracker` port |
| Artifacts + serving | `save_artifact` / `load_artifact` — versioned, integrity-checked artifact directory (see Standalone inference below) |
| ONNX export | `honestml.export_onnx(model, dir, sample=...)` — parity-gated, export-only bundle for external runtimes |
| Plugins | third-party models via `honestml.models` entry points (`docs/plugin-contract.md`) |

## Standalone inference

```python
from honestml import load_artifact

model = load_artifact("artifact_dir/")   # integrity-checked against the sha256 manifest
predictions = model.predict(X_new)
```

The artifact directory is self-contained — manifest, preprocessing schema and the
model body — and loads under the slim `honestml[inference]` install: no training
stack is imported. Boosting models can be saved with structural native bodies
(`model_format="native"`). **Trust model:** the default body is joblib/pickle —
load only artifacts you trust; native bodies contain no pickle (a non-boosting
estimator and the optional calibrator still ship as joblib).

## Reproducibility

Every run computes a **fingerprint** over the resolved config, data signature,
estimator set and library versions; the run report carries it together with the
full provenance (leaderboard, band, budget outcome, FS/HPO/ensemble decisions,
timings). Same inputs → same selection.

## Documentation

Documentation lives in `docs/` — quickstart, API reference, correctness guide and
the plugin contract; build it locally with `mkdocs serve`. Source and issue
tracker: <https://github.com/sukhov-is/HonestML>.

## Development

```bash
uv sync --extra dev --extra boosting --extra shap --extra pyarrow --extra mlflow
uv run pytest                 # full suite (onnx export tests also need `--extra onnx`, Python >=3.11)
uv run ruff check src tests; uv run mypy src/honestml; uv run lint-imports
```

The layered architecture (core ← adapters ← application ← composition) is enforced
by import-linter. See `docs/releasing.md` for the release pipeline and
`benchmarks/` for the honesty benchmark suite.

## License

MIT (see `LICENSE`).
