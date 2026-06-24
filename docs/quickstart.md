# Quickstart

honestml is a tabular AutoML library for binary and multiclass classification and
regression: it fits a leaderboard of candidate models and ships the honestly best
one. Every `python` block on this page is executed on every CI run, top to bottom,
in one namespace — the examples cannot rot.

## Install

```text
pip install honestml                  # lightweight core (linear/baseline models)
pip install "honestml[boosting]"      # + lightgbm/xgboost/catboost
pip install "honestml[all]"           # everything (optuna, mlflow, onnx, shap, ...)
```

## Fit a leaderboard, ship the honestly best model

```python
from sklearn.datasets import make_classification

from honestml import AutoML

X, y = make_classification(n_samples=200, n_features=8, n_informative=5, random_state=0)

model = AutoML(task="binary", models=("baseline", "linear"), random_state=0)
model.fit(X, y)

print(model.best_model_id_)        # the winner of the honest selection
print(model.leaderboard_)          # absolute, reproducible scores
proba = model.predict_proba(X)
```

The selection is *honest by default*: every candidate is scored on out-of-fold
predictions and ranked by its absolute metric. A bootstrap equivalence band
(`significance="bootstrap"`, the default) collects the candidates statistically
indistinguishable from the best, and the winner is the simplest member of that
band — ties break by compactness, then stability, then speed. The reported
scores are never refit-inflated. See the
[correctness guide](correctness-guide.md).

## Seeing progress (logging)

The library attaches a `NullHandler` to the `honestml` logger, so a long `fit`
prints nothing until you opt in:

```python
import logging

logging.basicConfig(format="%(levelname)s %(name)s: %(message)s")
logging.getLogger("honestml").setLevel(logging.INFO)
```

INFO gives stage-by-stage progress (selection, refit, finalize); cache reuse is
recorded in the run report (`run_report_["cache"]["reused"]`); WARNING surfaces
honesty-relevant events (skipped candidates, drift signals).

## Presets

A preset is a named, declarative partial config (data, not code). It fills only
the parameters you left unset — an explicit argument always wins:

```python
fast = AutoML(task="binary", models=("baseline", "linear"), random_state=0, preset="fast")
fast.fit(X, y)
print(fast.run_report_["preset"])   # {'name': 'fast', 'applied': ['cv']}
```

Built-ins: `fast` (3-fold CV), `balanced` (+ gated ensembling), `best`
(+ HPO via the `optuna` extra + ensembling). Honesty-controlling parameters
(`significance`, `finalize`, `run_mode`) are deliberately **not** presettable:
a preset can never silently downgrade the honest-selection contract.

## The run report: JSON, markdown, HTML

```python
import tempfile
from pathlib import Path

from honestml import render_report, save_run_report

out = Path(tempfile.mkdtemp())
save_run_report(model.run_report_, out)              # machine-readable provenance
md = render_report(model.run_report_, out, fmt="md") # human-readable summary
print(md.read_text(encoding="utf-8").splitlines()[0])
```

The rendered report summarizes the winner, the equivalence band, the full
leaderboard, and per-stage timings. `fmt="html"` embeds leaderboard/timing
charts when the `report` extra (matplotlib) is installed, and degrades
gracefully without it.

## Save, load, serve

```python
from honestml import load_artifact, save_artifact

art_dir = out / "artifact"
save_artifact(model.fitted_, art_dir)
loaded = load_artifact(art_dir)      # integrity-checked (sha256 manifest)
assert (loaded.predict(X) == model.predict(X)).all()
```

The artifact is a versioned directory (manifest + schema + model body) servable
by the slim `honestml[inference]` runtime — no training stack needed. Boosting
models can be persisted natively (`save_artifact(..., model_format="native")`),
and a supported subset exports to ONNX via `honestml.export_onnx` (parity-gated).

## Going further

```text
AutoML(budget=600)                                      # cooperative time budget, graceful degradation
AutoML(cache="runs/")                                   # resume: fingerprint-scoped candidate cache
AutoML(feature_selection=FeatureSelectionConfig(...))   # honest OOF feature selection
AutoML(hpo=HPOConfig(n_trials=50))                      # per-model Optuna tuning before selection
AutoML(ensemble=EnsembleConfig())                       # blend shipped only if significantly better
AutoML(tracker="mlflow")                                # post-fit experiment tracking (mlflow extra)
```

`FeatureSelectionConfig`, `HPOConfig`, and `EnsembleConfig` are importable from
`honestml`. The [API reference](api.md) documents every parameter; the
[correctness guide](correctness-guide.md) explains the selection contract.
