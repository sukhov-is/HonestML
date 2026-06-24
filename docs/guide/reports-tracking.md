# Run reports and experiment tracking

What a finished fit leaves behind: a versioned JSON report with the full
provenance of the run, and — only if you opt in — one record in an experiment
tracker. Every `python` block on this page is **self-contained**: copy any one
of them and it runs as-is, and every block is executed on each CI run, so the
examples cannot rot. The examples use the lightweight
`models=("baseline", "linear")` so they finish in seconds.

## The run report

After `fit`, `run_report_` is a single tracker-independent JSON document (plain
dicts, lists and scalars — `json.dumps` works on it directly), versioned by
`run_manifest_version` and evolving additively, so a consumer reads known keys
and ignores unknown ones. It records the selection outcome (`winner`,
`leaderboard`, `band`, `significance`, `holdout_score`, per-candidate
`failed`), the resolved inputs (`config`, `task`, `metric`, `preset`), every
opt-in stage (`feature_selection`, `hpo`, `ensemble`, `budget`, `cache`) and
the provenance trail (`honestml_version`, `run_fingerprint`, `timings`,
`serving`). Blocks for features you did not opt into are `None` or report
their off-state — never silently missing.

```python
from sklearn.datasets import make_classification

from honestml import AutoML

X, y = make_classification(n_samples=150, n_features=6, n_informative=4, random_state=0)

model = AutoML(task="binary", models=("baseline", "linear"), random_state=0).fit(X, y)
report = model.run_report_

print(sorted(report))
print(report["winner"], report["metric"], report["honestml_version"])
```

## The run fingerprint

`run_fingerprint` is the reproducibility contract: a hex SHA-256 over canonical
JSON of everything that can change a candidate's out-of-fold score — the
resolved config, the task and metric identity, a content digest of the data
(design matrix, target, row-aligned metadata, schema), the resolved estimator
set and the installed library versions. Same inputs give the same fingerprint
and therefore the same selection; the key is fail-closed, so any change to any
ingredient changes it. It is also the cache key for `cache=`/resume.
Post-selection observability — `tracker`, `finalize`, report rendering — is
deliberately outside the fingerprint, because it cannot change the model.

```python
from sklearn.datasets import make_classification

from honestml import AutoML

X, y = make_classification(n_samples=150, n_features=6, n_informative=4, random_state=0)

a = AutoML(task="binary", models=("baseline", "linear"), random_state=0).fit(X, y)
b = AutoML(task="binary", models=("baseline", "linear"), random_state=0).fit(X, y)

print(a.run_report_["run_fingerprint"] == b.run_report_["run_fingerprint"])  # True
print(a.run_report_["run_fingerprint"][:16], "...")
```

## Saving and rendering

`save_run_report(report, path)` writes the report as indented UTF-8 JSON and
returns the written path; when `path` is an existing directory the file is
`path/run_report.json`, and `overwrite=False` raises `FileExistsError` instead
of replacing an existing one. `render_report(report, path, fmt="md")` renders a
human-readable summary with the winner, the band, the leaderboard, per-stage
timings and the resolved config; `report` may be the `run_report_` mapping or a
path to a saved `run_report.json`, so save-then-render round-trips. Markdown
rendering needs nothing beyond the stdlib.

```python
import tempfile
from pathlib import Path

from sklearn.datasets import make_classification

from honestml import AutoML, render_report, save_run_report

X, y = make_classification(n_samples=150, n_features=6, n_informative=4, random_state=0)
model = AutoML(task="binary", models=("baseline", "linear"), random_state=0).fit(X, y)

out = Path(tempfile.mkdtemp())
json_path = save_run_report(model.run_report_, out)   # out/run_report.json
md_path = render_report(json_path, out, fmt="md")     # out/run_report.md

print("\n".join(md_path.read_text(encoding="utf-8").splitlines()[:7]))
```

`fmt="html"` writes a single self-contained file; with the `report` extra
installed it embeds leaderboard and timing charts as inline PNG, and without it
degrades gracefully (chart-less HTML plus a WARNING — never an `ImportError`):

```text
pip install "honestml[report]"   # matplotlib, used only for the HTML charts

render_report(model.run_report_, out, fmt="html")   # out/run_report.html
```

## MLflow tracking

Tracking is opt-in and post-fit: pass `tracker="mlflow"`, or a `TrackerConfig`
to set the experiment name, tracking URI, run name and tags. After a completed
fit, honestml logs exactly one MLflow run: the flattened resolved config as
params, the leaderboard scores, holdout score and stage timings as metrics,
provenance tags (`honestml.version`, `honestml.fingerprint`,
`honestml.winner`) and the full `run_report.json` as a run artifact. A missing
mlflow install fails fast *before* training starts, while a tracking failure
*after* the fit is downgraded to a WARNING — it can never destroy a finished
model. The adapter never mutates global mlflow state (no `set_tracking_uri`,
`set_experiment` or fluent `start_run`); everything goes through a client bound
to an explicit run id.

```text
pip install "honestml[mlflow]"

from honestml import AutoML, TrackerConfig

model = AutoML(
    task="binary",
    models=("baseline", "linear"),
    random_state=0,
    tracker=TrackerConfig(
        experiment="churn",
        tracking_uri="http://mlflow:5000",   # None defers to MLFLOW_TRACKING_URI -> file:./mlruns
        run_name="weekly-refresh",
        tags={"team": "risk"},
    ),
).fit(X, y)
# tracker="mlflow" is shorthand for TrackerConfig() — experiment "honestml", default URI
```

## Custom tracking backends

`tracker=` also accepts any object implementing the `ExperimentTracker` port —
a single method `log_run(report)`, called once per completed fit with a deep
copy of `run_report_`, so a mutating implementation cannot corrupt the facade's
own report. Implementations should ignore unknown keys, because the report
evolves additively. The same fail-soft rule applies: an exception raised by
your backend becomes a WARNING, not a failed fit.

```python
from sklearn.datasets import make_classification

from honestml import AutoML

class ListTracker:
    def __init__(self):
        self.runs = []

    def log_run(self, report):
        self.runs.append(report)

X, y = make_classification(n_samples=150, n_features=6, n_informative=4, random_state=0)

tracker = ListTracker()
AutoML(task="binary", models=("baseline", "linear"), random_state=0, tracker=tracker).fit(X, y)

print(len(tracker.runs), tracker.runs[0]["winner"])
```

The report describes the run; the model itself ships separately — see
[artifacts, serving and ONNX export](artifacts-serving.md).
