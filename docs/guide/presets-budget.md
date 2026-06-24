# Presets, budget and resume

How to trade thoroughness for speed without touching the honesty contract: named
presets fill the config, a budget caps the run, and a cache directory makes the
second identical `fit` nearly free. Every `python` block on this page is
**self-contained**: copy any one of them and it runs as-is — and every block is
executed on each CI run, so the examples cannot rot.

## Presets

A preset is a named, declarative partial config — data, not code. It fills
**only** the parameters you left unset, so an explicit argument always wins, and
the run is keyed and reported on the resolved values, not the preset name.
Built-ins: `fast` (3-fold CV), `balanced` (adds gated ensembling), `best` (adds
per-model HPO via the `optuna` extra, plus ensembling); a custom `Mapping` over
the same parameter surface works too. The honesty-controlling parameters —
`significance`, `finalize`, `run_mode` — are not presettable by construction: a
preset can never silently downgrade the honest-selection contract. Provenance
lands in `run_report_["preset"]` as `{"name": ..., "applied": [...]}`.

```python
from sklearn.datasets import make_classification

from honestml import AutoML

X, y = make_classification(n_samples=200, n_features=8, n_informative=5, random_state=0)

filled = AutoML(
    task="binary", models=("baseline", "linear"), preset="fast", random_state=0
).fit(X, y)
print(filled.run_report_["preset"], filled.run_report_["config"]["cv"]["n_splits"])

explicit = AutoML(  # an explicit cv=5 wins over the preset's cv=3
    task="binary", models=("baseline", "linear"), cv=5, preset="fast", random_state=0
).fit(X, y)
print(explicit.run_report_["preset"], explicit.run_report_["config"]["cv"]["n_splits"])
```

## The run budget

`budget=` accepts a float — wall-clock seconds, sugar for
`BudgetConfig(mode="time", time_budget_s=...)` — or a `BudgetConfig`: `mode` is
`"none"` (unbounded), `"time"` or `"trials"` (with `n_trials`, the candidate
count), and an orthogonal `memory_limit_mb` (process RSS, needs the `memory`
extra) composes with any mode. Enforcement is cooperative and per-candidate: the
check runs before each candidate starts, the time clock starts at the candidate
loop (reading the data is not billed), and HPO consumes from the same pool.
Degradation is graceful — when the budget runs out mid-run the remaining
candidates are skipped, the best model so far still ships, and the final refit is
never budget-gated. The outcome lands in `run_report_["budget"]`: the `mode`,
whether the run was `exhausted`, the `skipped` candidate ids and which axis
(`exhausted_by`) ended the run.

```python
from sklearn.datasets import make_classification

from honestml import AutoML, BudgetConfig

X, y = make_classification(n_samples=200, n_features=8, n_informative=5, random_state=0)

model = AutoML(
    task="binary",
    models=("baseline", "linear"),
    budget=BudgetConfig(mode="trials", n_trials=1),  # room for exactly one candidate
    random_state=0,
).fit(X, y)

print(model.best_model_id_)
print(model.run_report_["budget"])
```

## Cache and resume

`cache="some/dir"` turns on a per-candidate result store keyed by the **run
fingerprint**: a digest of the resolved config, the task and metric identity, the
estimator set, the compute-stack library versions and a content signature of the
training data. A second `fit` with all of that identical restores each
candidate's out-of-fold result instead of retraining it; change anything — a
row, the seed, a config field, a library version — and the fingerprint changes,
so a stale hit is impossible (a cold run next to older fingerprints logs that the
config or the data changed). Reuse is per candidate and every result is written
durably on completion, so an interrupted run *resumes*: the next `fit` recomputes
only the remainder. The truthful outcome is in `run_report_["cache"]` —
`enabled`, the `reused` ids and the `computed` ids.

```python
import tempfile

from sklearn.datasets import make_classification

from honestml import AutoML

X, y = make_classification(n_samples=200, n_features=8, n_informative=5, random_state=0)
cache_dir = tempfile.mkdtemp()

first = AutoML(
    task="binary", models=("baseline", "linear"), cache=cache_dir, random_state=0
).fit(X, y)
second = AutoML(
    task="binary", models=("baseline", "linear"), cache=cache_dir, random_state=0
).fit(X, y)

print(first.run_report_["cache"]["reused"])  # [] — a cold run computes everything
print(second.run_report_["cache"]["reused"])  # both candidates restored from the cache
print(second.best_model_id_ == first.best_model_id_)
```

All three knobs are provenance-first: the preset block, the budget outcome and
the cache outcome all land in `run_report_` — see the
[quickstart](../quickstart.md) for saving and rendering the report.
