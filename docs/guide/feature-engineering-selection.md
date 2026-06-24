# Feature engineering and selection

What honestml can derive from your columns before the models compete, and how it
prunes the feature set afterwards. Every `python` block on this page is
**self-contained**: copy any one of them and it runs as-is — and every block is
executed on each CI run, so the examples cannot rot. Everything here is
**opt-in**: `FEConfig` is a fixed catalog of transformers, all off by default,
and feature selection is off until you pass a `FeatureSelectionConfig`. Every
fitted spec serializes into `schema_`, so the exact same transformation is
applied at `predict` time.

## Target encoding

`FEConfig(target_encoding=True)` replaces each categorical with the smoothed
target mean of its category: `(sum_y + k·global_mean) / (count + k)`, where `k`
is `te_smoothing` (default 10) — larger values shrink rare categories harder
toward the global mean. During model selection the encoding is computed
**out-of-fold**: a row's encoded value never sees its own fold's target, so the
leaderboard is not inflated by target leakage; the shipped model carries the
full-train map, with unseen and null categories falling back to the global
mean. Target encoding is binary-classification-only — a multiclass or
regression run skips it gracefully with a WARNING while the rest of the catalog
still applies.

```python
import numpy as np
import pandas as pd

from honestml import AutoML, FEConfig

rng = np.random.default_rng(0)
n = 200
df = pd.DataFrame(
    {
        "amount": rng.normal(size=n),
        "city": rng.choice(["ams", "ber", "lis"], size=n),
    }
)
y = ((df["amount"] > 0) | (df["city"] == "ams")).astype(int)

model = AutoML(
    task="binary",
    models=("baseline", "linear"),
    feature_engineering=FEConfig(target_encoding=True, te_smoothing=5.0),
    random_state=0,
).fit(df, y)

print([c for c in model.schema_.features if c.endswith("_te")])
```

## Frequency encoding and categorical intersections

`frequency_encoding=True` adds, for each source categorical, a numeric
`<col>_freq` column holding the category's share of the training rows.
`intersections=True` pairs the source categoricals (alphabetical order) and
concatenates each pair into a new combined category `a__b` (nulls become
`__NA__`) — a cheap way to expose interactions like *device × country* to
linear models. The pair count is capped by `max_pairs` (default 50); when there
are more possible pairs, the list is truncated with a WARNING. Both transformers
work from the original categoricals only — derived intersection columns are not
re-fed into frequency or target encoding.

```python
import numpy as np
import pandas as pd

from honestml import AutoML, FEConfig

rng = np.random.default_rng(0)
n = 240
df = pd.DataFrame(
    {
        "amount": rng.normal(size=n),
        "device": rng.choice(["mobile", "desktop"], size=n),
        "country": rng.choice(["de", "fr", "pt"], size=n),
    }
)
y = ((df["device"] == "mobile") & (df["country"] == "de")).astype(int)

model = AutoML(
    task="binary",
    models=("baseline", "linear"),
    feature_engineering=FEConfig(frequency_encoding=True, intersections=True, max_pairs=10),
    random_state=0,
).fit(df, y)

print(sorted(c for c in model.schema_.features if c.endswith("_freq") or "__" in c))
```

## Datetime deltas via the task's report date

Datetime columns are a separate axis, driven by `Task(report_date=...)` rather
than `FEConfig`. When a report-date column is declared (or auto-detected by the
names `report_dt` / `report_date` / `feature_dt`), every other datetime column
becomes a numeric `<col>__days_to_report` feature — the whole-day difference
`report_date - column`. Datetime columns with no report date to anchor them are
dropped from the features with a WARNING, never fed to the models raw.

```python
import numpy as np
import pandas as pd

from honestml import AutoML, Task

rng = np.random.default_rng(0)
n = 160
df = pd.DataFrame(
    {
        "amount": rng.normal(size=n),
        "last_purchase": pd.to_datetime("2024-03-01")
        - pd.to_timedelta(rng.integers(0, 90, size=n), unit="D"),
        "report_dt": pd.to_datetime(["2024-03-01"] * n),
    }
)
y = (df["amount"] > 0).astype(int)

model = AutoML(
    task=Task(kind="binary", report_date="report_dt"),
    models=("baseline", "linear"),
    random_state=0,
).fit(df, y)

print([c for c in model.schema_.features if c.endswith("__days_to_report")])
```

## Selecting features

`feature_selection=FeatureSelectionConfig(...)` prunes the (FE-augmented)
feature set before the leaderboard is scored. The ranker strategies score each
feature individually: `"importance"` (tree-ensemble impurity importance),
`"random_probe"` (margin over injected random probe columns), `"null_importance"`
(real importance against a target-permuted background) and `"shap"` (SHAP values,
needs the `shap` extra — `pip install "honestml[shap]"`). Ranking is honest: on
every CV fold the ranker sees only that fold's training rows, the per-fold scores
are normalized and averaged, then `cutoff` turns them into a subset —
`"top_frac"` (default, keep the strongest 50%), `"top_k"` or `"auto"`, with a
`min_features` floor.

`"sequential"` is a different kind of strategy: a greedy wrapper that scores
whole subsets along a backward trajectory rather than ranking features, so
`cutoff` does not apply to it — it chooses its own feature count. Under the run's
`significance` setting (default `"bootstrap"`) it walks the full trajectory down
to the `seq_min_features` floor and keeps the **smallest** subset statistically
indistinguishable from the best one (paired-bootstrap band + Occam tie-break),
rather than the optimistic out-of-fold argmax; `significance="off"` restores the
plain argmax (with `seq_patience` early-stopping the descent). The band outcome
is disclosed as `seq_band` in the report.

The kept subset is attached to the schema (so training and inference can never
diverge) and disclosed in `run_report_["feature_selection"]`.

```python
import numpy as np
import pandas as pd

from honestml import AutoML, FeatureSelectionConfig

rng = np.random.default_rng(0)
n = 300
df = pd.DataFrame(
    {
        "signal_a": rng.normal(size=n),
        "signal_b": rng.normal(size=n),
        "noise_1": rng.normal(size=n),
        "noise_2": rng.normal(size=n),
        "noise_3": rng.normal(size=n),
        "segment": rng.choice(["a", "b", "c"], size=n),
    }
)
signal = df["signal_a"] + df["signal_b"] + (df["segment"] == "a")
y = (signal + rng.normal(scale=0.3, size=n) > 0.5).astype(int)

model = AutoML(
    task="binary",
    models=("baseline", "linear"),
    feature_selection=FeatureSelectionConfig(strategy="importance", cutoff="top_k", top_k=3),
    random_state=0,
).fit(df, y)

fs = model.run_report_["feature_selection"]
print(fs["strategy"], fs["n_selected"], fs["selected"])
```

## Comparing strategies with honest arbitration

`compare=(...)` runs several strategies and lets an arbiter pick one subset
winner instead of trusting any single ranker. With `arbitration="holdout"` (the
default) each strategy selects on part of the data and the subsets are scored
on an independent selection-holdout; `"nested"` refits each subset over K
folds and uses the significance band to prefer the most compact subset among
the statistically indistinguishable ones; `"auto"` resolves to the most honest
locus the data size can afford. The whole record — the strategies evaluated,
their per-strategy scores and the winner — lands in
`run_report_["feature_selection"]`, and only the winning subset ships in the
schema.

```python
import numpy as np
import pandas as pd

from honestml import AutoML, FeatureSelectionConfig

rng = np.random.default_rng(0)
n = 300
df = pd.DataFrame(
    {
        "signal_a": rng.normal(size=n),
        "signal_b": rng.normal(size=n),
        "noise_1": rng.normal(size=n),
        "noise_2": rng.normal(size=n),
    }
)
y = (df["signal_a"] + df["signal_b"] + rng.normal(scale=0.3, size=n) > 0).astype(int)

model = AutoML(
    task="binary",
    models=("baseline", "linear"),
    feature_selection=FeatureSelectionConfig(
        compare=("importance", "random_probe"), cutoff="top_k", top_k=2
    ),
    random_state=0,
).fit(df, y)

fs = model.run_report_["feature_selection"]
print(fs["winner"], fs["strategies_evaluated"])
print(fs["selected"])
```

Every decision on this page is recorded: the fitted FE specs live in `schema_`
and the resolved configs plus the selection outcome in `run_report_`. How the
pruned candidates are then scored and the winner chosen is covered in
[cross-validation and honest selection](cv-selection.md).
