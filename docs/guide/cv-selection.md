# Cross-validation and honest selection

How honestml splits your data and how the winner is chosen. Every `python` block
on this page is **self-contained**: copy any one of them and it runs as-is — and
every block is executed on each CI run, so the examples cannot rot. The examples
use the lightweight `models=("baseline", "linear")` so they finish in seconds;
everything shown applies unchanged to the boosting models.

## Picking a CV scheme

`cv` accepts an integer (the number of folds of the task's default scheme) or a
`CVConfig`. The default `scheme="auto"` resolves to stratified k-fold for
classification and plain k-fold for regression; `"holdout"` is a single shuffled
split. The full menu is `"kfold"` / `"stratified"` (i.i.d. shuffled folds),
`"group"` (repeated entities), `"holdout"` (one split), and the time-ordered
`"timeseries"` (row windows) and `"timeseries_period"` (calendar / Δt windows) —
each covered below. An unimplemented scheme or invalid parameter fails fast at
`fit`, never silently falls back.

```python
from sklearn.datasets import make_regression

from honestml import AutoML, CVConfig

X, y = make_regression(n_samples=200, n_features=6, noise=0.3, random_state=0)

model = AutoML(
    task="regression",
    models=("baseline", "linear"),
    cv=CVConfig(scheme="kfold", n_splits=3),
    random_state=0,
).fit(X, y)

print(model.best_model_id_, model.leaderboard_)
```

## Group-aware CV

With `scheme="group"`, rows that share a group label never span train and test —
the leakage guard for repeated entities (a customer with many rows, a patient
with many visits). Group labels are row-aligned metadata passed to `fit`, not a
feature; classification uses the stratified-group variant automatically. If a
`groups=` column is present but the scheme is not group-aware, honestml warns
about the leakage risk instead of silently accepting it.

```python
import numpy as np
from sklearn.datasets import make_classification

from honestml import AutoML, CVConfig

X, y = make_classification(n_samples=240, n_features=8, n_informative=5, random_state=0)
groups = np.arange(240) // 4  # 60 entities, 4 rows each

model = AutoML(
    task="binary",
    models=("baseline", "linear"),
    cv=CVConfig(scheme="group", n_splits=3),
    random_state=0,
).fit(X, y, groups=groups)

print(model.best_model_id_)
```

## Time-series CV with purge and embargo

`scheme="timeseries"` orders rows by the value of the `time=` column and scores
on expanding-window folds — train always precedes test. Sizes are in rows of the
time-ordered data: `n_test` is the test-window size per fold, `purge` drops rows
right before each test window and `embargo` skips rows right after earlier test
windows, so overlapping or delayed labels cannot leak across the split. When
labels mature over an interval, pass `label_time=` (the label end time) for the
full purge. A shuffling scheme over data that has a time axis triggers a
look-ahead warning.

```python
import numpy as np
from sklearn.datasets import make_classification

from honestml import AutoML, CVConfig

X, y = make_classification(n_samples=240, n_features=8, n_informative=5, random_state=0)
time = np.arange(240)  # any orderable axis: ints, timestamps, dates

model = AutoML(
    task="binary",
    models=("baseline", "linear"),
    cv=CVConfig(scheme="timeseries", n_splits=3, n_test=40, purge=5, embargo=5),
    random_state=0,
).fit(X, y, time=time)

print(model.best_model_id_)
```

## Calendar- and Δt-period folds

`scheme="timeseries_period"` walks forward over **periods** instead of rows: each fold tests a block of
whole periods and trains on all strictly earlier ones (expanding). Set `period` to `"month"`, `"week"`
(ISO, Monday-anchored) or `"day"` for a datetime `time=` axis, or `"delta"` with a `period_size` (the window
width) for a numeric axis. With period folds the integer knobs count **periods**, not rows: `n_test` is the
test width in periods, `step_periods` is the walk-forward step (defaults to `n_test`, i.e. adjacent tiles)
and `purge`/`embargo` are period gaps (the early-stopping tail `n_es` is the one exception — it always
counts rows). Empty periods never produce a fold, and the resolved period counts land in
`run_report_["cv"]`.

```python
import numpy as np
from sklearn.datasets import make_classification

from honestml import AutoML, CVConfig

X, y = make_classification(n_samples=360, n_features=8, n_informative=5, random_state=0)
time = np.arange("2021-01-01", "2022-01-01", dtype="datetime64[D]")[:360]  # ~12 months, daily

model = AutoML(
    task="binary",
    models=("baseline", "linear"),
    cv=CVConfig(scheme="timeseries_period", period="month", n_splits=3, n_test=2),
    random_state=0,
).fit(X, y, time=time)

print(model.best_model_id_, model.run_report_["cv"])
```

A "train 5 / test 2 months" recipe is `n_test=2` plus `max_train_periods=5` (the rolling cap below); a
numeric axis binned into fixed windows is `period="delta", period_size=...`.

## Wall-clock (Δt) gaps and rolling windows

On irregular axes (markets close at night and on weekends) a gap counted in rows spans a different real
duration each time. `purge_delta` and `embargo_delta` instead measure the gap **by time value** — in the
units the `time=` axis stores (for a datetime axis, its storage unit). They apply to both `"timeseries"`
and `"timeseries_period"` and are mutually exclusive with the integer `purge`/`embargo` on the same axis
(set one or the other, never both).

```python
import numpy as np
from sklearn.datasets import make_classification

from honestml import AutoML, CVConfig

X, y = make_classification(n_samples=240, n_features=8, n_informative=5, random_state=0)
time = np.arange(240.0)  # a numeric time axis

model = AutoML(
    task="binary",
    models=("baseline", "linear"),
    cv=CVConfig(scheme="timeseries", n_splits=3, n_test=40, purge_delta=5.0, embargo_delta=5.0),
    random_state=0,
).fit(X, y, time=time)

print(model.best_model_id_)
```

By default the train window **expands** — every fold trains on all earlier data. For non-stationary regimes
cap the lookback: `max_train_size` keeps only the last N rows (`"timeseries"`), `max_train_periods` the last
N periods (`"timeseries_period"`); leaving them unset keeps the expanding window.

```python
import numpy as np
from sklearn.datasets import make_classification

from honestml import AutoML, CVConfig

X, y = make_classification(n_samples=240, n_features=8, n_informative=5, random_state=0)
time = np.arange(240)

model = AutoML(
    task="binary",
    models=("baseline", "linear"),
    cv=CVConfig(scheme="timeseries", n_splits=3, n_test=40, max_train_size=80),
    random_state=0,
).fit(X, y, time=time)

print(model.best_model_id_)
```

## Weighting unequal periods

By default the leaderboard score is **pooled** — one metric over all out-of-fold rows, so a month with more
rows weighs more. With `weighting="period"` the score becomes the **macro-average over periods** (each
period counts equally), and the significance band aggregates the bootstrap by period to match. It needs a
time-ordered scheme and, under the default band, at least four periods with a defined metric (fewer fails
fast). A period whose metric is undefined — e.g. a single-class month for ROC AUC — is dropped, and
`run_report_["cv"]` reports the `weighting` mode plus `n_periods_used`.

```python
import numpy as np
from sklearn.datasets import make_classification

from honestml import AutoML, CVConfig

X, y = make_classification(n_samples=360, n_features=8, n_informative=5, random_state=0)
time = np.arange("2021-01-01", "2022-01-01", dtype="datetime64[D]")[:360]

model = AutoML(
    task="binary",
    models=("baseline", "linear"),
    cv=CVConfig(scheme="timeseries_period", period="month", n_splits=8, n_test=1, weighting="period"),
    random_state=0,
).fit(X, y, time=time)

cv = model.run_report_["cv"]
print(model.best_model_id_, cv["weighting"], cv["n_periods_used"])
```

## The equivalence band

All candidates are scored out-of-fold on one shared split, then a seeded paired
bootstrap builds an **equivalence band**: the set of candidates statistically
indistinguishable from the top scorer. The *simplest* band member ships — a more
complex model has to prove it is significantly better, not just luckier — and a
tie is disclosed, never hidden:

- `band_member_ids_` — who was in the band,
- `band_width_` / `band_unstable_` — how wide and how stable it was,
- `winner_by_tiebreak_` — whether the winner needed the simplicity tiebreak.

`significance="off"` disables the band and returns a pure argmax.

```python
from sklearn.datasets import make_classification

from honestml import AutoML

X, y = make_classification(n_samples=200, n_features=8, n_informative=5, random_state=0)

model = AutoML(task="binary", models=("baseline", "linear"), random_state=0).fit(X, y)

print(model.best_model_id_)
print(model.band_member_ids_, model.band_width_, model.winner_by_tiebreak_)
```

## Outer holdout and finalize

`outer_holdout` carves a fraction of the data once, before anything else runs.
Selection, tuning and calibration see only the remaining DEV part; the winner is
scored on the holdout exactly once and the score lands in `holdout_score_`. With
`finalize=True` (the default) the *shipped* model is then refit on all data —
after scoring, so the reported number stays a conservative estimate for the
model you deploy. The carve is scheme-aware (stratified for classification,
tail-of-time for time series).

```python
from sklearn.datasets import make_classification

from honestml import AutoML, CVConfig

X, y = make_classification(n_samples=400, n_features=8, n_informative=5, random_state=0)

model = AutoML(
    task="binary",
    models=("baseline", "linear"),
    cv=CVConfig(outer_holdout=0.25),
    random_state=0,
).fit(X, y)

print(model.best_model_id_, round(model.holdout_score_, 3))
```

## Probability calibration

Classification only, opt-in: `calibrate="sigmoid"`, `"isotonic"` or `"auto"`.
The calibrator is fit on out-of-fold predictions and is applied only when a
cross-fitted check shows it actually improves the proper loss — otherwise the
winner ships uncalibrated. `calibration_` reports what happened.

```python
from sklearn.datasets import make_classification

from honestml import AutoML, CVConfig

X, y = make_classification(n_samples=300, n_features=8, n_informative=5, random_state=0)

model = AutoML(
    task="binary",
    models=("baseline", "linear"),
    cv=CVConfig(calibrate="sigmoid"),
    random_state=0,
).fit(X, y)

print(model.calibration_)
proba = model.predict_proba(X)
print(proba.shape)
```

Everything on this page is recorded in the run report (`run_report_`): the
resolved CV config, the band, the holdout score and the calibration outcome —
see the [quickstart](../quickstart.md) for reports and artifacts.
