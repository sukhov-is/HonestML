# Data input

What honestml accepts at `fit`, what it infers from your data and how it treats
missing values. Every `python` block on this page is **self-contained**: copy any
one of them and it runs as-is — and every block is executed on each CI run, so
the examples cannot rot.

## Accepted inputs

`fit(X, y)` accepts a pandas DataFrame, a polars DataFrame or a 2-D numpy array;
`y` is any row-aligned array-like. A single boundary reader validates the input —
an empty frame, a length mismatch or an unsupported type raises
`SchemaValidationError` with a specific reason, never a bare `ValueError` deep
inside training. DataFrames keep their column names and may freely mix numeric,
string and datetime columns; a numpy array gets synthetic names `f0..f{n-1}` and
is typed per column like any other input.

```python
import numpy as np
import pandas as pd
import polars as pl

from honestml import AutoML

rng = np.random.default_rng(0)
n = 200
df = pd.DataFrame(
    {
        "age": rng.normal(40.0, 10.0, n),
        "plan": rng.choice(["free", "pro", "team"], n),
        "usage": rng.exponential(1.0, n),
    }
)
y = ((df["usage"] > 1.0) | (df["plan"] == "pro")).astype(int).to_numpy()

from_pandas = AutoML(task="binary", models=("baseline", "linear"), random_state=0).fit(df, y)
from_polars = AutoML(task="binary", models=("baseline", "linear"), random_state=0).fit(
    pl.DataFrame(df.to_dict(orient="list")), y
)
from_numpy = AutoML(task="binary", models=("baseline", "linear"), random_state=0).fit(
    df[["age", "usage"]].to_numpy(), y
)

print(from_pandas.best_model_id_, from_polars.best_model_id_, from_numpy.best_model_id_)
```

## What schema inference does

Each column gets a role from its dtype: strings become categorical features,
dates become datetime, floats become numeric. Integer columns are inspected, not
trusted: a low-cardinality integer (≤ 20 distinct values by default) is treated
as categorical, and a nearly-all-unique integer is dropped as an id-like column —
the thresholds are `Task` fields (`numeric_cat_max_unique`, `numeric_id_rate`,
`numeric_id_min_unique`). Every categorical column gets a category table fitted
on the training data and frozen into the schema: known categories map to stable
integer codes, nulls to a reserved code, and a value unseen at fit maps to a
reserved *unknown* code at predict — never to a wrong known category — so the
train↔inference encoding cannot drift. The two boosting models that support it,
CatBoost and LightGBM, consume these columns as *native* categorical features
(CatBoost via ordered target statistics, LightGBM via its `categorical_feature`
splits); `linear` and `baseline` treat the same codes as ordinal integers. The
fitted schema ships inside the model artifact, and an inference batch where over
10% of a column's values were unseen at train triggers a drift warning.

```python
import numpy as np
import pandas as pd

from honestml import AutoML

rng = np.random.default_rng(0)
n = 200
X = pd.DataFrame(
    {
        "income": rng.normal(0.0, 1.0, n),
        "city": rng.choice(["riga", "tallinn", "vilnius"], n),
        "rooms": rng.integers(1, 5, n),  # low-cardinality int -> categorical
    }
)
y = (X["income"] + (X["city"] == "riga") > 0.5).astype(int)

model = AutoML(task="binary", models=("baseline", "linear"), random_state=0).fit(X, y)

print({col: model.schema_.roles[col].value for col in X.columns})
print(model.schema_.categories["city"].categories)

X_new = X.head(5).assign(city="warsaw")  # a category unseen at fit
print(model.predict(X_new).shape)
```

## Declaring the task

`task` accepts a string — `"binary"`, `"multiclass"` or `"regression"` — or a
`Task` object; the string is sugar for `Task(kind=...)`. `Task` adds
`positive_label` (which class counts as positive for binary scoring; by default
label `1` when present, else the greatest label) and the auto-typing thresholds
above. The selection metric defaults per kind — `roc_auc` (binary), `log_loss`
(multiclass), `rmse` (regression) — and `metric=` overrides it by name:
`roc_auc`, `pr_auc`, `accuracy`, `log_loss`, `brier`, `ece`, `rmse`, `mae`. An
incompatible pair (a probability metric on a regression task, `pr_auc` on
multiclass) fails fast with `ConfigError` before any training runs.

```python
import numpy as np
from sklearn.datasets import make_classification

from honestml import AutoML, Task

X, y01 = make_classification(n_samples=200, n_features=8, n_informative=5, random_state=0)
y = np.where(y01 == 1, "churn", "stay")  # string labels work as-is

model = AutoML(
    task=Task(kind="binary", positive_label="churn"),
    metric="log_loss",
    models=("baseline", "linear"),
    random_state=0,
).fit(X, y)

print(model.classes_, model.run_report_["metric"])
print(model.predict_proba(X[:3]).shape)
```

## Row-aligned metadata

`sample_weight=` is passed to `fit` next to `y`: one weight per row, never a
feature. Weights flow through the whole honest pipeline — each fold's training,
the out-of-fold scoring that ranks candidates, calibration and the final refit —
so the leaderboard and the shipped model agree on what a row is worth. `groups=`
(group-aware CV) and `time=`/`label_time=` (time-series CV) are the same kind of
row-aligned metadata and are covered on the
[cross-validation page](cv-selection.md).

```python
import numpy as np
from sklearn.datasets import make_classification

from honestml import AutoML

X, y = make_classification(n_samples=200, n_features=8, n_informative=5, random_state=0)
weights = np.where(y == 1, 2.0, 1.0)  # the positive class counts double

model = AutoML(task="binary", models=("baseline", "linear"), random_state=0).fit(
    X, y, sample_weight=weights
)

print(model.best_model_id_, round(model.score(X, y, sample_weight=weights), 3))
```

## Missing values

honestml does not impute your data behind your back — it never rewrites the frame
you pass to `fit`, and categorical nulls get their own reserved code. Numeric NaN
reaches each model's boundary as-is, and every built-in then handles it on its own
terms: the boosting models (`catboost`, `lightgbm`, `xgboost`) split on NaN
natively, while `linear` and `baseline` carry a per-fold median imputer baked
inside their pipeline — fit fold by fold so it never leaks, and shipped with the
model so inference imputes the same way. No built-in is dropped for carrying NaN;
the skip-with-WARNING gate now fires only for a third-party plugin that declares it
cannot handle missing input. The example below therefore no longer needs the
`boosting` extra — `linear`/`baseline` alone tolerate NaN.

```python
import numpy as np
from sklearn.datasets import make_classification

from honestml import AutoML

X, y = make_classification(n_samples=150, n_features=6, n_informative=4, random_state=0)
rng = np.random.default_rng(0)
X[rng.random(X.shape) < 0.1] = np.nan  # 10% missing values

model = AutoML(task="binary", cv=3, random_state=0).fit(X, y)  # models=None: every installed model

print(sorted(e.model_id for e in model.leaderboard_))  # all installed models rank: linear/baseline impute, boosting splits on NaN
print(model.best_model_id_)
```

Everything inferred here is observable after `fit`: `model.schema_` carries the
roles and the frozen category tables, and the same schema ships inside the saved
artifact — see the [quickstart](../quickstart.md) for saving and serving it.
