# Hyperparameter tuning and ensembling

How honestml treats model hyperparameters and when it blends models. Every
`python` block on this page is **self-contained**: copy any one of them and it
runs as-is — and every block is executed on each CI run, so the examples cannot
rot. Both tuning and ensembling are **opt-in**; the default run trains every
candidate with fixed hyperparameters and ships a single model.

## Fixed, conservative defaults

By default no hyperparameter depends on your data. When the model zoo includes a
boosting model, each boosting fit uses **early stopping** (ADR-0080): the `n_es` tail
carved from every fold's training rows is held out as a validation set, the tree count
is raised to a generous ceiling and the library stops once validation stops improving.
The artifact manifest records the real `early_stopping` flag so the choice travels with
the model. A boosting fit *without* a validation tail (inner-CV HPO, or a fold too small
to spare one) falls back to a fixed `n_estimators=300` (`iterations` for CatBoost) and
logs a WARNING that the comparison may favor overfit settings. The linear and baseline
models are plain sklearn defaults. There is no data-size-driven adaptation of any kind —
what you see in the leaderboard is the untuned, reproducible baseline of each library.

```python
from sklearn.datasets import make_classification

from honestml import AutoML

X, y = make_classification(n_samples=240, n_features=8, n_informative=5, random_state=0)

model = AutoML(task="binary", models=("baseline", "lightgbm"), cv=3, random_state=0).fit(X, y)

print(model.best_model_id_, model.leaderboard_)
```

## Tuning is opt-in: HPOConfig

Passing `hpo=HPOConfig(...)` tunes each tunable model type on an **inner CV of
the development data, before the outer honest selection** — the tuned candidate
then competes in the regular leaderboard like any other, so tuning can never
bypass the selection guarantees. `backend="optuna"` is the only backend and
needs the `optuna` extra (`pip install "honestml[optuna]"`); `n_trials` is the
per-model search budget, `inner_cv` the fold count of the tuning objective, and
`models=None` tunes every selected type that declares a search space (baseline
and linear declare none). `keep_baseline=True` keeps the untuned factory in the
leaderboard next to the tuned one; `timeout_s` caps each model's search
wall-clock but makes the result non-deterministic — disclosed in the run
report.

```python
from sklearn.datasets import make_classification

from honestml import AutoML, HPOConfig

X, y = make_classification(n_samples=240, n_features=8, n_informative=5, random_state=0)

model = AutoML(
    task="binary",
    models=("baseline", "lightgbm"),
    cv=3,
    hpo=HPOConfig(n_trials=3, inner_cv=2),
    random_state=0,
).fit(X, y)

tuned = model.run_report_["hpo"]["tuned"]["lightgbm"]
print(tuned["chosen_params"])
print(round(tuned["inner_best_score"], 3), tuned["n_trials_run"])
```

## What the search explores and what is disclosed

Each model type declares its own search space; the tuner samples only those
keys, and a tuned tree count overrides the fixed 300:

- **CatBoost** — `depth` 4–10, `learning_rate` 0.01–0.3 (log),
  `iterations` 50–500 (step 50), `l2_leaf_reg` 1–10 (log), `subsample` 0.6–1.0,
  `one_hot_max_size` 2–64 (the one-hot↔target-statistics boundary for categoricals).
- **LightGBM** — `max_depth` 3–10, `learning_rate` 0.01–0.3 (log),
  `n_estimators` 50–500 (step 50), `reg_lambda` 0–10, `subsample` 0.6–1.0,
  `colsample_bytree` 0.5–1.0, plus the categorical-split regularizers
  `min_data_per_group` 10–300 and `cat_smooth` 1.0–50.0.
- **XGBoost** — `max_depth` 3–10, `learning_rate` 0.01–0.3 (log),
  `n_estimators` 50–500 (step 50), `reg_lambda` 0–10, `subsample` 0.6–1.0,
  `colsample_bytree` 0.5–1.0.

The whole tuning story is disclosed in `run_report_["hpo"]`: the per-model
`chosen_params`, inner score and trial count, the cost estimate
(`n_trials × inner_cv` fits per tuned model), `deterministic` (`False` under a
`timeout_s`), and the honesty flags — the selection OOF is computed
post-tuning, and tuning runs on the full feature width even when feature
selection is also enabled.

```python
from sklearn.datasets import make_classification

from honestml import AutoML, HPOConfig

X, y = make_classification(n_samples=200, n_features=8, n_informative=5, random_state=0)

model = AutoML(
    task="binary",
    models=("lightgbm",),
    cv=3,
    hpo=HPOConfig(n_trials=3, inner_cv=2),
    random_state=0,
).fit(X, y)

hpo = model.run_report_["hpo"]
print(hpo["deterministic"], hpo["cost_estimate_fits"], hpo["selection_oof_is_post_tuning"])
```

## Ensembling with a significance gate

`ensemble=EnsembleConfig()` blends the leaderboard candidates **after** the
honest selection, over their out-of-fold predictions — no extra refitting is
needed to evaluate the recipe. `method="caruana"` (the default) is greedy
ensemble selection with replacement plus seeded bagging (`size` caps the steps,
`n_bags` the bagging subsamples); `method="weighted"` solves for simplex
weights directly; `metric=None` blends on the run metric. The blend ships only
if it is **significantly better** than the best single model — the same
bootstrap significance gate selection uses — otherwise the single winner ships,
and the decision is always disclosed in `run_report_["ensemble"]`: `applied`,
the `member_ids` and `weights`, the OOF improvement `oof_delta`, and a
`gate_reason` such as `significant_improvement`, `equivalent_to_best`,
`worse_than_best` or `degenerate_recipe` (the weight search collapsed onto a
single member).

```python
from sklearn.datasets import make_classification

from honestml import AutoML, EnsembleConfig

X, y = make_classification(n_samples=240, n_features=8, n_informative=5, random_state=0)

model = AutoML(
    task="binary",
    models=("baseline", "linear"),
    ensemble=EnsembleConfig(),
    random_state=0,
).fit(X, y)

ens = model.run_report_["ensemble"]
print(ens["applied"], ens["gate_reason"])
print(ens["member_ids"], ens["weights"])
```

Tuned candidates and the ensemble decision both flow through the same honest
machinery described in
[cross-validation and honest selection](cv-selection.md), and every choice is
recorded in `run_report_`.
