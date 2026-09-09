# Hyperparameter tuning and ensembling

How honestml treats model hyperparameters and when it blends models. Every
`python` block on this page is **self-contained**: copy any one of them and it
runs as-is — and every block is executed on each CI run, so the examples cannot
rot. Both tuning and ensembling are **opt-in**; the default run trains every
candidate with fixed hyperparameters and ships a single model.

## Fixed, conservative defaults

Без HPO используются параметры фабрик; часть параметров backend может
разрешаться автоматически по данным. При наличии validation поддерживаемые
boosting-модели используют early stopping. Для IID CV ES выделяется внутри
train; временные схемы используют настроенный хвост. Это относится и к
внутренним HPO folds: inner-test остаётся оценочной частью.

Явные `n_estimators`/`iterations` задают потолок fit. Без явного значения
потолок составляет 1000 при ES и 300 без ES. Фактические iterations раскрываются
в отчёте; refit использует медиану фактических DEV rounds. Поэтому число
использованных деревьев может отличаться от заданного потолка.

```python
from sklearn.datasets import make_classification

from honestml import AutoML

X, y = make_classification(n_samples=240, n_features=8, n_informative=5, random_state=0)

model = AutoML(task="binary", models=("baseline", "lightgbm"), cv=3, random_state=0).fit(X, y)

print(model.best_model_id_, model.leaderboard_)
```

## Tuning is opt-in: HPOConfig

`hpo=HPOConfig(...)` настраивает модели с объявленным пространством параметров
на внутреннем CV внутри DEV. При включённом FS objective использует выбранные
признаки; последующая DEV-оценка остаётся оценкой после адаптивного поиска.
`backend="optuna"` требует extra `optuna`; `n_trials` задаёт предел trials на
семейство, `inner_cv` — число внутренних folds, `HPOConfig.models=None` — все
доступные для настройки семейства. В ограниченном поиске HPO работает с
выбранным семейством. `keep_baseline=True` сохраняет исходную фабрику рядом
с настроенной. `timeout_s` ограничивает поиск кооперативно и может изменить
число завершённых trials. Фактическая работа и решения доступны в отчёте.

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
post-tuning, and, when feature selection is also enabled, tuning runs on the
**post-selection** feature subset (the inner objective sees the pruned width,
not full DEV). Which set was tuned on is disclosed as `tuned_on` (`"fs_subset"`
or `"dev_full"`); `tuned_on_full_feature_space` is pinned `False` — the pre-1.1
full-width mismatch can no longer occur.

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
weights directly; `metric=None` blends on the run metric.

Gate сравнивает ансамбль с выбранной опорной одиночной моделью; она может
отличаться от лидера leaderboard по score. Если её предсказания недоступны,
опорой становится лучший доступный кандидат по метрике. Ансамбль применяется
при значимом улучшении относительно этой опоры. `run_report_["ensemble"]`
содержит `applied`, `member_ids`, `weights`, `oof_delta` и `gate_reason`.
Holdout применённого ансамбля нельзя автоматически вычитать из DEV score
опорной модели и называть полученную разность optimism ансамбля.

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
