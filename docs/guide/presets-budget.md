# Presets, budget and resume

Пресеты заполняют конфигурацию, бюджет ограничивает необязательную работу,
кеш переиспользует совместимые завершённые результаты. Подготовка, проверки
и refit могут выполняться при повторном fit. Примеры Python самодостаточны
и проверяются в CI.

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
extra) composes with any mode.

Ограничение кооперативное: начавшийся native fit не прерывается по жёсткому
deadline. HPO и выбор кандидатов используют общий бюджет времени; HPO trials
не расходуют счётчик кандидатов `BudgetConfig.n_trials`. При `SearchConfig`
часы учитывают уже измеренную подготовку, а `reserve_fraction` резервирует
время завершения. Final refit не ограничивается этим бюджетом. Если ни один
кандидат не завершён, результатом может быть `BudgetExhaustedError`.
В `run_report_["budget"]` раскрыты `mode`, `exhausted`, `skipped`, `exhausted_by`.

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

`cache="some/dir"` хранит результаты кандидатов по fingerprint разрешённой
конфигурации, данных, метрики, каталога моделей и версий вычислительного стека.
Переиспользуются совместимые завершённые записи; пропущенные, повреждённые
или несовместимые записи пересчитываются. `run_report_["cache"]` раскрывает
`enabled`, `reused` и `computed`.

Candidate/stage cache имеет формат 2. Отдельный HPO checkpoint формата 2
сохраняет историю trials с проверкой контекста и совместимости префикса.
Восстановление подготовки и trials не гарантирует нулевую стоимость следующего
fit или тот же итог при изменившемся фактическом времени. Каталог кеша должен
быть доверенным.

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

## Ограниченный поиск

`AutoML(search=SearchConfig(...))` включает отдельный режим поиска;
`preset="fast"` сам по себе его не включает. Начальные пробы используют
`max_rows=4096`; не более двух финалистов проходят подтверждение с
`confirmation_rows=65536`, `confirmation_folds=2`, `confirmation_iterations=256`.
Лимит строк суммируется по fit/ES/test folds одной семьи. Модельные пробы
сохраняют исходную ширину; `max_features` относится к FS.

Выбор по стоимости требует подтверждённой парной non-inferiority при
абсолютном `model_margin` (по умолчанию 0). Условный прогноз включает
поддержанные FS/HPO и оставшиеся CV/refit, но исключает полное wall-time.
Если компонент неизвестен, выбор по времени не заменяет лидера по качеству.
Одно явно разрешённое семейство пропускает модельные и стоимостные пробы;
одна FS-стратегия без `compare` сохраняет ограничения выбранной процедуры.
Итогом может стать широкий контроль с исходной фабрикой.

[Стоимость обучения](../training-performance.md) описывает измерения,
исключённые расходы и границы приёмки точности прогноза.

All three knobs are provenance-first: the preset block, the budget outcome and
the cache outcome all land in `run_report_` — see the
[quickstart](../quickstart.md) for saving and rendering the report.
