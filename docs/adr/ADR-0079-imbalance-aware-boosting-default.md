# ADR-0079 — Imbalance-aware дефолт бустингов: per-fold scale_pos_weight под rank-метрику

- **Статус:** **REVERTED (2026-06-13)** — реализовано, затем откатано в том же цикле.
- **Дата:** 2026-06-13

> ## Поправка (2026-06-13): решение откатано
>
> Ре-прогон showcase-ноутбука 04-credit-card-fraud (0.17% позитивов) вскрыл, что premise этого ADR
> неверна. Изолирующая диагностика на реальных fraud-данных (test pr_auc, es-хвост = 39 позитивов):
>
> ```
> backend     plain    +bal     +ES   +bal+ES
> catboost   0.8278  0.8187  0.8332   0.7196
> lightgbm   0.0123  0.0335  0.7263   0.0015   ← ES один спасает (0.012→0.726)
> xgboost    0.7899  0.8309  0.7862   0.8293
> ```
>
> **Настоящее лекарство от коллапса — early stopping (#2 / ADR-0080), а не `scale_pos_weight`.** ES
> сам по себе вытягивает все три бустинга (catboost 0.833, lightgbm 0.726, xgboost 0.786).
> `scale_pos_weight` помогает только xgboost (+0.04) и **катастрофически ломает catboost/lightgbm в
> связке с ES** (lightgbm стопается на 1-2 деревьях при spw=577; крошечный es-хвост уже «оптимален»).
> Выравнивание ES-метрики (auc/aucpr вместо logloss) связку НЕ спасает. Поскольку ES по умолчанию
> включён для бустингов, `scale_pos_weight` и ES всегда сосуществуют — значит директива чисто вредна.
>
> **Откат:** удалены `Capabilities.handles_imbalance`, `class_balance`-проброс
> (`build_boosting`/`_BoostingClassifier`/`_estimator_factory`), `_imbalance_kwargs`,
> метрико-зависимый гейт в `_select_estimators` и тесты #1. Дисбаланс лечится ES (ADR-0080).
> Юнит-тесты не поймали это, потому что гоняли синтетику с мягким балансом — ровно showcase-ноутбук
> и должен был поймать (и поймал).

---

_Исходное (отклонённое) решение ниже сохранено для провенанса._
- **Драйверы:** backlog finding #1 (нетюненые lightgbm/xgboost проваливаются на экстремальном дисбалансе ниже логрега); north-star честного зоопарка.
  Зависит от ADR-0020 (зоопарк/`build_boosting`), ADR-0061 §4 (tuned-params last), ADR-0019 (capabilities-фильтр).

## Контекст

На fraud (0.17% позитивов, `metric="pr_auc"`) нетюненые бустинги ранжируют хуже логрега (lightgbm 0.113, xgboost 0.026 против 0.700); Optuna полностью вытягивает их (→0.79), т.е. это **чисто дефолтная проблема**. У всех трёх библиотек есть `scale_pos_weight` (вес класса 1). Формула `n(класс0)/n(класс1)` по отсортированному `np.unique(y)` **библиотечно-консистентна** (lgbm/xgb/catboost все апвейтят класс 1), не требует `Task.positive` и ≈1.0 на балансе (само-нейтрализуется).

**Ключевая оговорка:** `scale_pos_weight` улучшает **ранжирование**, но искажает **калибровку вероятностей** и порог 0.5. Поэтому решение «балансировать ли» — **метрико-зависимо**, а адаптер метрики не знает (numpy-граница). Значит гейт принадлежит composition.

## Решение

### 1. Per-fold scale_pos_weight в адаптере
`_BoostingClassifier._imbalance_kwargs(y)` ([boosting.py](../../src/honestml/adapters/boosting.py)): при `class_balance=True` и бинарной задаче возвращает `{"scale_pos_weight": n0/n1}` (counts по `np.unique(y, return_counts=True)`; пусто если не 2 класса или класс пуст). Считается **пер-фолд** из переданного `y` (честно отражает реальный баланс фолда). `_make` укладывает его в kwargs **между** фикс-дефолтами и tuned `params`, поэтому tuned `scale_pos_weight`/`n_estimators` (HPO) **побеждает** (ADR-0061 §4). INFO-лог раз на backend (дедуп `_balanced_backends`). Регрессионная ветка наследует базовый `_imbalance_kwargs → {}`.

### 2. Метрико-зависимый гейт в composition
`_select_estimators` ([build.py](../../src/honestml/composition/build.py)) вычисляет
`balance_metric = task.kind=="binary" and metric.needs=="proba" and metric.greater_is_better`
— ровно `roc_auc`/`pr_auc` (rank-метрики). Proper scores (`log_loss`/`brier`/`ece` — `needs="proba"`, lower-is-better) и threshold-метрики (`accuracy` — `needs="class"`) остаются **без** балансировки (директива испортила бы калибровку/порог, который они награждают). Флаг `class_balance = balance_metric and caps.handles_imbalance` пробрасывается через `_estimator_factory` → `registry.build` → `build_boosting`.

### 3. Capability `handles_imbalance`
Новый аддитивный флаг `Capabilities.handles_imbalance` ([model_spec.py](../../src/honestml/core/ports/model_spec.py)); `_BOOST_CAPS=True`, линейные/baseline=False. `_estimator_factory` форвардит `class_balance` **только** когда оно True (т.е. только для `handles_imbalance`-моделей), поэтому лёгкие билдеры без `**kwargs` его никогда не получают.

### Границы scope
Только **бинарная классификация** и **нетюненый дефолтный путь**. Multiclass (нет однозначного `scale_pos_weight`) и линейные — вне scope. HPO-путь не получает директиву: тюнинг сам находит баланс (backlog #1 update подтверждает). Пользовательский `sample_weight` ортогонален (строковая ось) и применяется одновременно.

## Последствия

- **Положительные:** нетюненые бустинги на rank-метрике получают HPO-уровень устойчивости на дисбалансе «бесплатно»; на балансе — no-op; tuned/HPO не нарушен; lin/baseline не тронуты.
- **Отрицательные/компромиссы:** на proper-score/threshold-метриках дисбаланс не лечится дефолтом (осознанно — иначе регресс калибровки); multiclass-дисбаланс ждёт follow-up; INFO-лог фиксирует факт.
- **Слои:** `core` (capability), `adapters` (boosting), `composition` (гейт); import-linter не затронут.

## Проверки

`class_balance=True` + бинарный 9:1 → `scale_pos_weight=9.0`; на балансе → `1.0`; по умолчанию (`class_balance=False`) ключ отсутствует; tuned `scale_pos_weight` переопределяет дефолт; регрессия игнорирует директиву; gate в `build_default_components`: `pr_auc`/`roc_auc` → бустинг-фабрика armed, `log_loss`/`brier`/`accuracy` → off; линейный никогда не получает `class_balance`.
