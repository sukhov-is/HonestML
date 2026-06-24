# ADR-0011 — sklearn-совместимый фасад `AutoML`

- **Статус:** Proposed
- **Драйверы:** D2 (единый вход), D6; FR-10; NFR-9.
- **Воркстрим:** M2.

## Контекст

FR-10 требует единый фасад `AutoML(task=...).fit(X, y)` + `predict/predict_proba/
score`, `get_params/set_params`, совместимый с экосистемой sklearn (Pipeline,
`clone`). Прежний вход был процедурной функцией. Фасад — публичная
поверхность под SemVer (NFR-9).

## Рассмотренные варианты

1. **Процедурные функции** — не вписывается в sklearn-экосистему,
   нет `clone`/Pipeline.
2. **Свой интерфейс fit/predict без sklearn-баз** — потеряем `Pipeline`/`clone`/
   привычки пользователей.
3. **Наследник `sklearn.base.BaseEstimator` (+`ClassifierMixin`)** — даёт
   `get_params/set_params/clone`/Pipeline бесплатно при соблюдении контракта.

## Решение

Вариант **3**. `AutoML(BaseEstimator, ClassifierMixin)` в `composition/`:

- **`__init__`** хранит гиперпараметры **как переданы**, без вывода из данных и без
  сайд-эффектов (инвариант sklearn: `get_params` возвращает ровно конструктор-
  аргументы; `clone` работает). Параметры M2: `task`, `metric=None`, `cv=None`,
  `models=None`, `random_state=42`, `time_budget=None` (+опц. инъекция портов для
  тестов).
- **`fit(X, y, sample_weight=None)`**: `Reader.read` → `Dataset`; composition root →
  компоненты; `run_slice` → `SliceResult`; `refit_best`. Выставляет fitted-атрибуты
  с трейлинг-андерскором: `classes_`, `n_features_in_`, `feature_names_in_` (если
  pandas), `leaderboard_`, `best_estimator_`, `schema_`. Возвращает `self`.
- **`predict`/`predict_proba`/`score`**: `check_is_fitted` (иначе `NotFittedError`);
  вход через `Reader.read(X, schema=self.schema_)` (тот же препроцессинг train↔
  inference); проекция под метрику в `score`.
- **Совместимость:** проходит релевантный поднабор `sklearn.utils.estimator_checks`
  (get/set_params round-trip, `clone`, форма predict, `classes_`, работа в
  `Pipeline`). Осознанно НЕ гарантируем полный `check_estimator` (fit тяжёлый/
  бюджетно-недетерминирован) — отступления документируются (NFR-9).

## Последствия

- (+) Единый привычный вход; интеграция с Pipeline/`clone`/grid-search-обвязкой.
- (+) Фасад тонкий: вся логика в use-case/портах; он лишь адаптирует sklearn-контракт
  к `Reader`+`run_slice`.
- (−) Дисциплина sklearn-инварианта `__init__` (никаких вычислений) — компенсируется
  тестом get/set_params/clone.
- Границы: `get_params` для вложенных портов и полный estimator-checks-комплаенс
  уточняются по мере роста (M9); M2 — рабочий бинарный фасад.
