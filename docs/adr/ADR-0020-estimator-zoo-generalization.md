# ADR-0020 — Зоопарк эстиматоров: обобщение на multiclass/regression + бустинги через extras

- **Статус:** Accepted (реализован 2026-06-08, M3b)
- **Дата:** 2026-06-07
- **Драйверы:** DM3-2 (ленивость/упаковка), DM3-3 (обобщение за пределы binary); FR-3; NFR-2, NFR-5.
  north-star: D2/D6. Зависит от ADR-0019 (дескриптор/реестр), ADR-0006 (порты).
- **Воркстрим:** M3, дельта к ADR-0013 (M2-адаптеры).

## Контекст

M2-адаптеры (`estimators.py`) — `DummyEstimator`/`LinearEstimator`, оба `tasks=("binary",)`, sklearn-only,
без native cat, без ES. `Estimator`-порт уже generic по `kind`. Нужно: бустинги (CatBoost/LightGBM/XGBoost)
для всех типов задач + линейный и baseline для всех типов, через реестр. Принятые решения: **ES → M4**,
**native cat → отдельная дельта**. Значит обёртки M3 — без ранней остановки, коды-как-числа (`handles_cat=False`).

## Рассмотренные варианты

1. **Один адаптер-класс на библиотеку, ветвящийся по `kind` внутри** — компактно, но `Capabilities`
   статичны на класс, а `probabilistic` зависит от kind (классиф да, регрессия нет) → статич. тег станет
   неточным.
2. **Per-kind адаптеры** (`LinearClassifier`/`LinearRegressor`, `DummyClassifier`/`DummyRegressor`),
   а бустинг — один адаптер на библиотеку, выбирающий `*Classifier`/`*Regressor` по `kind` в `build()`,
   с **дескриптором на (библиотека)**, чьи `Capabilities.tasks` = все три, `probabilistic=True`
   (proba есть для классиф-ветки; на регрессии proba-метрика не выбирается фильтром, т.к. метрика
   регрессии `needs="value"`). **Выбран** (точные статич. теги для линейных/baseline; бустинг — единый
   дескриптор с ленивым выбором ветки).

## Решение

### 1. Линейные и baseline — per-kind, sklearn (лёгкое ядро)
- `LinearClassifier`: `LogisticRegression` (binary + multiclass; multinomial берётся sklearn-дефолтом
  для >2 классов). `Capabilities(tasks=("binary","multiclass"), probabilistic=True)`.
- `LinearRegressor`: `Ridge` (или `LinearRegression`). `Capabilities(tasks=("regression",),
  probabilistic=False)`. `predict` only.
- `BaselineClassifier`: `DummyClassifier(strategy="prior")`. `tasks=("binary","multiclass"),
  probabilistic=True`.
- `BaselineRegressor`: `DummyRegressor(strategy="mean")`. `tasks=("regression",), probabilistic=False`.

Существующие `DummyEstimator`/`LinearEstimator` (binary) **переименовываются/обобщаются** в эти классы
(минимальный диф: тот же код + расширенные `tasks`). Дескрипторы: `baseline`, `linear` (имя стабильно;
`build(task, random_state)` выбирает классиф/регр-вариант по `task.kind`).

### 2. Бустинги — extras, ленивый дескриптор на библиотеку
Имена дескрипторов: `catboost`, `lightgbm`, `xgboost`. `Capabilities(tasks=("binary","multiclass",
"regression"), handles_cat=False, handles_missing=<по библиотеке>, probabilistic=True)`. Модуль-адаптер
**не импортирует** библиотеку на верхнем уровне; `build(task, random_state, **params)`:
- по `task.kind` выбирает `*Classifier` (binary/multiclass) или `*Regressor` (regression);
- `import` библиотеки — внутри `build`; отсутствие пакета → `ImportError` → `MissingDependencyError`
  (через реестр, ADR-0019 §3);
- **seed (фикс R2-C6):** `random_state` пробрасывается в **нативный kwarg каждой библиотеки** (CatBoost
  `random_seed`, LightGBM/XGBoost `random_state`) — детерминизм обученной модели (NFR-4); граница
  недетерминизма (threads/GPU) документируется (north-star NFR-4);
- **без ES**: `X_val`/`y_val` игнорируются (как M2-обёртки на `fit ∪ es`, ADR-0010 §6); фикс-итерации с
  **консервативным дефолтом** (`n_estimators ≈ 200-300`, НЕ 1000 — чтобы без ES не systematically overfit);
  прогон логирует **WARNING** «бустинги без early stopping — сравнение предварительное» и помечает это в
  манифесте (фикс R1-5a: честность `select_best` до прихода ES в M4);
- категориальные **коды как numeric** (`design_matrix` без изменений; `cat_features` не передаётся);
- `predict_proba` — для классиф-ветки; `feature_importances`/`shap_values` — где доступно (role-интерфейсы).
- **Форма `feature_importances` (фикс R2-C8):** всегда **1-D `np.ndarray` длины n_features** — единообразно
  для linear и бустингов; для multiclass-линейного — агрегат `np.abs(coef_).mean(axis=0)` (среднее модулей
  по классам). Per-class матрица не возвращается (стабильная 1-D форма role-интерфейса; иначе breaking потом).
- **`handles_missing` зафиксирован (фикс R1-5b):** бустинги — **`True`** (CatBoost/LightGBM/XGBoost
  обрабатывают NaN **нативно и детерминированно, идентично train/inference** — одна и та же библиотека на
  обоих этапах; согласуется с `numeric_nan="keep"`, импутер не вставляется → нет train/serve skew).
  Линейные/baseline — **`False`** (NaN должны отсутствовать/импутироваться до модели; существующее
  ограничение M2, импутер — M6). Значение `handles_missing` — статично на дескриптор и не зависит от kind.

### 3. `probabilistic` — свойство классиф-ветки + инвариант (фикс R1-F3/4b)
Бустинг — **один дескриптор** на библиотеку (`tasks=(binary,multiclass,regression)`), `probabilistic=True`
означает «классиф-ветка вероятностна». Чтобы статич. тег не «врал» на регрессии:
1. **Совместимость task↔metric** (ADR-0021 §4): regression-таск + proba/class-метрика → `ConfigError` до
   фильтра. Значит фильтр `metric.needs∈proba ⇒ probabilistic` срабатывает только на классиф-тасках, где
   построенная ветка реально `ProbabilisticEstimator`.
2. **Пост-материализационный инвариант** (ADR-0021 §6 / `_run_candidate`): если для proba-метрики выбранный
   классиф-кандидат после `build()` **не** оказался `ProbabilisticEstimator` → явная ошибка/skip+WARNING,
   **не** тихая подмена проекции. Так дуализм «статич. флаг для фильтра ДО / `isinstance` ПОСЛЕ» безопасен.

### 4. Поведение фасада на regression (фикс R1-m2)
`AutoML(BaseEstimator, ClassifierMixin)` сохраняется (sklearn-совместимость не ломаем), но для regression:
`classes_` **не определяется**, `predict_proba` → `NotFittedError`/`ConfigError` (для regression нет
вероятностей), `predict` возвращает значения, `score` использует value-метрику. `fit` ветвится по
`task.kind` при установке атрибутов (`classes_` только для классификации). Полноценный `RegressorMixin`/
раздельный фасад — follow-up (sklearn `estimator_checks` для регрессии — отдельная задача), в M3 — рабочий
regression через единый фасад без классиф-атрибутов.

### 5. pyproject: extras + entry-points встроенных
- `[project.optional-dependencies]`: `catboost`, `lightgbm`, `xgboost` (и агрегат `boosting`).
- Встроенные модели/метрики/сплиттеры объявляются как дескрипторы внутри пакета (ADR-0019 §3); при желании
  дублируются в `[project.entry-points."honestml.models"]` для единообразия, но источник истины для встроенных —
  внутренний список (работает в editable-установке).

**Поправка реализации (2026-06-08, M3b) — доступность extras при дефолтном отборе.** Бустинг-дескрипторы
(`catboost`/`lightgbm`/`xgboost`) — встроенные (всегда в листинге `available_models`, ADR-0019 §7), но в
дефолтный отбор (`models=None`) попадают **только если их extra установлен**: дескриптор несёт
`requires=(<module>,)` (ADR-0019 §1-поправка), `build._select_estimators` фильтрует через
`registry.is_available` (`find_spec`, без импорта). Явный `models=("catboost",)` без extra →
`MissingDependencyError`. Это убирает «шумные» изоляции-падения на лёгком ядре и держит NFR-8 (лёгкое ядро
работает из коробки): дефолт без extras = `baseline`+`linear`; с `pip install honestml[boosting]` зоопарк
участвует автоматически. Решение (find_spec-фильтр) зафиксировано; зависит от изоляции падения
(ADR-0022) как фолбэка на отказ модели в рантайме.

## Последствия

- **Положительные:** зоопарк на все типы задач; лёгкое ядро без тяжёлых зависимостей (NFR-5/-8); точные
  статич. capability-теги; обёртки минимальны, без ES/native-cat (минимальный, честный объём).
- **Отрицательные / компромиссы:** бустинги без ES и без native cat в M3 — недо-используют свою силу
  (документировано как ограничение, закрывается M4/follow-up); per-kind классы множат количество классов
  (плата за точные теги). **Ручные гиперпараметры (`params`) пользователя — не в M3** (фикс R2-C3): нижний
  слой `registry.build(name, **kwargs)` их принимает, но публичного входа в фасаде нет; пользователь на
  консервативных дефолтах до M7 (HPO). Добавление `model_params` в фасад позже — **аддитивный** опц. kwarg
  (не breaking, NFR-9). Явно отложено в M7.
- **Влияние на слои:** все адаптеры — `adapters/`; зависят только от `core` портов; реестр (composition)
  знает дескрипторы. import-linter не нарушен.

## Проверки

`AutoML` обучает binary/multiclass/regression на синтетике через baseline+linear (regression → `rmse`,
ADR-0021 §5); бустинг-дескриптор без установленного extra → дискавери не падает, выбор → `MissingDependencyError`;
с установленным extra бустинг участвует; модуль-дескриптор бустинга не импортирует библиотеку (проверка
`sys.modules` до `build`); `Capabilities` линейных/baseline per-kind корректны (регрессия не `probabilistic`);
proba-метрика не выбирает `LinearRegressor`; **regression+proba-метрика → `ConfigError`** (фикс R1-4b);
**фасад regression: нет `classes_`, `predict_proba`→ошибка, `predict`/`score` работают** (фикс R1-m2);
бустинги без ES → WARNING «предварительное сравнение» в логе/манифесте (фикс R1-5a); `handles_missing`
бустингов=True даёт идентичный train/inference на NaN (фикс R1-5b).
