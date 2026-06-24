# ADR-0006 — Набор доменных портов и модель `Task`

- **Статус:** Proposed
- **Драйверы:** D1 (расширяемость), D2 (обобщение); FR-1..4/10, NFR-1/2.
- **Воркстрим:** M1.

## Контекст

Ядро должно задать контракты, на которых висят все адаптеры и use-cases, и снять
«бинарность» (D2). Текущий `BoostingModel` Protocol — зачаток, но узкий
(`predict_proba_positive`), а CV/метрика/задача не выделены в порты.

## Рассмотренные варианты

- **Protocol (structural) vs ABC (nominal)** для портов: Protocol даёт duck-typing
  и не навязывает наследование сторонним классам (важно для OSS-плагинов и
  sklearn-совместимости), ABC — явный контракт и `isinstance`. Конвергентно у
  FLAML/auto-sklearn/AutoGluon — тонкий ABC/Protocol с несколькими методами.
- **`Task` как enum** vs **`Task` как объект** (problem_type + метрика/лосс +
  правила сплита/типизации). FLAML/LAMA: объект — он связывает выбор метрики,
  стратификации и авто-типизации.

## Решение

**Порты — `typing.Protocol` (runtime_checkable где нужно)** для точек расширения
(duck-typing, не навязываем наследование), **ABC** — только для шаблонных базовых
классов с чистыми хелперами (напр. `TabularEstimator` с numpy-усреднением
предсказаний по фолдам). **Оркестрация CV** (вызов `CVSplitter`, цикл обучения)
живёт в **use-case**, а не в доменном ABC (R-16, Humble Object).

**`Task` — доменный объект** (Pydantic): `kind: Literal["binary","multiclass",
"regression"]` (+ задел на ranking/quantile), целевая `Metric`, дефолтная
стратегия сплита и правила авто-типизации. `Task` — корень обобщения.

### Доменные порты (M1 фиксирует контракты; реализации — M3+)

| Порт | Сигнатура (суть) | Заметки |
|---|---|---|
| `Task` | объект: kind, metric, split policy, typing policy | корень; снимает бинарность |
| `FeatureSchema` / `ColumnRole` | роли: numeric/categorical/datetime/text/target/group/folds (+метаданные) | типизированный контракт колонок |
| `Dataset` (Protocol) | columns, schema, select(cols), take(idx), **`to_numpy()`**, **`categorical_codes()`** (по schema-owned таблице категорий), `sample_weight` | polars-backend; **без `to_pandas()`** (ADR-0005, R-3); коды — из персистентной таблицы (R-2) |
| `Metric` (Protocol) | `score(y_true, y_pred, sample_weight=None) -> float`; `greater_is_better`; `needs: {proba\|threshold\|class\|value}`; `optimum` | `value` = регрессия (R-5); `sample_weight` (G2); единый Scorer для select/HPO/ensembling/отчёта |
| `CVSplitter` (Protocol) | `split(dataset) -> Iterator[Fold]`, где **`Fold(fit_idx, es_idx, test_idx)`** + пост-условие: индексы непересекаются, **при `group`-роли — непересекаются множества групп** (N-4), для time-series `max(times[fit∪es]) < min(times[test])` (overlap по значениям времени) с purge/embargo | механизм анти-ликеджа, не «по соглашению» (R-6/N-4) |
| `Estimator` (Protocol, **базовый**) | `fit(X, y, X_val, y_val, sample_weight=None)`, `predict(X)`, `feature_names`, `capabilities` | numpy на границе (ADR-0005); `sample_weight` (G2); обобщает `BoostingModel` |
| `ProbabilisticEstimator(Estimator)` | + `predict_proba(X)` | только классификация (LSP, R-4) |
| `SupportsFeatureImportance` | `feature_importances` | role-interface (ISP, R-4) |
| `SupportsShap` | `shap_values(X)` | role-interface (ISP, R-4); опционален |
| `ModelSpec` | `SearchSpace` + capabilities (tasks, **`handles_cat`**, **`handles_missing`**, needs_scaling, gpu, лимиты) | `handles_missing` (G1): доходят ли NaN до модели сырыми или импутируются раньше |
| `Budget` | `time_left()`, `consume()`, `exhausted`; **резерв под `memory_left()`** | время **и память** (G7); наполнение M5 |

**Role-interfaces вместо толстого порта (R-4):** базовый `Estimator` = только
`fit`/`predict` (взаимозаменяем для всех `Task.kind`, включая регрессию).
`predict_proba` — в под-порте `ProbabilisticEstimator` (выбирается по
`Task.kind`); `feature_importances`/`shap_values` — отдельные role-interfaces.
Use-case/leaderboard зависит только от нужного среза — без `hasattr`-ветвления.
`predict_proba_positive` (жёстко бинарный) **упраздняется**: проекцию выбирает
`Metric.needs` (`proba` даёт P(класса) нужной формы по `Task.kind`).

**Capabilities-декларация** у `Estimator`/`ModelSpec` (по образцу auto-sklearn
`get_properties`): какие `Task.kind` поддерживает, обрабатывает ли категории
(`handles_cat`) и пропуски (`handles_missing`), нужен ли скейлинг, лимиты по
строкам/колонкам — реестр (M3) использует для авто-пропуска неподходящих
компонентов.

**Контракт пропусков (NaN, G1):** `handles_missing=True` → NaN доходят до модели
сырыми (CatBoost/LightGBM умеют); иначе предобработка импутирует **до** модели,
причём **одинаково на train и inference** (инвариант, как для кодов категорий).
`FeatureSchema` фиксирует политику пропусков per-role; она сериализуется в артефакт.

**`sample_weight` (G2)** — опциональный сквозной параметр (`Estimator.fit`,
`Metric.score`, не перемешивается в `CVSplitter`); добавить позже без правки
сигнатур портов нельзя, поэтому фиксируется сейчас.

**Эволюция контракта порта (Day-2, D2-1):** порты — это публичный контракт для
сторонних плагинов (NFR-1/9). Breaking для плагина = добавление метода в базовый
Protocol или обязательного поля в capabilities. Правило: новые способности — через
**новый role-interface** (как `SupportsShap`), а не правку базового порта;
entry-points-namespace версионируется; депрекация метода порта — ≥1 минор с
предупреждением (как для фасада).

## Последствия

- (+) Расширяемость без форка: новый компонент = реализация Protocol + регистрация
  (реестр — M3, ADR в нём); ядро не знает адаптеров (NFR-1/2).
- (+) Обобщение на classification/regression/multiclass заложено в `Task`+`Metric`.
- (+) Чистое, синхронно-тестируемое ядро (NFR-3): порты не требуют I/O.
- (+) ISP/LSP соблюдены (R-4): базовый `Estimator` взаимозаменяем для всех задач
  (вкл. регрессию); опциональные способности — отдельные role-interfaces.
- (+) Анти-ликедж — механизм (`Fold` + валидатор непересечения), а не декларация
  (R-6); property-тесты — DoD M3 (базовый es-split) / M4 (purge/embargo).
- (−) Больше мелких портов (role-interfaces), но каждый узкий и явный.
- (−) Риск over-engineering портов — митигируется vertical slice M2 (реальное
  использование валидирует контракты до расширения зоопарка).
