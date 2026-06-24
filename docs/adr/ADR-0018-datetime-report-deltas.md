# ADR-0018 — Datetime-признаки как дельты до отчётной даты (report_dt) на границе Reader

- **Статус:** Accepted (принят и реализуется в составе M6a — см. ADR-0042 §6; design-gate ADR-0018 = GO)
- **Дата:** 2026-06-06 (принят 2026-06-09)
- **Драйверы:** D1 (качество/расширяемость), D6 (закалка); FR-1..6, NFR-1..5; RD-1..8.
- **Воркстрим:** datetime-report-deltas (дельта к M1a). Закрывает G-D1 (тихая потеря datetime).
  Переиспользует паттерн **ADR-0017/CategoryTable** (схема владеет спекой, адаптер материализует).

## Контекст

`pl.Date`/`pl.Datetime` → `ColumnRole.DATETIME`, но `FeatureSchema.features = numeric+categorical`
→ datetime молча выпадает из модели (G-D1). Принятое решение: превращать datetime в
**дельты до отчётной даты признаков (report_dt)** — `report_dt − col` в днях — без календарных/
cyclical компонент и без попарных date-date дельт. report_dt — per-row as-of cutoff (leak-safe).
Inference идёт через `Reader.read(X, schema=...)` (artifact.py:80) → спека FE обязана жить в
сериализуемой `FeatureSchema`, а считаться в адаптере.

## Рассмотренные варианты

### A. Где живёт трансформ
1. **Стадия в use-case (`run_slice`)** — нельзя: домен I/O-free, без polars (import-linter
   `usecases-independent-of-adapters`); датафрейм там уже материализован в `Dataset`.
2. **Отдельный transform-адаптер между Reader и Dataset** — лишний слой; дублирует то, что Reader
   уже делает для категорий (детект + материализация в схему/фрейм).
3. **Расширение `Reader`** — Reader уже единственный choke-point train+inference и уже фитит
   schema-owned спеки (`_fit_categories`). **Выбран.**

### B. Где хранить FE-спеку
1. **Пересчитывать на inference заново** — рассинхрон train↔inference (другой детект/набор), RD-4. Отвергнут.
2. **Параллельные поля вне схемы** — спека не уедет в artifact (inference её не увидит). Отвергнут.
3. **В `FeatureSchema` (сериализуемой)** — как `categories`. **Выбран.**

### C. Опорная дата
1. **Глобальная train-замороженная дата (max train)** — приемлемо как fallback, но это другой
   reference и риск дрейфа/непрозрачности; требуется per-row отчётная дата. Отвергнут как дефолт.
2. **Wall-clock now** — утечка/нестабильность. Отвергнут.
3. **Per-row report_dt из данных; нет колонки → skip+WARNING** (не фабриковать). **Выбран.**

## Решение

### 1. Контракт core (`FeatureSchema`) — аддитивно, как ADR-0017
Новая frozen-спека в `core/schema.py` (чистые данные, без polars):
```python
class DatetimeDeltaSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")
    report_date: str                          # имя report-колонки
    deltas: tuple[tuple[str, str], ...]       # упорядоченные пары (source_col, output_name); единица = дни
```
`FeatureSchema` получает `datetime_spec: DatetimeDeltaSpec | None = None` (дефолт `None`) +
helper `with_datetime_spec(...)`. `extra="ignore"` у `DatetimeDeltaSpec` (forward-compat).
Выходные delta-колонки получают роль **NUMERIC** в `roles` → автоматически входят в
`features`/`to_numpy` (без правок `design_matrix`). Источники datetime и report_dt **сохраняют
роль DATETIME** (исключены из features = потребляются). Новой роли не вводим.

### 2. Резолюция report_dt (адаптер, FR-2/FR-3)
- **(а) Явный override `Task.report_date`** — приоритет. Если задан, но колонка **отсутствует**
  или **не DATETIME-роли** → `SchemaValidationError` (**fail-loud**, R-m2): пользователь выразил
  намерение, молчать нельзя. Не деградировать в WARNING-skip.
- **(б) Авто-детект** (override не задан): среди DATETIME-колонок по узкому case-insensitive набору
  кандидатов (`report_dt`/`report_date`/`feature_dt`) — **ровно одно** совпадение → используется.
  0 совпадений или ≥2 → дельты не строятся + **WARNING** (перечень datetime-колонок без признаков).
- Reference **не фабрикуется** (no wall-clock, no train-max). Набор кандидатов — константа в
  адаптере (расширяема), не в core.

### 3. Вычисление дельты (адаптер, FR-1/FR-6)
Для каждой DATETIME-колонки ≠ report_dt: **обе колонки нормализуются к `pl.Date`**
(`.dt.date()`/`cast(pl.Date)`) **до вычитания**, затем `delta = (report_date − col).total_days()`
→ целые дни в **Float64 NUMERIC** (Float — чтобы хранить null как NaN), имя `f"{col}__days_to_report"`.
- **Формат-независимость (R-M2, критично):** нормализация к `Date` убирает внутридневную часть до
  вычитания, поэтому одна и та же сущность даёт **одинаковую дельту** независимо от того, прочиталась
  колонка как `Date` (parquet) или `Datetime` (csv с временем). Без этого `.days`/`total_days()` по
  `Datetime` усекал бы часы (`12h → 0`) и расходился с `Date`-входом (`1`) — тот же класс read-drift,
  что лечил ADR-0017 для категорий. Под-дневное разрешение сознательно отбрасывается («дни до отчёта»).
- **Коллизия имени:** проверяется **до** `with_columns` (имя выхода уже есть во `frame.columns`) →
  `SchemaValidationError` (а не тихая перезапись polars).
- **Signed:** отрицательное (col после report_date по календарю) сохраняется — легитимно «дни до
  события»; не клипуется (клиппинг — модельное мнение, вне объёма). Целые календарные дни → знак
  симметричен (нет floor/truncate-асимметрии Python `timedelta.days`).
- **null/NaT** в report_date или источнике → null-дельта (NaN), консистентно с `numeric_nan="keep"`.
  ⚠️ NaN потребляют только estimators с `handles_missing=True` (boosting, M3); M2 `LinearEstimator`
  (`handles_missing=False`) NaN не ест — это **существующее ограничение M2** (любой null-числовой при
  `numeric_nan="keep"` уже даёт NaN), не вводится этой дельтой. Политика null→NaN сохранена ради
  консистентности; не overclaim'им NaN-толерантность.
- **tz:** полноценная tz-нормализация — вне объёма (допущение: сопоставимые/наивные); ошибка
  приведения типов → доменная `SchemaValidationError` на границе.
- **Порядок** выходных признаков детерминирован: `_apply_datetime_deltas` (inference) **обязан**
  добавлять NUMERIC-роли строго в порядке `datetime_spec.deltas`; train записывает тот же порядок
  (порядок datetime-колонок входа). Это даёт `schema.features` train==inference (NFR-3) — а не
  косвенно через insertion-order `roles` (R-определённость).

### 4. Поток в Reader.read
- **Train (schema is None):** infer roles → детект report_dt (§2) → проверить отсутствие
  имён-выходов во `frame.columns` (коллизия → `SchemaValidationError`) → посчитать дельты, добавить
  NUMERIC-колонки во фрейм + роли (в порядке источников), записать `datetime_spec` → `_fit_categories`
  (categorical не затронуты). Нет report_dt → WARNING, спека `None`, datetime как прежде (не признак).
- **Inference (schema given):** `_apply_datetime_deltas(schema.datetime_spec)`:
  **сначала собственная валидация источников** — `schema.datetime_spec.report_date` и каждый
  `source` из `deltas` обязаны присутствовать во `frame.columns`, иначе **`SchemaValidationError`**
  (с именем отсутствующей колонки). Затем посчитать те же выходные колонки (в порядке `deltas`).
  Повторного детекта нет (FR-4). Порядок: материализация **до** `_validate_against_schema`.
  - **Почему отдельная валидация (R-B1, blocker):** `_validate_against_schema` проверяет только
    `schema.features` (numeric+categorical); источники дельт и report_date имеют роль **DATETIME** и
    в `features` **не входят** → существующий валидатор их НЕ покрывает. Без явного guard отсутствие
    источника на inference дало бы сырое polars-исключение или **тихий NaN-признак** (train≠inference
    деградация — ровно то, против чего фича). Поэтому `_apply_datetime_deltas` валидирует источники сам.

### 5. Публичный контракт
`Task.report_date: str | None = None` (аддитивное frozen-поле; round-trip в manifest;
sklearn-фасад не трогаем — override задаётся через `Task`). Reader читает `task.report_date`.

## Последствия

- (+) Datetime становится предсказательным признаком (дельта до отчётной даты) — закрыт G-D1;
  тихая потеря заменена на признаки или явный WARNING.
- (+) **Leak-safe относительно фабрикации reference** (per-row report_dt, не wall-clock/train-max).
  ⚠️ Это НЕ гарантирует корректность самого `report_dt`: если отчётная дата в данных производна от
  будущего/таргета — утечка возможна; корректность report_dt — **ответственность данных/пользователя**
  (R-m1), вне контроля Reader.
- (+) Стабильность train↔inference (schema-owned спека, нормализация к Date, фиксированный порядок
  выходов) + обратная совместимость (аддитивно, `ARTIFACT_VERSION` стабилен); слои чисты, как ADR-0017.
- (−) `FeatureSchema`/`Task` усложняются (одна спека + одно поле) + ветка в Reader.
- (−) **Forward-compat (downgrade) не поддержан** (R-forbid): `FeatureSchema`/`Task` имеют
  `extra="forbid"` на верхнем уровне (в отличие от вложенного `CategoryTable` `extra="ignore"` в
  ADR-0017) → **старый** билд, читающий **новый** artifact с `datetime_spec`/`report_date`, упадёт.
  Принято как ограничение Day-2: апгрейд (новый код / старый artifact) работает; даунгрейд — нет.
- (−) Принятые ограничения: tz-нормализация (RD-6), отрицательные дельты не клипуются (RD-5),
  авто-детект по узкому набору имён (RD-1), NaN не потребляется `handles_missing=False`-эстиматорами M2.
- **Инвариант C4 (явно, R-M1):** источники дельт и report_date **сохраняют роль DATETIME** ⇒
  `schema.datetime` непуст (минимум report_date) ⇒ `has_datetime=True` (facade.py:58) ⇒ C4
  lookahead-warning (build.py:133-138) продолжает работать. Закрепляется тестом «после построения
  дельт `schema.datetime` непуст». Конфликта с C4 нет.

## Проверки

- delta-признак строится и попадает в `to_numpy`/`design_matrix` (FR-1); **Date-вход == Datetime-вход
  дают одинаковую дельту** (R-M2, FR-1/NFR-3); авто-детект + override; **override на отсутствующую/
  не-DATETIME колонку → `SchemaValidationError`** (R-m2, FR-2); report_dt ∉ `schema.features` (RD-8, FR-2);
  нет report_dt → skip+WARNING, неоднозначность → WARNING (FR-3); inference по schema даёт идентичные
  признаки + JSON round-trip спеки + `schema.features` train==inference (FR-4/NFR-3); **inference без
  источника/без report_dt → `SchemaValidationError`** (R-B1, FR-4); схема без спеки грузится (FR-5);
  signed (отрицательная сохранена) + null source → NaN + **null report_dt → вся строка NaN** (FR-6);
  **после построения дельт `schema.datetime` непуст** (R-M1, инвариант C4). `lint-imports` 3/3; `core`
  без polars (NFR-1); WARNING/INFO в `caplog` (NFR-2); `Task` round-trip с `report_date`, `clone(AutoML)` (NFR-5).
