# ADR-0028 — Публичный API оси времени: `fit(time=, label_time=)` + `Dataset.time()`

- **Статус:** Accepted (реализован 2026-06-08, M4b)
- **Дата:** 2026-06-08
- **Драйверы:** DM4-2 (time-series CV без утечки); FR-M4-6/7, NFR-M4-3/5/6. Паттерн ADR-0025
  (публичное объявление колонки) + ADR-0023 (`Dataset.groups()`). Обслуживает ADR-0027.
- **Воркстрим:** M4b. Контракт-change: новый порт-метод + публичный kwarg + новая роль.

## Контекст
`TimeSeriesSplitter` (ADR-0027) упорядочивает фолды по **значению** времени и нуждается в доступе
к оси времени, но `Dataset` (`dataset.py`) такого аксессора не имеет (есть только `groups()`).
`Reader` не присваивает роль времени. Полный de-Prado purge требует label-end-time `t1`
(опционально). Нужен единый индексно-выровненный источник времени — как
`groups()` для групп (ADR-0023) и как публичное объявление group-колонки (ADR-0025).

## Рассмотренные варианты
1. **Переиспользовать `ColumnRole.DATETIME`** как ось CV. Конфликтует с datetime-report-deltas
   (ADR-0018): DATETIME — это datetime-**признаки** (M6); ось CV — иная роль. Несколько datetime-
   колонок ⇒ неоднозначность. Отвергнут.
2. **Pre-sorted строки** (предположить, что вход отсортирован). Хрупко, тихо ломается. Отвергнут.
3. **Выделенная роль `TIME` + публичный `time=` (паттерн ADR-0025)** + опц. `label_time=`. **Выбран.**

## Решение

### 1. Новая роль + порт-аксессоры (core, аддитивно)
- `ColumnRole.TIME` — новый член enum, ось CV (отлична от `DATETIME`-признака, M6).
- `FeatureSchema.time -> str | None` (как `.group`): колонка роли TIME.
- `Dataset.time() -> np.ndarray | None` — значения time-колонки **в порядке строк**, индексно
  выровнены с `design_matrix` (как `groups()`); реализуется в `PolarsDataset`.
- `Dataset.label_time() -> np.ndarray | None` — опц. label-end-time `t1` для полного purge; **name-
  based** (как `sample_weight`/`weight_col`), присутствует только когда задан. Не роль (вторичные
  метаданные).

### 2. `Reader` — kwargs `time`, `label_time` (зеркально `groups`, ADR-0025)
`read(..., groups=None, time=None, label_time=None)`. Константы `_TIME_COL="__time__"`,
`_LABEL_TIME_COL="__label_time__"`.
- `time` крепится `_attach_column` (валидация длины → `SchemaValidationError`); в `_infer_schema`
  ветка `__time__ → ColumnRole.TIME`. `label_time` крепится как `__label_time__` (без роли),
  читается `Dataset.label_time()`.
- **Граничные guard'ы (NFR-M4-3, паттерн ADR-0025 §4):** null/NaN в `time` (или `label_time`, если
  задан) → `SchemaValidationError` (ось времени должна быть полной — null ломает порядок). `time`
  должен быть сортируемым (datetime/числовой), иначе `SchemaValidationError`. Если `label_time` задан
  без `time` → `ConfigError`. (Опц.) `t1 >= t_event` поэлементно — иначе `SchemaValidationError`.
- TIME/`__label_time__` исключены из признаков (не NUMERIC/CATEGORICAL).
- **Граничные guard'ы (фикс R2-completeness):** `time=`/`label_time=` заданы, но резолвнутая схема НЕ
  `timeseries` ⇒ ось времени объявлена, но не используется для разбиения → **WARNING** (look-ahead,
  ADR-0027 §3, расширено на `has_time`); `label_time` особенно «нем» вне timeseries — WARNING делает
  это видимым (а не молчит).

### 3. `facade.fit` — проводка
`fit(X, y, sample_weight=None, groups=None, time=None, label_time=None)` → `read(..., time=time,
label_time=label_time)`; `has_time = ds.schema.time is not None` передаётся в
`build_default_components`; `scheme='timeseries'` требует `has_time` (иначе `ConfigError`, ADR-0027 §3).
`__init__` не трогается (sklearn-инвариант). predict/score — без time (train-only, как groups).

### 4. Координация с datetime-report-deltas (ADR-0018)
`TIME` (ось CV, M4) и `DATETIME` (datetime-признаки, M6) — **разные роли**. datetime-колонка как ось
CV объявляется `time=` → `__time__`/TIME (не признак); datetime-колонка-признак → роль DATETIME
(M6). Конфликта нет; документируется. M4 НЕ реализует datetime-FE. **Уточнение (фикс R1-consistency):**
ADR-0018 (datetime-report-deltas) — статус `Proposed`, НЕ реализован; координация «TIME ≠ DATETIME» —
на уровне будущего намерения (DATETIME-FE = M6). Фактическое текущее поведение DATETIME — исключение
из `features` (`schema.features = numeric+categorical`); **M4b ни на какую DATETIME-FE-ветку в рантайме
НЕ рассчитывает** (реализатору не искать несуществующую логику).

## Последствия
- **Положительные:** time-CV достижим end-to-end из `AutoML.fit`; единый индексно-выровненный
  источник времени (как `groups()`); опц. `t1` открывает полный purge; numpy/pandas/polars
  единообразно; `__init__`/clone не тронуты; время не утекает в признаки и не нужно на inference.
- **Отрицательные/компромиссы:** новый enum-член + порт-методы (аддитивно, но это контракт-change →
  ADR); `label_time` — name-based (асимметрия с role-based `time`, оправдана: вторичные метаданные).
- **Влияние на слои:** `core` (роль/порт-методы, аддитивно), `adapters/{reader,polars_dataset}`,
  `composition/facade`. Сериализуемая `roles` несёт TIME (валидный enum, аддитивно) — `ARTIFACT_VERSION`
  не меняется (NFR-M4-5). import-linter не нарушен.

## Проверки
- `Reader.read(X, y, time=t)` → `schema.time=="__time__"`, `ds.time()` построчно равно `t`, не в
  features; `AutoML.fit(X, y, time=t)` + `scheme='timeseries'` отрабатывает end-to-end (FR-M4-6).
- null/NaN в `time` → `SchemaValidationError`; несортируемый `time` → `SchemaValidationError`;
  `label_time` без `time` → `ConfigError`; `len(time)!=n_rows` → `SchemaValidationError`.
- `label_time=t1` → `ds.label_time()` построчно равно `t1`; полный purge (ADR-0027) использует его
  (FR-M4-7).
- save→load→predict: схема с TIME-ролью грузится, predict без `time` работает (NFR-M4-5, паттерн
  ADR-0025).
- Обратная совместимость: `fit(X, y)` без `time` не меняется.
