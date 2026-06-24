# ADR-0099: Rolling / ограниченный lookback (max_train) для time-series CV

- **Статус:** Accepted (реализован, этап 2 — 2026-06-23)
- **Дата:** 2026-06-23
- **Драйверы:** D-5 (нестационарные режимы — ограниченный lookback); FR-5, NFR-2/5.
  Расширяет ADR-0027 (там expanding помечен дефолтом, rolling — отложенный future-field).

## Контекст
`TimeSeriesSplitter` — **expanding-only**: train всегда `order[: ts - purge]` от позиции 0
(`splitters.py:414`). Для нестационарных биржевых режимов нужен train на **последних N
периодах/строках** (rolling), чтобы не учить на устаревшем режиме. ADR-0027 явно отложил это.

## Рассмотренные варианты
1. **Только expanding (как сейчас).** Не покрывает нестационарность; пользователь взял rolling
   (направление 3). Недостаточно.
2. **Rolling по умолчанию.** Меняет дефолтную семантику `timeseries` (нарушает NFR-5 и гарантию
   ADR-0027). Отвергнут.
3. **Opt-in `max_train_*`, expanding — дефолт.** Аддитивно; дефолт сохраняет ADR-0027. **Выбран.**

## Решение

### 1. `CVConfig` (core, аддитивно)
- `max_train_periods: int | None = None` — нижняя граница train в **периодах** (для
  `timeseries_period`). `max_train_size: int | None = None` — нижняя граница train в **строках**
  (для `timeseries`). Оба `gt=0`. `None` → expanding (текущее поведение).
- Гейт в `build`: `max_train_periods` требует `timeseries_period`; `max_train_size` требует
  `timeseries`; иначе `ConfigError`.
- **Проводка в конструкторы (G14):** `TimeSeriesSplitter.__init__` получает новый параметр
  `max_train_size`, `PeriodTimeSeriesSplitter.__init__` — `max_train_periods` (сейчас конструктор
  `TimeSeriesSplitter` их не принимает — это новое звено `build`→конструктор, явный пункт DoD).

### 2. Сплиттеры (adapters)
- **`TimeSeriesSplitter`:** нижняя граница train становится `max(0, ts - purge - max_train_size)`
  вместо жёсткого 0 (правка единственного выражения `splitters.py:414`); при `None` — 0 (expanding).
- **`PeriodTimeSeriesSplitter`:** train ограничивается периодами
  `[test_start_period - max_train_periods, test_start_period)` (при `None` — от первого периода); `purge`
  затем срезает верх окна до `test_start_period - purge`, поэтому при `purge>0` эффективный train =
  `max_train_periods − purge` периодов (в row-схеме purge и `max_train_size` вычитаются совместно:
  `ts - purge - max_train_size`).
- **es-хвост и feasibility (R-5, F8):** для `TimeSeriesSplitter` row-формула достаточности
  обобщается с учётом `max_train_size`. Для `PeriodTimeSeriesSplitter` row-формула
  `first_test - purge < n_es + 1` (`splitters.py:397`) **неприменима** — периоды неравны по числу
  строк, число строк train известно только ПОСЛЕ материализации. Поэтому feasibility считается **по
  числу строк train после gather**: `train_rows.size ≥ n_es + min_fit_rows` (`min_fit_rows ≥ 2`),
  иначе `SchemaValidationError` с понятным текстом. Внимание на ловушку `train[:-0]` — при клампе
  `n_es→0` срез `[:-0]` numpy даёт ПУСТОЙ fit (не «весь train»); явный guard + тест. `_carve_es`
  уже отвергает `train.size < 2` — переиспользуется, но порог fit поднимается до `min_fit_rows`.
  **Реакция per-fold (G5):** недостаточный train у любого фолда — **fail-run** (`SchemaValidationError`),
  а не тихий пропуск фолда (консистентно с fail-fast; молчаливый skip изменил бы число фолдов
  незаметно). Сводный инвариант `periods→folds→valid-blocks` — в `07-design-review.md`.
  (Номера строк здесь ориентировочные — `split()` ≈ 386-419, гейт ≈ 397-401, G14.)
- `validate_fold` (value-based overlap) не меняется — rolling лишь уменьшает train, порядок и
  отсутствие пересечения сохраняются.

## Последствия
- **Положительные:** нестационарные/биржевые режимы выразимы; expanding — дефолт (ADR-0027 цел);
  правка локальна (нижняя граница train + обобщённый feasibility-гейт).
- **Отрицательные/компромиссы:** слишком малое окно может оставить мало fit-строк — закрывается
  feasibility-гейтом (R-5); ещё два зноба (аддитивно, дефолт-None).
- **Влияние на слои:** `core` (поля+гейт), `adapters` (нижняя граница train в обоих сплиттерах).
  Слои/контракты не нарушены.

## Проверки
- `max_train_periods=12` → train фолда покрывает ≤12 последних периодов перед тестом;
  `max_train_size=N` → ≤N строк (FR-5).
- При урезанном train es-хвост непуст и `fit` непуст, иначе понятная ошибка (R-5).
- Дефолт (`None`): фолды байт-в-байт как expanding до фичи (NFR-5).
- value-based overlap сохраняется при rolling (NFR-2).
