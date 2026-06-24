# ADR-0042 — FE-спеки в FeatureSchema: сериализация, inference, совместимость; принятие ADR-0018 для datetime

- **Статус:** Accepted (реализован в M6a, 2026-06-09; inference collision-guard добавлен по ревью F3, `09-review.md`)
- **Дата:** 2026-06-09
- **Драйверы:** DM-2 (train==inference), DM-4 (аддитивность/кэш); FR-FE-2/4/5/6/7, NFR-FE-3/4. Наследует
  schema-owned спеки (ADR-0005/0017), artifact-контракт (ADR-0012), run-fingerprint (ADR-0035), datetime
  (ADR-0018).
- **Воркстрим:** M6a (FE-дельта).

## Контекст
FE даёт ценность только если **train==inference**: то, что обучено на train, должно примениться на inference
**без пересчёта** и **детерминированно**. Прецедент — `CategoryTable` (схема владеет, artifact возит, Reader
применяет). Нужно распространить это на FE-спеки и решить совместимость artifact/кэша + интегрировать datetime
(ADR-0018, designed-not-implemented).

## Рассмотренные варианты (где живёт FE-спека)
1. **Пересчитывать FE на inference заново.** Рассинхрон train↔inference (другой набор/детект) — нечестно (RD-4
   ADR-0018). **Отвергнут.**
2. **Отдельные файлы карт рядом с artifact**. Не уедут консистентно, вне единого источника истины
   препроцессинга. **Отвергнут.**
3. **Аддитивные frozen-поля в `FeatureSchema`** (как `categories`/`datetime_spec`). **Выбран** — единый
   сериализуемый источник, Reader применяет, train==inference гарантирован.

## Решение

### 1. Спеки — frozen-данные в `core/schema.py`, аддитивно на `FeatureSchema`
```python
class TargetEncodingSpec(BaseModel):   # frozen, extra="ignore"
    encodings: dict[str, dict[str, float]]   # col -> {code_str -> smoothed_mean}
    global_mean: float
    smoothing: float                         # k из FEConfig.te_smoothing на фите; нужен для воспроизв./аудита
class FrequencyEncodingSpec(BaseModel): # frozen, extra="ignore"
    frequencies: dict[str, dict[str, float]] # col -> {code_str -> freq}
class IntersectionSpec(BaseModel):      # frozen, extra="ignore"
    pairs: tuple[tuple[str, str], ...]       # упорядоченные (a, b); выход a__b — отдельная CATEGORICAL-колонка
```
`FeatureSchema` получает `target_encoding: TargetEncodingSpec | None = None`,
`frequency_encoding: FrequencyEncodingSpec | None = None`, `intersections: IntersectionSpec | None = None`
(+ `datetime_spec` из ADR-0018) и copy-update helper'ы `with_target_encoding`/`with_frequency_encoding`/
`with_intersections`/`with_datetime_spec` (как `with_categories`). `FEConfig` — публичный тип, экспортируется в
`honestml.__all__` рядом с `CVConfig`/`BudgetConfig` (operational §1). Все спеки — дефолт `None`/пусто → **старый
artifact грузится**.
Выходы FE получают роль **NUMERIC** (TE/freq) или **CATEGORICAL** (пересечения) → автоматически входят в
`schema.features` (без правок `design_matrix`).

**Ключи карт TE/freq (фикс ревью R1/A10):** ключ — `code_str = str(<int-код CategoryTable>)` (коды уже
материализованы `CategoryTable.encode`). `null_code`/`unknown_code`/непокрытые коды **НЕ** хранятся в карте →
на lookup'е `dict.get(code_str, global_mean)` (TE) / `0.0` (freq) — детерминированный fallback. JSON и так
приводит int-ключи к строкам → хранение `str`-ключей **устраняет** дрейф коэрсии (train==inference). Lookup —
`dict.get` (порядок ключей **не важен**); run-fingerprint канонизирует порядок (`sort_keys`, ADR-0035) — порядок
итерации карт нигде не влияет на результат (фикс minor про JSON-порядок). Round-trip-тест обязателен.

**Кодирование пересечений (фикс ревью R1/A5 + R2-completeness/C1):** правило выбора пар **закреплено и
детерминировано** — `pairs = list(combinations(sorted(schema.categorical), 2))[:max_pairs]` (лексикографически
по имени колонки, **без seed**), усечение до `max_pairs` логирует **WARNING** (FR-FE-5); `< 2` категориальных →
пустой `IntersectionSpec`, **без** WARNING (отлично от усечения). Для каждой пары `_fit_intersections` строит
колонку `a__b` (concat кодов/значений), **проверяет коллизию имени** (`a__b` уже во `frame.columns` →
`SchemaValidationError`, не тихая перезапись polars), затем кодирует её как обычную категорию — **отдельная
`CategoryTable`** в `FeatureSchema.categories[f"{a}__{b}"]` (резервы null/unknown наследуются).
`schema.categories` растёт на число пар; кардинальность ограничена `max_pairs` (один гейт, до генерации).

### 2. Reader: фит на train, применение на inference (граница)
- **Train (schema=None):** после `_infer_schema` → построить датетайм-дельты (ADR-0018) → построить
  intersection-категории (+`CategoryTable`) → `_fit_categories` → `_fit_frequency` → `_fit_target_encoding`
  (full-train, ADR-0041 §3). Порядок выходов **детерминирован** (по порядку спек/колонок), записывается в схему.
- **Inference (schema given):** `_apply_datetime_deltas` → `_apply_intersections` → `_apply_frequency` →
  `_apply_target_encoding` — каждый **сам валидирует наличие источников** (роли источников вне `features`)
  **до** `_validate_against_schema`; отсутствие источника/коллизия имени выхода → `SchemaValidationError`
  (fail-loud, не тихий NaN) — паттерн ADR-0018 §4.
- **Порядок фиксирован, без цепочки (фикс ревью R1/nit + R2):** трансформеры **независимы** — каждый берёт
  **исходные** колонки (intersections/freq/TE — над source-categorical, **не** над выходами datetime/друг друга);
  цепочки (FE поверх FE) в M6a нет. Порядок одинаков train/inference и обеспечивает детерминизм.
  **Приоритет роли datetime (фикс ревью R2):** `pl.Date/pl.Datetime → DATETIME` — **окончательно** (как в
  `_infer_schema`); DATETIME-колонка **исключена** из source-categorical → не участвует ни в TE, ни в
  intersections. datetime-выход (`days_to_report`) — NUMERIC, **не** категория → в пересечения тоже не входит
  (нет цепочки). Так коллизия datetime↔категория исключена по построению.
- **Симметрия skip vs fail-loud (фикс ревью R1-leakage/A9):** «skip+WARNING» (datetime без report_dt) и
  «fail-loud» **не противоречат** — это разные случаи. Если спека **отсутствует** (`datetime_spec is None`:
  artifact обучен без дельт) → inference дельты **не строит** (симметрично train) — ошибки нет. Fail-loud
  срабатывает **только** когда спека **есть**, но её источник **отсутствует** на inference (нарушен контракт
  train==inference). Аналогично TE/freq/intersections: `None`-спека → no-op симметрично; спека-с-отсутствующим-
  источником → `SchemaValidationError`. Граничные кейсы — в матрице.

### 3. Совместимость artifact (`ARTIFACT_VERSION` не бампается)
Поля схемы **аддитивны** (дефолт None) → **новый код читает старый artifact** (поля отсутствуют → off).
`ARTIFACT_VERSION` остаётся **1** (контракт расширен вперёд-совместимо для нового кода). **Downgrade не
поддержан** (старый код, новый artifact с FE-полями → падение из-за `FeatureSchema extra="forbid"`) —
**принятый предел Day-2** (тождественно ADR-0018 §5, R-FE-ART-COMPAT). Bump версии не требуется (старые artifact
читаются; ломается только обратное чтение, что и так не гарантировалось).

### 4. Кэш / run-fingerprint (NFR-FE-4)
`RunConfig.fe` (ADR-0040) и `Task.report_date` (ADR-0018) входят в run-fingerprint **автоматически**
(`compute_run_fingerprint` хеширует `RunConfig` + `Task`, ADR-0035) → изменение FE-конфига даёт **другой**
кэш-ключ (нет stale-hit, R-FE-CACHE). Дополнительно `data_signature` уже покрывает данные. Явная проверка в
матрице.

### 5. Наблюдаемость (FR-FE-7) и порядок признаков
FE-конфиг — в `report["config"]` через config-дамп (вход); применённые трансформеры/добавленные признаки видны
через `schema.features` (исход). `RUN_MANIFEST_VERSION` **не** бампается.

**Закреплённый порядок `schema.features` (фикс ревью R2-completeness/C2, NFR-FE-3):** `FeatureSchema.features`
строится **явной конкатенацией блоков** (не итерацией `roles`-dict), порядок один train==inference при **любом**
подмножестве включённых FE:
```
features = original_numeric
         ⊕ datetime_deltas (в порядке источников DATETIME)
         ⊕ frequency_outputs (в порядке source-categorical)
         ⊕ target_encoding_outputs (в порядке source-categorical)
         ⊕ original_categorical
         ⊕ intersection_outputs (в порядке пар IntersectionSpec)
```
Каждый блок может быть пуст (FE выключен) — порядок остальных не меняется. `FeatureSchema.features` обновляется
учитывать FE-выходные роли в этом фиксированном порядке (правка `core/schema.py`: `features` больше не
`numeric + categorical`, а блочная конкатенация выше; legacy без FE даёт прежний `numeric + categorical`). Тест:
два прогона / одно подмножество FE → идентичная позиция каждого признака.

### 6. datetime — принять ADR-0018 (binding, фикс ревью R1/A4)
datetime-FE в M6a = **ADR-0018 как есть** (граница Reader, `days_to_report`, per-row leak-safe, авто-детект/
`Task.report_date`, Date-нормализация R-M2). Календарные/cyclical-компоненты — **вне объёма** (defer,
OQ-DT-CAL).
**Binding:** ADR-0018 — **авторитетный источник** datetime-логики; M6a её **реализует** (`DatetimeDeltaSpec`,
`Task.report_date`, `Reader._apply_datetime_deltas` перед `_fit_categories`). **Статус ADR-0018 переведён в
`Accepted`** (design-gate ADR-0018 уже = GO; M6a — его реализация) — снято с «Proposed», чтобы дельта
трассировалась от принятого решения, а не от обсуждаемого. Тест-план datetime — в ADR-0018; матрица M6a
ссылается на него (FR-FE-2), без дублирования критериев. Конфликта с FE-каталогом нет (datetime — отдельная
per-row ось, ADR-0040 §4); intersections — **только** категория×категория (§1), NUMERIC-дельта в пересечения
**не** входит (нет цепочки FE, §2).

## Последствия
- (+) train==inference для всего FE; единый сериализуемый источник; старый artifact грузится; FE в fingerprint
  (нет stale-hit); переиспользование `CategoryTable`/Reader-паттерна; `design_matrix` не меняется.
- (−/компромисс) Downgrade artifact не поддержан (Day-2, как ADR-0018); `FeatureSchema` обрастает полями
  (4 спеки) — оправдано единым источником истины.
- **Влияние на слои:** спеки — `core`; фит/apply — `adapters/reader`; конфиг/fingerprint — `composition`;
  `ARTIFACT_VERSION`/`RUN_MANIFEST_VERSION` не тронуты.

## Проверки
- `FeatureSchema` с FE-спеками — JSON round-trip; старый artifact (без полей) грузится; `predict` фасад==artifact.
- inference без источника/report_dt → `SchemaValidationError` (не тихий NaN); коллизия имени выхода →
  `SchemaValidationError`.
- `schema.features` train==inference (порядок+кардинальность); другой FE-конфиг → другой `run_fingerprint`.
- `ARTIFACT_VERSION`/`RUN_MANIFEST_VERSION` не изменены; FE-конфиг в `report["config"]`.
- datetime: delta строится и в `design_matrix`; Date==Datetime-вход; override-miss → `SchemaValidationError`;
  нет report_dt → skip+WARNING; `schema.datetime` непуст после построения (инвариант C4, ADR-0018).
