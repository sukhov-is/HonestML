# ADR-0040 — Архитектура FE-пайплайна: водораздел размещения по target-зависимости + публичный конфиг

- **Статус:** Accepted (реализован в M6a, 2026-06-09; см. `08-plan.md`/`09-review.md`)
- **Дата:** 2026-06-09
- **Драйверы:** DM-1 (анти-ликедж), DM-3 (слои/Humble Object), DM-5 (подключаемость); FR-FE-1/2/3/4/5,
  NFR-FE-2. Наследует Reader-границу (ADR-0005), OOF run_slice (ADR-0010), cross-fit-прецедент (ADR-0030/0031),
  datetime (ADR-0018).
- **Воркстрим:** M6a (FE-дельта).

## Контекст
M6a добавляет FE-трансформеры (datetime, target-encoding, frequency, intersections) с жёстким анти-ликеджем.
Главное решение — **где фитится каждый трансформер**, чтобы не протечь и не нарушить слои/`train==inference`.
Research установил водораздел по **зависимости от таргета** (`00-research.md` §4).

## Рассмотренные варианты (размещение)
1. **Весь FE на границе Reader** (schema-owned, включая TE). TE — target-зависим: глобальный фит на full-train
   **протекает** в CV-test → оптимистичный OOF (R-FE-LEAK). **Отвергнут** для TE.
2. **Весь FE per-fold в `run_slice`.** Для target-**независимых** (datetime/freq/intersections) фолды не нужны
   (они leak-safe на границе) → лишняя сложность, ломает простоту `train==inference`, дублирует материализацию.
   Over-engineering. **Отвергнут.**
3. **Водораздел по target-зависимости.** target-**независимый** (datetime/freq/intersections) — на **границе
   Reader** (schema-сериализуемые спеки, в `design_matrix`); target-**зависимый** (TE) — **кросс-фит OOF в
   `run_slice`** для оценки (ADR-0041) + full-train спека в схеме для refit/inference. **Выбран** — минимально,
   корректно, переиспользует прецедент `crossfit_calibrate`.

## Рассмотренные варианты (форма расширения)
A. **Новый рантайм-плагин-порт `FeatureTransformer`** (fit/transform, реестр как у Estimator). Гибко, но для
   **фиксированного** каталога из 4 трансформеров — преждевременная абстракция (YAGNI). **Отложен** (seam для
   third-party FE — future, симметрично будущему FeatureSelector).
B. **Фиксированный конфигурируемый каталог** (M6a): спеки-данные в `core` + чистый cross-fit в `application` +
   fit/apply в `Reader`. Минимальный диф, расширяемо позже без breaking (спеки аддитивны). **Выбран.**

## Решение

### 1. Водораздел размещения (ядро)
| Трансформер | target-зависим | Где фитится (train) | Где применяется (inference) |
|---|---|---|---|
| datetime → `days_to_report` (ADR-0018) | нет | `Reader` (граница), `DatetimeDeltaSpec` в схеме | `Reader._apply_datetime_deltas` |
| frequency-encoding | нет | `Reader` (граница), `FrequencyEncodingSpec` | `Reader._apply_frequency` |
| intersections (`A__B`) | нет | `Reader` (граница), `IntersectionSpec` (+CategoryTable пересечения) | `Reader._apply_intersections` |
| **target-encoding** | **да** | **OOF cross-fit в `run_slice`** (оценка) **+** full-train `TargetEncodingSpec` (refit/inference) | `Reader._apply_target_encoding` (full-train спека) |

Target-независимые трансформеры дают NUMERIC/CATEGORICAL-колонки **на границе** → автоматически входят в
`schema.features`/`design_matrix` (без правок `design_matrix`), симметрично train/inference. TE — единственный,
кому нужен per-fold OOF (ADR-0041).

### 2. FE вычисляется один раз за прогон, разделяется кандидатами (NFR-FE-5)
FE преобразует **признаки** (X), независимо от эстиматора. Поэтому:
- target-независимый FE — часть `design_matrix` (строится один раз, `x_full`), общий для всех кандидатов;
- OOF-TE-аугментация (ADR-0041) считается **один раз** перед циклом кандидатов и разделяется ими — **только для
  оценки** (refit/inference используют full-train-TE на границе, ADR-0041 §3, не OOF-аугментацию).
Не фитить FE заново на каждого кандидата (R-FE-PERF).

**Гейт `oof_fold_index` (фикс ревью R1/A6):** `run_slice` строит `oof_fold_index` **безусловно** при
`fe.target_encoding` (а не только при `capture_proba`/refinement) — условие
`if capture_proba or selection == "refinement" or fe.target_encoding:`. Это предусловие OOF-TE-кросс-фита.

**Взаимодействие с кэшем M5 (фикс ревью R1/A2):** `FEConfig` входит в **run-fingerprint** (ADR-0042 §4) → весь
кэш-каталог скоупится фингерпринтом: другой FE → другой фингерпринт → другой каталог → **stale-hit невозможен**
(R-FE-CACHE). Per-candidate cache кэширует исход кандидата, посчитанный поверх **уже аугментированного** (тем же
FE) `x_full`; при reuse OOF-TE не пересчитывается (она часть входа, общего для прогона под этим фингерпринтом).
Изменение FE → новый каталог → кандидаты считаются заново (с новой OOF-TE).

### 3. Слои (Humble Object)
- `core/schema.py`: frozen-спеки `TargetEncodingSpec`/`FrequencyEncodingSpec`/`IntersectionSpec`
  (+ `DatetimeDeltaSpec` из ADR-0018) — **чистые данные**, без polars.
- `adapters/reader.py` (+`polars_dataset.py`): фит спек на full-train, материализация на границе (polars).
- `application`: чистый `crossfit_encode` (numpy, зеркало `crossfit_calibrate`) для OOF-TE — за контрактом, без
  polars; `run_slice` не именует адаптер. import-linter 3/3.

### 4. Публичный конфиг (sklearn-инвариант)
- **`FEConfig`** (`core/config.py`, frozen, рядом с `CVConfig`/`BudgetConfig`): `target_encoding: bool=False`,
  `te_smoothing: float=10.0`, `frequency_encoding: bool=False`, `intersections: bool=False`,
  `max_pairs: int=50`. Несётся в `RunConfig.fe: FEConfig` (дефолт all-off).
- Фасад: `AutoML(..., feature_engineering: FEConfig | None = None)` — verbatim в `__init__`, резолв в `fit`;
  невалидный → `ConfigError` (guard, как `run_mode`). Дефолт `None` → off → M5 неизменен (FR-FE-1).
- **datetime** управляется **отдельно** через `Task.report_date` (ADR-0018): per-row граничный трансформер,
  резолвится в `Reader` (авто-детект/override), не часть `FEConfig`-каталога. Маппинг документируется.

## Последствия
- (+) Корректный анти-ликедж (TE — OOF; остальные leak-safe на границе); минимальный диф; переиспользование
  `design_matrix`/`crossfit_calibrate`/`CategoryTable`/Reader-границы; дефолт сохраняет M5.
- (−/компромисс) Плагин-порт для third-party FE отложен (фиксированный каталог) — расширение позже, аддитивно.
- (−/компромисс) Два конфиг-поверхности (datetime через `Task.report_date`, остальное через `FEConfig`) —
  оправдано разной природой (per-row граница vs конфигурируемый каталог); маппинг в docs.
- **Влияние на слои:** спеки — `core`; фит/материализация — adapters; OOF-cross-fit — application; конфиг/резолв
  — composition. `ARTIFACT_VERSION` — см. ADR-0042 (аддитивно, без bump).

## Проверки
- FE=off (дефолт) → leaderboard/artifact идентичны M5; `clone`/`Pipeline` сохраняют `feature_engineering`.
- Включённый target-независимый FE → колонки в `design_matrix` (n_features растёт), `predict` работает.
- target-encoding идёт через OOF (ADR-0041), не через граничный глобальный фит (тест размещения).
- FE считается один раз (счётчик материализаций) — не на каждого кандидата.
- `lint-imports` 3/3 KEPT; `core` без polars; невалидный FE-конфиг → `ConfigError`.
