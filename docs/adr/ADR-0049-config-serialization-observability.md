# ADR-0049 — Конфиг compare (аддитивно к M6b), сериализация победителя, наблюдаемость, версии

- **Статус:** Accepted (реализован в M6c, 2026-06-10; конфиг-дельта + `run_report`-блок, версии не бампнуты)
- **Дата:** 2026-06-10
- **Драйверы:** DM-C4 (аддитивность/совместимость/наблюдаемость), DM-C6 (детерминизм). FR-FSC-1/5/6/7,
  NFR-FSC-4/5. Наследует `FeatureSelectionConfig`/run-fingerprint (ADR-0035), сериализацию subset (ADR-0045),
  run-report (ADR-0033/0037).
- **Воркстрим:** M6c.

## Контекст
M6b только что отгрузил `FeatureSelectionConfig.strategy: FSStrategy` как **публичный контракт**. M6c добавляет
сравнение нескольких стратегий, новые имена стратегий и параметры — нужно **аддитивно**, не ломая M6b-конфиг, с
backward-compat artifact и без бампа версий.

## Решение

### 1. Конфиг — аддитивная дельта (`core/config.py`)
```python
FSStrategy = Literal["importance", "random_probe", "null_importance", "sequential"]   # расширен

class FeatureSelectionConfig(BaseModel, frozen, extra="forbid"):
    strategy: FSStrategy = "importance"                      # M6b: single-path (compare=None)
    compare: tuple[FSStrategy, ...] | None = None            # M6c: список для сравнения (≥1, уникальные)
    selection_holdout: float = Field(0.25, gt=0.0, lt=1.0)   # доля DEV под арбитраж (ADR-0048)
    # cutoff (M6b, для ranker-стратегий): cutoff/top_k/top_frac/min_features
    n_probes: int = 3                                        # random_probe (M6b)
    n_runs: int = Field(30, ge=1)                            # null_importance перестановки (ADR-0047 §1)
    null_percentile: float = Field(95.0, gt=0.0, lt=100.0)   # null_importance порог
    seq_min_features: int = Field(1, ge=1); seq_patience: int = Field(2, ge=1)   # sequential плато
    random_state: int | None = None
```
- **Семантика:** `compare=None` → single-path по `strategy` (тождественно M6b; `strategy` может быть **любым**
  именем `FSStrategy`, включая M6c-новые). `compare=(...)` → compare-режим (ADR-0046/0048), `strategy` игнорируется
  в выборе (но валиден). **Один элемент** в `compare` ⇒ арбитраж не запускается (как N=1) — для **обоих** видов
  портов, включая lone `compare=("sequential",)` (отбор на full-DEV, без carve; ADR-0046 §3, уточнено R1-A11).
- **Валидация** (`model_validator`, → `ConfigError` через guard): `compare` непуст и **без дубликатов**; все имена
  ∈ `FSStrategy`; существующий `_check_cutoff` (top_k требует top_k). **cutoff применяется только к
  ranker-стратегиям** (`importance`/`random_probe`/`null_importance`); subset-selector (`sequential`)
  **игнорирует** cutoff (возвращает subset сам по `seq_min_features`/плато) — **тихо, не ошибка** (документировано
  в docstring конфига; per-strategy-override cutoff отложен; уточнено R1-A9). Стратегии оцениваются **независимо**,
  union/intersection их subset'ов **не** применяется — арбитр выбирает один (ADR-0048).
- **Per-strategy seed (детерминизм, DM-C6, уточнено R1-A10):** `random_state` пред-заполняется фасадом из
  `self.random_state` (как M6b). Seed стратегии — **стабильный** (не Python `hash()`, он рандомизирован между
  процессами): `seed_s = int.from_bytes(blake2b(f"{name}:{random_state}".encode(), digest_size=4).digest(),
  "big")`. Гарантия: тот же `(name, random_state)` → тот же `seed_s` на всех платформах/запусках; разные имена →
  разные seed (изоляция A↔B). Дубликаты имён запрещены валидатором (нет коллизий внутри прогона).
- **Публичный экспорт (R1-A8):** `FeatureSubsetSelector` — в `core/ports/__init__.py` и `honestml.core.ports`
  (как `FeatureRanker`); расширенный `FSStrategy` и новые поля — публичны (док в docstring `FeatureSelectionConfig`).
- Паттерн полей — как M6b (плоские strategy-специфичные поля, ср. `n_probes`).

### 2. Сериализация — artifact хранит ТОЛЬКО победителя (как M6b)
`FeatureSchema.selected_features` = subset **победителя** (M6b-поле, без изменений). Per-strategy-скоры **не**
сериализуются в artifact (транзиентная наблюдаемость, §3). `refit_best`/`design_matrix`/inference — **без правок**
(проекция M6b на `selected_features`). Старый artifact (без поля/M6b) грузится. **`ARTIFACT_VERSION` = 1** не бампается.

### 3. Наблюдаемость — `run_report["feature_selection"]` расширен аддитивно (`application/run_report.py`)
```jsonc
"feature_selection": {
  "strategy": "<winner>",            // M6b-ключ = победитель (back-compat)
  "strategies_evaluated": ["importance", "sequential", ...],
  "per_strategy": { "importance": {"n_selected": 7, "arb_score": 0.831}, ... },
  "winner": "sequential", "selected": [...], "n_selected": 9
}
```
- Single-path (`compare=None`): блок как M6b (`strategy`/`n_selected`/`selected`); `per_strategy` из одного
  элемента, `winner` = `strategy`. Дефолт-off → ключ `null`/отсутствует (отчёт как M6a). FS-конфиг (стратегии +
  параметры) — в `report["config"]` через config-дамп. **`RUN_MANIFEST_VERSION` не бампается** (аддитивные ключи,
  ADR-0037). Старые парсеры игнорируют новые ключи (JSON-safe).
- **Программный доступ (R2-completeness):** тот же исход доступен на фасаде как атрибут (паттерн M6b
  `run_report_`/`holdout_score_`) — без отдельного парсинга словаря; `selected`/`per_strategy`/`winner`
  восстановимы из `run_report_["feature_selection"]`. Опц. **stability** (Jaccard-overlap subset'ов между
  стратегиями, ADR-0044 §5) — зарезервированный аддитивный ключ `per_strategy`-блока (сигнальная диагностика,
  не инвариант; формат — на реализации).

### 4. Кэш / run-fingerprint (NFR-FSC-4) — два чётких случая (уточнено по ревью R1-A2)
`RunConfig.fs` уже в run-fingerprint через `model_dump` (ADR-0035). Поведение — **строго по двум случаям**:
- **`fs=None` (дефолт-off):** `FeatureSelectionConfig` **не инстанцируется**, в дампе `"fs": null` — **в точности
  как в M6b**. Новые поля живут **внутри** `FeatureSelectionConfig`, поэтому при `None` в дамп **не попадают**. →
  fingerprint **идентичен M6b**, **кэш M6b НЕ инвалидируется** дефолтными прогонами. (Отличие от M6a→M6b, где в
  `RunConfig` добавилось **само** поле `fs` → дамп менялся даже при off.) ✔
- **FS включён** (хоть M6b-`strategy=...`, хоть M6c-`compare=...`): `model_dump` сериализует **все** поля
  `FeatureSelectionConfig`, включая новые с дефолтами (`compare`, `selection_holdout`, `n_runs`, …) → дамп
  отличается от M6b → **другой fingerprint** → cache-miss (не ошибка). **Контент** (subset/модели) при том же
  `strategy` идентичен M6b; различается лишь кэш-ключ — тождественно эффекту M6a→M6b. В Release Notes.

FR-FSC-1 «идентично M6b» — про **результат** (контент leaderboard/artifact), не про строку fingerprint.

## Последствия
- (+) M6b-конфиг продолжает работать (single-path); дельта аддитивна; artifact лёгкий (только победитель); версии
  не тронуты; наблюдаемость полная (все N + решение); дефолт-off **не** инвалидирует кэш M6b.
- (−/компромисс) Включённый FS-прогон получает другой fingerprint, чем M6b (cache-miss, как M6a→M6b);
  per-strategy-скоры не в artifact (по дизайну — транзиентны); downgrade artifact не поддержан (Day-2, как ADR-0045).
- **Влияние на слои:** конфиг/спека — `core`; наблюдаемость — `application`; прикрепление победителя — `composition`.
  `ARTIFACT_VERSION`/`RUN_MANIFEST_VERSION` не тронуты.

## Проверки
- M6b-конфиг (`strategy="importance"`, без `compare`) → тот же subset, что M6b; `compare=(...)` → compare-путь.
- Невалидный `compare` (пустой/дубли/неизвестное имя) → `ConfigError`; `sequential` в `compare` + cutoff → cutoff
  игнорируется (не падает).
- `run_report["feature_selection"]` несёт `strategies_evaluated`/`per_strategy`/`winner` (аддитивно); single-path
  → как M6b; off → `null`. `ARTIFACT_VERSION`/`RUN_MANIFEST_VERSION` не изменены; старый artifact грузится.
- `fs=None` → fingerprint == M6b (кэш не инвалидируется); включённый FS → другой fingerprint (cache-miss).
