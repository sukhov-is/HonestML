# ADR-0034 — Публичный тумблер significance (honest-on по умолчанию, явный off)

- **Статус:** Accepted (реализован в M5c, 2026-06-09)
- **Дата:** 2026-06-09
- **Драйверы:** DM5-3 (честность по умолчанию, но управляемая); FR-M5-5, NFR-M5-6. Наследует
  `BootstrapSignificanceTest`/`NoSignificanceTest` (ADR-0007/0026), sklearn-инвариант `__init__`
  (ADR-0011). Реализует отложенное M4 (`operational.md`: `significance: Literal["bootstrap","off"]` → M5).
- **Воркстрим:** M5 (ядро прогона).

## Контекст
M4 сделал честный band **дефолтом-ON** (`build.py:115` безусловно строит `BootstrapSignificanceTest`),
но **публичной поверхности отключить его нет** — только программно передать инертный `NoSignificanceTest`
в `run_slice` (`build.py:113` коммент: «Programmatic opt-out is M5»). Пользователю нужен публичный,
clone-безопасный контроль: оставить честный отбор (дефолт) или вернуть чистый argmax.

## Рассмотренные варианты
1. **Только программно** (статус-кво) — инертный тест внутрь `run_slice`. Не публично, не через фасад.
   **Отвергнут** (FR-M5-5 требует публичную поверхность).
2. **Публичный параметр фасада + поле `RunConfig`** — `significance: Literal["bootstrap","off"]="bootstrap"`
   в `__init__` (verbatim, clone-safe) → `RunConfig.significance` (аддитивно, для truthful-манифеста) →
   `build_default_components` выбирает `Bootstrap` vs `No`. **Выбран** (планка: дефолт честный, off явный).
3. Булев `honest: bool=True`. Менее расширяемо (будущие режимы значимости) и менее truthful в манифесте
   (литерал-режим читаемее). **Отвергнут** в пользу строкового литерала.

## Решение

### 1. Параметр фасада `significance` (verbatim, sklearn-инвариант)
`AutoML(..., significance: Literal["bootstrap","off"] = "bootstrap")` — хранится в `__init__` как есть (без
вычислений, ADR-0011), участвует в `get_params/set_params/clone`/`Pipeline`. **Дефолт `"bootstrap"`
сохраняет поведение M4** (honest-on).

### 2. Аддитивное поле `RunConfig.significance` (truthful-манифест)
`RunConfig.significance: Literal["bootstrap","off"] = "bootstrap"` (аддитивно, frozen/extra=forbid;
дефолт = honest-on). Фасад собирает `RunConfig(seed, cv, budget, significance, model_types)` → значение
попадает в **run-report** (ADR-0033, ключ `significance`) как **резолвнутое** (truthful, NFR-M5-6).
> **Провенанс — в run-report, НЕ в артефакте (фикс R2-completeness, симметрия с budget):** резолвнутый
> significance-режим живёт **только в run-report** — симметрично budget-исходу (ADR-0032 §6). Артефакт его
> **не** само-раскрывает: для `"off"` band вырождается в lone-anchor (как честный band из 1 модели) — это
> **принятый** компромисс (run-report — спутник артефакта, несёт полный провенанс прогона в одном месте;
> provenance-в-артефакте отдельным аддитивным ключом → future, при нужде). НЕ оставляем асимметрию
> «budget в артефакте, significance — нет»: **оба** провенанса в run-report.

### 3. Резолв в composition
`build_default_components(..., significance="bootstrap"|"off")`:
- `"bootstrap"` → `BootstrapSignificanceTest(resolved, seed=random_state, n_boot=_DEFAULT_N_BOOT)` (как M4).
- `"off"` → `NoSignificanceTest()` (инертный): `run_slice._wants_oof` → `False` (OOF-захват под значимость
  **не форсится**), `equivalence_band` даёт вырожденный band = чистый argmax (нет членства,
  `winner_by_tiebreak=False`).
Никакой новой логики отбора — переиспользуются существующие порты/ветки.

### 4. Взаимодействие с refinement/calibration (M4d)
`selection="refinement"`/`calibrate` (ADR-0030/0031) **ортогональны** тумблеру: refinement меняет
ранжирование (score кандидатов), significance — членство в band. `"off"` отключает только band (argmax по
текущему score, включая refined). Калибровка победителя не зависит от тумблера. Конфликтов нет.

## Последствия
- **Положительные:** публичный, clone-безопасный контроль честности; дефолт — честный (north-star);
  резолвнутый режим виден в манифесте (truthful); переиспользует `No/BootstrapSignificanceTest` (нулевая
  новая логика отбора); расширяемо (будущие режимы — новый литерал).
- **Отрицательные/компромиссы:** `"off"` возвращает оптимистичный argmax-выбор (смещён вверх на near-ties,
  Cawley-Talbot) — это **осознанный пользовательский выбор**, помеченный в манифесте, не молчаливый.
- **Влияние на слои:** параметр — `facade`; поле — `core/config` (аддитивно); резолв — `composition/build`;
  `application`/`core`-порты значимости без изменений. import-linter не нарушен. `honestml.__all__` не
  расширяется (литерал — не новый публичный тип).

## Проверки
- Дефолт `AutoML().fit()` строит band (поведение M4 не изменено) — существующий сьют зелёный.
- `significance="off"` → чистый argmax: нет band-членства, `winner_by_tiebreak=False`, OOF-захват под
  значимость не форсится (FR-M5-5).
- `clone/get_params` сохраняют `significance`; `set_params(significance="off")` работает.
- Резолвнутый режим (`bootstrap`/`off`) виден в `run_report_`/манифесте (truthful, NFR-M5-6).
- Ортогональность: `significance="off"` + `selection="refinement"` → argmax по refined-score, без band.
