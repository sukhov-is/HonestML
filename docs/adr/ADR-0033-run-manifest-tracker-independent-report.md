# ADR-0033 — Tracker-независимый run-manifest / отчёт прогона (G-O1)

- **Статус:** Accepted (реализован в M5b, 2026-06-09)
- **Дата:** 2026-06-09
- **Драйверы:** DM5-2 (наблюдаемость прогона без трекера); FR-M5-4/6, NFR-M5-4/6. Наследует
  `RunContext.manifest()` (`context.py:58`), резолвнутый-config паттерн (ADR-0016), run-vs-artifact split
  (M4 operational, R1-COMP-1). Закрывает gap **G-O1**.
- **Воркстрим:** M5 (ядро прогона).

## Контекст
Сегодня нет **всегда-доступного, tracker-независимого отчёта прогона как контракта** (G-O1, `roadmap.md:48`):
есть лишь artifact-манифест (`save_artifact`, про модель) и in-memory `RunContext.manifest()` =
`{config, timings}` (`context.py:58`), который **никто не эмитит** и чьи `timings` в M2+-пути **пусты**.
Пользователь после `fit` не видит сводки прогона (резолвнутый scheme/seed/бюджет, winner, исход бюджета)
без MLflow. M4 operational **развёл** run-манифест (config+timings, `RunContext`) и artifact-манифест
(результаты модели) — нужно наполнить и **эмитнуть** первый.

## Рассмотренные варианты
1. **Только artifact-манифест** (как сейчас). Смешивает run- и model-контракты; G-O1 не закрыт; нет
   budget-исхода/резолвнутого-config на уровне прогона. **Отвергнут.**
2. **Run-manifest как отдельная поверхность поверх `RunContext.manifest()`** — наполнить `timings`,
   обогатить **исходом прогона** (winner/leaderboard-сводка, band-исход, budget-исход, резолвнутый
   significance/scheme), версионировать **своим** `RUN_MANIFEST_VERSION` (отдельно от `ARTIFACT_VERSION`),
   эмитить публичным `run_report_` + опц. файлом. **Выбран** (планка MLJAR `README`/leaderboard).
3. Внешний трекер (MLflow) как источник истины. Нарушает «tracker-независимо»; трекинг — порт M8.
   **Отвергнут** для G-O1.

## Решение

### 1. Чистый ассемблер run-report в `application` (Humble Object)
Чистая функция `build_run_report(*, run_config, timings, result, budget_outcome, significance_mode,
versions) -> dict` в `application` (без I/O, синхронно тестируема, NFR-M5-3/6). Соединяет:
- **резолвнутый** `RunConfig` (scheme/purge/embargo, seed, `BudgetConfig`, **`model_types`** —
  сериализуется **полностью**, R1-CONS-info) — ADR-0016 truthful;
- **`timings`** из `RunContext` (см. §2);
- **исход прогона** из `SliceResult`: `best_model_id`, leaderboard-сводка (id/score/rank), band-исход
  (`band_member_ids`/`band_unstable`/`winner_by_tiebreak`, M4), **budget-исход** (`budget_exhausted`,
  `skipped_by_budget`, ADR-0032), резолвнутый significance-режим (ADR-0034);
- **версии** (`honestml_version`, ключевые lib-версии) + `run_manifest_version: RUN_MANIFEST_VERSION`.
`SliceResult`/honesty-результаты НЕ дублируются в core — отчёт собирается над ними в application.
- **JSON-проекция (фикс R1-COMP3/DOM-info):** ассемблер возвращает **только JSON-примитивы** —
  `skipped_by_budget`/`band_member_ids` как `list[str]`, score/время через `float(...)`, `budget` через
  `BudgetConfig.model_dump(mode="json")`. **numpy-несущие поля `SliceResult`** (`oof_fold_index: np.ndarray`,
  `candidates` с oof-массивами) в отчёт **не входят** (как `LeaderboardEntry`, уже plain-`float`). Тест
  `test_run_report_json_serializable` гоняет `json.dumps` на `SliceResult` с непустым `oof_fold_index`.

#### Схема `run_report_` v1 — контракт top-level ключей (фикс R2-MAJ)
`run_report_` — **версионируемая публичная** поверхность; имена ключей v1 = контракт (потребители делают
`.get("<exact>")`). Базис — `RunContext.manifest()` (`{config, timings}`, context.py:58); ассемблер
**оборачивает** его исходом. v1:

| ключ | тип | источник |
|---|---|---|
| `run_manifest_version` | `int` (=`RUN_MANIFEST_VERSION`=1) | константа |
| `honestml_version` | `str` | `importlib.metadata` |
| `config` | `dict` | `RunConfig.model_dump(mode="json")` — **полный** (seed/cv-резолвнут/budget/significance/model_types) |
| `timings` | `dict[str, dict[str,float]]` | `RunContext.timings` (ключ `"run"` → `selection`/`refit`, §2) |
| `winner` | `str` | `result.best_model_id` |
| `leaderboard` | `list[dict]` | `[{model_id, score:float, rank}]` (проекция `LeaderboardEntry`) |
| `band` | `dict` | `{member_ids:list[str], unstable:bool, width:int, winner_by_tiebreak:bool}` (M4) |
| `budget` | `dict` | `{mode, exhausted:bool, skipped:list[str]}` (ADR-0032 §4/§6 — **здесь** живёт budget-провенанс) |
| `significance` | `str` | резолвнутый `"bootstrap"`/`"off"` (ADR-0034 — **здесь** живёт significance-провенанс) |

Эволюция — **только аддитивные** ключи + `.get`-fallback (operational §1); бамп `RUN_MANIFEST_VERSION` —
лишь при смене семантики существующего ключа.

### 2. Эмиссия в `composition`/`facade` + точки `timed_stage` (фикс R1-COMP2/ASIS2)
- **Наполнение `timings` — в фасаде** (composition): `run_slice` и `refit_best` — **чистые use-cases**,
  не само-таймятся; фасад оборачивает их `ctx.timed_stage("run","selection")` вокруг `run_slice` и
  `ctx.timed_stage("run","refit")` вокруг `refit_best` (единый key `"run"`; `total_time("run")` = сумма
  стадий). Так `timings` непуст (FR-M5-4 п.4) с конкретным местом вставки, а не «подразумеваемо».
- Публичный атрибут **`self.run_report_: dict`** после `fit` (как `band_*_`/`calibration_`).
- **Контракт `save_run_report` (фикс R2-MAJ):** `save_run_report(report: dict, path: str | Path, *,
  overwrite: bool = True) -> Path`. `path` — **файл** (по умолчанию имя `run_report.json`; если передан
  каталог — пишет в `path/run_report.json`). `overwrite=False` + существующий файл → `FileExistsError`.
  Кодировка **`utf-8`**; содержимое — чистый `json.dumps(report, indent=2)`. Возвращает записанный путь.
  (Отлично от `save_artifact`, который берёт каталог и пишет несколько файлов — здесь один JSON.)
- `RunContext.manifest()` остаётся **базисом** (config+timings); ассемблер оборачивает его исходом из
  `SliceResult`. Само ядро `core` без изменений (отчёт — application/composition concern).

### 3. Версионирование — отдельно от артефакта (NFR-M5-4)
`RUN_MANIFEST_VERSION = 1` (новая константа в `application`/`composition`). Run-report — **отдельная
поверхность**: `ARTIFACT_VERSION` (=1) **не меняется**. Эволюция отчёта — **аддитивные** ключи +
`.get`-fallback на чтении (паттерн M4-манифеста), без бампа артефакта (R-MAN).

### 4. Truthful + budget/degradation видимы (NFR-M5-6, FR-M5-6)
Отчёт пишет **резолвнутое** (фактический scheme после `auto`, фактический significance-режим, фактический
бюджет), не запрошенное. **Budget-исход явен:** `budget_exhausted` + список `skipped_by_budget`;
within-budget → `False`/пустой список. Деградация «видна, а не подразумевается».

## Последствия
- **Положительные:** G-O1 закрыт — сводка прогона без трекера; чёткое разделение run- vs artifact-контракта;
  чистый ассемблер тестируем на фейковом `SliceResult`; `ARTIFACT_VERSION` не тронут; основа (`RunContext.
  manifest()`) переиспользована.
- **Отрицательные/компромиссы:** ещё одна версионируемая поверхность (`RUN_MANIFEST_VERSION`) — но
  аддитивная и дешёвая; `timings` требует обернуть стадии `timed_stage` (мелкая инструментизация фасада).
- **Влияние на слои:** ассемблер — `application` (чист); эмиссия/сериализация + `run_report_` — `composition`/
  `facade`; `core` (`RunContext`) — без изменений. import-linter не нарушен.

## Проверки
- После `fit` `run_report_` несёт резолвнутый scheme/seed/`BudgetConfig` + winner + band-исход +
  budget-исход; сериализуется в JSON без трекера (FR-M5-4).
- `timings` непуст (как минимум суммарное время прогона) — `timed_stage` вокруг отбора/refit.
- Budget-исход правдив: исчерпание → `budget_exhausted=True` + непустой `skipped_by_budget`; within →
  `False`/пусто (FR-M5-6).
- `ARTIFACT_VERSION` не изменён; run-report несёт `run_manifest_version`; старое чтение отчёта толерантно к
  отсутствующим ключам (`.get`).
- `build_run_report` — чистая, тестируется на фейковом `SliceResult`/`RunContext` без обучения (NFR-M5-3).

## Impl-note (M5, 2026-06-09)
- Сигнатура сведена к `build_run_report(*, run_config, timings, result)`: `budget_outcome`,
  `significance_mode` и `versions` из §1 НЕ передаются отдельными аргументами, а выводятся внутри —
  budget-режим/significance из **резолвнутого** `run_config` (truthful), budget-исход (`exhausted`/`skipped`)
  из `result`, версии из `importlib.metadata`. Контракт схемы-v1 (таблица ключей) соблюдён дословно; это
  устранение дублирования (DRY), не смена решения.
- `save_run_report` — в `composition/run_report.py`, экспортируется верхнеуровнево (`from honestml import
  save_run_report`); `RUN_MANIFEST_VERSION` — в `application/run_report.py`.
