# ADR-0039 — Memory-enforce: ортогональный кооперативный лимит RSS (best-effort)

- **Статус:** Accepted (реализован 2026-06-09)
- **Дата:** 2026-06-09
- **Драйверы:** DM-2 (memory как ортогональная ось с graceful degradation), DM-3 (слои + опциональная
  зависимость); FR-MEM-1/2/3, NFR-RM-2/3/4. Наследует порт `Budget.memory_left()` (ADR-0006, уже в контракте;
  реализуется здесь) и **аддитивно расширяет порт членом `exhausted_reason`** (§1/§5); per-candidate gate +
  graceful degradation + time-best-effort (ADR-0032 §1/§2/§3/§6), run-report (ADR-0033), инъекцию зонда `clock`
  в `RunBudget` (ADR-0032 §2). Расширяет множество значений `BudgetExhaustedError.mode` (ADR-0008) на `"memory"`.
- **Воркстрим:** M5 (memory дельта).

## Контекст
`BudgetConfig.memory_limit_mb` объявлен, но **не enforced** (мёртвое поле). Нужно его активировать как **лимит
памяти прогона**, не ломая дефолт, слои и lightweight-core, и **честно** относительно среды: hard-kill одного
`fit` невозможен (нет SIGALRM на Windows, kill нативного C-кода небезопасен — ADR-0032). Память — **ортогональна**
времени/трайлам (прогон может быть ограничен и по времени, **и** по памяти).

## Рассмотренные варианты
1. **Новое значение `BudgetMode = "memory"`.** Сделало бы память **взаимоисключающей** с time/trials (один
   `mode`) — неверно: память композируется с любым режимом. **Отвергнут.**
2. **Декомпозиция `RunBudget` → `TimeBudget`/`TrialBudget`/`MemoryBudget`+`CompositeBudget`** (SRP-note
   ADR-0032). Чище по SRP, но это **рефакторинг работающего M5-кода** (RunBudget оттестирован, отгружен) →
   расширение объёма дельты. **Отложен** (future SRP-улучшение); для дельты — минимальный диф.
3. **Память как ортогональная проверка в существующем `RunBudget`** (поверх любого `mode`), инжектируемый
   `mem_probe` (RSS), gate ИЛИ-объединяет mode-exhausted и memory-exhausted. **Выбран** (минимально, порт не
   трогается, поле `memory_limit_mb` наконец используется, переиспользует gate+degradation).
4. **`resource.getrusage` / `tracemalloc`.** `resource` — нет на Windows; `tracemalloc` — только Python-heap,
   **не** видит C-аллокации бустингов (главный потребитель). **Отвергнуты** (R-MEM-DEP).
5. **Hard-cap: subprocess-изоляция / `pynisher` / kill mid-fit** (планка auto-sklearn/MLJAR). Среда исключает
   (Windows/C-код); переусложнение для дельты. **Отвергнут** (future, если появится изоляция).

## Решение

### 1. Память — ортогональная проверка в `RunBudget`; порт получает `exhausted_reason`
`RunBudget(config, *, clock=…, mem_probe: Callable[[], float] = _default_rss_mb)`. `mem_probe` возвращает RSS
процесса в МБ (по образцу инжекции `clock` — Humble Object, тест на фейке без psutil). Контракт `exhausted`
(порядок проверки фиксирован — **mode первым, memory вторым**):
```
exhausted = _mode_exhausted() or (memory_limit_mb is not None and mem_probe() >= memory_limit_mb)
```
`memory_left()` = `memory_limit_mb - mem_probe()` (или `None`, если лимит не задан) — реализует уже
существующий метод порта `Budget.memory_left()` (раньше `None`). **Отрицательный `memory_left()` валиден**
(перерасход) — симметрично `time_left()<0` (закреплено `test_run_budget.py`), реализация **не** клампит к 0
(иначе теряется наблюдаемость величины перерасхода) (фикс m2). Память **ортогональна** `mode`: работает под
`none`/`time`/`trials` (`mode="time"`+`memory_limit_mb` — оба ограничения; `mode="none"`+`memory_limit_mb` —
memory-only).

**Порт `Budget` расширяется ОДНИМ аддитивным членом — `exhausted_reason: str | None`** (фикс M3, R-LEAK-
порта): что фактически исчерпано (`"time"`/`"trials"`/`"memory"`/`None`), симметрично уже присутствующему
`memory_left()`. Это снимает структурную утечку адаптерного поля в use-case: `run_slice` читает
`budget.exhausted_reason` **по контракту порта**, а не `getattr` адаптер-атрибута. `exhausted_reason` следует
тому же порядку (фикс M1, политика приоритета): возвращает mode-ось (`time`/`trials`), если mode исчерпан,
иначе `"memory"`, иначе `None` — детерминированно, согласовано с порядком `exhausted` и c4-flowchart (при
одновременном исчерпании time+memory причина = `time`: явный, документированный приоритет «явный бюджет
пользователя важнее вторичного guard», не произвол).

**Стоимость зонда (фикс m3):** `_default_rss_mb` конструирует `psutil.Process()` **один раз** (в замыкании/
конструкторе зонда), не на каждый `mem_probe()`; per-candidate частота вызовов (≈ число кандидатов) приемлема.

### 2. Сборка budget при memory под `mode="none"` (R-MEM-NONE) + граничный случай baseline
`facade._build_budget` строит `RunBudget`, если `config.mode != "none"` **ИЛИ** `config.memory_limit_mb is not
None` (иначе gate инертен и память не enforced). Валидатор `BudgetConfig` уже допускает `memory_limit_mb` под
любым mode (guard «none» запрещает только `time_budget_s`/`n_trials`).

**Граничный случай «лимит ниже baseline RSS» (фикс M2):** под памятью gate проверяется **до** старта первого
кандидата (в отличие от ленивого `t0` time-mode, где первый всегда стартует — инвариант ADR-0032 §3 здесь **не
универсален**). Если baseline RSS процесса (numpy/polars/sklearn + материализованный `design_matrix`) уже ≥
лимита, первый кандидат не стартует → 0 завершённых → `BudgetExhaustedError(reason="memory")`. Это **намеренно
и корректно** (лимит неудовлетворим), не дефект. Для actionability `RunBudget` при заданном memory-лимите
логирует **WARNING** при построении, если baseline ≥ лимита («memory_limit_mb=X below process baseline ~Y MB;
no candidate can start») — диагностика, **не** hard-`ConfigError` (RSS дрожит → флаки-падение хуже). Ссылка
ADR-0032 §3 «первый всегда стартует» уточняется: инвариант относится к time-mode; memory может дать
0-completed легитимно.

### 3. Зонд RSS и опциональная зависимость (NFR-RM-4)
`_default_rss_mb` импортирует `psutil` (`psutil.Process().memory_info().rss / 1024**2`) — **только в адаптере**.
`psutil` — **опциональный extra**: `pyproject` `[memory] = ["psutil>=5.9"]`, **добавляется в агрегат `[all]`**,
**не** добавляется в `[dev]` (чтобы тест «отсутствие psutil» был исполним через мок, не зависел от среды; gate-
тесты идут на инжектированном `mem_probe` без psutil — NFR-RM-2).

**Точка падения — единственная, при построении `RunBudget` (фикс M4):** если `memory_limit_mb is not None`
**и** `mem_probe` — дефолтный (не инжектирован), `RunBudget.__init__` (через `_default_rss_mb`-factory)
пытается импортировать psutil **немедленно** → при отсутствии `MissingDependencyError("memory",
package="psutil")`. При **инжектированном** `mem_probe` импорт не делается (тесты psutil-free). Так разрешены и
«когда падать» (на сборке budget, до цикла кандидатов, не на верхнем `import honestml` — lightweight-core
ADR-0001), и «как мокать отсутствие» (тест дефолтного зонда с подменённым `import`).

**Точное сообщение (фикс B2):** используется **существующий** контракт `MissingDependencyError("memory",
package="psutil")` (`core/exceptions.py`), дающий
`"Optional dependency 'psutil' for feature 'memory' is not installed. Install it with: pip install
honestml[memory]"`. Конструктор исключения **не** расширяется; ранее процитированная произвольная строка —
ошибочна (исключение не принимает custom-текст). Критерий приёмки NFR-RM-4 проверяет тип+`extra="memory"`+
`package="psutil"`, **не** дословный нестандартный текст.

RSS — **абсолютный process-footprint** (простой/правдивый); искажение нативными тредами бустингов
документировано как best-effort (R-MEM-RSS).

### 4. Реакция на исчерпание = graceful degradation ADR-0032 (переиспользуется)
Memory-exhausted в per-candidate gate ведёт себя как time/trials (`slice.py`): кандидат **не стартует**
(skip), `budget_exhausted=True`. Пост-цикл (ADR-0032 §3): ≥1 завершённый → best-so-far + **refit всегда**
(final-fit не gated, отгружается рабочая модель); 0 завершённых + skip → `BudgetExhaustedError`; 0 + все упали
→ `FitFailedError`. Реакция **не дублируется** — тот же код gate/пост-цикла.

**Причина в `BudgetExhaustedError` (фикс B1):** на 0-completed пути `run_slice` передаёт **первым аргументом**
(метка `mode` в `BudgetExhaustedError.__init__`, `core/exceptions.py`) значение `budget.exhausted_reason`
(порт, §1) — т.е. **ось исчерпания** (`time`/`trials`/`memory`), а не `BudgetConfig.mode`. Так сообщение для
memory-only прогона (`mode="none"`+`memory_limit_mb`) станет `"budget (memory) exhausted: …"` вместо неверного
`"budget (none) exhausted"`. Для time/trials `exhausted_reason` == `mode` → существующее поведение/тесты
(напр. `test_…::match="time"`) **не регрессируют**. **Contract-note (ADR-0008, фикс B1-minor):** строковое поле
`BudgetExhaustedError.mode` теперь может принимать значение `"memory"` (ось исчерпания) сверх `BudgetMode` —
аддитивное расширение множества значений публичного поля исключения; сообщение `"budget (memory) exhausted: …"`.
Сигнатура исключения **не** меняется (первый позиционный — по-прежнему строка-метка).

### 5. Truthful-наблюдаемость (FR-MEM-3, аддитивно к ADR-0033)
`exhausted_reason` — теперь **член порта `Budget`** (§1), не адаптер-атрибут. `run_slice` при degraded пишет
`SliceResult.exhausted_by: str | None` из `budget.exhausted_reason` (по контракту порта — без `getattr`-утечки,
фикс M3); поле аддитивно. `build_run_report` budget-блок += **только `exhausted_by`** (исход). **`memory_limit_mb`
НЕ дублируется** в budget-блоке (фикс m4): лимит — это **вход**, живёт в `report["config"]["budget"]` через
config-дамп (источник истины, ADR-0033 §1 «config vs outcome»); budget-блок несёт только исход
(`mode`/`exhausted`/`skipped`/`exhausted_by`). **`RUN_MANIFEST_VERSION` не бампается** (аддитивный ключ;
потребители — `.get`). `build_run_report` не принимает `budget`-объект (сигнатура без него) — `exhausted_by`
приходит **через `SliceResult`** (заполнено в `run_slice`, где доступен порт), не из report-ассемблера.

**Взаимодействие memory × cache (M5-resume, фикс m5):** gate (`budget.exhausted`) проверяется **до** cache-hit
(`slice.py` порядок), и память **не освобождается** между кандидатами (operational §4). Значит после превышения
RSS все последующие кандидаты — **включая cache-hit'ы** — пропускаются, хотя hit обучения не запускает. Это
**намеренно и корректно**: raison d'être лимита — не держать больше, а не «сколько вычислили»; в отличие от
time (где hit near-instant проходит gate, ADR-0037 §2), memory-gate после превышения закрыт необратимо.

### 6. Контракт детерминизма (NFR-RM-3)
Memory-enforce — **best-effort/недетерминирован** (наследует ADR-0032 §6): RSS зависит от ОС/аллокатора/тредов
→ между машинами невоспроизводим; исход — в run-report, **не** в артефакте (`ARTIFACT_VERSION` не трогается).
**Peak-within-fit (R-MEM-PEAK):** кооперативный gate ловит RSS только **между** кандидатами; кандидат, чей
собственный пик превышает лимит, не предотвратим (hard-kill невозможен). Это **документированный предел**,
не выдаётся за hard-cap; subprocess-изоляция → future.

## Последствия
- **Положительные:** `memory_limit_mb` наконец enforced; ортогонально mode; порт не тронут; переиспользует
  gate+degradation; psutil опционален и ленив; дефолт неизменен; truthful-исход в отчёте.
- **Отрицательные/компромиссы:** best-effort (peak-within-fit не ловится, R-MEM-PEAK — принято); абсолютный RSS
  искажается тредами (R-MEM-RSS — документировано); `RunBudget` остаётся одним классом с ортогональной memory-
  веткой (SRP-декомпозиция отложена — осознанный минимальный диф); новая опциональная зависимость psutil.
- **Влияние на слои:** `mem_probe`/psutil/RSS — `adapters/run_budget`; gate ИЛИ memory + `exhausted_by` —
  `application/slice` (через порт); `_build_budget`-условие + extra — `composition`/`pyproject`; **порт
  `Budget` получает ОДИН аддитивный член `exhausted_reason: str | None`** (`core/ports/budget.py`, симметрично
  `memory_left()`), фейки-budget в тестах добавляют его одной строкой; `core` иначе не трогается. import-linter
  не нарушен.

## Проверки
- Фейковый `mem_probe` (RSS ≥ лимита) → gate пропускает следующего кандидата (тест без psutil); `memory_left()`
  = `limit - rss` (в т.ч. **отрицательный** при перерасходе, не клампится); `exhausted_reason == "memory"`.
- Композиция: memory работает под `mode="time"`/`"trials"`/`"none"` (на фейках); при одновременном исчерпании
  time+memory `exhausted_reason == "time"` (приоритет mode); `mode="none"`+`memory_limit_mb` → активный budget
  (не `None`).
- `memory_limit_mb` с дефолтным зондом без psutil → `MissingDependencyError(extra="memory", package="psutil")`
  при построении `RunBudget` (мок отсутствия импорта); `import honestml` без psutil не падает; дефолт (`None`) и
  инжектированный `mem_probe` psutil не требуют.
- Baseline ниже лимита: WARNING при построении budget; 0-completed-by-memory → `BudgetExhaustedError` c
  `mode`-меткой `"memory"` (сообщение `"budget (memory) exhausted: …"`).
- Graceful degradation: ≥1 завершённый при memory-skip → success + refit + `predict` работает; 0 + memory-skip
  → `BudgetExhaustedError` (причина «memory»). Для time/trials причина == mode (`match="time"` не регрессирует).
- Run-report: budget-блок несёт `exhausted_by` (`memory`/`time`/`trials`/`null`); `memory_limit_mb` — в
  `config["budget"]` (не дублируется в budget-блоке); `RUN_MANIFEST_VERSION` не изменён; старое чтение толерантно.
- Memory × cache: после превышения RSS последующие cache-hit'ы пропускаются (память не освобождается) — тест на
  фейках (предзаполненный кэш + fake `mem_probe`).

## Impl-notes (2026-06-09)
- Реализовано как спроектировано (`adapters/run_budget.py` `mem_probe`/`_default_rss_mb`/`exhausted`/
  `exhausted_reason`/`memory_left`/baseline-WARNING; `core/ports/budget.py` `exhausted_reason`;
  `application/slice.py` `exhausted_by`; `composition/facade.py` `_build_budget`; `application/run_report.py`
  budget-блок; `pyproject.toml` `[memory]`). Проверки: `tests/unit/test_run_budget.py` (memory-gate/
  `memory_left`<0/композиция/приоритет/none+memory/baseline/`MissingDependencyError`),
  `tests/unit/test_run_slice.py` (`test_memory_degraded_best_so_far`/`test_memory_zero_completed_raises_memory`/
  `test_memory_cache_hits_skipped_after_exceedance`), `tests/unit/test_run_report.py`,
  `tests/unit/test_facade.py::test_memory_limit_requires_psutil`, `tests/unit/test_ports_contracts.py`.
- **Уточнение (фикс ревью Q1):** ось исчерпания **захватывается в момент исчерпания** в цикле `run_slice`
  (`exhausted_by`, first-capture-wins) и переиспользуется и для `SliceResult.exhausted_by`, и для метки
  `BudgetExhaustedError` — вместо позднего перечтения `budget.exhausted_reason` после цикла. Так значение
  устойчиво к немонотонности RSS (память могла «просесть» между skip'ом и концом цикла). `or "budget"` —
  недостижимый last-resort.
- **Уточнение приёмки NFR-RM-4:** `test_public_api.py::test_top_level_import_is_lightweight` **не** проверяет
  отсутствие psutil в `sys.modules` — `joblib` (транзитивно через sklearn) тянет psutil при наличии,
  независимо от honestml, и тест-среда (системный Python) его имеет. Ленивость psutil **нашего** кода покрыта
  `test_run_budget.py::test_memory_limit_requires_psutil` (импорт только при заданном лимите с дефолтным
  зондом) + доказательством lightweight-импорта в venv без psutil.
