# ADR-0032 — Enforcement бюджета (time/trials) + graceful degradation «верни лучшее на сейчас»

- **Статус:** Accepted (реализован в M5a-engine/M5a-wire, 2026-06-09)
- **Дата:** 2026-06-09
- **Драйверы:** DM5-1 (сквозной бюджет + деградация); FR-M5-1/2/3, NFR-M5-1/2/3. Наследует `Budget`-порт
  (ADR-0006, `budget.py:13`), `run_slice`/OOF (ADR-0010), `BudgetExhaustedError` (G7, `exceptions.py:89`).
- **Воркстрим:** M5 (ядро прогона).

## Контекст
`Budget`-порт, `BudgetConfig` и `BudgetExhaustedError` есть, но **полностью инертны**: ни enforcement, ни
адаптера, ни raise-сайта; фасадный `time_budget` хранится, но не течёт в `RunConfig.budget`
(`facade.py:61`). `run_slice` гоняет ВСЕХ кандидатов во внешнем цикле `for name, factory in
estimators.items()` (`slice.py:254`) без оглядки на бюджет. Нужно: дешёвый, портируемый enforcement
time/trials и поведение «на исчерпании — вернуть лучшее на сейчас, не упасть». Среда Windows: **нет
`SIGALRM`**, kill нативного sklearn/boosting C-кода небезопасен.

## Рассмотренные варианты
1. **Hard-preemptive timeout** (signal.alarm / поток / процесс-kill одного `fit`). Прерывает один долгий
   кандидат, но: нет SIGALRM на Windows, kill C-кода = риск повреждённого состояния, недетерминизм,
   подпроцесс — тяжело. **Отвергнут для M5** (вне объёма; кооператив достаточен).
2. **Кооперативный per-candidate gate** — проверка `budget.exhausted` **между кандидатами** во внешнем
   цикле `run_slice`; на исчерпании — стоп, ранжирование завершённых. Портируемо, детерминированные точки
   останова, нет полу-обученного состояния, graceful best-so-far; цена — overshoot ≤ один кандидат.
   **Выбран** (планка FLAML/AutoGluon).
3. Per-fold gate (финее). Прерывает кандидата на полпути → частичный OOF, спец-обработка покрытия;
   усложняет без явной нужды. **Отложен** (гранулярность кандидата достаточна; per-fold — future).

## Решение

### 1. Кооперативный per-candidate gate в `run_slice` (фикс инертности)
`run_slice` принимает **опциональный** `budget: Budget | None = None` (порт, не конкретный — контракт
usecases-independent-of-adapters). Внешний цикл (gate-skip остатка через `continue`, чтобы заполнить
**полный** `skipped_by_budget` — FR-M5-6; trial считается **только за завершённого** кандидата):
```
for name, factory in estimators.items():
    if budget is not None and budget.exhausted:
        skipped.append(name); budget_exhausted = True; continue   # пропустить остаток (не break)
    try:
        cand = _run_candidate(...); candidates.append(cand)       # успех
    except _CandidateFailed as exc:
        failed.append(...); continue                               # отказ НЕ тратит трайл
    if budget is not None: budget.consume(cand.train_time)         # один consume = один ЗАВЕРШЁННЫЙ трайл
```
**Источник `seconds` (фикс R2):** `consume(cand.train_time)` — переиспользует уже измеренное per-candidate
время (`Candidate.train_time`, slice.py:448), без дублирующего тайминга во внешнем цикле; для trials/none
значение advisory (важно число вызовов), для будущего time-`consume`/`CompositeBudget` — корректный источник.
**`budget_exhausted`** захватывается **булевой переменной в цикле** (не live-перечтением `budget.exhausted`
после цикла) — детерминированно, тестируемо на фейковых часах, и питает `SliceResult.budget_exhausted` (§4).
`budget=None` → ветка не активна → **поведение M3/M4 без изменений** (NFR-M5-4). Overshoot ограничен одним
`_run_candidate` (NFR-M5-1). Gate — **перед** стартом кандидата. **`consume` — после успеха, до ветки
`except`** (фикс R1-DOM4): упавший кандидат не сжигает трайл (`n_trials` = число **завершённых**, не
стартовавших). Перебор остатка `continue` (не `break`) нужен, чтобы собрать всех пропущенных (R1-CONS-info).
- **Gate — ТОЛЬКО в `run_slice` (отбор); `refit_best` НЕ gated (фикс R2-completeness):** финальный refit
  победителя (`facade.py:122`) **всегда выполняется** (graceful degradation обязана вернуть рабочую модель —
  FR-M5-2 «predict работает»), бюджетом не прерывается. Под `mode="time"` тот же `RunBudget` жив во время
  refit, но refit не gated → **реальный overshoot на уровне `fit()` = один in-flight кандидат + финальный
  refit победителя** (NFR-M5-1 уточнён). `refit_best` лишь **таймится** (`timed_stage`, ADR-0033 §2), не
  gated.

### 2. Адаптер `RunBudget` (реализует существующий порт; порт НЕ меняется)
`adapters`: `RunBudget(config: BudgetConfig, *, clock: Callable[[], float] = time.perf_counter)`
реализует `Budget` **без расширения порта** (фикс NFR-M5-3):
- **`mode="none"` (unbounded, дефолт):** `exhausted=False` всегда; `time_left()=inf`. Кодирует «без
  границ» **как явный режим конфига** (см. §5a), чтобы манифест был правдив (фикс R1-major).
- **`mode="time"`:** **ленивый старт часов** — `t0=clock()` фиксируется при **первом** обращении к
  `exhausted` (= вход в цикл кандидатов в `run_slice`), **не** при конструировании (фикс R1-CA3): так
  setup (reader/holdout-carve) и `design_matrix`/split самого `run_slice` **не биллятся** —
  «time_budget = время прогона кандидатов». `time_left() = time_budget_s - (clock()-t0)`;
  `exhausted = time_left() <= 0`. **Идемпотентность `t0` (фикс R2):** gate читает `exhausted` **на каждой**
  итерации (N+1 раз); `t0` захватывается **ровно один раз** — `if self._t0 is None: self._t0 = clock()`;
  последующие чтения переиспользуют (иначе часы сбрасывались бы каждым кандидатом и бюджет никогда не истёк
  бы). Тест: два `exhausted` до сдвига фейковых часов → `t0` стабилен.
- **`mode="trials"`:** считает **вызовы `consume`** как завершённые трайлы — `exhausted = trials_done >=
  n_trials`; `time_left() = inf`. **Контракт (фикс R1-CA2/DOM5):** gate зовёт `consume` **ровно один раз
  на завершённого кандидата**; `RunBudget` трактует один `consume` как один трайл **независимо от значения
  `seconds`** (аргумент advisory); в `mode="time"`/`"none"` `consume` — no-op (исчерпание clock-/режим-
  производно). Тест закрепляет: trials-исчерпание зависит от **числа** вызовов, не от `seconds`.
- **`memory_left()` → `None`** (memory-enforce отложен, FR-M5 границы; `psutil` не тянем).
- **Инъекция clock** делает логику бюджета **синхронно тестируемой на фейковых часах** без обучения
  (Humble Object, NFR-M5-3). Конкретный `time.perf_counter` — только в адаптере (NFR-M5-5).

> **SRP-замечание (R1-CA-info, принято):** `RunBudget` — один адаптер с режим-ветками. Для M5 (2 режима +
> none) — приемлемо (честное переиспользование 4-методного порта). При добавлении memory-enforce/per-fold
> (operational §5) предпочесть **композицию** мелких `Budget` (`TimeBudget`/`TrialBudget`/`CompositeBudget`)
> вместо роста режим-флага — порт и сигнатура gate это позволяют без изменений.

### 3. Graceful degradation (семантика) + явное правило нуля завершённых (фикс R1-blocker)
- **≥1 завершённый кандидат на исчерпании:** прогон **успешен**; band/тай-брейк (M4) и leaderboard
  считаются **по завершённым** (`candidates` и так содержит только завершённых) → best-so-far. Без исключения.
- **0 завершённых — явное пост-цикловое правило** (вместо сегодняшнего безусловного `if not candidates:
  raise FitFailedError`, slice.py:280): после цикла, если `candidates` пуст — ветвление на **захваченном в
  цикле булеве `budget_exhausted`** (не live-перечтение `budget.exhausted`, фикс R2 — детерминированно):
  ```
  if budget_exhausted and skipped:        # бюджет урезал прогон до пустоты
      raise BudgetExhaustedError(<режим>, completed=0, skipped=len(skipped), failed=len(failed))
  raise FitFailedError(failed)            # иначе — отказ моделей (как M3)
  ```
  **Решение смешанного случая** (`skipped` непуст И были упавшие): → **`BudgetExhaustedError`** — прогон
  трактуется как **бюджет-деградированный** (бюджет пропустил часть кандидатов), независимо от того, упал ли
  единственный стартовавший (фикс R2-minor: формулировку «не получил честного шанса» смягчаем — критерий =
  «бюджет пропустил кандидатов», а не «никто не стартовал»). Сообщение (фикс R1-COMP-info): режим бюджета +
  «0 завершено до исчерпания; N пропущено, M упало», по образцу `FitFailedError` (`exceptions.py:83`) —
  аддитивный `__init__`.
- **Достижимость ветки (фикс R2-minor):** при `mode="time"` ленивый старт (§2) гарантирует, что **первый**
  кандидат всегда стартует (на первом `exhausted` часы только запускаются) → «0 завершённых под time»
  возможно **лишь** если все стартовавшие **упали**, пока бюджет исчерпался (= смешанный случай). При
  `mode="trials"` `n_trials ≥ 1` (валидатор `gt=0`). Т.е. ветка `BudgetExhaustedError` достижима именно
  через смешанный случай — это **не** defensive-код для невозможного состояния (scope_constraints), путь явен.

### 4. Наблюдаемость (аддитивно в `SliceResult`)
`SliceResult += skipped_by_budget: tuple[str, ...] = ()` и `budget_exhausted: bool = False` (дефолты —
прогон в рамках бюджета, M3/M4 без изменений). Заполняются `run_slice`. Питают run-manifest (ADR-0033) и
`*_`-атрибуты фасада (FR-M5-6). `LeaderboardEntry` не трогается (аддитивно вне frozen-entry, как band).

### 5a. Кодирование «без границ» в `BudgetConfig` (фикс R1-major, 4 ревьюера)
Сегодня `BudgetMode = Literal["time","trials"]` и валидатор требует конкретный лимит → **unbounded
непредставим**, а `RunConfig.budget` дефолт = `BudgetConfig(n_trials=50)` (config.py:75, инертен, никто не
читает). M5 **начинает читать** `RunConfig.budget` → дефолт молча урезал бы прогон до 50 трайлов и манифест
был бы неправдив. **Решение (аддитивно):**
- `BudgetMode = Literal["none","time","trials"]` — добавлен **`"none"`** (unbounded). Валидатор
  `_check_mode_params` для `"none"` **не требует** лимитов, но **симметрично запрещает stray-лимиты**
  (фикс R2-minor): `mode="none"` + (`time_budget_s` или `n_trials` не `None`) → `ValueError` — иначе манифест
  показал бы противоречивое `{mode:"none", n_trials:50}`, реинтродуцируя ту самую неправдивость, которую
  фикс закрывает. `RunBudget(mode="none").exhausted` = всегда `False`.
- **`RunConfig.budget` дефолт меняется** `BudgetConfig(n_trials=50)` → **`BudgetConfig()` (mode="none")**.
  **Поведение-сохраняюще** (поле сегодня не читается — research §1.1), но это **смена дефолта frozen-
  exported-модели** → фиксируется здесь как осознанная (обновить round-trip-тесты `test_core_config`).
- Манифест пишет `budget` **как есть** (`{"mode":"none"}` для unbounded) — правдиво (ADR-0033 §4,
  NFR-M5-6); `budget=None` на фасаде ⇄ `BudgetConfig(mode="none")` в `RunConfig` ⇄ `RunBudget` inert ⇄
  `budget=None` в `run_slice`. Расхождения «манифест vs прогон» нет.

### 5. Публичный budget-API фасада + проводка
- **Фасад:** параметр **`budget: float | BudgetConfig | None = None`** (зеркало `cv: int | CVConfig |
  None`): `float` → `BudgetConfig(mode="time", time_budget_s=budget)`; `BudgetConfig` → как есть
  (trials/time/none); **`None` → `BudgetConfig(mode="none")`** (unbounded, §5a). Хранится verbatim в
  `__init__` (ADR-0011), резолв — в `fit`. **Инертный `time_budget`-параметр удаляется** (никогда не
  работал — pre-release tidy; поведенчески no-op, но **сигнатурно breaking**: `AutoML(time_budget=…)` даст
  `TypeError`, миграция `time_budget=x`→`budget=x`; impl-grep подтвердил: тесты на имя не завязаны —
  R1-COMP-info).
- **Проводка:** `fit` строит `RunConfig(budget=resolved_budget_config, …)` и при `mode!="none"` —
  `RunBudget(resolved, clock=time.perf_counter)` (через composition), передаёт `budget=` в `run_slice`
  (при `"none"` — `budget=None`, gate инертен). Резолвнутый `BudgetConfig` попадает в run-manifest
  (ADR-0033) — truthful. (Манифест сериализует **полный** `RunConfig`, включая `model_types`, — R1-CONS-info.)

## Последствия
- **Положительные:** дешёвый портируемый бюджет (планка FLAML/AutoGluon `time_budget`); graceful «лучшее
  на сейчас»; порт `Budget` НЕ меняется (адаптер считает трайлы через `consume`); аддитивно (бюджет
  опционален → M3/M4 без изменений); кооператив тестируем на фейковых часах.
- **Отрицательные/компромиссы:** overshoot ≤ один кандидат (один долгий `fit` перебирает бюджет — R-OVER,
  принято/задокументировано); time-бюджет **недетерминирован между машинами** (R-DET — контракт, см. §6);
  per-fold gate и hard-preempt отложены; memory-enforce отложен.
- **Влияние на слои:** `Budget`-порт — `core` (без изменений); **`BudgetConfig` — `core` (аддитивно:
  `mode="none"`, дефолт `RunConfig.budget`, guard stray-лимитов)**; `RunBudget`-адаптер — `adapters`; gate +
  `SliceResult`-поля + пост-цикловое правило + `BudgetExhaustedError.__init__` — `application`/`core`; сборка
  бюджета и проводка — `composition`/`facade`. **Artifact-поверхность НЕ трогается** (budget-исход в
  run-report, не в артефакте — R2). import-linter не нарушен (`run_slice` берёт порт; `time` только в
  адаптере). `ARTIFACT_VERSION` не меняется.

### 6. Контракт детерминизма + blast radius на артефакт (NFR-M5-2, фикс R1-DOM3 + R2)
- **`mode="trials"`/`"none"` — полностью детерминированы:** `RunBudget` не читает часы → тот же seed →
  тот же leaderboard/winner/**артефакт**/манифест (кроме абсолютных таймингов).
- **`mode="time"` — недетерминизм затрагивает НЕ только отчёт, но и АРТЕФАКТ.** Wall-clock решает, **какие
  кандидаты завершатся** → `candidates` → `leaderboard`/`best_model_id`/band/`refit_best` → сериализованный
  `FittedModel` (`model.joblib`/`leaderboard.json`). Тот же seed на двух машинах → **возможно разный
  winner/модель**. **Контракт явно фиксирует:** под `mode="time"` воспроизводимость **на уровне артефакта**
  (winner/leaderboard/model) **не гарантируется между машинами** — осознанный компромисс ради предсказуемого
  времени (документируется в публичных docs M9). `trials`/`none` — воспроизводимость сохраняется.
- **Само-раскрытие — в run-report, НЕ в артефакте (фикс R2-MAJ, option b):** budget-исход (`mode`,
  `budget_exhausted`, `skipped_by_budget`) и significance-режим живут в **run-report** (ADR-0033) — он и есть
  «как прошёл прогон». **Artifact-манифест НЕ трогается** (нет проводки `FittedModel.budget`/save/load —
  это было бы scope-creep; `ARTIFACT_VERSION`=1 не меняется). Симметрично: significance-провенанс — **тоже
  только в run-report** (ADR-0034/NFR-M5-6), чтобы провенанс прогона жил в одном месте, а не асимметрично
  (R2-completeness). Trade-off: загруженная в отрыве от run-report модель не само-раскрывает budget/
  significance — **принято** (run-report — спутник артефакта; полный provenance-в-артефакте → future, при
  нужде аддитивным ключом по M4-паттерну).

## Проверки
- `mode="trials", n_trials=k` при N>k кандидатах → ровно `k` **завершённых** в leaderboard, остальные в
  `skipped_by_budget`, `budget_exhausted=True` (фейковый бюджет, без обучения); упавший кандидат **не**
  тратит трайл (тест failure-под-trials).
- `mode="trials"`-исчерпание зависит от **числа** `consume`, не от значения `seconds` (R1-CA2/DOM5).
- `mode="time"` с малым бюджетом на фейковых часах → стоп после текущего кандидата (overshoot ≤ 1),
  `budget_exhausted=True`; **часы стартуют лениво** при первом `exhausted` (setup не биллится, R1-CA3).
- `mode="none"`/не задан → все кандидаты бегут, `skipped_by_budget=()`, `budget_exhausted=False`; манифест
  `budget={"mode":"none"}` (поведение M3/M4 не изменено) — существующий сьют зелёный.
- **0 завершённых под бюджетом** (`budget_exhausted` ∧ `skipped`≠∅, смешанный случай) → `BudgetExhaustedError`
  (режим+счётчики); **0 завершённых без бюджета** (все упали) → `FitFailedError` — правило различает (R1-blocker).
- ≥1 завершённый на исчерпании → `fit` успешен, `best_model_id_` из завершённых, band по завершённым;
  **`refit_best` победителя всегда выполняется** (не gated) → `predict` работает (R2).
- `BudgetConfig(mode="none", n_trials=50)` → `ValueError` (stray-лимит запрещён, R2); `BudgetConfig()` →
  `mode="none"` round-trip ок.
- Два прогона `mode="trials"` один seed → идентичные leaderboard/best_model_id/манифест/`predict(X)`
  (наблюдаемые, не байты `model.joblib`, R2-info) (NFR-M5-2).
- `lint-imports` 3/3 KEPT; `RunBudget` на фейковых часах (ленивый `t0` идемпотентен); `core` без новых
  тяжёлых импортов; `ARTIFACT_VERSION` не тронут (artifact-поверхность не меняется — budget в run-report).

## Impl-note (M5, 2026-06-09)
- `RunBudget` несёт публичный `mode`; `run_slice` читает его при `BudgetExhaustedError` через
  `getattr(budget, "mode", "budget")` — порт `Budget` не расширен (структурное чтение только для сообщения).
- `BudgetConfig.mode` дефолт переведён на `"none"` (а не отдельный дефолт у `RunConfig.budget`), поэтому
  `BudgetConfig()` = unbounded; `RunConfig.budget` = `default_factory=BudgetConfig`. Guard `"none"` запрещает
  stray `time_budget_s`/`n_trials` (memory_limit_mb не enforced ни в одном режиме — вне guard).
- Бюджет резолвится в фасаде (`_resolve_budget`/`_build_budget`), а не в `build_default_components`: budget —
  не ML-компонент, ему нужен clock-инъект на уровне composition-root; `build` остаётся про метрику/сплиттер/
  модели. Проверки — `test_facade.py` (caps/none/clone) вместо `test_build_components`.
