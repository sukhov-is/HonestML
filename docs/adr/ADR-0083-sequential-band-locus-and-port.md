# ADR-0083: Band-выбор числа признаков в sequential — локус, контракт порта, переиспользование

- **Статус:** Proposed
- **Дата:** 2026-06-19
- **Драйверы:** D-1 (честный выбор по умолчанию), D-2 (переиспользование band-машинерии),
  D-3 (Humble Object / анти-утечка), D-6 (single минует арбитраж). Источник: FR-1,
  FR-2, FR-6, NFR-2, NFR-3. Наследует ADR-0026/0046/0052/0053.

## Контекст
`SequentialSelector.select` возвращает один субсет — argmax OOF-скора по обратной
траектории ([feature_selectors.py:47-61](../../../../src/honestml/adapters/feature_selectors.py#L47)).
Чтобы выбрать наименьший статистически-неотличимый субсет (band+Occam), нужен **пул**
кандидатов (несколько субсетов разного размера), а band-машинерия
(`equivalence_band`/`select_best`) требует на кандидата **OOF-вектор**, `oof_mask`,
`y`, `block_index`, `policy`, `SignificanceTest`
([selection_policy.py:84-142](../../../../src/honestml/core/selection_policy.py#L84)).
Силы: (1) адаптер обязан остаться Humble Object — видеть только индексы и скалярный
`score_subset`, не сырые строки/OOF (ADR-0046); (2) статистика и анти-утечка живут в
`application` (ADR-0044/0053); (3) `compare_features` при одной стратегии уходит в
`_compare_single` **до** nested-арбитража ([feature_compare.py:959](../../../../src/honestml/application/feature_compare.py#L959)),
значит band нельзя вешать на `_nested_winner`.

## Рассмотренные варианты
1. **Band внутри адаптера (расширить порт vector-scorer).** `score_subset` отдаёт
   `(score, oof, mask)`, адаптер сам зовёт `equivalence_band`. — **Минусы:** ломает
   Humble Object (адаптер видит OOF-векторы), тянет `core.selection_policy` и
   `SignificanceTest` в адаптер, нарушает локус анти-утечки. Отвергнут.
2. **Адаптер возвращает траекторию; band — в `application` (`_select_one`).**
   `select` возвращает упорядоченную траекторию субсетов (жадный спуск на **скалярном**
   `score_subset`); application скорит каждую точку `make_oof_vector_scorer`, оборачивает
   в `Candidate`, зовёт `select_best`. — **Плюсы:** Humble Object цел (адаптер видит
   только индексы), band в application как в ADR-0053, точный образец — `no_selection_gate`
   ([feature_compare.py:99-169](../../../../src/honestml/application/feature_compare.py#L99)).
   **Выбран.**
3. **Инжектить «выбиратель» `choose: Callable[[list[subset]],subset]` в `select`.**
   Адаптер делает спуск и зовёт инъекцию. — Возврат не меняется, но адаптер
   оркестрирует прикладной выбор через колбэк (мутный SRP), и патиенс-стоп остаётся
   связан с band. Отвергнут в пользу (2) как менее чистый.

## Решение
Принят **вариант 2**.

### 1. Контракт порта (минимальная правка, без shim)
`FeatureSubsetSelector.select` и `SequentialSelector.select` возвращают **траекторию**
— `tuple[tuple[int, ...], ...]`: упорядоченную последовательность посещённых субсетов
строго убывающего размера (полный набор → … → пол). Жадный выбор «какой признак
выкинуть» по-прежнему по **скалярному** `score_subset` (Humble Object). Единственная
реализация (`SequentialSelector`) и единственный потребитель (`_select_one`) правятся
вместе — без флагов/легаси (CLAUDE.md: меняем код, не плодим shim). `best_keep`/argmax
из адаптера **удаляется** — выбор финального субсета переезжает в application.

### 2. Локус band — `_select_one` (общий для single и compare)
В `_select_one` при `isinstance(strategy, FeatureSubsetSelector)`
([feature_compare.py:235](../../../../src/honestml/application/feature_compare.py#L235)):
1. строим скалярный `score_subset = make_oof_scorer(...)` (жадный спуск, как сейчас);
2. `trajectory = strategy.select(..., score_subset=...)`;
3. строим `vector_scorer = make_oof_vector_scorer(...)` на **тех же** фолдах;
4. на каждый субсет траектории — `score, oof, mask = vector_scorer(subset)`;
   `Candidate(id=<size-key>, score, n_features=len(subset), oof_pred=oof, oof_mask=mask)`;
5. `winner = select_best(candidates, policy, significance_test, y, block_index=groups,
   sample_weight=sw)`; возвращаем субсет победителя.

Это зеркало `_nested_winner`/`no_selection_gate`. `core.selection_policy` уже
импортируется в `feature_compare` ([feature_compare.py:31](../../../../src/honestml/application/feature_compare.py#L31))
(нужно дополнить импорт `select_best` — он там ещё не значится) — новых межслойных рёбер нет.

**Траектория — ВНУТРЕННЯЯ для `_select_one` (фикс RIPPLE-007).** Адаптер возвращает
траекторию, но `_select_one` потребляет её немедленно и резолвит в **один** субсет.
Поэтому контракт «вниз по стеку» не меняется: `_compare_single`/`_compare_holdout`/
`_compare_nested`/`_compare_per_fold` и `_nested_winner`/`_score_procedure` по-прежнему
получают от `ctx.select` **единичный** субсет (`tuple[int,...]`). Смена типа возврата
порта (на траекторию) видна **только** паре «адаптер ↔ `_select_one`»; дальше ряби нет.
Единственное дополнительное «эхо» — наблюдаемость: `ctx.select`/`_select_one` для
wrapper-ветки отдают пару `(subset, BandResult | None)`, и эту `BandResult` родительский
`_compare_*` кладёт в `CompareOutcome.seq_band` (ADR-0086). Для ranker-ветки —
`(subset, None)`. Затронутые наблюдаемостью места перечислены в ADR-0086 §1.

### 3. Проводка `significance_test`/`policy` в `_CompareCtx` — БЕЗУСЛОВНО (фикс WIRING-006/C-006)
`_CompareCtx` ([feature_compare.py:591](../../../../src/honestml/application/feature_compare.py#L591))
получает поля `significance_test: SignificanceTest | None` и `policy: SelectionPolicy
| None`; `compare_features` (уже принимает оба аргумента,
[feature_compare.py:929-930](../../../../src/honestml/application/feature_compare.py#L929))
кладёт их в ctx; `ctx.select`/`_select_one` получают их как параметры. `groups`
(структурная метка для `block_index`) уже протягивается в `ctx.select` как `grp`.

**Важно:** проводка **безусловна** — `significance_test`/`policy` идут в `_CompareCtx`
для **всех** режимов, в т.ч. single-strategy. Сейчас они инжектятся только в nested-ветке
([feature_compare.py:980-983](../../../../src/honestml/application/feature_compare.py#L980)),
а `_compare_single` ([feature_compare.py:674-692](../../../../src/honestml/application/feature_compare.py#L674))
их не видит. Поскольку one-strategy `sequential` уходит именно в `_compare_single` (D-6),
без безусловной проводки band там **молча не активируется**. ⇒ Composition обязан
передавать их в `compare_features` во всех путях (в Components они уже строятся,
[build.py:173-181](../../../../src/honestml/composition/build.py#L173)); `assert` в
nested-ветке остаётся, но это не единственная точка получения.

### 3b. Точные точки правки composition (фикс WIRING-UNSPEC)
Риск «тихого no-op» в single-strategy реален, поэтому фиксируем где именно:
- `compare_features` уже принимает `significance_test`/`policy`
  ([feature_compare.py:929-930](../../../../src/honestml/application/feature_compare.py#L929)),
  но **вызывается** из `run_slice` (composition-провязка): этот вызов обязан передавать
  их **всегда** (не только когда арбитраж nested). В Components они уже построены
  ([build.py:92-94,173-181](../../../../src/honestml/composition/build.py#L92)).
- `assert policy is not None and significance_test is not None`
  ([feature_compare.py:980-983](../../../../src/honestml/application/feature_compare.py#L980))
  остаётся как инвариант **nested**-ветки, но это больше **не единственная** точка их
  использования: `compare_features` кладёт оба в `_CompareCtx`, и `_compare_single`/
  `ctx.select` берут их оттуда. Для off-режима значение — `NoSignificanceTest` (не None).

### 3a. Взаимодействие с ADR-0054 (nested_per_fold) — band на уровне фолда
Поскольку band живёт в `_select_one`, а per-fold реселекция (ADR-0054) вызывает тот же
`_select_one` внутри каждого внешнего фолда, band по траектории **автоматически**
применяется на каждом фолде при `sequential` под `nested_per_fold`. seed на фолд —
существующий `_strategy_fold_seed` (blake2b, ADR-0054 §4); `block_index` на фолд —
срез `groups[tr]` (тот же per-row источник, §2 ADR-0085). Это **наследуемое** поведение,
кода сверх §2/§3 не требует; фиксируется как явный инвариант (а не сюрприз).

### 4. Условие активации — на существующем тумблере `significance`
Band активен ⇔ `significance != "off"` (передан реальный `BootstrapSignificanceTest`).
При `off` composition передаёт `NoSignificanceTest` ⇒ `equivalent→False` ⇒ band =
{anchor} ⇒ `select_best` сводится к argmax — **без отдельной ветки** (FR-2).
В отличие от ADR-0053, **nested-режим не требуется**: OOF-вектор берётся прямо с
селекционных фолдов (детали и граница честности — ADR-0085). Новых конфиг-полей нет
(D-5); раздельный рычаг «band-для-sequential» — возможное future, не сейчас.

## Последствия
- **Положительные:** убирается последний голый argmax в FS; полное переиспользование
  band-ядра (core без правок); единая точка для single+compare; Humble Object и
  анти-утечка целы; контракт-change off-пути отсутствует.
- **Отрицательные / компромиссы:** ломающая смена типа возврата порта (контролируемо:
  1 реализация + 1 call-site, R-PORT); band висит на глобальном `significance`
  (нет раздельного управления, D-5); дефолт-ON меняет исход sequential по умолчанию
  (R-DEFAULT, наблюдаемо). Остаточный winner's curse — ADR-0085 §5.
- **Влияние на слои:** `core/ports/feature_subset_selector.py` — смена возврата
  (порт); `adapters/feature_selectors.py` — траектория вместо argmax; `application/
  feature_compare.py` — band в `_select_one` + поля `_CompareCtx`; `core.selection_policy`
  — **без изменений**. import-linter `usecases-independent-of-adapters` — KEPT.

## Проверки
- FR-1: на сконструированных OOF меньший субсет неотличим от пика → выбран меньший
  (`tests/unit/test_feature_selection.py` / новый band-кейс).
- FR-2: `significance="off"` → субсет идентичен текущему argmax (regression-тест).
- NFR-2: адаптер получает только индексы (наследник `test_feature_selectors.py:60-68`);
  permutation-тест анти-утечки.
- NFR-3: `lint-imports` — контракт KEPT.
