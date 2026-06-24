# ADR-0059 — Per-fold агрегация degenerate-block статистики

- **Статус:** Accepted (M6f, design-gate pending)
- **Драйвер:** DM-F2 (FR-FSF-3; NFR-FSF-4/5)
- **Связано:** ADR-0050 (structure-aware null), ADR-0054 (per-fold re-selection), ADR-0055 (time_window),
  ADR-0053 §3 (единый source-label блока), M6e-review **S3** (закрываемая находка).

## Контекст
M6e `null_block_stats` считает фрагментацию блоков **один раз на full-DEV** (`slice.py:391-405`). В режиме
`nested_per_fold` каждый outer-фолд переотбирает на fold-train, где строк меньше (`(K−1)/K · |DEV|`) ⇒ блоки
**мельче и вырожденнее**, чем на full-DEV. Full-DEV-метрика **занижает реальную per-fold фрагментацию** —
оптимистично для продукта «честно лучшая модель» (M6e-review S3, диспозиция «Day-2»). Плюс текущий
degenerate-count — **O(n_blocks·n)** (`slice.py:394`, маска по полному `y` на каждый блок; M6e-review O3).

## Рассмотренные варианты
1. **Оставить full-DEV-только.** — ❌ занижает фрагментацию, S3 не закрыт.
2. **Заменить full-DEV на per-fold.** — ❌ ломает существующую наблюдаемость/тесты; full-DEV всё ещё
   полезен как общий контекст. Несовместимо с аддитивностью.
3. **Аддитивные per-fold агрегаты рядом с full-DEV + векторизация счётчика.** — ✅ честно, аддитивно,
   обратносовместимо, попутно снимает O(n_blocks·n).

## Решение (Вариант 3)
### §1 Per-fold агрегаты
В per-fold цикле (`feature_compare._score_procedure`/`_per_fold_winner`) для каждого outer-фолда со
**структурными блоками** считать degenerate-долю на **fold-train** и агрегировать по фолдам в **аддитивные**
ключи `null_block_stats`:
- `per_fold_degenerate_mean: float` — средняя по фолдам доля вырожденных блоков;
- `per_fold_degenerate_max: float` — худший фолд;
- `per_fold_n_blocks_mean: float` — среднее число блоков на fold-train.
Full-DEV-ключи (`n_blocks/mean_block_size/degenerate_blocks/block_mode/block_window`) **сохранены без
изменений**. Per-fold-ключи присутствуют **только** при `arbitration_effective ∈ {nested_per_fold,
per_fold_partial_c5_inner}` и наличии структурных блоков (NFR-FSF-4; R-PFAGG: нет ложной per-fold честности
вне per-fold пути).

### §1a Канал возврата (явный плумбинг, уточнено после R1)
`null_block_stats` строится в `run_slice` (slice.py:391-411) на full-DEV **до** вызова `compare_features`,
а per-fold цикл живёт **внутри** `compare_features` → нужен явный обратный канал:
1. `CompareOutcome` (feature_compare.py:49-64) получает аддитивное поле
   `per_fold_block_stats: dict[str, float] | None = None`.
2. Затронуто **3 функции / 2 fixed-arity возврата** (уточнено после R2): `_score_procedure` (где живёт
   `groups[tr]` на fold-train, feature_compare.py:451) считает degenerate-долю **только по не-degraded
   фолдам** (degraded делают `continue` до блоков) через хелпер §2 и расширяет свой возврат-tuple
   per-fold-блок-статой; `_per_fold_winner` агрегирует **только статы стратегии-победителя** (НЕ всех
   сравниваемых — иначе double-count по стратегиям, R-PFAGG) и расширяет свой возврат-tuple;
   `compare_features` кладёт результат в `CompareOutcome.per_fold_block_stats` (только на per-fold ветке;
   иначе `None`). Тест `test_per_fold_degenerate_aggregated_across_folds` фиксирует **winner-агрегат**.
3. `run_slice` **после** получения `outcome` (slice.py:428+) **мёржит** `outcome.per_fold_block_stats` в
   локальный `null_block_stats` (до сборки `SliceResult`). Если `null_block_stats is None` (не было блоков)
   и per-fold-агрегаты есть — создаётся dict с per-fold ключами.
Без этого канала per-fold ключи физически не доходят до run-report (тест `*_in_report` не на что вешать).

### §2 Векторизованный degenerate-count (O(n)) — общий хелпер
Модуль-хозяин — `application/feature_selection.py` `_degenerate_counts(block_labels, y) -> int` (рядом со
`structure_labels` — единый источник block-семантики, ADR-0053 §3); импортируется И в `slice.py` (full-DEV),
И в `feature_compare.py` (per-fold). Оба — application, импорт внутрислойный (деп-граф не нарушен; одна
векторизация, без двух копий).
- Реализация: один проход — `np.add.at`/`np.bincount` подсчитывают per-block число уникальных классов
  (для бинарного «класс==1»: блок вырожден, если в нём все `y` равны), без `y[labels==b]`-маски на каждый
  блок (снимает O(n_blocks·n) из M6e-review O3).
- **Краевой случай (после R1):** 1-строчный блок (частый при `time_window`-densify, feature_selection.py:42)
  имеет 1 класс ⇒ `size<2` ⇒ **degenerate** — векторная реализация даёт ту же `1` на таких блоках. Тест-
  эквивалентность включает `time_window`-кейс с 1-строчными/переменными блоками, иначе «тождество»
  непроверяемо на самом нагруженном пути.

### §3 Источник блока
Per-fold блок берётся из **того же** `structure_labels`, что и null-permutation/significance на fold-train
(единый source-label, ADR-0053 §3) — никакого второго определения блока (R-PFAGG / консистентность).

### §4 Наблюдаемость и не-двойной-учёт
Per-fold-агрегаты — **отдельные именованные** ключи; run-report выводит их аддитивно рядом с full-DEV
(`run_report._feature_selection_report`). Full-DEV и per-fold **не суммируются** и не путаются (разные
ключи + разный смысл: общий контекст vs реальная per-fold фрагментация).

## Последствия
- **+** Честная метрика фрагментации, отражающая реальные per-fold условия; закрывает S3.
- **+** Снимает O(n_blocks·n) → O(n) (O3).
- **−:** добавляет per-fold подсчёт в горячий цикл — но O(n) на фолд, без рефитов (NFR-FSF-5).
- **−/риск R-PFAGG:** путаница full-DEV↔per-fold — снято раздельными ключами + тестом отсутствия per-fold
  ключей вне per-fold пути.
- **Не-объём:** агрегаты для `nested` (не per-fold) — не нужны (subset фиксирован на DEV, фрагментация =
  full-DEV); per-fold-перевзвешивание значимости по фрагментации — future (только наблюдаемость здесь).
