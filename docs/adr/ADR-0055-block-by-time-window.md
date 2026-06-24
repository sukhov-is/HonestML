# ADR-0055 — Блок-по-time-окну для структурного null (нерегулярные ряды)

- **Статус:** Принят (дизайн M6e; реализация — скил implementation). Питается SPIKE-M6e-validity.
- **Дата:** 2026-06-10
- **Драйверы:** DM-E2 (структурный null валиден под нерегулярными интервалами), DM-E3 (численное
  подтверждение). FR-FSE-5/6/7, NFR-FSE-4. **Уточняет ADR-0050** (structure-aware null) — не отменяет:
  rank-биннинг остаётся дефолтом для регулярных рядов.
- **Воркстрим:** M6e.

## Контекст
ADR-0050 §2/§5 строит структурный блок timeseries как **равно-СЧЁТНЫЙ** rank-блок:
`block = argsort(argsort(times)) // null_block_size` — `null_block_size` **строк** на блок. Допущение «фикс.
число строк ≈ фикс. окно времени» **верно лишь при ~равномерной частоте**; под нерегулярными интервалами
временной горизонт блока «плавает» (в плотной зоне блок = час, в разрежённой = месяц), intra-block
автокорреляция непостоянна → within-block null деградирует неравномерно. ADR-0050 §5 явно отнёс
**блок-по-окну → M6e** как честную границу. SPIKE-M6e-validity показал: баланс «подавление спурьёзного ↔
genuine-power» зависит от **состава** блока — однородный временной горизонт важен.

Сырые `times` (`Dataset.time()`) **уже** в scope на call-site `structure_labels` (`slice.py:322-323,373-377`);
текущий код их выбрасывает (оставляет лишь ранг). ⇒ блок-по-окну — **локальная** правка `times`-ветки, без
нового плумбинга в application.

## Рассмотренные варианты
1. **Оставить rank-биннинг (M6d).** Прост, но под нерегулярностью горизонт блока плавает — структурная
   валидность null/significance деградирует. Недостаточно для DM-E2.
2. **Авто-детект нерегулярности и неявный свитч.** Магия/непрозрачность; ломает воспроизводимость и
   труфулнес манифеста (что именно ран). Отвергнут.
3. **Явный режим биннинга opt-in (`block_mode`), окно по Δt, дефолт rank.** Прозрачно, аддитивно,
   back-compat. **Выбран.**

## Решение

### 1. Конфиг — аддитивный режим биннинга (`core/config.py`)
```python
class FeatureSelectionConfig(BaseModel, frozen, extra="forbid"):
    ...
    null_block_size: int = Field(50, ge=2)                              # M6d: rows per rank-block
    null_block_mode: Literal["rank", "time_window"] = "rank"           # M6e; "rank" = M6d-дефолт
    null_block_window: float | None = Field(None, gt=0)                # M6e: Δt окна для time_window
```
- Дефолт `"rank"` ⇒ **тождественно M6d**. Поля **внутри** `FeatureSelectionConfig` ⇒ `fs=None` → fingerprint
  M6b (NFR-FSE-7).
- **Жёсткая ошибка — в `_check_config`** (→ `ConfigError`, pydantic-валидатор бросает): `null_block_mode=
  "time_window"` **требует** `null_block_window` (иначе нет Δt). **Dead-config WARNING — в composition, не в
  валидаторе** (фикс R1-consistency): при `null_block_mode="rank"` заданный `null_block_window` **игнорируется**
  с **WARNING** при resolve (pydantic-валидатор только raise'ит, WARNING'и dead-config живут в composition —
  наследует паттерн M6d ADR-0052 §1). `group`-схема игнорирует оба (блок=группа).

### 2. Биннинг по окну (`application/feature_selection.py::structure_labels`)
Расширить сигнатуру: `structure_labels(groups, times, block_size, *, mode, window)`. Ветка `times is not None`:
```python
if mode == "time_window":
    raw = ((times - times.min()) / window).astype(np.int64)   # окно по фактическому Δt
    _, dense = np.unique(raw, return_inverse=True)             # уплотнить в 0..k-1 (пустые окна выброшены)
    return dense.astype(np.int64)
# mode == "rank" (M6d):
rank = np.argsort(np.argsort(times, kind="stable"), kind="stable")
return (rank // block_size).astype(np.int64)
```
- **Граница-доверие валидатору (фикс R1-clean-arch-m5):** `structure_labels` **полагается** на config-валидатор
  (`null_block_window is not None` под `time_window`) и **не** перепроверяет `window` (никакой защитной ветки
  `if window is None` — это нарушило бы `scope_constraints` «не валидировать невозможные состояния, доверять
  внутреннему коду»). Валидация — на границе (`_check_config`), трансформ — чистый.
- **Уплотнение меток обязательно** (R-WINEMPTY): нерегулярные ряды дают пустые окна; `np.unique(...,
  return_inverse=True)` отдаёт плотные 0..k−1 ⇒ downstream-контракт (плотный int64 label-per-row) **неизменен**,
  адаптеры/significance без правок (NFR-FSE-4). `for g in np.unique(groups)` в ранкере итерирует только
  непустые блоки.
- **Контракт downstream неизменен** ⇒ `NullImportanceRanker._permute_target` и block-bootstrap `block_index`
  значимости работают как есть.

### 3. Единый источник — рипл в significance автоматически (ADR-0053 §3, FR-FSE-6)
`structure_labels` — **один** per-row массив и для null-перестановки (срез `[train_idx]`), и для block-bootstrap
`block_index` арбитража (срез `[mask]`, `feature_compare.py:337`). Смена `block_mode`/`window` рипплит в
bootstrap-блоки significance **консистентно** (один источник; срезы одного массива). **Leaderboard-band**
(CV-fold-id, `slice.py:329-336`) — иное определение, **не затрагивается** (намеренно, ADR-0053 §3 R2-F-R2-4).

### 4. Наблюдаемость (FR-FSE-7, аддитивно)
`null_block_stats` дополняется `block_mode` и `block_window` (или `block_size` для rank), сохраняя
`n_blocks`/`mean_block_size`/`degenerate_blocks`. Degenerate-WARNING при >50% (`slice.py:387-392`)
**переиспользуется**: под `time_window`/нерегулярностью 1-строчных/degenerate-class окон больше (R-WINEMPTY) →
WARNING становится более нагруженным, что и нужно (сигнал «окно слишком узкое»). Версии не бампаются.
- **Граница наблюдаемости при `nested_per_fold` (фикс R2-completeness):** `null_block_stats` считается **один
  раз на всём DEV** (`slice.py:378-392`), **до** per-fold цикла. Под `nested_per_fold`+`time_window` реальная
  null-перестановка идёт на срезе `groups[tr]` — глобально-уплотнённая метка, срезанная к outer-train, **фрагментирует
  блоки** (здоровый на DEV блок может стать 1-строчным в фолде → больше identity-перестановок). ⇒ `null_block_stats`
  отражает **full-DEV**, не per-fold; документируется явно, а per-fold честность отражается `fold_subset_jaccard`
  (ADR-0054). (Агрегация degenerate-stats по фолдам — возможное Day-2-уточнение.)

### 5. Честная граница (нормативный владелец численного вывода validity, фикс R1-completeness)
- **ADR-0055 — нормативный владелец численной границы φ** (FR-FSE-8/NFR-FSE-8): SPIKE-M6e-validity численно
  подтвердил, что within-block null подавляет ложный KEPT спурьёзного признака сильнее uniform, и преимущество
  **значимо при сильной автокорреляции** (нагляднее при φ≈0.9), при экстремальной (φ≈0.95) метод **консервативен**
  (теряет genuine-power из-за коллапса SNR). Это **принятое документированное свойство** within-block null
  (мотивирует блок-по-окну: однородный горизонт улучшает баланс), а не дефект. Load-bearing evidence остаётся
  **group-решающим** (M6d) + структурный вывод; ts — численно подтверждён направленно.
- `window` — в единицах `times`; пользователь обязан знать временную шкалу. Плохой `window` (слишком узкий →
  много degenerate; слишком широкий → блок ≈ весь ряд, null почти не структурен) ловится degenerate-WARNING и
  `n_blocks` в `null_block_stats`. rank остаётся дефолтом — `time_window` для тех, у кого ряд нерегулярен.
- `time_window` не «исправляет» SNR-границу при экстремальной автокорреляции (SPIKE-M6e-validity §3) — он лишь
  делает временной горизонт блока **однородным**; power-vs-conservativeness баланс остаётся свойством данных.

## Последствия
- (+) Структурный null/significance валиден под нерегулярными рядами (однородный временной горизонт блока);
  локальная правка одной ветки `structure_labels`; downstream-контракт неизменен; полный back-compat (rank
  дефолт); единый источник ⇒ null и significance не расходятся.
- (−/компромисс) Требует знания временной шкалы (`window`); больше degenerate-окон под узким окном (ловится
  WARNING); не снимает SNR-границу метода (§5).
- **Влияние на слои:** конфиг — `core`; биннинг — `application` (`structure_labels`); адаптеры/significance —
  **без правок** (контракт неизменен). `import-linter` 3/3 KEPT.

## Проверки
- Нерегулярный ряд: `time_window` → блоки равной **длительности** (разной мощности); `rank` → равной
  **мощности** (разной длительности); метки плотны 0..k−1 при пустых окнах (FR-FSE-5, NFR-FSE-4).
- `block_mode="rank"` (дефолт) → метка тождественна M6d (эквивалентность-тест); `group`-схема игнорирует режим.
- Та же метка идёт в `NullImportanceRanker` и в `equivalence_band` (срезы одного массива, FR-FSE-6);
  leaderboard `block_index` (fold-id) не меняется.
- `time_window` без `null_block_window` → `ConfigError`; `null_block_window` при `rank` → WARNING.
- `null_block_stats` несёт `block_mode`/`block_window`; degenerate-WARNING на сконструированном узком окне.
