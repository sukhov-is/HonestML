# ADR-0053 — Significance-aware winner арбитража: Occam-тай-брейк «неотличим от лучшего → компактнейший»

- **Статус:** Принят (реализован в M6d, 2026-06-10; `_nested_winner`→`Candidate`/`equivalence_band`, `block_index`
  из `structure_labels`, `winner_rule` в run_report). **Зависит от ADR-0052** (нужен OOF-вектор nested-арбитража).
- **Дата:** 2026-06-10
- **Драйверы:** DM-H4 (честный winner вместо argmax по шуму). FR-FSH-8, NFR-FSH-1/3. Наследует
  `equivalence_band`/`select_best`/`Candidate`/`SelectionPolicy` (ADR-0007/0026), `SignificanceTest`
  block-bootstrap (M1-контракт, M4 «честная версия»).
- **Воркстрим:** M6d.

## Контекст
M6c-арбитраж — **голый argmax** `arb_score` (`feature_compare.py:290`, строгий `>`, ничьи → первая стратегия).
При близких стратегиях это выбор «по шуму»: разница в третьем знаке метрики не значима, но argmax детерминированно
берёт одну. Дифференциатор продукта — **«честно лучшая модель»**: среди статистически **неотличимых** FS-стратегий
честно выбрать **простейшую** (меньше признаков), а не случайно-лучшую по зашумлённому скору. Механизм **уже есть**
— `equivalence_band` + Occam tie-break (`selection_policy.py:86-199`), используемый для выбора модели; он **инертен**
для арбитража, пока нет OOF-вектора (single-holdout даёт лишь скаляр). ADR-0052 (nested) даёт OOF-**вектор** на DEV
→ band становится применим.

## Решение

### 1. Условие активации (back-compat)
Significance-aware winner включается **только** при **обоих**: `arbitration="nested"` (есть OOF-вектор) **и**
`significance != "off"` (включён реальный `BootstrapSignificanceTest`). Иначе — **argmax** как M6c:
| `arbitration` | `significance` | winner арбитража |
|---|---|---|
| `holdout` | любой | **argmax** (M6c; single-holdout — только скаляр, band неприменим) |
| `nested` | `off` (`NoSignificanceTest`) | **argmax** (band пуст, `equivalent→False`) |
| `nested` | вкл. | **band + Occam tie-break** (компактнейший среди неотличимых) |

`NoSignificanceTest.equivalent→False` ⇒ band = {anchor} ⇒ `select_best` сводится к argmax **автоматически** —
отдельной ветки «off» не нужно, переиспользуется существующая семантика.

### 2. Маппинг стратегий → `Candidate` и вызов `select_best` (`application/feature_compare.py`)
nested-арбитраж (ADR-0052 §2) уже даёт на каждый subset: усреднённый `arb_score` **и** OOF-вектор предсказаний
на DEV (через `make_oof_scorer`, который возвращает pooled-OOF). Обернуть:
```python
candidates = [
    Candidate(id=name, score=arb_score[name], n_features=len(subset[name]),
              oof_pred=oof_vec[name], oof_mask=oof_mask[name])
    for name in strategies
]
winner = select_best(candidates, policy, significance_test, y_dev,
                     block_index=block_index, sample_weight=sw_dev)
```
- `policy.greater_is_better` — из метрики; `tie_break = ("n_features","stability","train_time")` (Occam:
  **компактность** → стабильность → скорость, как для выбора модели). **`n_features` первым** ⇒ среди неотличимых
  побеждает subset с **меньшим** числом признаков (DM-H4, дифференциатор).
- **Полное равенство tie-break (фикс R2-C2 — band=все, равный `n_features`/stability/speed):** если несколько
  кандидатов совпадают по **всем** ключам, финальный fallback — **argmax-anchor** (лучший по `arb_score`), затем
  лексикографически по имени стратегии. Это детерминированная гарантия (наследует `rank`-стабильность
  `selection_policy.py:83` — сортировка с `c.id` тай-брейком). Случай «band=все» валиден (все стратегии
  статистически равны) → берётся компактнейший, при равенстве — anchor.
- `y_dev` — таргет DEV, выровненный по общей OOF-маске (band владеет единой маской пересечения, R2-B1/B2 уже
  решён в `equivalence_band`). `oof_pred`/`oof_mask` — из **sibling-scorer** `make_oof_vector_scorer` (ADR-0052 §2,
  возвращает `(score, oof_pred, oof_mask)`); **metric-ready** вектор (P(positive)/proba/класс — как `Candidate`
  ожидает, ADR-0026 §3) строится теми же `_fold_proba`/`project_for_metric`, что leaderboard. (Голый
  `make_oof_scorer->float` для этого недостаточен — см. ADR-0052 §2.)

### 3. `block_index` для block-bootstrap — единый источник со scheme (NFR-FSH-1, фикс R1-F8)
`BootstrapSignificanceTest` (M4) использует **блочный**/кластерный bootstrap. `block_index` арбитража берётся из
**той же** централизованной `_structure_labels(dataset, scheme, null_block_size)` (ADR-0050 §3), что и
null-перестановка: `timeseries` → ранг-биннинг `Dataset.time()`; `group` → `Dataset.groups()`; i.i.d. → `None`
(обычный bootstrap). **Единый источник** гарантирует, что дефект/уточнение семантики блока правится в **одном**
месте и не расходится между null и значимостью (каскад R1-F8 закрыт). ⇒ тест значимости арбитража **уважает** ту
же структуру зависимости, что и null/CV.
- **Инвариант проводки (фикс R2-F-R2-3):** `_structure_labels` — **per-row over full DEV**; null-путь срезает её
  `[train_idx]` (train-часть фолда), band-путь — общей OOF-маской `[mask]`. Это разные срезы **одного** per-row
  массива → консистентны по построению; реализация **не** строит блок «по позиции в срезе» (это был дефект
  R1-B2).
- **Почему структурный блок, а не fold-id арбитражных фолдов (фикс R2-F-R2-4):** leaderboard-band использует
  `block_index` = CV-fold-id (M4). Арбитражная band берёт **структурный** блок (time-rank/group), а **не** fold-id
  арбитражных фолдов, **сознательно**: block-bootstrap должен ресэмплить по **истинной** структуре зависимости
  данных (автокорреляция/группа), а не по случайному арбитражному разбиению — иначе CI значимости не уважает
  зависимость. Это **иная** дефиниция блока, чем у leaderboard-band, и это намеренно.

### 4. Наблюдаемость (FR-FSH-9, аддитивно)
`run_report["feature_selection"]` (при nested+significance) аддитивно несёт band-исход: `band_members`
(неотличимые от лучшего), `winner_by_tiebreak` (winner ≠ anchor по argmax), `band_unstable`
(anchor-чувствительность) — из `BandResult`. **`winner_rule` (фикс R2-C6):** явный ключ ∈ `{argmax_holdout,
argmax_band_empty, band_tiebreak}` — различает «argmax т.к. holdout-режим», «argmax т.к. band пуст
(significance off / одна стратегия)», «Occam-тай-брейк внутри band». Версии не бампаются; старые парсеры
игнорируют новые ключи.

### 5. Честная граница: остаточный winner's curse (фикс R1-F7)
Anchor выбирается argmax по `arb_score`, а band строится `equivalence_band`'ом на **тех же** OOF-векторах →
точечная оценка и оценка разброса **не независимы**: bootstrap ресэмплит те же OOF-остатки, что определили
лидера. ⇒ band **смягчает**, но **не устраняет** winner's curse (anchor систематически на удачной стороне шума
своего OOF, band вокруг него слегка смещён). Это **документированное ограничение**, не дефект: band всё равно
строго консервативнее голого argmax (M6c), и Occam-тай-брейк выбирает компактнейший **из неотличимых**, что
честнее, чем argmax по третьему знаку. Полная развязка (независимый held-out для оценки значимости) — Day-2/M6e.
Формулировка «честный winner» — в смысле «честнее argmax», без претензии на устранение curse.

## Последствия
- (+) Честнее argmax: среди статистически равных FS-стратегий — **простейшая** (Occam), а не argmax по шуму;
  переиспользованы `equivalence_band`/`select_best` (core без изменений); block-bootstrap уважает структуру;
  полный back-compat (argmax при holdout/off).
- (−/компромисс) Работает только в nested-режиме (требует OOF-вектор от sibling-scorer, ADR-0052 §2) —
  задокументировано; добавляет bootstrap-стоимость (мала относительно N×K фитов); требует включённого
  `significance`; **остаточный winner's curse** (score и band на одном OOF) — смягчён, не устранён (§5).
- **Влияние на слои:** маппинг/вызов — `application`; `Candidate`/`equivalence_band`/`select_best`/порт
  `SignificanceTest` — `core` (без изменений, переиспользование); резолв теста/политики — `composition`
  (уже есть, ADR-0007/M4). `import-linter` 3/3 KEPT.

## Проверки
- `nested`+`significance=on`, две статистически **неотличимые** стратегии → побеждает **меньший** subset (тест
  на сконструированных OOF-векторах); три стратегии, одна явно лучше → побеждает она (band = {anchor}).
- `holdout` или `significance=off` → **argmax** (back-compat, тест эквивалентности с M6c).
- `block_index` для `timeseries`/`group` передан и согласован со scheme (структурный тест); `y_dev`/`oof_pred`
  выровнены по общей маске (наследует тесты `equivalence_band`).
- `run_report` несёт `band_members`/`winner_by_tiebreak` при nested+significance; отсутствуют при holdout/off.
