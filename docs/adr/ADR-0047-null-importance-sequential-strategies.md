# ADR-0047 — Стратегии `null_importance` (ranker) и `sequential` (subset-selector): методология + анти-ликедж

- **Статус:** Accepted (реализован в M6c, 2026-06-10; `NullImportanceRanker`/`SequentialSelector`, ts/group→`ConfigError`)
- **Дата:** 2026-06-10
- **Драйверы:** DM-C2 (расширение каталога), DM-C3 (анти-ликедж wrapper), DM-C5 (стоимость), DM-C6 (детерминизм).
  FR-FSC-2/3, NFR-FSC-1/3/5/6. Наследует OOF-спайн/нормировку (ADR-0044), порты (ADR-0046), `crossfit_encode`-паттерн.
- **Воркстрим:** M6c.

## Контекст
M6b отложил `null_importance` и `sequential` (ADR-0044 §2). research §3 установил их **port-fit**: первая — за
`FeatureRanker`, вторая — за `FeatureSubsetSelector` (ADR-0046). Нужна **методология** обеих, дешёвая (extra-free)
и с честным анти-ликеджем.

## Решение

### 1. `null_importance` — за портом `FeatureRanker` (`adapters/feature_rankers.py`)
Перестановочный фильтр против случайного фона. На обучающей матрице фолда (`fit⊕es`):
```
imp_real = ranker_model.fit(x, y).feature_importances              # ExtraTrees (M6b рэнкер-модель)
for r in range(n_runs):                                            # перемешать ТАРГЕТ обучающей части
    y_perm = rng.permutation(y)                                    # rng = RandomState(random_state + r)
    null[r] = ranker_model.fit(x, y_perm).feature_importances
score = imp_real - np.percentile(null, p, axis=0)                  # сигнатура actual − null_p (p=95)
```
- **Выход — per-feature score-вектор** (знаковый margin) → штатный `rank()→ndarray`, штатный спайн/cutoff.
  Знаковый ⇒ проходит `_normalize_fold` как pass-through (без L1, как `random_probe`); `auto_threshold = 0.0`
  (бьёт фон). Параметры: `n_runs` (дефолт **30**; консервативно снижен ради цены, §3 — **уточнено R1-A13**:
  `null_percentile=95` на 30 семплах = ~28-я порядковая статистика, грубовато → гранулярность хвоста растёт с
  `n_runs`; 30 — консервативный баланс цена/стабильность, пользователь может поднять), `null_percentile` (дефолт
  95). **Рэнкер-модель**, не кандидаты (estimator-agnostic, R-FS-RANKER-MODEL).
- **Анти-ликедж — как `importance` (зеркало M6b):** фит **только** на `fit⊕es`-части фолда; перестановка
  затрагивает **обучающий** таргет (создаёт нулевую гипотезу), `test`-часть фолда не используется в `rank`. →
  property: перестановка таргета в **test**-части фолда **не меняет** `imp_f` (рэнкинг из `fit⊕es`).
- **Ограничение перестановки → fail-loud (фикс R2-MUST-FIX-1, усилено относительно R1-A4):** равномерная
  `rng.permutation(y)` предполагает **обмениваемость** таргета. На **timeseries** (автокорреляция) и **group**
  (внутригрупповая зависимость) она разрушает структуру → нулевое распределение нереалистично, `null_importance`
  **недооценит** важность лаговых/групповых признаков (ошибка II рода). **Критично:** в отличие от wrapper-оптимизма
  `sequential` (снят внешним holdout, ADR-0029), это смещение **не имеет корректирующего якоря** — оно
  **проникает прямо** в замороженный subset → молчаливое выбрасывание реально важных признаков. Поэтому
  **не WARNING, а `ConfigError`**: при `scheme∈{timeseries, group}` выбор `null_importance` (в `strategy` или
  `compare`) **отклоняется** (guard, как прочая FS-валидация). Метод для этих scheme — **M6d** (structure-aware
  перестановка внутри временных блоков / групп через `Dataset.groups()`). Для i.i.d. (disjoint/stratified) — штатно.
- Детерминирован при фикс. `random_state` (seeded `RandomState`); `import honestml` не тянет внешних пакетов.

### 2. `sequential` (backward-elimination) — за портом `FeatureSubsetSelector` (`adapters/feature_selectors.py`)
Жадная обёртка: старт с полного набора, на каждом шаге **уронить** признак, чьё удаление лучше всего улучшает
(или меньше всего ухудшает) **инъектируемый** OOF-скор; стоп — плато (нет улучшения `patience` шагов) или
`min_features`:
```
keep = list(range(n_features)); best = score_subset(keep)
while len(keep) > min_features:
    cand = argmax_j score_subset(keep \ {j})                       # пробует удалить каждый оставшийся
    if score_subset(keep\{cand}) < best - tol and stalled: break   # плато → стоп
    keep.remove(cand); best = max(best, score_subset(keep))
return tuple(sorted(keep))
```
- **Политика — в адаптере**, leakage-критичный `score_subset` — **инъекция из `application`** (ADR-0046 §1): фит
  на `fit⊕es`, скоринг `test`-части, агрегат по фолдам. Адаптер видит **только скаляр** → не обучается на тесте.
- **Граница честности (R-FSC-SEQ-LEAK, честно, без overclaim):** per-fold фит/скоринг **чисты** (фит и оценка
  на **разных** частях фолда — нет train-on-test). **Но** жадный выбор **оптимизирует агрегированный OOF-скор** →
  это **wrapper-оптимизм**: качественно **иной** род оптимизма, чем у ранжеров (ранжер сортирует важности и **не**
  оптимизирует CV-метрику; wrapper — оптимизирует, как HPO/выбор модели). **Не** утверждаем эмпирически, что он
  *по магнитуде* сильнее (это не измерено spike'ом, фикс R1-A14) — лишь что он **присутствует** там, где у
  ранжеров его нет. Перестановка `test`-таргета **изменит** скор шага (в отличие от ранжеров) → property для
  `sequential` иной: «per-fold scorer фитит только на `fit⊕es`» + «внешний holdout (ADR-0029) несмещён (отбор
  только на DEV)». Inner-оптимизм **снят внешним holdout**; арбитраж — на **независимой** поверхности (ADR-0048).
- Стоимость `O(n_features² · n_folds)` (worst-case) — opt-in, WARNING (§3); детерминирован при seed; floor ≥1.

### 3. Стоимость и предупреждения (NFR-FSC-3)
| Стратегия | Фитов рэнкер-модели / scorer | Замечание |
|---|---|---|
| `importance`/`random_probe` (M6b) | `n_folds × 1` | дёшево |
| `null_importance` | `n_folds × (1 + n_runs)` | ~`5×31` при n_runs=30 → WARNING с оценкой |
| `sequential` | `≈ O(n_features² · n_folds)` `score_subset`-вызовов, каждый = `n_folds` фитов | дорого → WARNING; cap по `min_features` |
Все — **пре-процессинг вне budget trials** (как M6b); включение дорогой стратегии логирует WARNING с числом
ожидаемых фитов. Внешних зависимостей нет (sklearn ExtraTrees + numpy).

## Последствия
- (+) Каталог расширен двумя дешёвыми (extra-free) стратегиями за **правильными** портами; анти-ликедж
  ранжирующей `null_importance` доказуем как M6b; wrapper `sequential` честно ограничен (per-fold чист, оптимизм
  снят holdout); детерминизм.
- (−/компромисс) `null_importance` дороже (n_runs-множитель); `sequential` несёт wrapper-оптимизм и `O(n²)` —
  принято, документировано, opt-in, арбитраж на независимой поверхности.
- **Влияние на слои:** обе стратегии — `adapters`; нормировка/`score_subset`/спайн — `application`; порты — `core`.

## Проверки
- `null_importance`: golden — скор = `imp_real − percentile(null, p)` на фейк-данных; знаковый pass-through в
  `_normalize_fold`; `auto_threshold=0`; детерминизм при seed; **property**: перестановка таргета в `test`-части
  фолда не меняет вклад фолда; счётчик фитов = `n_folds × (1+n_runs)`.
- `sequential`: на фейк-`score_subset` (монотонная функция от набора) возвращает ожидаемый subset; floor ≥1;
  порядок `schema.features`; детерминизм; **инъекция**: адаптер вызывает только `score_subset` (не получает
  `test`-строк); per-fold scorer фитит на `fit⊕es` (тест анти-ликедж-границы).
- Оба реализуют свой порт (runtime-checkable), `import honestml` не тянет shap/boosting-extra (`test_import_*`).
