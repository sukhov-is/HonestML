# ADR-0046 — Архитектура honest-compare: мульти-стратегийный резолв + второй порт `FeatureSubsetSelector`

- **Статус:** Accepted (реализован в M6c, 2026-06-10; `FeatureSubsetSelector` + `compare_features`, см. `09-review.md`)
- **Дата:** 2026-06-10
- **Драйверы:** DM-C2 (два порта: score- vs wrapper-методы), DM-C3 (Humble Object для wrapper), DM-C4
  (аддитивность). FR-FSC-1/3/4, NFR-FSC-2. Наследует порт `FeatureRanker`/спайн (ADR-0043/0044), DI-резолв
  (ADR-0016 паттерн), `Fold`/`validate_fold` (ADR-0010).
- **Воркстрим:** M6c (honest-compare + новые стратегии).

## Контекст
M6b: **один** порт `FeatureRanker` (`rank(одна матрица)→scores`) + спайн `select_features` (per-fold OOF-цикл →
агрегация → cutoff) → **один** subset. M6c добавляет (а) **сравнение нескольких** стратегий и (б) стратегию
`sequential`, которая по research §3 **не ложится** в `rank()→scores`: это жадная обёртка
`(estimator+scorer+folds)→subset`. Нужно решить **форму расширения** (как добавить wrapper-стратегию, не сломав
анти-ликедж-модель ADR-0043) и **структуру оркестрации** сравнения.

## Рассмотренные варианты (порт для wrapper-метода)
1. **Втиснуть `sequential` в `FeatureRanker.rank()`.** Невозможно: метод выдаёт subset, требует estimator/scorer/
   folds и control-flow, а не per-feature вектор по одной матрице. **Отвергнут.**
2. **Variant-A: `select(x,y,folds,estimator,scorer)→subset` владеет всем** (как отвергнутый ADR-0043 §вариант-A).
   Анти-ликедж-механика (какие фолды, фит без `test`) **уезжает в адаптер** → дублирование leakage-критичной
   логики, нетестируемо на фейке. **Отвергнут** (та же причина, что в ADR-0043).
3. **Второй порт `FeatureSubsetSelector` с ИНЪЕКТИРУЕМЫМ чистым OOF-scorer из `application`.** Адаптер несёт
   только **политику** (что ронять / критерий плато), а leakage-критичный фит/скоринг остаётся в `application`
   (Humble Object). **Выбран.**

## Решение

### 1. Второй порт `FeatureSubsetSelector` (`core/ports/feature_subset_selector.py`)
```python
@runtime_checkable
class FeatureSubsetSelector(Protocol):
    name: str
    def select(
        self,
        x: np.ndarray,
        y: np.ndarray,
        folds: Sequence[Fold],
        *,
        categorical: np.ndarray,
        score_subset: Callable[[Sequence[int]], float],   # инъекция из application (анти-ликедж OOF)
        random_state: int,
        sample_weight: np.ndarray | None = None,
    ) -> tuple[int, ...]: ...                              # отсортированный subset индексов, floor ≥1
```
`score_subset(indices)→float` — **инъектируемый** чистый OOF-scorer (фит на `fit⊕es` каждого фолда, предсказание
`test`-части, агрегат — единственное место с фолд-механикой, в `application`). Адаптер **получает скаляр** на
кандидата-subset, **никогда** не видит сырые `test`-строки → не может обучиться на тесте (Humble Object, DM-C3).
Контракт: выход непуст (floor ≥1), отсортирован по позиции `schema.features`, детерминирован при `random_state`.
Порт — чистая numpy/Callable-сигнатура (домен без sklearn/polars, `core-independence`).

> **Граница честности (см. ADR-0047 §2):** `score_subset` агрегирует **OOF**-скор (использует `test`-таргет
> через метрику) → жадный выбор `sequential` **оптимизирует CV-метрику** (inner-оптимизм, как HPO/выбор модели) —
> **иной род** оптимизма, чем у ранжеров (те не оптимизируют метрику); о большей *магнитуде* не утверждаем
> (R1-A14). Per-fold фит/скоринг **чисты** (без train-on-test); inner-оптимизм снят внешним holdout (ADR-0029).
> Это явно документируется, арбитраж — на **независимой** поверхности (ADR-0048).
>
> **Закрытость порта от test-строк (Humble Object, фикс R1):** `score_subset` конструируется в `application`
> как замыкание `lambda idx: oof_score(x_full, y, folds, idx, …)` — захватывает данные, но **вызывается адаптером
> с одними индексами столбцов**; сырые `test`-строки в область видимости адаптера **не попадают** → адаптер
> структурно не может обучиться на тесте (проверяется тестом: `score_subset` вызван только с tuple индексов).

### 2. Два порта, единый каталог, мульти-резолв (`composition/build.py`)
- `FeatureRanker` — score-стратегии (`importance`, `random_probe`, **`null_importance`**): `rank()→scores`,
  обрабатываются спайном `select_features` (ADR-0044).
- `FeatureSubsetSelector` — wrapper-стратегии (**`sequential`**): `select()→subset`, обрабатываются
  параллельным application-путём (`select_subset`, §3) с инъекцией `score_subset`.
- Резолвер `_resolve_strategies(task, fs) -> list[tuple[name, FeatureRanker | FeatureSubsetSelector]]` строит
  **по одному адаптеру на имя** из списка стратегий (ADR-0049). `run_slice`/use-case **не именуют** конкретный
  адаптер (зависимость через порты) → `usecases-independent-of-adapters` KEPT.

### 3. Оркестрация сравнения (`application/feature_compare.py`, новый)
Драйвер `compare_features(x_full, y, folds, *, strategies, categorical, arbiter, config, sample_weight) ->
CompareOutcome` (чистый, в `application`):
1. Для каждой стратегии получить subset на **DEV** (анти-ликедж per-strategy сохранён): ranker → `select_features`
   (ADR-0044), subset-selector → `select_subset` (инъекция OOF-`score_subset`). N subset'ов.
2. **Арбитраж** (ADR-0048): оценить каждый subset на **независимой** поверхности (selection-holdout) → выбрать
   **один** subset-победитель. Детерминированно.
3. Вернуть `CompareOutcome(winner_name, winner_subset, per_strategy=[(name, n_selected, arb_score)])` →
   `SliceResult`/`run_report` (ADR-0049). Один subset едет в refit/artifact.
**Один путь для N=1** (тождественно M6b, **уточнено R1-A11**): при **одной** стратегии — **любого** вида порта,
включая lone `compare=("sequential",)` — арбитраж/`sel_holdout`-carve **не** запускаются; `compare_features`
сводится к прямому `select_features` (ranker) или `select_subset` на **full-DEV** (subset-selector). N=1 ⇒ нет carve.
`arbiter`/carve подаются в `compare_features` как **инъекция из composition** (callable, как `splitter` в
`run_slice`, ADR-0016) — `application` не импортирует адаптер carve (ADR-0048 §1, `layers` KEPT).

### 4. Слои
Порты (`FeatureRanker`, `FeatureSubsetSelector`) + конфиг/спека — `core`; спайн/`select_subset`/`compare_features`/
арбитр/`score_subset` — `application` (чистые, тест на фейк-стратегиях); адаптеры стратегий (фит модели,
жадная политика) — `adapters`; мульти-резолв — `composition`. `import-linter` 3/3 KEPT.

## Последствия
- (+) Open/Closed: новые стратегии добавляются за **подходящим** портом без правки use-case; wrapper-методы
  получают «дом», не ломая rank-модель; анти-ликедж остаётся в одном месте (инъекция scorer); единый драйвер
  сравнения; N=1 ⇒ M6b без изменений.
- (−/компромисс) Второй порт + драйвер сравнения — рост поверхности `application`/`core` (оправдан: wrapper не
  выразим rank-портом без variant-A); `sequential` несёт более сильный inner-оптимизм (ADR-0047 §2, снят holdout).
- **Влияние на слои:** два порта в `core`; вся leakage-критичная механика в `application`; адаптеры — Humble.

## Проверки
- `null_importance` реализует `FeatureRanker`, `sequential` — `FeatureSubsetSelector` (runtime-checkable); оба
  резолвятся в `build`; `run_slice` не именует адаптер (`lint-imports` KEPT).
- `compare_features` на **фейк-стратегиях** (без обучения): N subset'ов → один победитель, детерминирован;
  N=1 ⇒ путь тождествен M6b (тот же subset).
- `select_subset` передаёт адаптеру только `score_subset`-скаляры (адаптер не получает `test`-строк) — тест
  инъекции; property анти-ликеджа per-fold (ADR-0047 §Проверки).
