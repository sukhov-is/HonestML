# ADR-0043 — Архитектура отбора признаков: порт `FeatureRanker`, водораздел размещения, публичный конфиг, один subset на прогон

- **Статус:** Accepted (реализован в M6b, 2026-06-10; см. `08-plan.md`/`09-review.md`)
- **Дата:** 2026-06-09
- **Драйверы:** DM-3 (слои/Humble Object), DM-5 (подключаемость), DM-6 (стоимость); FR-FS-1/2/7, NFR-FS-2/5.
  Наследует Reader-границу/`design_matrix` (ADR-0005/0013), OOF run_slice (ADR-0010), cross-fit-прецедент
  (ADR-0030/0041), реестр компонентов (ADR-0019/M3), публичный конфиг (ADR-0011/0040).
- **Воркстрим:** M6b (Feature Selection — вторая половина M6).

## Контекст
M6b вводит **отбор признаков за портом** с анти-ликеджем.
Главные решения: **форма порта**, **где фитится/агрегируется отбор** (слои), **как часто** (один subset vs
per-candidate) и **публичная конфигурация**. Объём — **спайн + дешёвый каталог**: порт +
OOF-агрегатор + сериализация + конфиг + DI и две самодостаточные стратегии (`importance`, `random_probe`);
SHAP/null-importance/sequential — defer M6c (аддитивно за тем же портом).

## Рассмотренные варианты (форма порта)
A. **`FeatureSelector.select(x, y, folds) → subset`** — стратегия владеет всем: фолды, рэнкинг, отсечение,
   floor. Гибко, но **каждая** стратегия обязана сама правильно реализовать анти-ликедж-цикл и floor →
   дублирование самого опасного кода в каждом адаптере (R-FS-LEAK размазан). **Отвергнут.**
B. **Раздельно «рэнкер scores → общий cutoff» (`FeatureRanker.rank` + спайн-агрегатор):** стратегия отвечает
   только за **«как оценить важность по одной обучающей матрице»**; **анти-ликедж-цикл по фолдам, агрегацию и
   отсечение+floor централизует чистый спайн в `application`**. Один раз правильно — для всех стратегий.
   Симметрия `Metric.score` (оценивает) / `SelectionPolicy` (решает). **Выбран.**

## Рассмотренные варианты (форма расширения)
1. **Рантайм-плагин-порт + реестр** (как Estimator, ADR-0019). Для растущего каталога рэнкеров оправдан;
   дефолт-резолв в `composition`. **Выбран** (минимальная версия: реестр имён→адаптер).
2. Хардкод if-ветки по имени стратегии в use-case. Нарушает `usecases-independent-of-adapters`. **Отвергнут.**

## Решение

### 1. Порт `FeatureRanker` (core/ports) — «оцени важность по одной обучающей матрице»
```python
@runtime_checkable
class FeatureRanker(Protocol):
    name: str
    def rank(self, x: np.ndarray, y: np.ndarray, *, categorical: np.ndarray,
             random_state: int, sample_weight: np.ndarray | None = None) -> np.ndarray:
        """Per-feature importance (len == n_features, higher=better) for ONE training matrix.

        `x` — (n_fit, n_features) обучающая матрица (numeric-блок ⊕ cat-коды, layout `design_matrix`);
        `categorical` — bool-маска (n_features,) категориальных колонок (native handling рэнкер-модели);
        `sample_weight` — (n_fit,)|None, веса строк (рэнкер-модель учитывает, как кандидаты, `slice.py:496`);
        чистая функция фолдов — **не** видит test-строк (их не передают). Детерминирована по `random_state`
        (рэнкер использует **только** `np.random.RandomState(random_state)`, иных источников randomness нет).
        Возвращаемые важности **масштаб-сопоставимы между вызовами** (нормированы/фолд-относительны), чтобы
        агрегация по фолдам не доминировалась магнитудой одного фолда (ADR-0044 §1, фикс ревью A4): `importance`
        отдаёт L1-нормированные важности (доля суммарного gain); `random_probe` — margin относительно зондов
        того же фита (уже фолд-относителен). Спайн дополнительно нормирует неотрицательный вектор (см. §2).
        **Инварианты выхода (фикс ревью R2):** `len == x.shape[1]`, без NaN/inf (нарушение → `ValueError`);
        `importance` — `≥ 0`, `random_probe` — знаковый margin. Пустой `x` (`n_fit == 0`) → `ValueError`.
        Имя стратегии — атрибут `name` (для реестра/наблюдаемости, ADR-0045 §3).
        """
```
Рэнкер **не** управляет фолдами и **не** режет — он считает вектор важностей по поданной обучающей матрице.
Анти-ликедж обеспечивает вызывающий спайн (ADR-0044), подавая **только** train-часть фолда.

### 2. Спайн `select_features` (application) — чистый numpy, зеркало `crossfit_encode`
```python
def select_features(x_full, y, folds, *, ranker, categorical, cutoff, random_state,
                    sample_weight=None, min_features=1) -> tuple[int, ...]:  # индексы оставленных колонок
    scores = np.zeros(x_full.shape[1]); k = 0
    for fold in folds:                                 # folds — ТОТ ЖЕ список, что в run_slice (ADR-0044 §1)
        train_idx = fold.fit_idx if fold.es_idx.size == 0 else np.concatenate([fold.fit_idx, fold.es_idx])
        sw = sample_weight[train_idx] if sample_weight is not None else None
        imp = ranker.rank(x_full[train_idx], y[train_idx], categorical=categorical,
                          random_state=random_state, sample_weight=sw)
        scores += _normalize_fold(imp)                 # масштаб-инвариантность по фолдам (фикс ревью A4)
        k += 1
    agg = scores / k                                   # средний нормированный скор по фолдам
    return apply_cutoff(agg, cutoff, min_features)     # ADR-0044 §3; гарантирует ≥1
```
`folds` — **тот же** список (из `splitter.split(dataset)`), что использует `run_slice` для оценки; передаётся
в спайн **параметром** (спайн **не** пере-split'ит) → фолды отбора тождественны оценочным (ADR-0044 §1,
R-FS-FOLD-ALIGN). `train_idx = fit_idx ⊕ es_idx` — **те же** строки, на которых учатся кандидаты
(`slice.py:492-494`), без `test_idx`. `_normalize_fold` приводит неотрицательный вектор важностей к доле
(L1: `imp / imp.sum()`); **граничный случай** `imp.sum()==0` (все важности нули) → возвращает нулевой вектор
(деления на ноль нет, фикс ревью R2) — фолд с большей магнитудой не доминирует агрегацию (фикс ревью A4);
знаковые скоры (`random_probe` margin) уже фолд-относительны и проходят без перенормировки. Спайн I/O-free, тестируется на фейк-`FeatureRanker`
без обучения (NFR-FS-2). `categorical` выводится из схемы: первые `len(numeric)` колонок `x_full` — numeric,
остальные — cat-коды.

### 3. Водораздел размещения (слои)
| Компонент | Слой | Обоснование |
|---|---|---|
| порт `FeatureRanker`, `FeatureSelectionConfig`, спека subset (ADR-0045) | `core` | сигнатуры/данные, без polars/sklearn |
| `select_features` + `apply_cutoff` (фолд-цикл, агрегация, отсечение, floor) | `application` | чистый numpy за портом; `run_slice` не именует адаптер |
| стратегии `importance`/`random_probe` (фит рэнкер-модели, важности) | `adapters` | зависят от GBDT/обучения; реализуют порт |
| резолв дефолтного `FeatureRanker` по имени | `composition` (`build.py`) | DI, как estimators |

import-linter 3/3: `core` чист; `application` за портом; стратегии-адаптеры реализуют `core`-порт.

### 4. Один subset на прогон (не per-candidate)
Отбор вычисляется **один раз** перед циклом кандидатов и разделяется ими — как OOF-TE M6a (ADR-0040 §2): FE и
отбор преобразуют **признаки** (X), не зависят от эстиматора-кандидата. Рэнкер использует **отдельную дешёвую
дефолт-модель** (GBDT), а не каждого кандидата → subset **estimator-agnostic** (R-FS-RANKER-MODEL принят by
design: оценивает «полезность признака», не подгоняет под конкретного кандидата; документировано). Per-candidate
отбор отвергнут: дорого и ломает stage-cache (cacheable-unit = кандидат, ADR-0036). Число фитов рэнкер-модели =
`folds × 1` (NFR-FS-5).

**Бюджет (закрытие OQ-FS-BUDGET, фикс ревью):** отбор — **пре-процессинг входа** (часть подготовки `x_full`,
до цикла кандидатов), **НЕ** учитывается в бюджете trials/time. Бюджет (ADR-0032/0039) гейтит **кандидатов**;
рэнкинг считается один раз перед циклом и в счётчик trials не входит (как OOF-TE/датетайм-материализация M6a).
Это согласовано с «один subset на прогон»: при `budget.mode="trials"` число trial'ов = число кандидатов, не
зависит от включённости отбора. Проверка — счётчик budget не растёт от FS.

### 5. Публичный конфиг (sklearn-инвариант)
```python
class FeatureSelectionConfig(BaseModel):   # core/config.py, frozen, extra="forbid"
    strategy: Literal["importance", "random_probe"] = "importance"
    cutoff: Literal["top_k", "top_frac", "auto"] = "top_frac"
    top_k: int | None = Field(default=None, ge=1)       # для cutoff="top_k"
    top_frac: float = Field(default=0.5, gt=0.0, le=1.0) # для cutoff="top_frac"
    min_features: int = Field(default=1, ge=1)           # floor (инвариант §F9, FR-FS-5)
    n_probes: int = Field(default=3, ge=1)               # random_probe
    random_state: int | None = None                      # None → наследует RunConfig.seed
```
Несётся в `RunConfig.fs: FeatureSelectionConfig | None = None` (**дефолт None = отбор выключен** → M6a/M5
неизменны). Фасад: `AutoML(..., feature_selection: FeatureSelectionConfig | None = None)` — verbatim в
`__init__`, резолв в `fit`; невалидный (не `FeatureSelectionConfig`/None) → `ConfigError` (guard, как
`feature_engineering`). Отбор **task-kind-agnostic** (importance/probe не требуют binary, в отличие от TE) —
работает для classification и regression. **Cutoff для `random_probe` (уточнено по ревью):** дефолтный cutoff
стратегии `random_probe` — `auto` (порог margin>0, ADR-0044 §3); `top_k`/`top_frac` **допустимы** (применяются
поверх усреднённого margin как score-ранга, тот же `apply_cutoff`), **не** ошибка валидации, но если среди
оставленных `top_k`/`top_frac` есть отрицательный margin (признак **хуже** шума) — спайн логирует **WARNING**
(фикс ревью R2: пользователь видит, что cutoff оставил под-шумовые признаки; рекомендуется `auto`). `importance`
по умолчанию — `top_frac=0.5`. Невалидное **имя** стратегии/политики → `ConfigError`; несовместимых комбинаций
нет (любой cutoff применим к любому score-вектору).

**Публичная поверхность (фикс ревью R2):** `FeatureSelectionConfig` экспортируется в `honestml.__all__` рядом с
`FEConfig`/`CVConfig` (симметрия M6a, FR-FS-1); порт `FeatureRanker` — в `honestml.core.ports` (как
`Estimator`/`CVSplitter`). Параметр `AutoML(feature_selection=…)` — часть `get_params`/`clone` (frozen →
immutable, безопасно для `Pipeline`/HPO).

## Последствия
- (+) Анти-ликедж и floor — в **одном** месте (спайн), не в каждой стратегии; каталог расширяется аддитивно за
  портом; subset считается раз, estimator-agnostic; дефолт сохраняет M6a/M5; FS task-kind-agnostic.
- (−/компромисс) Рэнкер-модель ≠ кандидаты (subset под рэнкер, не под победителя) — принято by design
  (один subset на прогон); полноценный per-model отбор — вне объёма.
- (−/компромисс) Дорогие стратегии (SHAP/null-imp/sequential) отложены — каталог из 2 дешёвых; расширение M6c.
- **Влияние на слои:** порт/конфиг/спека — `core`; спайн+cutoff — `application`; стратегии — `adapters`;
  резолв — `composition`. `ARTIFACT_VERSION`/`RUN_MANIFEST_VERSION` — ADR-0045 (аддитивно, без bump).

## Проверки
- Обе стратегии реализуют `FeatureRanker` (runtime-check), резолвятся в `build`; невалидное имя/конфиг →
  `ConfigError`; `run_slice` не импортирует адаптер-стратегию (`lint-imports` 3/3 KEPT).
- `select_features` на фейк-рэнкере и массивах (без обучения) = ручной расчёт (golden); рэнкер видит ровно
  `fit ⊕ es` (не `test`).
- Дефолт `feature_selection=None` → leaderboard/artifact **идентичны** M6a; `clone`/`Pipeline` сохраняют параметр.
- Число фитов рэнкер-модели = `n_folds` (счётчик), не зависит от числа кандидатов.
