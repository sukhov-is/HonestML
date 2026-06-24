# ADR-0088: Контракт estimator для нативных категорий и вычисление индексов

- **Статус:** Proposed
- **Дата:** 2026-06-22
- **Драйверы:** D-1 / FR-3, NFR-5

## Контекст
Роутинг (ADR-0087) требует передать native-capable модели **позиции
категориальных колонок** в матрице. Контракт estimator (`core/ports/estimator.py`)
сейчас: `fit(X, y, X_val, y_val, sample_weight)` + атрибут `feature_names`,
ставится извне перед fit (`slice.py:745`). Есть готовый паттерн маркер-ролей
(`SupportsEarlyStopping`, `SupportsNativeModel`), которые use-case читает через
`isinstance`. Категориальные индексы должны:
- доходить только до моделей, которым нужны (не менять сигнатуру `fit` для всех);
- быть **FE/FS-aware**: только CATEGORICAL-роль (исходные категории + пересечения
  `a__b`), без numeric-выходов FE (`_te`, `_freq`, datetime), корректно **после**
  проекции `selected_features` (наивный срез `len(numeric):` неверен — R-6).

## Рассмотренные варианты
1. **Расширить `Estimator.fit` параметром `categorical_features`** — затрагивает
   ВСЕ реализации (linear/baseline/boosting), большинство его игнорирует; шире
   нужного, нарушает «минимальный контракт».
2. **Передавать индексы внутри `X`** (структурный массив/DataFrame) — ломает
   float64-numpy-границу (ADR-0005) для всех.
3. **(выбрано) Маркер-роль `SupportsNativeCategorical` + инъекция атрибута**, по
   образцу `feature_names`/`SupportsEarlyStopping`:
   - новый Protocol в `core/ports/estimator.py`:
     ```python
     @runtime_checkable
     class SupportsNativeCategorical(Protocol):
         supports_native_categorical: bool
         categorical_indices: list[int]
     ```
   - `run_slice` для `isinstance(est, SupportsNativeCategorical)` ставит
     `est.categorical_indices = <indices>` перед `fit` (рядом с
     `est.feature_names = …`); остальные не трогаются;
   - индексы вычисляются методом **схемы** (core), а не в use-case.

## Решение
- Ввести `SupportsNativeCategorical` (маркер-роль, аналог `SupportsEarlyStopping`),
  реализуемый обёрткой бустинга **per-backend** (True для catboost/lightgbm).
- **Контракт маркера (явно):**
  - `categorical_indices: list[int]` — Python-список int (как `feature_names: list[str]`),
    значения ∈ `[0, n_features)`; **ставит use-case перед `fit`**, обёртка сама не
    инициализирует (до инъекции — не читается).
  - **Пустой список — легитимен:** native-capable модель на датасете без категорий
    получает `[]` → материализация (ADR-0089) становится no-op, поведение
    эквивалентно пути кодов (важно для edge-case «нет категорий»).
  - Маркер **ортогонален** `SupportsEarlyStopping`/`SupportsNativeModel`/
    `ProbabilisticEstimator`: это независимые атрибуты на одной обёртке, конфликтов
    диспетчеризации нет (как `feature_names` сосуществует с ES).
  - **Forward-compat:** старая обёртка из прежнего pick\'а не имеет атрибута →
    `isinstance(est, SupportsNativeCategorical)` = False → путь кодов (мягкая
    деградация, не падение).
- Добавить в `FeatureSchema` (core) метод вычисления категориальных индексов по
  ролям на **итоговом** наборе фич (с учётом `selected_features`):
  ```python
  def categorical_indices(self) -> list[int]:
      feats = self.selected_features or self.features
      cat = set(self.categorical)            # original_categorical ⊕ intersections
      return [i for i, f in enumerate(feats) if f in cat]
  ```
  Это включает `a__b` и исключает `_te`/`_freq`/datetime (они в `self.numeric`),
  и автоматически корректно после FS-проекции (R-6).
- `run_slice`/`refit_best` ставят `est.categorical_indices = schema.categorical_indices()`
  для native-capable кандидата перед `fit`; адаптер читает их в `_make`/материализации
  (ADR-0089).
- **`refit_best` согласован по построению:** он обучает на `design_matrix(dataset)`,
  который проецирует на те же `selected_features` (`slice.py:194-207`) и ставит
  `est.feature_names` на тот же спроецированный список; `categorical_indices()`
  использует те же `selected_features` → индексы на refit и на CV-фолдах
  идентичны, расхождения «fit на полном vs проекция» нет.
- **Индексы считаются на ИТОГОВОМ (после FS-проекции) наборе фич** — по
  `selected_features` (если задан), иначе по `features` — методом-по-ролям.
  Это **не** переиспользует наивную FS-маску `categorical_mask[len(numeric):]`
  (`slice.py:442-443/482-483`): та маска корректна на **до-проекционной** матрице
  и применяется только в ранжировании feature-selection; индексы для `est.fit`
  берутся на **спроецированном** `feature_names`, где numeric-блок мог укоротиться
  (часть numeric отброшена FS) и срез `len(numeric):` дал бы сдвиг (R-6). Членство
  по ролям корректно при любой контигуальности блока.
- Индексы вычисляются **один раз** на спроецированной матрице и идентичны на
  train, OOF, refit и inference (источник — одна и та же `schema`/`selected_features`,
  забейканные в артефакт, ADR-0016) — это и обеспечивает FR-4 на уровне контракта.

## Последствия
- **Положительные:** контракт `fit` не меняется → нулевая регрессия остальных
  моделей; знание о ролях остаётся в core (схема — источник истины); переиспользует
  существующий паттерн инъекции; индексы FS/FE-корректны по построению.
- **Отрицательные / компромиссы:** ещё один внешне-устанавливаемый атрибут
  estimator (как `feature_names`) — слабая «временная» связанность use-case↔estimator;
  принимается как уже существующий в кодовой базе паттерн.
- **Влияние на слои/границы:** Protocol и метод схемы — в `core` (без ML-импортов,
  NFR-5); инъекция — в application; реализация маркера — в adapters. Зависимости
  внутрь; новый контракт import-linter не требуется.

## Проверки
- Юнит-тест `categorical_indices()` на схеме с FE (`_te`/`_freq`/`a__b`) и заданным
  `selected_features`: множество совпадает с позициями CATEGORICAL-колонок; срез
  `len(numeric):` дал бы иной (неверный) результат (FR-3, R-6).
- Тест: `isinstance(boosting_catboost, SupportsNativeCategorical)` → True;
  `isinstance(linear, SupportsNativeCategorical)` → False.
- `uv run lint-imports` зелёный; `core/ports/estimator.py` без ML-импорта (NFR-5).
