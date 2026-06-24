# ADR-0051 — Стратегия `shap`: ранжер на TreeExplainer за lazy-extra `shap`

- **Статус:** Принят (реализован в M6d, 2026-06-10; `ShapRanker` lazy-extra, `tree_path_dependent`,
  `_mean_abs_per_feature` нормализует list/2-D/3-D, `shap_max_samples`). Питается research §2.
- **Дата:** 2026-06-10
- **Драйверы:** DM-H5 (расширение каталога без раздувания базы). FR-FSH-4/5, NFR-FSH-2/3/4/7. Наследует порт
  `FeatureRanker`/спайн/нормировку (ADR-0043/0044), estimator-agnostic ранжер-модель (ADR-0043 §4), lazy-extra
  паттерн (`MissingDependencyError`).
- **Воркстрим:** M6d.

## Контекст
roadmap §3.1 относит SHAP-рэнкер к каталогу `FeatureSelector`-стратегий; M6c отложил его в M6d как **внешнюю**
зависимость (NFR-FSC-6, `test_import_does_not_pull_shap` уже зелёный). Инфраструктура **уже заложена**:
`pyproject.toml:38` объявляет `shap = ["shap>=0.44"]` (в `all`), `MissingDependencyError(extra)` готов,
тест-страж импорта существует. Нужно добавить стратегию `"shap"` тем же портом и спайном, что M6b-ранжеры, не
протащив `shap` в `import honestml`.

## Решение

### 1. `ShapRanker` — `FeatureRanker` на собственной ранжер-модели (`adapters/feature_rankers.py`)
Как `ImportanceRanker`/`NullImportanceRanker`: фитит **отдельную дешёвую** `ExtraTrees(n_estimators=100,
n_jobs=1)` (estimator-agnostic, ADR-0043 §4 — **не** кандидат-эстиматор), затем считает SHAP на её деревьях:
```python
def rank(self, x, y, *, categorical, random_state, sample_weight=None, groups=None):  # groups игнор (ADR-0050)
    model = _fit_ranker_model(self._task, x, y, random_state, sample_weight)   # ExtraTrees, как M6b
    import shap                                                                 # LAZY (см. §2)
    explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
    sv = explainer.shap_values(x if self._max_samples is None else x[:self._max_samples])
    return _mean_abs_per_feature(sv)                                            # (n_features,), ≥ 0
```
- **`feature_perturbation="tree_path_dependent"`** (подтверждено через context7 для `shap>=0.44`): точный
  полиномиальный Tree SHAP **без фоновых данных** → детерминирован, **без** сабсэмплинга фона (в отличие от
  `interventional`). ⇒ **поля `shap_max_bg_samples` не требуется** (interventional-режим потребовал бы скрытой константы размера фоновой выборки).
- **Агрегация `_mean_abs_per_feature`:** `mean(|shap|, axis=0)` по строкам. Multiclass: для пина `shap>=0.44`
  `TreeExplainer.shap_values` на sklearn-ансамбле возвращает **список массивов по классам** (`len == n_classes`,
  каждый `(n, n_features)`); агрегируем `mean(|·|)` по строкам **и** по классам в 1-D длины `n_features` (как
  `feature_importances_` для multiclass). Адаптер **нормализует** оба возможных формата (список или 3-D ndarray
  `(n, n_features, n_classes)` у новых версий) → форма проверяется тестом `test_shap_ranker_multiclass`.
  Бинарный/регрессия → прямой `mean(|shap|, axis=0)`. (R-SHAPAPI: формат инкапсулирован в адаптере, проверяется
  тестом — не «слепое» доверие версии.)
- **Разведение с `SupportsShap` (R1-m5):** в `core` есть role-interface `SupportsShap.shap_values(X)`
  (`core/ports/estimator.py`) — это контракт **эстиматора-кандидата**, НЕ ранжера. `ShapRanker` его **не**
  использует: он строит SHAP на **собственной** ExtraTrees (estimator-agnostic, ADR-0043 §4) через
  `shap.TreeExplainer`. Два «SHAP» — разные уровни (model-level role vs ranker-level стратегия), не путать.
- **Выход ≥ 0 ⇒ L1-ветка спайна** `_normalize_fold` (как `importance`); `auto_threshold(n) = 1/n` («выше
  равномерной доли», как `ImportanceRanker`). Контракт порта (длина, finite) соблюдён — спайн/cutoff **без
  изменений**.
- **Опц. cost-cap `shap_max_samples`** (число объясняемых строк фолда): на широких/высоких данных SHAP-проход
  дорог; cap **детерминирован** (берём первые N строк, без случайного сабсэмплинга → воспроизводимость). Дефолт
  `None` (объяснять всю train-часть фолда). Это **cost-knob**, не фон.

### 2. Lazy-extra — `import shap` внутри `rank()`/конструктора (паттерн `run_budget`)
- **На уровне модуля `feature_rankers.py` `import shap` ЗАПРЕЩЁН** (упадёт `test_import_does_not_pull_shap`).
  Импорт — **внутри** метода: `try: import shap; except ImportError as exc: raise
  MissingDependencyError("shap") from exc` → сообщение `pip install honestml[shap]` (`core/exceptions.py`).
- **Где ловить:** в `ShapRanker.__init__` (fail-fast при сборке компонента, как `run_budget._default_rss_mb`),
  чтобы конфиг-ошибка всплывала на построении пайплайна, а не в середине FS. (Сам тяжёлый проход — в `rank`.)
- `pyproject` **не трогаем** (extra `shap` уже есть). `ShapRanker` экспортируется в `adapters/__init__.py`
  (импорт **класса** лёгкий; `shap` лежит lazy внутри `__init__`/`rank`).
- **Хрупкость eager-инстанцирования (фикс R1-M4):** инвариант — `ShapRanker` **не инстанцируется на уровне
  модуля** нигде в `composition`/`adapters.__init__` (только внутри `_make_strategy` при выборе стратегии). Иначе
  `__init__`-`import shap` сработал бы при `import honestml`. Усилить страж: добавить тест `import honestml.adapters`
  **не** тянет `shap` (сильнее текущего `import honestml`, `test_public_api.py:59-63`).

### 3. Интеграция — единая точка `_make_strategy` (`composition/build.py`)
- `FSStrategy` (`core/config.py:27`) расширяется: `Literal[..., "shap"]`.
- `_make_strategy(task, fs, name)` получает ветку `if name == "shap": return ShapRanker(task,
  max_samples=fs.shap_max_samples)` с lazy-импортом класса внутри функции. Это **единственная** точка маппинга —
  и одиночный путь (`_resolve_feature_ranker`), и compare (`_resolve_strategies`) не разъезжаются.
- SHAP попадает в compare по ветке `FeatureRanker` (`Strategy = FeatureRanker | FeatureSubsetSelector`)
  **автоматически**. **CV-scheme-гард не нужен** (TreeExplainer на tree-модели применим при любой scheme; SHAP
  считается на train-части фолда, как `importance`).
- `FeatureSelectionConfig` получает опц. `shap_max_samples: int | None = Field(default=None, gt=0)` (cost-cap,
  §1). Аддитивно (frozen/extra=forbid), внутри `FeatureSelectionConfig` → fingerprint-нейтрально при `fs=None`.

### 4. Стоимость и детерминизм (NFR-FSH-2/3)
| Аспект | Значение |
|---|---|
| Фиты ранжер-модели | `n_folds × 1` (как `importance`) + Tree SHAP-проход на фолд |
| Порядок цены | дешевле `null_importance` (`n_folds×(1+n_runs)`) и `sequential` (`O(n²)`); SHAP-проход полиномиален по деревьям |
| Детерминизм | ExtraTrees seeded + `tree_path_dependent` без случайности ⇒ воспроизводим при seed |
| Память (R12) | широкие данные → SHAP может быть дорог по памяти; `shap_max_samples` как cost-cap |

## Последствия
- (+) Каталог расширен популярной SHAP-стратегией за **правильным** портом, без правки спайна/cutoff; база не
  тянет `shap` (NFR-FSH-4); детерминизм без скрытого сабсэмплинга; единая точка инстанцирования.
- (−/компромисс) Внешняя тяжёлая зависимость (opt-in, lazy); SHAP-проход добавляет цену к `importance`-уровню;
  `interventional`-режим (с фоном) **не** вводим в M6d (tree_path_dependent достаточно и детерминирован) — это
  возможное Day-2-расширение.
- **Влияние на слои:** стратегия — `adapters` (Humble, lazy-import); порт/конфиг — `core`; спайн/нормировка —
  `application` (без изменений); резолв — `composition`. `import-linter` 3/3 KEPT.

## Проверки
- `ShapRanker` реализует `FeatureRanker` (runtime-checkable); `import honestml` не тянет `shap`
  (`test_import_does_not_pull_shap` зелёный); отсутствие пакета → `MissingDependencyError("shap")` при сборке.
- На малых данных `rank` возвращает неотрицательный вектор длины `n_features`, проходит L1-нормировку; multiclass
  агрегируется в 1-D (тест multiclass); детерминизм при seed; `shap_max_samples` ограничивает объясняемые строки.
- `strategy="shap"` и `compare=(...,"shap")` резолвятся в `build` через `_make_strategy` (одна точка).
