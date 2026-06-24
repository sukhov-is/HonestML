# ADR-0066 — Standalone-граница инференса + лёгкая dep-граница

- **Статус:** Accepted (M8, design-gate pending)
- **Драйвер:** DM-80 (продакшн-выход) — FR-SRV-2, FR-SRV-5; NFR-SRV-1 (← NFR-8)
- **Связано:** ADR-0012 (единый путь `FittedModel`, baseline — запрет второго пути), ADR-0005 (FeatureSchema),
  ADR-0018 (datetime-дельты), ADR-0019/0020 (реестр — только обучение), `.importlinter` (контракты слоёв).
  Закрывает E1/E2 (дублирующий прототипный inference-модуль).

## Контекст
`FittedModel` (в `composition.artifact`) импортирует **barrel-фасады** `honestml.adapters` и
`honestml.application`, а их `__init__.py` делают **eager-импорт всего пакета**: ranker-модели (sklearn
cluster/ensemble), significance, tuner-обёртка, Caruana/Weighted-поиск, splitters, budget, cache,
feature-selectors, metrics (sklearn.metrics), polars_dataset. Итог: `load_artifact`+`predict` тянет почти
весь training-стек (минус ленивые optuna/shap/boosting). `honestml/__init__` и `composition/__init__` тоже
eager. Extra `inference`/`serving` нет (NFR-8 пробел). Прототипный inference-модуль — **второй, дублирующий** путь
(per-type препроцессинг + datetime), ровно дрейф, против которого ADR-0012 вводил единый `FittedModel`;
по факту его препроцессинг субсумирован `FeatureSchema`, datetime-дельты — `_apply_datetime_deltas`.

## Рассмотренные варианты
1. **Статус-кво (`pip install honestml` + load_artifact)** — ❌ для `predict` тянется весь training-стек;
   NFR-8 не выполнен; slim-serving невозможен.
2. **Воскресить/держать zero-dep прототипный inference-модуль** — ❌ второй путь инференса = дрейф train/inference
   (запрещён ADR-0012); дублирует datetime; per-type препроцессинг уже субсумирован схемой. Поддерживать два
   пути — источник рассинхрона контракта.
3. **Ленивые barrel-`__getattr__` (PEP 562) + минимальный inference-seam + extra `inference`, единый путь,
   ретайр прототипа** — ✅ `from honestml.adapters import Reader` импортирует только `adapters.reader`, не
   training-подмодули; import-конус `FittedModel` сжимается до {core, reader+polars_dataset,
   slice.design_matrix, metrics (лениво, на `.score()`)}; публичные имена **те же** (бэк-совместимо);
   один путь инференса; прототип ретайрнут.

## Решение (Вариант 3)

### §1 Ленивые barrel-`__getattr__` (PEP 562) — и что РЕАЛЬНО держит slim-конус (правка R1)
`honestml/__init__.py`, `adapters/__init__.py`, `application/__init__.py`, `composition/__init__.py` —
перевести на ленивый `module.__getattr__(name)` + явные `__all__`/`__dir__`. Публичные имена сохраняются;
подмодуль импортируется при **первом обращении к атрибуту**. Прямой импорт подмодуля
(`import honestml.adapters.tuning`) продолжает работать (lazy не ломает явные импорты). `lint-imports`-контракты
сохраняются (направление слоёв не меняется — меняется лишь момент импорта).

**Точный механизм (R1):** slim-конус держат **две** разные вещи, не одна:
- **Ленивость `adapters/__init__` — критична:** `from honestml.adapters import Reader` иначе eager-тянет
  ranker-модели (`sklearn.cluster`/`ensemble`), `significance`, `OptunaTuner`-обёртку, ensembler-поиск,
  splitters, budget, cache, feature_selectors. `__getattr__` отсекает их — грузится только `adapters.reader`
  (+`polars_dataset`).
- **Чистота `application` — инвариант, не ленивость:** обращение к `design_matrix` всё равно импортирует
  **весь** `application.slice`, а он eager-тянет `application.{calibration,feature_encoding,feature_selection}`
  + `core.ports.splitter`. Конус остаётся slim **потому что эти подмодули pure** (core+numpy, без `adapters`/
  optuna/shap) — гарант контракт `usecases-independent-of-adapters` (`.importlinter`), а **не** `__getattr__`
  application. Поэтому cone-тест (NFR-SRV-1) обязан ассертить **отсутствие `honestml.adapters.*` обучающих
  подмодулей** (tuning/ensembling/significance/splitters/feature_selectors) в `sys.modules` после
  `import design_matrix`+`load_artifact` — это ловит будущую регрессию чистоты `application`, которую
  ленивость не защищает.

### §2 Минимальный inference-seam + отложенная метрика
`FittedModel.predict/predict_proba` тянут только `Reader` (+`polars_dataset`), `design_matrix`,
`align_proba`/`resolve_positive`, core, joblib — без sklearn.metrics/обучающих адаптеров. **`resolve_metric`
откладывается** (конкретика R1): сейчас `artifact.py` импортирует `resolve_metric` на уровне модуля
(`from honestml.adapters import ... resolve_metric`, строка 23) и жадно зовёт его в `load_artifact` (250) →
`adapters.metrics` → `sklearn.metrics` уже при импорте `artifact`. Нужно убрать `resolve_metric` **и** из
module-top импорта, **и** из жадного вызова; `FittedModel.metric` материализуется лениво (property/внутри
`_score_dataset`) при первом `.score()`. Тогда чистый `predict` не импортирует `sklearn.metrics`. Built-in
estimator (Linear/Baseline) тянет sklearn только если это его модель; бустинг-артефакт тянет свой пакет для
unpickle.

### §3 Extra `inference` + явный `joblib` (правка R2 — **pandas обязателен**)
`[project.optional-dependencies] inference = ["numpy", "pandas", "polars", "scikit-learn", "pydantic",
"joblib"]` — реальный runtime-минимум для load+predict. **`pandas` ОБЯЗАТЕЛЕН** (правка R2-blocker): единственный
inference-путь `Reader` делает безусловный module-top `import pandas` и зовёт `pd.isna`/`pl.from_pandas`
(reader.py:16,164,201,206) — без pandas импорт Reader падает, cone-тест FR-SRV-2 не проходит. `sklearn` нужен
для built-in Linear/Baseline (и `.score()`); `joblib` объявлен **явно** (сейчас транзитивен через sklearn).
Per-model runtime (catboost/lightgbm/xgboost) ставится своим extra при бустинг-артефакте.

**Честная природа «slim» (правка R2):** этот extra ≈ core-deps + явный `joblib` (core уже =
numpy/pandas/polars/sklearn/pydantic). Выигрыш serving не в выкидывании core, а в том, что (1) training-only
extras (optuna/shap/mlflow/onnx/matplotlib/boosting-обучение) **не ставятся** (они и так отдельные extras —
`pip install honestml` их не тянет), и (2) **ленивый import-конус** (§1/§2): даже при установленном sklearn
обучающие adapter-модули (rankers→sklearn.cluster/ensemble, tuner, significance) **не исполняются** на
predict. Поэтому NFR-SRV-1 проверяется ассертом **не-импорта** (`sys.modules`) в стандартном окружении, а не
отдельной slim-установкой (см. §2/NFR-SRV-1). `inference`-extra — удобный/документированный alias этого
runtime-набора (делает `joblib` явным), не средство «обрезать» core.

### §4 Единый путь + ретайр прототипного inference-модуля (E1/E2)
Единственный standalone-путь — `honestml.load_artifact(dir).predict(X)` (тот же `FittedModel`, что у фасада →
паритет train==inference, NFR-SRV-4). **прототипный inference-модуль удаляется**: per-type препроцессинг
субсумирован `FeatureSchema`/`CategoryTable`, datetime-дельты — `schema.datetime_spec`+`_apply_datetime_deltas`
(симметрично); календарные компоненты (month/quarter/dow) — сознательный не-goal (ADR-0018). `FittedModel`
остаётся в `composition` (минимальный диф; import-конус теперь ленив) — вынос в отдельный `honestml.serving`
рассмотрен и отклонён (больший диф без выгоды после §1). **Уточнение R1:** прототип читает **иной, более не
производимый** формат артефакта,
которого нет в текущем публичном API → удаление безопасно (ничто не импортирует, вне wheel); паритет
доказывается **не** на артефактах старого формата, а тестом `load_artifact(dir).predict == fitted_.predict` на
актуальном артефакте.

## Последствия
- **+** NFR-8 закрыт: slim-serving окружением `honestml[inference]`; один путь инференса (нет дрейфа).
- **+** Бэк-совместимо: публичные имена и явные импорты подмодулей не меняются; `lint-imports` KEPT.
- **+** E1/E2 закрыты: дублирующий прототип ретайрнут, паритет доказывается тестом.
- **−/R-SRVDEP:** ленивый `__getattr__` требует корректных `__all__`/`__dir__` (IDE/автокомплит, `from x
  import *`); риск циклов/регрессии импортов → gate: полный suite + тест import-конуса (`sys.modules`) +
  `lint-imports`.
- **−/R-SRVLEGACY:** удаление прототипа теряет исторический источник (вне wheel; паритет доказан) — принято.
- **−:** отложенная метрика — небольшое усложнение `FittedModel` (ленивое поле); `.score()` без окружения
  метрики даст явную ошибку об отсутствующем sklearn (граница).

## Day-2 (committed)
- Выделенный пакет `honestml.serving` (если inference-поверхность вырастет) — сейчас избыточно.
- Нативная/ONNX-сериализация (убирает sklearn/боустинг-unpickle из конуса) → **M8b** (SPIKE-3).
