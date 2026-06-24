# ADR-0070 — Нативная сериализация бустингов (opt-in, кросс-версийно стабильная)

- **Статус:** Accepted (M8b-design). Драйверы: DM-B2, DM-B6. Питается **SPIKE-0003** (round-trip exact).
  Реализует порт ADR-0069.
- **Контекст:** тело модели сейчас — pickle (`joblib.dump`). Pickle/«memory snapshot» **не гарантирует**
  загрузку между версиями библиотек (xgboost docs прямо: backward-compat — только для `save_model`, не для
  pickle). Это риск долговечности артефакта в проде. SPIKE-0003 показал: нативные форматы round-trip-ятся
  **точно** (max|Δ|=0.0): xgboost `.json/.ubj` и catboost `.cbm` — exact + полный sklearn-API; lightgbm text —
  exact по значениям, classification требует тонкого адаптера; sklearn нативного формата не имеет.

## Решение

### §1. Opt-in, дефолт неизменен
Нативная сериализация включается `save_artifact(..., model_format="native")` — **keyword-only аргумент только
`save_artifact`** (персистенция user-driven: фасад его не несёт, провенанс формата — в манифесте `model_type`,
правка R2). **Дефолт остаётся `joblib`** → нулевое изменение M8-поведения (NFR-SER-1); флип дефолта — будущее
roadmap-решение, не здесь. Адаптеры-сериализаторы — в `adapters/serializers.py`, библиотека импортится **лениво**
внутри `save`/`load`.

### §2. XGBoost (`model_type="xgboost"`)
`save`: `native_model().save_model(dir/"model.ubj")` (UBJSON — дефолт ≥2.1, компактный, backward-compat).
`load`: `XGBClassifier()/XGBRegressor().load_model(...)` → оборачивается обратно в `_Boosting*`-обёртку;
`classes_` восстанавливаются нативно (SPIKE: `[0,1]`/`[0,1,2]` ок) и сверяются с `manifest["classes"]`.

### §3. CatBoost (`model_type="catboost"`)
`save`: `native_model().save_model(dir/"model.cbm", format="cbm")` (нативно поддерживает категории, быстрее
ONNX на x86-64). `load`: `CatBoost*().load_model(...)` → ре-обёртка; sklearn-API и `classes_` восстановлены.

### §4. LightGBM (`model_type="lightgbm"`)
`save`: `native_model().booster_.save_model(dir/"model.txt")` (текст). `load`:
- **regression** — `lgb.Booster(model_file=...)`; `Booster.predict ≡ predict` (SPIKE exact) → прямая обёртка.
- **classification** — `Booster` не несёт `predict_proba`/`classes_`; загрузчик оборачивает его в тонкий
  адаптер `_NativeLgbmClassifier`, синтезирующий `predict_proba` из сырого выхода (**binary:**
  `column_stack([1-p, p])` в порядке `classes`; **multiclass:** `(n,K)` как есть) и берущий `classes_` из
  `manifest["classes"]`. Значения — **точные** (SPIKE max|Δ|=0.0): это восстановление API-формы, не fidelity.

### §5. Контракт инференса не меняется
Любой `load` возвращает объект, реализующий `Estimator`/`ProbabilisticEstimator`; путь `FittedModel`
(`design_matrix → estimator.predict[_proba]`) и выравнивание классов (`align_proba` по глобальному `classes`)
— как для joblib. `classes` уже в манифесте (ADR-0024) — источник для ре-обёртки. Ре-обёртка **per-serializer**
(правка R2), через **явную adapter-side фабрику**, а **не** через текущий fit-driven `__init__` (boosting.py:111,
захватывает модель только в `fit` → нужен новый публичный `from_native`-classmethod):
- **xgb/cat** → `_Boosting{Classifier,Regressor}.from_native(native_model, classes, backend)` (восстановленный
  sklearn-API объект);
- **lgbm-clf** → отдельный тип `_NativeLgbmClassifier(booster, classes)` (§4); **lgbm-reg** → прямая обёртка
  `Booster`.

То есть «фабрика» — это **разные пути на сериализатор**, не одна униформа; все живут в `adapters` — composition
концретную обёртку не трогает (её знает только реестр).

### §6. Отсутствие библиотеки на загрузке
Манифест аддитивно несёт `required_extra` (например `"xgboost"`). Если на загрузке библиотека отсутствует —
явная **`MissingDependencyError`** с именем extra (как registry `requires`-gating через `find_spec`,
ADR-0020 §5 — поправка к ADR-0019 §1), а не падение в недрах. Сам факт «нужен рантайм X» виден из манифеста
до десериализации.

### §7. Integrity / scope / carve-outs
Нативный файл попадает в `checksums.files` и проходит verify до загрузки; anti-traversal по basename.
**Carve-outs из native (задокументированы, не молча):**
- **Калибратор** (`calibrator.joblib`) **остаётся joblib** (маленький sklearn-объект; нативная сериализация
  калибратора — вне scope).
- **sklearn-эстиматоры** (`baseline`/`linear`) в режиме `native` прозрачно остаются joblib (нативного формата нет).
- **Ансамбль (`BlendedEstimator`)** не реализует `SupportsNativeModel` → в режиме `native` он бы прозрачно ушёл в
  joblib, **но это бы тихо за-pickle-ило гетерогенный блендинг бустингов** (ровно то, что NFR-SER-5/DM-B2 хотят
  убрать). Поэтому (правка R2): `model_format="native"` на `BlendedEstimator` → **не молча** — либо явный
  **per-member** native (каждый член в свой нативный файл + дескриптор блендинга: веса/классы/`model_type` членов),
  либо документированный joblib-carve-out с **WARNING** «ансамбль отгружается pickle-телом». **Решение M8b:**
  per-member native — **Day-2** (отдельный срез); в M8b ансамбль в `native` → joblib-тело **с WARNING** (раскрыто),
  ONNX-экспорт ансамбля → отказ (ADR-0071 §2).

## Последствия
- **+** Долговечность/портабельность артефакта: тело модели в документированно-стабильном формате, не pickle
  (NFR-SER-5); load-back **точный** (точнее ONNX-float32).
- **+** Opt-in → дефолт безопасен; поднабор расширяем по одному формату.
- **−** Сложность: per-lib адаптеры + реконструкция LightGBM-classification (тонкий адаптер); поверхность тестов.
- **−** Native не покрывает калибратор и sklearn-модели (joblib остаётся в смешанном артефакте) — приемлемо,
  задокументировано.
- **Day-2:** при флипе дефолта на native — миграционная заметка; фикстур-артефакт в тестах подтверждает
  кросс-версийную загрузку (operational).
