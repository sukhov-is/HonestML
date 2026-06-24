# ADR-0071 — ONNX-экспорт: export-only, поддерживаемый поднабор, паритет-gate

- **Статус:** Accepted (M8b-design). Драйверы: DM-B3, DM-B4, DM-B5, R-SER-PREPROC. Питается **SPIKE-0003**.
- **Контекст:** roadmap §M8 — «ONNX-экспорт (поддерживаемый поднабор)». SPIKE-0003 (shipped-конфиг 300 деревьев):
  linear/lgbm/xgb/cat конвертируются с паритетом proba **≤ 5.7e-7** и согласием меток **1.0 (0 рассогласований)**
  (даже с ordinal-кат-столбцом — R3 снят); **baseline (Dummy) не имеет skl2onnx-конвертера**; ONNX-деревья
  **float32** (приближённо, но крошечно); CatBoost ONNX **export-only**. Нативный формат (ADR-0070) уже даёт
  **точный** load-back и легче по рантайму → ONNX незачем грузить обратно в `FittedModel`. **Роль ONNX —
  внешний serving** (другой рантайм/язык).

## Рассмотренные варианты
1. **ONNX как load-back `model_type`** (грузится в `FittedModel` через onnxruntime). Тянет `onnxruntime` в
   критический путь + бремя паритет-гейтинга на каждой загрузке; при этом native уже даёт **точный** load-back.
   Дублирование с худшей точностью. ❌
2. **ONNX export-only** (выбран): отдельная функция `export_onnx`, артефакт для внешних рантаймов; основной
   loaded-back артефакт остаётся native/joblib. Без runtime-dep в горячем пути, паритет проверяется один раз на
   экспорте.

## Решение

### §1. `export_onnx(model, directory, *, sample)` — отдельный канал (composition), не `model_type`
Производит **standalone ONNX-бандл** для внешнего serving: `model.onnx` (граф эстиматора над числовой
`design_matrix`) + копия `schema.json` + `onnx_manifest.json` (контракт препроцессинга + паритет-репорт).
**Не** грузится обратно в `FittedModel` (для load-back есть native/joblib). Адаптеры-конвертеры — в
`adapters/onnx_export.py`, onnx-тулинг импортится **лениво** (extra `onnx`+`onnxruntime`).

**`sample` — обязателен** (правка R2): паритет-gate (§3) сверяет ONNX с нативной моделью на **реальных данных**, а
`FittedModel` обучающую `design_matrix` **не хранит** (только estimator/schema/task/classes/leaderboard). Поэтому
сырые строки `sample` прогоняются через `model._read`/`design_matrix` (тот же препроцессинг), затем
`float32`-матрица — в `onnxruntime`. **Без `sample` honest-gate выполнить нельзя** — это не опциональный пропуск,
а контракт сигнатуры (keyword-only, без дефолта).

### §2. Поддерживаемый поднабор (из SPIKE-0003), конвертеры и **имена выходов** (per-converter)
Имена выходных тензоров ONNX **различаются по конвертеру** (правка R2) — паритет-чек (§5) читает их **строго по
имени конвертера**, не по единому глобальному имени и не позиционно:

| Семейство | Конвертер | Имена выходов (clf: label / proba; reg) | Примечание |
|---|---|---|---|
| linear (logistic/ridge) | `skl2onnx.to_onnx(..., options={"zipmap":False})` | `label` / `probabilities`; reg `variable` | чистый proba-тензор (skl2onnx 1.20 реально эмитит `label`/`probabilities`, не `output_*` — **проверено прогоном** M8b-2) |
| lightgbm | `onnxmltools.convert_lightgbm(..., zipmap=False)` | `label` / `probabilities`; reg `variable` | |
| xgboost | `onnxmltools.convert_xgboost(...)` | `label` / `probabilities`; reg `variable` | без zipmap-параметра |
| catboost | нативный `native_model().save_model(format="onnx")` | `label` / `probabilities`; reg `predictions` | export-only; output-shape квирк + **proba = ZipMap seq-of-maps** → нормализуется в плотную `(n,K)` (§5) |
| **baseline (Dummy)** | — | — | **не поддержан** → явная ошибка |
| **ensemble (`BlendedEstimator`)** | — | — | **не поддержан** → явная ошибка (см. ниже) |

`baseline` → `SchemaValidationError("baseline is not ONNX-exportable")` (тривиальная prior/mean-модель, не
serving-цель). **`BlendedEstimator`** (shipped-ансамбль, ADR-0064) → `SchemaValidationError("ensemble is not
ONNX-exportable")` — гетерогенный взвешенный блендинг не покрывается single-estimator-конвертерами; явный отказ
вместо невнятной ошибки конвертера (нативное поведение ансамбля — ADR-0070 §7). Вход в граф — `float32`;
классификатор — `zipmap=False`. Имена выходов выше — контракт против дрейфа версии конвертера (R-SER-VERSION),
проверяется CI-тестом per-converter.

### §3. Паритет-gate на экспорте (честность, DM-B4)
После конверсии — прогон `onnxruntime` на **переданном `sample`** (§1) и сверка с нативной моделью **до** записи
бандла (пороги — SPIKE-0003 на **shipped-конфиге 300 деревьев**, запас ~18–70×):
- **proba** — max|Δ| ≤ **1e-5** (наблюдалось ≤5.7e-7); **регрессия** — |Δ| ≤ **1e-4** abs / rtol 1e-3
  (≤1.4e-6): превышение → **`SchemaValidationError`** (hard-fail, не отгружаем расходящийся граф) с метрикой.
- **метки — boundary-aware** (правка R1-AD1, вместо хрупкого `==1.0`): рассогласование `argmax(onnx)≠
  argmax(native)` допустимо **только** при native top-2 gap ≤ **2·proba-допуска** (доброкачественный тай в
  float32-шуме → **WARNING**); любое рассогласование с gap > 2·допуска → **`SchemaValidationError`**.
  - **Обоснование «2·» и остаточный риск (правка R2):** граница консервативна — это hard-ceiling proba-допуск
    (1e-5), а **не** измеренный шум (~5.7e-7), поэтому окно тай-WARNING ≤ 2e-5 шире фактического float32-шума.
    Следствие: гипотетический реальный дрейф с top-2 gap в `(шум, 2e-5]` будет помечен **WARNING**, а не отказом.
    Поэтому: (1) gross-дрейф независимо ловит proba-hard-fail (max|Δ|>1e-5); (2) **каждый** тай-WARNING виден
    человеку. **Поправка M8b-2 (run-verified):** standalone `export_onnx` выполняется пост-fit и `run_report`
    недоступен — человеческий канал это **`logger.warning` + `onnx_manifest.parity`**
    (`label_verdict`/`n_tie_warnings`); дублирование в `run_report` вернётся, только если экспорт когда-либо
    появится во fit-time-потоке. SPIKE: 0 рассогласований на 300 деревьях/boundary-dense → окно фактически
    пустое, правило — защита от будущего тай-флипа без ложного отказа честного экспорта.

### §4. Препроцессинг — вне ONNX-графа + схема `onnx_manifest` (R-SER-PREPROC)
Граф покрывает **только эстиматор** (числовая матрица → предсказание). `FeatureSchema → design_matrix`
(порядок столбцов, ordinal-кодирование категорий, **проекция отобранных фич**) **не входит** в граф. При активном
feature-selection **ширина входа графа** `initial_type=FloatTensorType([None, n])` `n` = ширина **проецированной**
`design_matrix` (post-selection), а `feature_order`/`columns` манифеста перечисляют **проецированные** фичи в
порядке `design_matrix` (иначе внешний потребитель подаст не то число столбцов).

**`onnx_manifest.json` — конкретный контракт** (правка R2; аддитивен по `onnx_manifest_version`):
```
{
  "onnx_manifest_version": 1,
  "schema_ref": "schema.json",
  "feature_order": ["col0", ...],            // проецированный порядок design_matrix
  "columns": [{"name": "col0", "dtype": "float32", "ordinal": true}, ...],
  "classes": [0, 1] | null,                  // null для регрессии
  "conversion": {"method": "skl2onnx|lightgbm|xgboost|catboost", "tool_versions": {...}},
  "calibration": {"applied": true|false, "method": "...|null"},   // см. ниже
  "parity": {"proba_max_abs": 5.7e-7, "reg_max_abs": null,
             "label_verdict": "ok|warning", "n_tie_warnings": 0, "n_validation_rows": 400}
}
```
Внешний потребитель **обязан воспроизвести `design_matrix`** по схеме — раскрыто в бандле/README. «ONNX
несамодостаточен по препроцессингу» — задокументированное ограничение, не дефект.

**Калибратор — вне графа (честный дисклоуз, правка R2):** при наличии калибратора `FittedModel.predict_proba`
отдаёт **калиброванные** вероятности, а ONNX-граф оборачивает **сырой** эстиматор, и паритет-gate сверяет ONNX vs
**сырой** native-proba. Чтобы не было тихого расхождения «граф ≠ то, что отгружает артефакт» (DM-B4):
`onnx_manifest.calibration` фиксирует `applied=true`+метод, README бандла предупреждает «граф — **до** калибровки;
потребитель применяет калибровку отдельно». (Экспорт калибровочного отображения в ONNX — Day-2.)

### §5. Чтение выходов ONNX — строго по имени (per-converter), без позиционного фолбэка
Паритет-чек читает нужный тензор **строго по имени, ожидаемому для конкретного конвертера** (таблица §2:
clf — все четыре эмитят `probabilities`; регрессия — `variable` у skl2onnx/onnxmltools, `predictions` у catboost)
и решейпит к `(n,K)`; ZipMap seq-of-maps (catboost) нормализуется в плотную матрицу. **Без позиционного фолбэка** (как в spike-харнессе `od.get(..., out[-1])`): если ожидаемый
выход **отсутствует** (дрейф версии конвертера, R-SER-VERSION) → явная `SchemaValidationError("unexpected ONNX
output schema: <name> missing")`. Отдельно у **CatBoost** объявленная форма выхода `label`/`predictions` квирк-ная
(onnxruntime WARNING), но значения верны — поэтому **не доверяем** объявленной форме `label`, берём `probabilities`/
`predictions` по имени и решейпим. Так регресс конвертера ловится CI-gate детерминированно для **всех четырёх**
семейств, а честный skl2onnx-экспорт не падает ложно из-за чужого имени.

### §6. Зависимости и версии
В extra `onnx` добавить `onnxruntime` (нужен для паритет-чека на экспорте и внешнему потребителю). Пины — из
**совместимого набора SPIKE-0003**: onnx 1.21 / onnxruntime 1.23.2 / skl2onnx 1.20 / onnxmltools 1.16
(валидированы SPIKE-3 на locked lgbm 4.6.0 / xgb 3.2.0 / cat 1.2.10 / sklearn 1.7.2). CI-gate паритета ловит
регресс конвертера (R-SER-VERSION).

### §7. Публичный API + slim-конус (NFR-SER-3, FR-SER-5)
`export_onnx` реэкспортируется ленивым barrel-`__getattr__` (как `save_artifact`/`load_artifact`): **аддитивная
дельта** — `'export_onnx': '.composition'` в `_SUBMODULES`, запись в `__all__`, `TYPE_CHECKING`-импорт; `__getattr__`
onnx-тулинг **не** тянет на резолве имени — onnx импортится строго в теле `export_onnx`.

`import honestml`, **конструкция фасада** и `load_artifact(...).predict(...)` **не** импортируют onnx-тулинг.
`model_format` — **keyword-only аргумент только `save_artifact`** (персистенция user-driven: пользователь сам зовёт
`save_artifact(model.fitted_, dir, model_format=...)`); фасад его **не** несёт и реестр/onnx жадно не импортирует
(провенанс формата — в **манифесте** `model_type`/`required_extra`, не в фасаде). `application` не затрагивается; 3
import-linter-контракта KEPT. Import-конус-тест (расширение ADR-0066) проверяет отсутствие
`onnx`/`onnxmltools`/`skl2onnx` в `sys.modules` после `import honestml`, **после конструкции `AutoML(...)`** и
после `load_artifact(...).predict(...)`.

## Последствия
- **+** Кросс-рантайм/кросс-язык serving поднабора; паритет **доказан** и гейтится; честность «не отгружать
  расходящийся граф».
- **+** Export-only → нет `onnxruntime` в горячем пути, нет паритет-бремени на загрузке; native остаётся точным
  load-back.
- **−** float32-приближение (крошечное, в пределах допуска) — задокументировано; baseline не экспортируется.
- **−** Препроцессинг — на стороне потребителя (по schema) — задокументированное ограничение export-only.
- **Day-2:** при спросе на ONNX-load-back — отдельный ADR (добавит `onnxruntime` в конус + load-back-обёртку);
  пины onnx-тулинга обновляются по матрице совместимости с зоопарком.
