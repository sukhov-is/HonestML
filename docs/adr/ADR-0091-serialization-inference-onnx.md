# ADR-0091: Сериализация, inference-паритет и ONNX-политика для нативных категорий

- **Статус:** Proposed
- **Дата:** 2026-06-22
- **Драйверы:** D-5, D-6 / FR-4, FR-5, FR-6, NFR-7; опирается на SPIKE-0004, SPIKE-0005

## Контекст
Inference идёт через тот же `design_matrix` → `FittedModel.predict*` (ADR-0024),
что даёт train==inference при кодах. С нативным входом нужно, чтобы predict
применял **то же** категориальное представление, что и fit. Из research:
- **CatBoost** на predict требует те же int/string cat-колонки → обёртка должна
  знать категориальные индексы и на inference (для int-каста) → их надо **хранить**.
  LightGBM бакает категории в booster и принимает float64 (округление) → доп.
  состояние не требуется, но индексы полезно хранить для единообразия.
- Дефолт сериализации — **joblib** (пиклит обёртку целиком, ADR-0070) → нативная
  модель и её `categorical_indices` сохраняются «бесплатно». Native `.cbm`/`.txt`
  — opt-in; `from_native` восстанавливает обёртку (`boosting.py:188-218`).
- **ONNX** нативные категории не держит: CatBoost — невозможно (issue #863,
  numeric-only); LightGBM — `onnxmltools` разворачивает в `==`-цепочки (дорого),
  отложено. SPIKE-0003 показал паритет ONNX↔native **именно на ordinal-кодах**.

## Рассмотренные варианты (ONNX)
1. **Поддержать ONNX-native сразу** (CatBoost невозможен; LightGBM-unfold) —
   CatBoost блокирует целиком; LightGBM дорог и требует полноценного re-spike +
   паритет-гейта; вне разумного объёма v1.
2. **Молча падать в native при запросе ONNX** — нарушает прозрачность (NFR-6):
   пользователь думает, что получил ONNX.
3. **(выбрано) Явный гейт:** нативно-категориальная модель сериализуется joblib
   (дефолт) или native; ONNX-экспорт такой модели **отклоняется понятной ошибкой**;
   ONNX-native откладывается отдельным backlog-пунктом (re-spike, SPIKE-0005).

## Решение
- **Inference-паритет (FR-4):** обёртка применяет ту же материализацию (ADR-0089) и
  на predict: CatBoost — int-каст cat-колонок по сохранённым `categorical_indices`;
  LightGBM — float64 как есть (booster содержит категории). Индексы пересобираются
  из той же спроецированной матрицы → совпадают с train по построению.
- **Сериализация (FR-5):**
  - joblib (дефолт): обёртка с `categorical_indices` пиклится целиком — round-trip
    «из коробки».
  - native (`.cbm`/`.txt`): `native_model().save_model(...)` сохраняет категориальную
    структуру; `from_native` восстанавливает **обёртку**, а её `categorical_indices`
    проставляет **сериализатор `load`** из манифеста (`_NativeBoostingSerializer.load`),
    у которого есть доступ к `manifest` — `from_native` формат манифеста не знает.
    **Место хранения:** `manifest.json`, поле
    `categorical_indices: list[int] | null` рядом с `model.<ext>`; для joblib он же
    внутри пикла обёртки. Для не-нативных моделей формат не меняется (NFR-7).
  - **Версионирование — без bump'а `artifact_version`.** Поле добавляется
    **аддитивно** (как существующий `early_stopping: bool`, `artifact.py:243`),
    `artifact_version` остаётся `1`. Это сознательно: загрузчик гейтит версию
    **строгим** `!=` (`artifact.py:322-325`), поэтому bump сломал бы чтение прежних
    артефактов. Чтение: `categorical_indices = manifest.get("categorical_indices")`
    → отсутствует ⇒ `None`/`[]` ⇒ путь кодов.
  - **Запись решения роутинга (репликабельность, NFR-6):** в manifest добавляется
    аддитивное поле `native_categorical: {backend, n_cat} | null` (по образцу
    `early_stopping`), чтобы по артефакту было видно, обучалась ли модель нативно и
    на скольких категориях — не только в логах.
- **ONNX-политика (FR-6):** при ONNX-экспорте (`export_onnx()`; гейт в
  `adapters/onnx_export.py::convert` **до** конвертера — ONNX не `model_type`,
  поэтому `save_artifact` его не пишет, ADR-0071) модели, обученной нативно
  (`categorical_indices` непуст) — **для обоих бэкендов**
  (CatBoost и LightGBM) — путь экспорта поднимает специфичное исключение
  `NativeCategoricalONNXUnsupportedError` (наследник существующего семейства ошибок
  сериализации) с сообщением «native categorical models are not ONNX-exportable on
  v1 (CatBoost: catboost#863; LightGBM: deferred to ONNX re-spike); use joblib or
  native format». Гейт срабатывает **до** вызова конвертера → молчаливого неверного
  графа нет. На v1 native-cat ⇒ joblib/native only.
- **Обратная совместимость:** ранее сохранённые артефакты не содержат
  `categorical_indices` → `manifest.get(...)` даёт `None` ⇒ модель на кодах,
  поведение прежнее (NFR-7). **Downgrade не поддерживается:** нативный артефакт
  новой версии, прочитанный старой библиотекой (без native-cat), приведёт к
  громкому отказу (CatBoost отвергает float в cat / несовпадение обёртки при
  unpickle), а не к тихо неверному предсказанию — фиксируется в CHANGELOG.

## Последствия
- **Положительные:** train==inference сохранён (FR-4); дефолтный joblib-путь несёт
  native-cat без спец-кода; ONNX-ограничение явно и безопасно; политика «native =
  load-back, ONNX = export-only» (SPIKE-0003) не нарушена.
- **Отрицательные / компромиссы:** нативно-категориальные модели теряют ONNX-экспорт
  (документированное ограничение, как в backlog); новое поле артефакта
  `categorical_indices` (аддитивно, опционально).
- **Влияние на слои/границы:** сериализаторы/`FittedModel` — composition/adapters;
  гейт ONNX — там же, где выбор формата; портов не трогает.

## Проверки
- SPIKE-0004: round-trip joblib и native ≤ 1e-6 для CatBoost/LightGBM (FR-5);
  train↔inference паритет (FR-4).
- SPIKE-0005: CatBoost-native→ONNX отклоняется; гейт ловит native-cat до конвертера
  (FR-6).
- Тест: загрузка старого артефакта без `categorical_indices` → модель на кодах,
  поведение неизменно (NFR-7).
- Тест: сообщение гейта ONNX присутствует и специфично (NFR-6, FR-6).
