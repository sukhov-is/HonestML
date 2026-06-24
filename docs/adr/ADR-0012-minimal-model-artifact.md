# ADR-0012 — Минимальный версионированный ModelArtifact + standalone-предсказание

- **Статус:** Proposed
- **Драйверы:** D3 (воспроизводимость), D6; FR-11, FR-15; NFR-9/11; E1.
- **Воркстрим:** M2 (минимум) → M8 (полный артефакт/ONNX/zero-dep).

## Контекст

FR-11 требует версионированный артефакт: save/load и **standalone-инференс** с
препроцессингом, идентичным train. Ранее per-type save/load и отдельный inference-модуль дублировали datetime/категориальный контракт (E1).
M2 фиксирует **минимальный** артефакт, чтобы замкнуть slice, и задаёт версионируемый
контракт, который M8 расширит (не переписывая).

## Рассмотренные варианты

1. **Pickle всего фасада** (FLAML-стиль) — хрупко к версиям sklearn/либ; известная
   боль. Отказ.
2. **Per-type save/load + отдельные файлы препроцессинга** — работает, но
   дублирует препроцессинг-контракт (E1) и не использует schema-owned
   `CategoryTable` из M1.
3. **Директория: manifest + сериализованная `FeatureSchema` + нативный файл модели**
   — препроцессинг = одна сериализуемая `FeatureSchema` (с `CategoryTable`),
   устранение дублирования E1; модель — в нативном формате (устойчивее pickle).

## Решение

Вариант **3**. Артефакт — директория:
- `manifest.json`: `artifact_version` (int, начинаем с 1), `honestml_version`,
  `task` (сериализованный `Task`), `metric`, `best_model_id`, `created_at` (передаётся,
  не генерится в домене).
- `schema.json`: сериализованная `FeatureSchema` (роли + **`CategoryTable`**) —
  единственный носитель препроцессинг-контракта (коды категорий train↔inference,
  ADR-0005). Никаких отдельных `categories.json`/`encoder.joblib`.
- `model.<ext>`: нативный формат адаптера (sklearn → `joblib`; бустинг → нативный,
  M3). Тип/файл указаны в manifest.
- `leaderboard.json`: результат для самодокументации (FR-14, минимум).

**Save/Load:** `save_artifact(slice_result, dir)` / `load_artifact(dir) ->
FittedModel`. `load` проверяет `artifact_version` → при несовпадении
**`SchemaValidationError`** (или будущий `ArtifactVersionError` в M8) с понятным
сообщением, а не тихий сбой в predict (закрывает находку ревью M1 про version-check).

**Standalone-predict (M2):** через **лёгкое ядро** `honestml` (core+`Reader(schema=
schema_)`+адаптер модели) — без обучающего тяжёлого стека, но с импортом `honestml`.
`predict(X)` = `Reader.read(X, schema)` → numpy+коды → `model.predict[_proba]`. Препроцессинг унифицирован через `FeatureSchema`.

**Граница M2 vs M8 (R4):** truly-zero-import standalone (копируемый файл), ONNX-экспорт, полный `ModelArtifact` (манифест прогона,
окружение/fingerprint для replay, FR-15) — **M8**. M2 даёт минимальный
forward-совместимый контракт (через `artifact_version`).

## Последствия

- (+) Устойчивее pickle-всего; препроцессинг — один сериализуемый источник истины
  (E1 закрыт по существу).
- (+) `artifact_version` даёт forward-совместимость и явную ошибку (NFR-9/11).
- (+) train==inference препроцессинг гарантирован переиспользованием `FeatureSchema`.
- (−) sklearn-модель грузится через joblib (pickle) — для M2 приемлемо с пометкой
  «доверенный источник»; целостность/подпись и нативные не-pickle форматы — M8.
- (−) «standalone» в M2 = лёгкое ядро, не zero-dep — явно задокументировано (OQ5).

## Уточнения контрактов (ревью, фаза 8)

- **`FittedModel.predict[_proba]` (F-minor):** загруженная модель переиспользует
  тот же путь, что и фасад — `Reader.read(X, schema)` → `X = hstack([to_numpy(),
  categorical_codes()])` (ADR-0013) → проекция `project_for_metric` (ADR-0010). Так
  препроцессинг и выбор positive-столбца идентичны train↔inference и facade↔
  standalone.
- **Диспетчер сериализации (F-minor):** `manifest.model_type` → загрузчик
  (`joblib` для sklearn в M2; нативные форматы бустингов — M3). Это forward-точка.
- **Источник метаданных (F-minor):** `created_at` и `honestml_version` передаёт
  composition-слой (`importlib.metadata.version("honestml")`), не домен (чистота).
- **Безопасность (M1-находка):** docstring `load_artifact` дублирует пометку
  «загружай только из доверенного источника» (joblib=pickle); целостность/подпись —
  M8.
- **`leaderboard.json`:** сериализуется как `list[LeaderboardEntry]` (схема —
  ADR-0010 §Уточнения п.8).
