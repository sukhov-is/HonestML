# ADR-0024 — Inference-путь (`FittedModel`/артефакт) multiclass + regression aware

- **Статус:** Accepted (реализован 2026-06-08, M3b)
- **Дата:** 2026-06-07
- **Драйверы:** DM3-3 (обобщение за пределы binary); FR-3, FR-5. north-star: D2/D5. Зависит от ADR-0012
  (минимальный артефакт), ADR-0021 (проекция/метрики), ADR-0020 (зоопарк/regression).
- **Воркстрим:** M3, дельта к ADR-0012. **Введён по находке ревью R2-C1.**

## Контекст

`FittedModel` (`composition/artifact.py`) — **единый inference-путь** фасада и standalone-load — **binary-only**:
- `score` (`:64-73`) берёт `predict_proba(x)[:, self._positive_index()]` (один столбец позитива) и зовёт
  `project_for_metric(metric, proba=…, pred=…)` **без** `kind`/multiclass-ветки;
- `_positive_index` (`:82-84`) — бинарная семантика;
- `load_artifact` (`:147`) делает `classes=np.asarray(estimator.classes_)` — для regression-эстиматора
  атрибута `classes_` **нет** → load падает;
- `resolve_metric(metric_name)` (`:146`) — без `classes`/`average` (нужно для multiclass, ADR-0021 §4) и
  без regression-метрик (ADR-0021 §5).

Без правки FR-3 (multiclass/regression end-to-end через фасад) **недостижим**: фасад обучит, но
`score`/`predict_proba`/`save`/`load` вернут мусор или бросят. ADR-0021/operational ошибочно полагали
«artifact.py не трогаем» — формат не меняется, но **код inference-пути** обязан стать kind-aware.

## Решение

### 1. `FittedModel.classes` — опционально (None для regression)
Тип `classes: np.ndarray | None`. Классификация — заполнено (`np.unique(y)`); regression — `None`.

### 2. `predict_proba`
- Классификация: возвращает `(n, K)`, **выровненный к `self.classes`** (та же `align_proba`-семантика,
  ADR-0021; единый глобальный порядок). Для binary совместимо (K=2).
- Regression: `SchemaValidationError("regression model has no probabilities")` (как фасад, ADR-0020 §4).

### 3. `score` — через kind-aware проекцию
Использует `project_for_metric(metric, *, proba, pred, kind=self.task.kind)` (ADR-0021 §3):
- binary + proba-метрика → 1-D `P(positive)`; multiclass + proba-метрика → `(n,K)` выровненный;
  classification + class-метрика → `pred`; regression → `pred` (value).
Метрика конструируется с `classes`/`average`: `resolve_metric(metric_name, classes=self.classes,
average=…)` (ADR-0021 §4). `_positive_index` остаётся только для binary-ветки.

### 4. Артефакт: аддитивные ключи манифеста, `ARTIFACT_VERSION` не меняется
- `save_artifact` дописывает в манифест: `classes` (список меток, для классификации; отсутствует/`null`
  для regression) и `metric_average` (если задан). Это **аддитивные** ключи.
- `load_artifact`: `classes` берёт из манифеста (если есть), иначе fallback на `estimator.classes_`
  **только если атрибут есть** (классиф), иначе `None` (regression) — **фикс падения load на regression**.
  `resolve_metric(metric_name, classes=…, average=…)`.
- **Back-compat:** старый (M2/binary) артефакт без новых ключей грузится как прежде (`classes` из
  `estimator.classes_`, `average=None`). `ARTIFACT_VERSION` остаётся `1` (только добавление опц. ключей
  манифеста + поведение чтения; downgrade-ограничение — как ADR-0012/operational).

## Последствия

- **Положительные:** FR-3/FR-5 действительно end-to-end (фасад **и** standalone-артефакт работают на
  binary/multiclass/regression); inference-путь единый и kind-aware; формат артефакта не ломается.
- **Отрицательные / компромиссы:** `FittedModel`/`save`/`load` затронуты (объём M3 растёт — но это
  обязательная часть FR-3, не опция); манифест получает 2 опц. ключа (аддитивно); ONNX/подпись/полный
  манифест — по-прежнему M8.
- **Влияние на слои:** изменения — `composition/artifact.py` (inference-путь); зовёт `project_for_metric`
  (application) и `resolve_metric` (adapters) — уже разрешённые направления. `core` не затронут.

## Проверки

multiclass: `fit`→`predict_proba` даёт `(n,K)`, `score` валиден, `save`→`load`→`score` воспроизводит;
regression: `fit`→`predict` значения, `predict_proba`→`SchemaValidationError`, **`save`→`load` не падает**
(нет `classes_`), `score` через value-метрику; binary: бит-в-бит как до дельты (load старого артефакта
работает); манифест со старыми ключами (без `classes`/`metric_average`) грузится (back-compat).
