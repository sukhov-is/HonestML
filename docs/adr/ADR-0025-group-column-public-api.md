# ADR-0025 — Публичный API объявления group-колонки: `groups=` в `fit`

- **Статус:** Accepted (реализован 2026-06-08, M3 follow-up)
- **Дата:** 2026-06-08
- **Драйверы:** DM-FU-1 (end-to-end достижимость group-CV); наследует DM3-4 (корректность
  group-leakage). FR-4b (дельта FR-4).
- **Воркстрим:** M3 follow-up, дельта к ADR-0023 (group-CV) и ADR-0011 (фасад) / ADR-0005
  (Reader, граница данных). Замыкает отложенный в `M3/06-traceability.md` пункт.
- **Основание дельты:** механизм ADR-0023 **реализован и смержен** (M3c, статус
  `✅ реализован 2026-06-08`, подтверждено кодом `adapters/splitters.py`,
  `application/slice.py`, `composition/build.py`) — дельта опирается на реальный код, а не
  на бумажное `Proposed`-решение.

## Контекст

ADR-0023 реализовал group-aware CV целиком, но `Reader` не присваивает
`ColumnRole.GROUP` (выводит только NUMERIC/CATEGORICAL/DATETIME/TARGET). Поэтому
group-CV достижим лишь через **вручную собранную** `FeatureSchema` с GROUP-ролью, а
через `AutoML(...).fit()` — нет. Фасад уже считает `has_group = ds.schema.group is not
None` и роутит `scheme="group"`; не хватает только публичного способа **проставить**
эту роль.

Существует прецедент-симметрия: `sample_weight` уже передаётся в `fit(X, y,
sample_weight=...)` и крепится `Reader` как зарезервированная колонка `__sample_weight__`
(`_WEIGHT_COL`) через `_attach_column`; `_infer_schema` её пропускает (не признак).
sklearn-каноника group-CV — `groups=` как метаданные на строку (`cross_val_score(...,
groups=)`, `GridSearchCV.fit(X, y, groups=)`, `splitter.split(X, y, groups)`).

## Рассмотренные варианты

1. **`groups=` массив в `fit` (Option A).** Метаданные на строку; `Reader` крепит
   `__group__` с ролью GROUP — точно как `sample_weight`. **Выбран.**
   - **+** sklearn-идиома; переиспользует `_attach_column` (DRY); единообразно для
     numpy/pandas/polars `X`; не трогает `__init__`/clone/get_params; группа не часть `X`,
     не попадает в признаки; симметрия с уже существующим `sample_weight`.
   - **−** `groups` — не гиперпараметр ⇒ не переносится через `clone`/`Pipeline`
     автоматически; обёртка в `GridSearchCV` требует sklearn metadata-routing/`fit_params`
     для проброса `groups` (стандартное для любого потребителя `groups=`, не breaking).
2. **`group_column="имя"` в `__init__` (Option B).** Группа — именованная колонка
   **внутри** `X`; конструкторный гиперпараметр; `Reader` помечает её GROUP и исключает
   из признаков. **Отвергнут.**
   - **+** гиперпараметр (clone/get_params/set_params, переживает клон); декларативно
     (как AutoGluon).
   - **−** требует именованных колонок (numpy → нужен `feature_names`); группа физически
     лежит в `X` (риск утечь в признаки, если забыть исключить); менее sklearn-идиоматично;
     рассинхрон при выравнивании именованных колонок.
3. **Только ручная `FeatureSchema` (статус-кво).** Отвергнут: это не публичный API,
   разрыв сохраняется.

## Решение (Option A)

### 1. `Reader` — новый kwarg `groups`
`Reader.read(X, y, *, schema=None, feature_names=None, sample_weight=None, groups=None)`.
Реализуется **зеркально `sample_weight`**:
- константа `_GROUP_COL = "__group__"` (рядом с `_WEIGHT_COL`);
- если `groups is not None`: `_attach_column(frame, _GROUP_COL, groups, "groups")` —
  валидация длины (`len != n_rows → SchemaValidationError`), крепится в тот же frame **до**
  ветки схемы (как weight) ⇒ построчное выравнивание гарантировано;
- `_infer_schema`: для `col == _GROUP_COL` → `roles[col] = ColumnRole.GROUP` (continue;
  не признак, не таргет, не категория). `_fit_categories` его не трогает (не categorical).

**Inference-ветка (schema задан):** безопасность держится на **двух независимых фактах**:
(1) `FittedModel._read` вызывает `Reader.read(..., schema=...)` **без** `groups` ⇒ на
inference `__group__` вообще не прикрепляется; (2) даже если бы прикрепился —
`_validate_against_schema` валидирует лишь **наличие required-`features`** (GROUP в них не
входит) и лишние колонки не запрещает, а `design_matrix` собирает только numeric⊕categorical
(GROUP не трогает). Точно как `sample_weight`. Сохранённая схема несёт GROUP-запись в
`roles` — это валидный enum-член внутри `dict`, грузится `model_validate_json` без правки
`ARTIFACT_VERSION`.

### 2. `AutoML.fit` — новый kwarg `groups`
`AutoML.fit(X, y, sample_weight=None, groups=None)`: прокидывает
`self._reader(task).read(X, y, sample_weight=sample_weight, groups=groups)`. Дальше
`has_group = ds.schema.group is not None` (уже есть) становится `True` ⇒ готовый роутинг
`scheme="group"` срабатывает. `__init__` **не меняется** (sklearn-инвариант сохранён,
ADR-0011). `predict`/`predict_proba`/`score` — без изменений (group — train-only).

### 3. Объявление группы ≠ авто-выбор group-CV (сохранение ADR-0023 §3)
`groups=` **только** проставляет GROUP-роль. Чтобы CV стала group-aware, пользователь
ставит `cv=CVConfig(scheme="group")` **явно**. Поведение комбинаций (всё уже в ADR-0023,
теперь достижимо публично, ничего не пересматриваем):
- `scheme="group"` + есть группа → group-aware CV (по `task.kind`: классификация →
  `StratifiedGroupKFold`, регрессия → `GroupKFold`).
- `scheme="group"` + **нет** группы → `ConfigError` (build, ADR-0023 §4).
- есть группа + резолвнутая **не-group** схема (`stratified`/`holdout`/`kfold`, в т.ч.
  `auto`→default) → **WARNING** «есть group-колонка, но CV не group-aware; используйте
  scheme='group'» (ADR-0023 §3, `build.py`). Авто-переключение **отвергнуто**: молча
  переопределять явный `cv` — хуже предсказуемости; возможная эргономическая надстройка —
  отдельным будущим ADR, не здесь (не правим §3 молча).
- **Механизм WARNING — `logging.warning` (как ADR-0023 §3 / ADR-0016 §5), не
  `warnings.warn`.** Тест проверяет через `caplog` (логгер), а не `pytest.warns` — фиксируем
  явно, чтобы тест целил в верный механизм.

### 4. Null-группы на публичной границе (фикс R-5; не ослаблять DM3-4)
`groups=` расширяет поверхность входа: теперь группа приходит **любым** pandas/polars/numpy
вводом через публичный kwarg, а не собирается экспертом вручную. Существующий guard
`_has_null_groups` (`splitters.py`, ADR-0023) ловит `float`-NaN и `object`-None, но **не**
`pandas.NA`/`Int64`-null (`np.asarray(pd.array([...], "Int64"))` → object-массив с `pd.NA`,
который ≠ `None` и ≠ `float('nan')`) ⇒ null **проскочил бы** в `StratifiedGroupKFold.split`,
ломая анти-ликедж тихо/неинформативно.

**Решение:** null-детекция группы — **на границе `Reader`** (где публичный вход и входит),
pandas/polars-aware (`pd.isna`-семантика или polars `.is_null()` после attach), **fail-fast
`SchemaValidationError`** «groups contains null/NaN; groups must be complete» — до того как
группа дойдёт до CV, и **независимо от схемы** (закрывает и «null + не-group схема молча
бессмысленны»). Сплиттер-guard ADR-0023 **остаётся** как defense-in-depth для пути ручной
схемы; сообщения консистентны. Это **дополняет**, а не пересматривает ADR-0023.

## Последствия

- **Положительные:** group-CV работает end-to-end из `AutoML.fit` (FR-4b закрыт);
  sklearn-идиома; переиспользование `_attach_column` (DRY, минимальный диф ≈ 1 kwarg в
  `Reader` + 1 ветка в `_infer_schema` + 1 kwarg-проброс в фасаде); единообразие для
  numpy/pandas/polars; группа не утекает в признаки и не нужна на inference; clone/get_params
  не затронуты.
- **Отрицательные / компромиссы:** `groups` — не гиперпараметр (не переносится `clone`/
  `Pipeline` без metadata-routing) — принято как стандартная sklearn-семантика `groups=`;
  объявление группы не включает group-CV автоматически — компенсируется WARNING (R-1).
- **Влияние на слои:** правки только в `adapters/reader.py` (+kwarg, +ветка роли,
  +null-guard границы) и `composition/facade.py` (+kwarg-проброс). `core` **не меняется**
  (`ColumnRole.GROUP`, `FeatureSchema.group`, `Dataset.groups()` — уже из ADR-0023).
  import-linter не нарушен.
- **Публичный контракт (DoD):** `groups` документируется в docstring `AutoML.fit` и
  `Reader.read` (это публичный API). `score(X, y)` **намеренно без** `groups`: это финальная
  метрика на переданных данных, не CV — групповой оценки на hold-out здесь нет (осознанное
  ограничение). `n_features_in_` от объявления `groups` **не меняется** (GROUP вне
  `features`) — фиксируется регресс-тестом.
- **Day-2 / совместимость:** аддитивно (новый kwarg default `None`); существующие вызовы
  не ломаются. `ARTIFACT_VERSION` **не трогаем**: схема может теперь нести GROUP-запись в
  `roles` (валидный enum-член, аддитивно к dict); старые артефакты без GROUP грузятся
  по-прежнему. Group — train-only, в инференс-манифесте роль не мешает.

## Проверки (→ тесты `implementation`)

- `Reader.read(X, y, groups=g)` → `schema.group == "__group__"`; `ds.groups()` построчно
  равно `g`; `"__group__" not in schema.features`.
- `AutoML(cv=CVConfig(scheme="group")).fit(X, y, groups=g)` завершается; затем `predict(X)`
  (без `groups`) работает — inference не требует group-колонки (R-2).
- **save→load→predict** (а не только fit→predict в одной сессии): артефакт со схемой,
  несущей GROUP-роль, грузится и предсказывает на `X` без `__group__` — пинит, что
  inference-безопасность держится на сериализации схемы (R-2/R-3).
- `AutoML(cv=CVConfig(scheme="group")).fit(X, y)` без `groups` → `ConfigError` (ADR-0023 §4).
- `fit(X, y, groups=g)` при не-group резолвнутой схеме → **WARNING** (через `caplog`,
  логгер), обучение идёт (R-1).
- `len(groups) != n_rows` → `SchemaValidationError` (граница, R-4).
- **`groups` c null/NaN, включая pandas-nullable (`pd.NA`/`Int64`)** → `SchemaValidationError`
  на границе `Reader`, независимо от схемы (R-5; анти-ликедж не роняется молча).
- numpy-`X` + `groups` работает (группа — отдельный массив, не зависит от имён колонок).
- `n_features_in_` не меняется при передаче `groups` (GROUP вне `features`).
- Обратная совместимость: `fit(X, y)` и `fit(X, y, sample_weight=w)` — без изменений
  поведения (регресс-тесты).
