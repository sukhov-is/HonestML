# ADR-0045 — Сериализация subset в `FeatureSchema`, inference-проекция через `design_matrix`, совместимость artifact/кэш

- **Статус:** Accepted (реализован в M6b, 2026-06-10; `design_matrix` selection-aware, см. `09-review.md`)
- **Дата:** 2026-06-09
- **Драйверы:** DM-2 (train==inference), DM-4 (аддитивность/кэш), DM-6 (наблюдаемость); FR-FS-4/6/7,
  NFR-FS-3/4. Наследует schema-owned спеки (ADR-0005/0042), artifact-контракт (ADR-0012),
  `design_matrix`-choke-point (ADR-0013 §F9), run-fingerprint (ADR-0035).
- **Воркстрим:** M6b (Feature Selection).

## Контекст
Отбор даёт ценность только при **train==inference**: подмножество, выбранное на train, должно примениться на
inference **без пересчёта рэнкинга** и **детерминированно**. Прецедент — FE-спеки M6a (схема владеет, artifact
возит, Reader/`design_matrix` применяет). Нужно решить **где живёт subset**, **как он проецируется на inference
с минимальным дифом** и **совместимость** artifact/кэша.

## Рассмотренные варианты (где живёт subset)
1. **Отдельный файл-список рядом с artifact.** Вне единого источника истины препроцессинга; рассинхрон.
   **Отвергнут.**
2. **Поле на `FittedModel`/манифесте.** Subset — про препроцессинг входа модели, а его источник истины —
   `FeatureSchema` (как `categories`/FE-спеки); раздвоение владельца. **Отвергнут.**
3. **Аддитивное frozen-поле `FeatureSchema.selected_features`** + проекция в `design_matrix`. Единый
   сериализуемый источник, Reader/`design_matrix` применяют, predict-путь **не меняется**. **Выбран.**

## Решение

### 1. Спека subset — аддитивное frozen-поле `FeatureSchema` (core/schema.py)
```python
selected_features: tuple[str, ...] | None = None   # None → все признаки (legacy/off)
```
+ copy-update helper `with_selected_features(names)` (как `with_categories`/`with_*` M6a). **Именованный**
список (не позиционная маска) — устойчив к порядку (R-FS-COLDRIFT). Дефолт `None` → **старый artifact грузится**
(поле отсутствует → отбора нет). Subset хранит признаки в порядке `schema.features` (FR-FS-7).

### 2. Проекция — в `design_matrix` (один choke-point), predict-путь без правок
`design_matrix` строит **полный** numeric ⊕ cat-коды в порядке `schema.features`, затем **проецирует** на subset
**когда он задан**:
```python
def design_matrix(dataset) -> np.ndarray:
    full = np.hstack([numeric, codes])                       # как сейчас; §F9-guard на полном наборе
    sel = dataset.schema.selected_features
    if sel is None:
        return full
    name_to_col = {f: i for i, f in enumerate(dataset.schema.features)}
    try:
        keep = [name_to_col[f] for f in sel]                 # порядок subset (= порядок features, §1)
    except KeyError as e:
        raise SchemaValidationError(f"selected feature {e} absent from design matrix")  # fail-loud, FR-FS-4
    return full[:, keep]
```
Так **все** потребители (`run_slice`/`refit_best`/inference `predict`) получают спроецированную матрицу из
**одного** места → train==inference по построению, predict-путь artifact **не трогается** (DM-2, минимальный
диф). `refit_best` ставит `est.feature_names = list(schema.selected_features or schema.features)`.

**Порядок вычисления subset (явные шаги, нет двойной проекции — уточнено по ревью):**
1. `Reader` отдаёт в `run_slice` схему **без** `selected_features` (`None`).
2. `run_slice` строит `x_full = design_matrix(ds)` → **полный** (None ⇒ проекции нет; нужен для рэнкинга).
3. `run_slice` вызывает `subset = select_features(x_full, y, folds, …)` (ADR-0044) — **один раз**, до цикла
   кандидатов.
4. `select_features` вернул **индексы** `idx: tuple[int,...]`; `run_slice` проецирует в памяти:
   `x_eval = x_full[:, idx]`, пересчитывает категориальную маску `cat_eval = categorical[list(idx)]` (см. ниже)
   и конвертирует в **имена**: `subset_names = tuple(feature_names[i] for i in idx)`. leaderboard/band/кандидаты
   — на `x_eval` с `cat_eval`. `SliceResult` возвращает `subset_names`.
5. **После** прогона фасад прикрепляет subset к схеме: `schema_sel = schema.with_selected_features(subset_names)`.
6. `refit_best`/artifact идут по `schema_sel` → здесь `design_matrix` **проецирует** автоматически. На
   inference — то же `schema_sel`.

Полная матрица строится **только** на шаге 2 (до выбора subset); проекция применяется **ровно один раз** на
каждом пути (в run_slice — явно шаг 4; на refit/inference — внутри `design_matrix`). Двойного применения нет.

**Категориальная граница после проекции (фикс ревью R2):** native cat-handling (LightGBM/CatBoost) требует
знать, какие колонки `x_eval` категориальны. Проекция **сохраняет** это: `cat_eval = categorical[list(idx)]` —
маска проецируется тем же набором индексов, что и матрица, в том же порядке → выравнено по столбцам. Если отбор
исключил **все** категориальные (или все numeric) — это валидно (остаётся ≥1 признак, §F9 floor); cat-маска
просто становится всё-`False`/всё-`True`. На refit/inference маска восстанавливается из `schema_sel`
(первые `len(numeric∩subset)` — numeric, остальные — cat) — тот же инвариант, что в `design_matrix`.

### 3. Композиция (facade) и наблюдаемость
- `AutoML.fit`: после `run_slice` (вернул subset в `SliceResult`) → `schema_sel = schema.with_selected_features(
  subset)`; refit и artifact используют `schema_sel`. Дефолт-off (`fs=None`) → subset не вычисляется, схема без
  поля (идентично M6a).
- **Наблюдаемость (FR-FS-6):** FS-конфиг (стратегия + cutoff + параметры) — в `report["config"]` через
  config-дамп `RunConfig.fs`; исход — `report["feature_selection"]` (аддитивно): `strategy`, `n_selected`,
  `n_total`, и (опц.) список `selected`/`dropped`. Выбранные/отброшенные восстановимы как
  `schema_sel.selected_features` vs `schema.features`; `n_features` в leaderboard отражает размер subset.
  Дефолт-off → ключ `feature_selection` отсутствует/`null` (отчёт как M6a). `RUN_MANIFEST_VERSION` **не**
  бампается. (Версионирование самой стратегии — вне M6b; future при появлении third-party стратегий, M6c.)

### 4. Совместимость artifact (`ARTIFACT_VERSION` не бампается)
Поле `selected_features` аддитивно (дефолт `None`) → **новый код читает старый artifact** (поля нет → все
признаки). `ARTIFACT_VERSION` остаётся **1** (вперёд-совместимо). **Downgrade не поддержан** (старый код +
новый artifact с полем → падение `extra="forbid"`) — принятый предел Day-2 (тождественно ADR-0042 §3).

### 5. Кэш / run-fingerprint (NFR-FS-4)
`RunConfig.fs` входит в run-fingerprint **автоматически** (хеш `RunConfig`, ADR-0035) → смена стратегии/политики
→ другой кэш-ключ (нет stale-hit, R-FS-CACHE). **Замечание о совместимости fingerprint:** добавление поля
`fs: … = None` в `RunConfig` меняет сериализованный дамп (`"fs": null`) → fingerprint M6b ≠ M6a для логически
того же прогона. Это **ожидаемо** и **тождественно** тому, как M6a добавил `fe` (ADR-0042 §4): **содержимое**
leaderboard/artifact при `fs=None` идентично M6a (тот же `x_full`, те же модели), различается лишь кэш-ключ.
FR-FS-1 «идентично M6a» относится к **результату** (leaderboard/artifact content), **не** к строке fingerprint
(формулировка FR-FS-1 уточнена по ревью R2: «контент идентичен, кэш-ключ отличается»).

**IMPACT — молчаливая инвалидация кэша M6a (фикс ревью R2, в Release Notes):** так как дефолтный `RunConfig()`
теперь несёт `"fs": null`, **весь существующий кэш прогонов M6a перестаёт совпадать** по fingerprint — это
**cache-miss (не ошибка)**, прогоны пересчитываются с нуля. Для активных проектов это материальная стоимость
(часы вычислений). Тождественно эффекту добавления `fe` в M6a, но **должно быть явно** в Release Notes
апгрейда. Альтернатива «исключать `fs=None` из дампа fingerprint» отвергнута: усложняет универсальный
`hash(RunConfig.model_dump())` спец-логикой ради разовой миграции — **не стоит** (Day-2-предел принят).

## Последствия
- (+) Единый сериализуемый источник subset; predict-путь artifact **не меняется** (проекция в одном
  choke-point); старый artifact грузится; FS в fingerprint (нет stale-hit); переиспользование
  schema/`design_matrix`-паттерна; train==inference по построению.
- (−/компромисс) `design_matrix` становится selection-aware (одна ветка `if sel is None`) — оправдано единым
  choke-point и нетронутым predict; downgrade artifact не поддержан (Day-2, как ADR-0042).
- **Влияние на слои:** спека — `core`; проекция — `application` (`design_matrix`); прикрепление/наблюдаемость —
  `composition`. `ARTIFACT_VERSION`/`RUN_MANIFEST_VERSION` не тронуты.

## Проверки
- `FeatureSchema` с `selected_features` — JSON round-trip; старый artifact (без поля) грузится и предсказывает
  (subset = все).
- `predict` фасада == artifact на тех же данных; subset с признаком, отсутствующим в `design_matrix`, →
  `SchemaValidationError` (не тихий сдвиг колонок).
- `design_matrix`: `selected=None` → полный (идентично M6a); заданный subset → проекция в порядке `features`;
  §F9 (≥1) сохранён (floor ADR-0044 §3).
- Другой FS-конфиг → другой `run_fingerprint`; `ARTIFACT_VERSION`/`RUN_MANIFEST_VERSION` не изменены; FS-конфиг
  в `report["config"]`; число признаков в leaderboard уменьшается при включённом отборе.
