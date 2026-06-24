# ADR-0027 — TimeSeriesCV: value-based порядок + purge + embargo + непустой es

- **Статус:** Accepted (реализован 2026-06-08, M4b)
- **Дата:** 2026-06-08
- **Драйверы:** DM4-2 (time-series CV без утечки); FR-M4-5/7/8, NFR-M4-3. Снимает M3-заглушки
  (ADR-0016 §C4 явно отдал это в M4). Использует `Dataset.time()` (ADR-0028).
- **Воркстрим:** M4b. Дельта к ADR-0013 (сплиттеры), ADR-0016 (резолвер).

## Контекст
`CVConfig` уже несёт `scheme='timeseries'`, `purge`, `embargo`, `n_test`, `n_es`, но
`build._resolve_splitter` жёстко отвергает их (`UnsupportedSchemeError`, `build.py:116-119,147-151`).
`validate_fold` имеет проверку порядка `max(fit)<min(es)<min(test)`, но **по индексам** и
гейтится непустым `es`; `slice.py:205` не передаёт `time_ordered=True`. Все сплиттеры эмитят
`es=_EMPTY`. Реальной TimeSeriesCV нет. Решение: **date gap+embargo (дефолт), полный
de-Prado purge по `t1` — опционально** (ADR-0028).

## Рассмотренные варианты
1. **sklearn `TimeSeriesSplit(gap=)`** напрямую. Позиционный (не value-based), `gap` односторонний,
   нет embargo-после-теста, нет purge-by-horizon, игнорит y/groups. Недостаточно. Отвергнут как
   единственное решение (переиспользуем идею expanding+gap).
2. **Кастомный `TimeSeriesSplitter`** (de Prado-стиль): value-based порядок по `dataset.time()`,
   expanding-окно, purge (gap до теста) + embargo (gap после), непустой `es`. **Выбран.**

## Решение

### 1. `TimeSeriesSplitter` (adapters/splitters.py), маркер `time_ordered = True`
`split(dataset) -> Iterator[Fold]`. Параметры: `n_splits`, `n_test`, `n_es`, `purge`, `embargo`.
- **Value-based порядок:** берёт `t = dataset.time()` и (опц.) `t1 = dataset.label_time()`
  **ИСКЛЮЧИТЕЛЬНО через порт `Dataset`** (не читает `__time__`/`__label_time__` по имени из frame —
  фикс R1-clean: сплиттер не связывается с деталью Reader). Строит **перестановку строк по
  возрастанию времени**; фолды нарезаются по этому порядку, НО `Fold.fit_idx/es_idx/test_idx`
  содержат **исходные индексы строк** (инвариант ADR-0023: индексы выровнены с
  `design_matrix(dataset)`; сплиттер не переупорядочивает датасет). Порт-аксессоры `time()`/
  `label_time()` индексно выровнены с `design_matrix` (документируется в docstring порта рядом с
  `groups()`).
- **Expanding-окно** (дефолт; rolling/`max_train_size` — аддитивный future-field, вне M4): для
  фолда k тест = следующие `n_test` (по времени) после накопленного train.
- **purge** = выбросить из train последние `purge` сэмплов (по времени) перед тестом (обобщение
  sklearn `gap`). **embargo** = выбросить `embargo` сэмплов train сразу ПОСЛЕ теста (серийная
  корреляция; Lopez de Prado). Единица — **счётчик сэмплов в порядке времени** (sklearn-семантика
  `gap`). **Гарантия = time-OVERLAP, не time-magnitude (фикс R1-adv):** purge/embargo удаляют
  смежные по времени сэмплы, а инвариант, который проверяется (§2), — «ни один train-сэмпл по
  ВРЕМЕНИ не попадает в тестовый интервал» (нулевое пересечение), а НЕ «временной зазор ≥ N единиц».
  Горизонт-утечка метки (label сверх теста) закрывается опц. `t1` (FR-M4-7), не счётчиком.
- **Непустой `es`** (фикс RM4-8, FR-M4-8): из хвоста train-периода каждого фолда карвится `n_es`
  последних (по времени) сэмплов в `es_idx`. Обучение на `fit∪es`
  как сейчас; ES-использование бустингами → M7 (хвост зарезервирован честно).

### 2. `validate_fold` — проверка по ЗНАЧЕНИЯМ времени (фикс «индексной» проверки)
Порт расширяется аддитивно: `validate_fold(fold, *, groups=None, time_ordered=False, times=None)`
(параметр `gap` НЕ вводится — фикс R1-adv «единицы сэмплы vs время»). При `time_ordered=True and
times is not None` проверяет **нулевое пересечение по времени**: `max(times[fit ∪ es]) <
min(times[test])` (ни один обучающий сэмпл, включая `es`, по времени не попадает в тестовый
интервал — `es` входит в обучение через `fit∪es`, slice.py:297-299, поэтому проверяется
`fit∪es`, а не только `fit`). Это делает анти-ликедж **проверяемым механизмом** на реальных датах
(NFR-M4-3, R-6). **Существующая ИНДЕКСНАЯ ветка `time_ordered` (`splitter.py:66-70`) ЗАМЕНЯЕТСЯ**
value-based (не дополняется — чтобы не было двух семантик флага; ветка сейчас мертва, регресс-риск
нулевой). **Точка вызова — единственная, в `run_slice` (`slice.py:204-205`), per-fold ДО цикла
кандидатов**: `slice.py` детектит маркеры через `runtime_checkable`-протоколы
`isinstance(splitter, TimeOrderedSplitter)` и `isinstance(splitter, GroupAwareSplitter)` (проверяют те же
атрибуты-маркеры `time_ordered`/`group_aware`) и передаёт
`times=dataset.time()`. Сам зазор purge/embargo обеспечивается СПЛИТТЕРОМ (удалением смежных
сэмплов), а `validate_fold` лишь подтверждает отсутствие пересечения.

### 3. `build._resolve_splitter` — снятие заглушек
- Убрать безусловный отказ `purge>0|embargo>0` (`:116-119`); вместо него: purge/embargo допустимы
  **только** при резолвнутом `scheme='timeseries'`, иначе `ConfigError` («purge/embargo требуют
  scheme='timeseries'»).
- Ветка `scheme='timeseries'` → `TimeSeriesSplitter(n_splits, n_test, n_es, purge, embargo)` (вместо
  отказа `:147-151`), требует `dataset.time() is not None` (иначе `ConfigError`, как group без группы).
- Резолвнутый scheme + purge/embargo пишутся в манифест (паттерн ADR-0016 — truthful manifest).
- `task.default_cv_scheme` НЕ возвращает timeseries — только явный `CVConfig.scheme='timeseries'`.
- **Look-ahead WARNING расширяется на роль TIME (фикс R1-consistency):** условие WARNING при
  шафлящей схеме = `has_datetime ∨ has_time` (а не только `has_datetime`/роль DATETIME) — иначе
  пользователь, объявивший ось через `time=` (роль TIME) и поставивший `scheme='stratified'/'kfold'`,
  не получит предупреждения об утечке времени. `facade` передаёт `has_time = ds.schema.time is not
  None` в `build`.

### 4. Скор leaderboard = pooled-OOF (≈ multi-window mean)
Скор как сейчас — метрика на пуле OOF по всем тестовым окнам (каждый фолд кладёт свои строки в
OOF). При равных `n_test` это ≈ среднее по окнам (AutoGluon multi-window), но переиспользует
существующий OOF-путь без новой машинерии.

> **Реализация M4b (отклонения, обоснованные):**
> - **embargo — cross-fold.** В forward-chaining expanding train всегда по времени ПЕРЕД тестом,
>   поэтому «embargo сразу после теста» для текущего фолда пусто. Единственная непустая семантика:
>   исключить зону `[end_test_j, end_test_j+embargo)` после РАННИХ тест-окон из train ПОЗДНИХ фолдов
>   (серийная корреляция, де Прадо). Реализовано так; проверено `test_timeseries_embargo_excludes_post_test_zone`.
> - **NFR-M4-3 leakage-probe → прямые fold-композиционные инварианты.** Вместо вероятностного probe
>   (флаки, требует обучения реальной модели и тонкой настройки порога [0.45,0.55]) анти-ликедж
>   проверяется ТОЧНО и детерминированно тремя инвариантами состава фолда: value-based overlap
>   (`validate_fold`), purge-magnitude (ровно `purge` сэмплов в зазоре), `t1`-horizon (train со
>   label-overlap удалён). Это сильнее probe — проверяет сам механизм, а не его статистическое следствие.
> - **block_index для TS-значимости.** `slice.py` строит fold-block `block_index` (id фолда на OOF-строку)
>   при `time_ordered`, закрывая шов с ADR-0026 §2 (band на TS-OOF → fold-block bootstrap, не iid).

## Последствия
- **Положительные:** roadmap-TimeSeriesCV закрыт; анти-ликедж по датам — проверяемый; `es`
  зарезервирован честно; manifest truthful; значимость получает временно-упорядоченный OOF для
  block-bootstrap (ADR-0026 §2).
- **Отрицательные/компромиссы:** purge/embargo в единицах сэмплов (не wall-clock) — для регулярных
  рядов точно, для нерегулярных — приближение (документируется); rolling-окно отложено; полный
  label-horizon purge — опц. (ADR-0028).
- **Влияние на слои:** новый сплиттер — `adapters`; аддитивное расширение `validate_fold` — `core`-порт
  (обратносовместимо, новые kwargs с дефолтами); проводка `times` — `application/slice`; роутинг —
  `composition`. import-linter не нарушен.

## Проверки
Две РАЗНЫЕ гарантии разведены (фикс R2-major «overlap vs magnitude»):
- **(1) Overlap-инвариант** — `validate_fold(time_ordered=True, times=t)` (БЕЗ `gap`): property-тест
  «нулевое пересечение времени train(`fit∪es`)↔test» (FR-M4-5). Это проверяет порядок/отсутствие
  пересечения, но НЕ величину purge.
- **(2) Purge/embargo-MAGNITUDE** — **отдельный property-тест НА СПЛИТТЕРЕ** (не в validate_fold):
  assert, что между `max(train_t)` и `min(test_t)` удалено ровно `purge` ближайших по времени сэмплов
  (и `embargo` после теста); при `t1` — удалены все train-строки с label-overlap. Без `t1` при
  expanding-окне `purge>0` меняет СОСТАВ train, но overlap-инвариант его «не видит» — документируется
  (purge применён ≠ purge проверен overlap'ом).
- ~~**Leakage-probe — операционализирован (фикс R2-major):** probe-признак `X_probe[i] = y[i]`
  (label-leak) ... `roc_auc ∈ [0.45,0.55]` ...~~ **ЗАМЕНЕНО impl-note (b):** вероятностный probe флаки
  (требует обучения реальной модели + тонкой настройки порога) и логически слаб (`X=y` даёт AUC=1.0 при
  любом CV — target-leak, не time-leak). NFR-M4-3 проверяется ТОЧНЫМИ детерминированными
  fold-композиционными инвариантами: value-based overlap (`validate_fold`), purge-magnitude (ровно
  `purge` сэмплов в зазоре), `t1`-horizon (train со label-overlap удалён) — они проверяют сам механизм.
- Фолды несут `es_idx≠∅`; порядок по времени, индексы — исходные (FR-M4-8, инвариант ADR-0023).
- `scheme='timeseries'` без `dataset.time()` → `ConfigError`; purge/embargo при не-timeseries →
  `ConfigError`; резолв+purge/embargo в манифесте.
- `timeseries` воспроизводим при фикс. конфиге (NFR-M4-2).
