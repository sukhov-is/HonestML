# ADR-0096: CV-схема `timeseries_period` и `PeriodTimeSeriesSplitter`

- **Статус:** Accepted (реализован)
- **Дата:** 2026-06-23
- **Драйверы:** D-1 (нарезка по периодам), D-2 (анти-ликедж), D-7 (datetime+числовая ось);
  FR-1/2/3/7/8/9, NFR-1/2/3/4/6/7/8/9. Расширяет ADR-0027 (timeseries) и ADR-0028 (ось времени).

## Контекст
`timeseries` (ADR-0027) упорядочивает строки по значению времени, но размеры окон — счётчики
строк (`adapters/splitters.py:395-407`). «Трейн = N месяцев / тест = M месяцев» и биржевые ряды
требуют нарезки по **периодам/длительности**. Нужна аддитивная схема, сохраняющая проверяемый
анти-ликедж и переиспользующая существующие абстракции (`Fold`, `time_ordered`, `validate_fold`,
`label_time`-purge, es-carve), не ломая текущее поведение и не меняя версию артефакта.

## Рассмотренные варианты
1. **Переинтерпретировать `n_test` как «периоды» в существующем `TimeSeriesSplitter`.** Минимум кода,
   но ломает семантику (две интерпретации одного поля), не даёт календарного floor и densify пустых
   периодов, смешивает row- и period-логику в одном классе. Отвергнут.
2. **Period-key как group-ось и переиспользование group-CV.** Group-CV шаффлит и не time-ordered;
   всё равно нужен ordered-group вариант; семантика времени размазывается по двум подсистемам.
   Отвергнут (см. отклонённый вариант 4 в обсуждении с пользователем).
3. **Новая закрытая схема `timeseries_period` + отдельный `PeriodTimeSeriesSplitter`** (паттерн
   диспетчера `_resolve_splitter`), переиспользующий маркер/валидацию/es-carve. **Выбран.**

## Решение

### 1. `CVConfig` (core, аддитивно)
- В `CVScheme` Literal добавить `"timeseries_period"`.
- Новые поля: `period: Literal["month","week","day","delta"] | None = None`;
  `period_size: float | None = None` (ширина Δt в единицах **числовой** оси, **только** для
  `period="delta"`); `step_periods: int | None = None` (None → `=n_test`, смежная непересекающаяся
  плитка). `n_splits` переиспользуется (число фолдов).
- **Переиспользование существующих знобов (F5, без `test_periods`):** под `timeseries_period`
  ширина теста — это уже существующее `n_test` (его описание уже «test-fold size (folds or
  periods)»; docstring уточняется). Единое **правило единиц:** целочисленные `n_test`/`purge`/
  `embargo` — в натуральной единице схемы (строки для `timeseries`, **периоды** для
  `timeseries_period`); `n_es` — всегда строки (внутренняя деталь ES, не окно CV). Фиксированной
  глубины train (`train_periods`) НЕТ — train expanding (или ограничен `max_train_periods`, ADR-0099).
- Duration-альтернатива зазора — `purge_delta`/`embargo_delta` (ADR-0097); `max_train_*`/`weighting`
  вводятся в ADR-0099/0098.
- **Валидатор `CVConfig` (новый `model_validator`, прецедент BudgetConfig/FeatureSelectionConfig):**
  field-coherence `period_size` обязателен ⇔ `period="delta"` бросает `ValueError` → у прямого
  `CVConfig(...)` это pydantic `ValidationError` (как Budget/FS, test_core_config.py:163), на путях
  `RunConfig.parse`/preset-dict — `ConfigError` (G2/F15). Схемо-зависимые гейты (`period`/
  `period_size`/`step_periods`/`max_train_periods` требуют `timeseries_period`) — в `build` после
  резолва `auto` → `ConfigError` (паттерн `build.py:677-680`). **`n_test` гейта НЕ имеет** — валиден
  под обеими time-схемами (строки/периоды), G1.

### 2. `PeriodTimeSeriesSplitter` (adapters/splitters.py), маркер `time_ordered = True`
`split(dataset) -> Iterator[Fold]`:
- `t = dataset.time()`; `None` → `SchemaValidationError` (как `TimeSeriesSplitter`).
- **Валидация dtype↔period (F6):** `month/week/day` требуют `np.issubdtype(t.dtype, np.datetime64)`;
  `delta` требует **числовую** ось и `period_size>0`. Несоответствие → `SchemaValidationError` (R-2).
  (Datetime+delta запрещён — ширина зависела бы от единицы хранения `time()`.)
- **Ключ периода построчно (векторно, O(n)):** `month`/`day` — floor через
  `t.astype("datetime64[M|D]")`; `week` — ISO (понедельник): day-floor + целочисленная арифметика
  от понедельника (НЕ `datetime64[W]`, привязанный к четвергу — F12); `delta` —
  `((t - t.min())/period_size).astype(int64)` (паттерн `feature_selection.structure_labels:42-50`).
  Затем **densify** в плотные id `0..P-1` по возрастанию (`np.unique(..., return_inverse=True)`) —
  пустые периоды не оставляют дыр и не сдвигают индексацию исходных строк (R-3).
- **Walk-forward по оси периодов:** **последний** фолд заканчивается на последнем периоде `P`
  (последние периоды — тест, как expanding в ADR-0027), а более ранние фолды отсчитываются назад на
  `step`: `first = (P - n_test) - (n_splits-1)*step` (`step = step_periods or n_test`); для фолда k тест
  = периоды `[first + k*step, first + k*step + n_test)`; train = периоды строго раньше начала теста
  (expanding; нижняя граница — ADR-0099). Строки фолда = строки соответствующих периодов.
  `step_periods > n_test` допустимо (контролируемый gap — периоды в дырах не тестируются, G13);
  `step_periods < n_test` даёт перекрытие тест-окон. (Уточнение при реализации, ревью этапа 1:
  формула `first = P - n_splits*n_test` верна только при `step = n_test`; общий вид выше совпадает с
  ней в этом частном случае, но при `step ≠ n_test` не выводит окна за ось периодов — иначе пустое
  тест-окно роняло бы сырой `ValueError` / молча усекалось бы, нарушая FR-3/FR-8.)
- **Анти-ликедж переиспользуется:** строки train **сортируются по `t`** (как `order` в
  `TimeSeriesSplitter`, F10) и фильтруются `t[train] < test_min_t` (строгая отсечка по значению,
  как `splitters.py:416`), и `t1[train] < test_min_t` при `label_time` (de-Prado horizon purge).
  Зазор `purge`/`embargo` (в периодах) или Δt — ADR-0097.
- **es-хвост:** `_carve_es` переиспользуется НА ОТСОРТИРОВАННОМ по времени train (иначе
  `train[-n_es:]` не «последние по времени», F10): n_es последних строк train в `es_idx`,
  клампинг до ≥1 fit-строки.
- **Feasibility-гейт:** `first - purge < 1` (нужен ≥1 train-период перед первым тест-окном, с учётом
  `step` и `purge` в периодах) → `SchemaValidationError`; дополнительная проверка по числу строк train
  ПОСЛЕ материализации — в ADR-0099 §2 (F8). (Уточнение при реализации: гейт учитывает `step`/`purge`,
  а не только `total` — иначе `step>n_test` или `purge`, опустошающий первый фолд, давали бы сырой
  `ValueError` / вводящее в заблуждение сообщение es-carve вместо понятной ошибки, NFR-8.)

### 3. `build._resolve_splitter` и сопутствующие пути (composition)
- Новая ветка `elif scheme == "timeseries_period"`: требует `has_time` (иначе `ConfigError`,
  зеркало `timeseries`); строит `PeriodTimeSeriesSplitter(...)`.
- Гейты схемо-зависимых знобов (паттерн `build.py:677-680`): `period`/`period_size`/`step_periods`/
  `max_train_periods` при не-`timeseries_period` → `ConfigError`. `period` обязателен при
  `timeseries_period`. **`n_test` НЕ гейтится** (валиден под обеими схемами, G1/F5). Резолвнутая
  схема + параметры → манифест (truthful, FR-8).
- **`outer_holdout_carve`/`_timeseries_carve` (F1, BLOCKER):** сейчас диспетчер срабатывает строго
  на `scheme == "timeseries"` (`splitters.py:313`), иначе уходит в **шаффлящий** `HoldoutSplitter` →
  тихая утечка во времени под period-схемой → **обобщить на `scheme in ("timeseries",
  "timeseries_period")`**. **Под `timeseries_period` карв period-aware (G4):** строит тот же
  period-ключ (floor); holdout = последние периоды, покрывающие ≥`fraction` строк; зазор = `purge`
  **в периодах** ИЛИ `purge_delta` (Δt, value-based) — единица согласована со схемой, не «строки».
  Тот же диспетчер обслуживает FS selection-holdout (`_make_selection_carve`, `build.py:617-624`) и
  outer-holdout (`facade._carve_holdout`).
- **FS авто-резолвер (F9/G11):** правка нужна в `resolve_fs_defaults` (`build.py:512`, ветка
  `scheme == "timeseries" and purge == 0` → `arbitration='holdout'`) — обобщить на `scheme in
  ("timeseries","timeseries_period")` и заменить условие на `is_purged = purge>0 or purge_delta is
  not None` (учесть, что период-граница сама даёт разделение). `_resolve_block_mode` правки НЕ
  требует — он ветвится по наличию массива `times`, схемо-независим (G11). Согласование FS-структуры
  `null_block_*` с `period` — независимо (обосновано в ADR-0098 §4).
- `task.default_cv_scheme` **не** возвращает period-схему (только явный выбор) — как timeseries.
- Look-ahead WARNING (`build.py:732-738`) уже покрывает `has_time` — период-схема не шаффлит,
  предупреждения не вызывает.

### 4. Метаданные нарезки в отчёт (application/run_report, G6)
FR-8 (truthful manifest) для главного результата фичи требует канала наружу: `model_dump` несёт лишь
**входной** конфиг, а число densified периодов, число фолдов и число дропнутых пустых/невалидных
периодов вычисляются внутри `PeriodTimeSeriesSplitter`/`run_slice` и сейчас выбрасываются. Решение
(аддитивно, прецедент `native_routing`/`fs_resolution`): опциональное поле `SliceResult.cv_split:
dict | None` (`{n_periods, n_folds, n_dropped_empty, period, weighting}`), заполняемое для
time-ordered схем; `build_run_report` сериализует его в отчётный блок `cv`. `LeaderboardEntry`
(`extra='forbid'`) и формат артефакта не трогаются (G7).

## Последствия
- **Положительные:** «трейн N / тест M периодов» выразимо end-to-end; календарные и Δt-окна;
  анти-ликедж — тот же проверяемый механизм; нулевой bump версии артефакта; диспетчер/валидация/
  es-carve переиспользованы (NFR-7).
- **Отрицательные/компромиссы:** `n_test`/`purge`/`embargo` меняют единицу под период-схемой
  (периоды) — документируется правилом единиц; `week` = ISO-понедельник (реализуется явно, не
  numpy `[W]`); `delta` ограничен числовой осью — для **биржевого** ряда с фиксированным Δt-окном
  подайте время как числовую (epoch) ось через `time=`, либо используйте календарные `day`/
  `week`/`month` (G13); таймзоны/DST вне области. **God-config (F13,
  accept):** `CVConfig` разрастается, но существующие time-series знобы уже плоские на `CVConfig` —
  ради консистентности CV-параметры остаются плоскими (не выносятся в под-модель, в отличие от
  ортогональных каталогов `fs`/`hpo`/`ensemble`); рост поверхности — R-8.
- **Влияние на слои:** `core` (config — аддитивно), `adapters` (новый сплиттер), `composition`
  (диспетчер). Граф зависимостей внутрь; `.importlinter` не требует allowlist (новый класс в уже
  покрытом `adapters`).

## Проверки
- `scheme='timeseries_period'` без `time=` → `ConfigError`; с datetime `time=` отрабатывает
  end-to-end; манифест несёт резолвнутую схему и параметры (FR-1/8).
- Календарь на числовой оси → `SchemaValidationError`; `delta` без `period_size` → `ValueError`
  валидатора (FR-9, R-2).
- Unit на datetime-фикстуре: строки одного месяца/недели в одном бакете; пустой период не даёт
  фолда (NFR-3, R-3).
- property-тест value-based overlap для периодных фолдов (NFR-2, FR-7); недостаток периодов →
  понятная ошибка (FR-3, R-4-смежн.).
- `uv run pytest`/`mypy`/`ruff`/`lint-imports` зелёные без правок существующих тестов схем (NFR-5/6).
