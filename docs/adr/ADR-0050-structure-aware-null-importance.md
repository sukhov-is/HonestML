# ADR-0050 — Structure-aware перестановка `null_importance` для timeseries/group + контракт-change `FeatureRanker.rank`

- **Статус:** Принят (реализован в M6d, 2026-06-10; `NullImportanceRanker._permute_target` within-block,
  `structure_labels` ранг-биннинг, `rank(+groups)`, гард снят, `null_block_stats`). Питается **SPIKE-M6d-validity**.
- **Дата:** 2026-06-10
- **Драйверы:** DM-H1 (структурная честность null под ts/group). FR-FSH-1/2, NFR-FSH-1/3/6/7. Наследует
  методологию `null_importance`/i.i.d.-ограничение (ADR-0047 §1), спайн анти-ликеджа (ADR-0044), `Dataset.groups()`
  /`Dataset.time()` (ADR-0023/0028), обратносовместимый kwarg порта (паттерн `validate_fold`, ADR-0027 §2).
- **Воркстрим:** M6d.

## Контекст
M6c (ADR-0047 §1) **отклоняет** `null_importance` при `scheme∈{timeseries, group}` через `ConfigError`:
равномерная `rng.permutation(y)` предполагает **обмениваемость** таргета, которая ложна при автокорреляции
(timeseries) и внутригрупповой зависимости (group). Это смещение **не имеет корректирующего якоря** (проникает
прямо в замороженный subset) → выбрана fail-loud-заглушка, а метод отложен в M6d. Теперь нужно **заменить
отклонение на корректный structure-aware null**: перестановку **внутри** структуры, где обмениваемость
правдоподобна.

**Проблема проводки (research §1.1):** перестановка — деталь адаптера (`NullImportanceRanker.rank`), но адаптер
**не видит** структуру строк: порт `FeatureRanker.rank(x, y, *, categorical, random_state, sample_weight)` не
несёт меток блока/группы, а спайн срезает их на `train_idx`. ⇒ метку надо **прокинуть** в адаптер.

## Рассмотренные варианты (проводка структуры)
1. **Тащить `Dataset` в спайн/адаптер.** Нарушает `core-independence` (спайн — чистый numpy) и Humble-границу.
   **Отвергнут.**
2. **Глобальный стейт / отдельный канал.** Непрозрачно, неконтрактно. **Отвергнут.**
3. **Обратносовместимый kwarg `groups: np.ndarray|None=None` в `FeatureRanker.rank`; метки готовит
   `application` (где есть `Dataset`), спайн срезает `groups[train_idx]` и передаёт — как `sample_weight`.**
   **Выбран** (минимальный контракт-change, паттерн `validate_fold`).

## Решение

### 1. Контракт-change порта `FeatureRanker.rank` (`core/ports/feature_ranker.py`)
```python
def rank(self, x, y, *, categorical, random_state, sample_weight=None,
         groups: np.ndarray | None = None) -> np.ndarray: ...
```
- `groups` — **метка блока/группы на строку** `x` (index-aligned), `None` ⇒ i.i.d. (равномерная перестановка,
  **тождественно M6c**). Чистый numpy (без sklearn/Dataset) → `core-independence` KEPT.
- **Обратносовместимо (R-PORT):** `importance`/`random_probe`/`shap` принимают и **игнорируют** `groups`
  (равномерные ранжеры структуры не используют). Существующие/плагинные ранжеры с дефолтом `None` работают
  как прежде (FR-FSH-2).

### 2. Структурная перестановка (`adapters/feature_rankers.py`, `NullImportanceRanker`)
Заменить `rng.permutation(y)` на перестановку **внутри** структуры, когда `groups` задан:
```python
def _permute(y, groups, rng):
    if groups is None:
        return rng.permutation(y)                     # i.i.d. (M6c)
    out = y.copy()
    for g in np.unique(groups):                       # внутри каждого блока/группы
        idx = np.where(groups == g)[0]
        out[idx] = rng.permutation(y[idx])
    return out
```
- **group:** `groups` = `Dataset.groups()` срез фолда → перестановка внутри группы (обмениваемость внутри группы
  правдоподобна).
- **timeseries:** `groups` = **индекс временного блока**, построенный по **временно́му рангу** строк (фикс
  R1-B2/F1). Критично: `Dataset.time()` идёт в **порядке строк**, который **не** отсортирован, а строки фолда —
  произвольное подмножество (`TimeSeriesSplitter` упорядочивает через `np.argsort(t)`). Поэтому блок строится
  **не по позиции в срезе**, а по рангу времени: `rank = argsort(argsort(t[train_idx]))`, затем
  `block = rank // null_block_size` (смежные **по времени** блоки). Это вычисление — деталь `application` (§3),
  где доступен `Dataset.time()`; адаптер получает уже готовую метку-на-строку. Перестановка внутри блока сохраняет
  крупномасштабную автокорреляцию в null (между блоками), разрушая локальную внутриблочную.
- **Неравные интервалы времени (фикс R1-F2, честная граница):** блок из `null_block_size` **строк** соответствует
  фиксированному окну во **времени** только при ~равномерной частоте наблюдений. При нерегулярных интервалах
  (gaps/бурсты) временно́й горизонт блока плавает → автокорреляционная структура внутри блока непостоянна. Это
  **независимое** от §5 ограничение; документируется там же. Блок-по-времени-окну (вместо по-строкам) — возможное
  M6e-уточнение.
- Скор-формула не меняется: `imp_real − percentile(null, p)`; знаковый pass-through; `auto_threshold=0` (ADR-0047).
  Перестановка затрагивает только обучающую часть фолда (`fit⊕es`) — анти-ликедж наследуется (NFR-FSH-1).

### 3. Проводка меток — полный контракт-change `application` (фикс R1-B1/M1)
Проводка проходит через **3 уровня вызовов** и затрагивает **оба** FS-пути — это явный объём, не «лишь
прокинуть»:
- **Подготовка `block_labels`** (там, где доступен `Dataset`): единая чистая функция
  `_structure_labels(dataset, scheme, null_block_size) -> np.ndarray | None` — для `group` `dataset.groups()`;
  для `timeseries` ранг-биннинг `dataset.time()` (§2); иначе `None`. Эта **одна** функция — общий источник и для
  null-перестановки (здесь), и для `block_index` block-bootstrap (ADR-0053 §3) → семантика блока **централизована**
  (фикс R1-F8).
- **Контракт-change сигнатур (`application`):** `select_features(+groups: np.ndarray|None=None)` и
  `_select_one(+groups=...)` получают метки; спайн срезает `groups[train_idx]` перед `ranker.rank(..., groups=...)`
  (как `sample_weight[train_idx]`). Спайн остаётся **чистым numpy** (Dataset не входит) → `core-independence`/
  `usecases-independent-of-adapters` KEPT.
- **Оба пути FS** (фикс R1-B1, single-ranker не забыт):
  - **compare-путь** (`compare_features`→`_select_one`→`select_features`): метки готовит `compare_features`
    (имеет `dataset`).
  - **single-ranker путь** (`run_slice`, ветка `feature_ranker is not None`, минует `compare_features`): метки
    готовит **`run_slice`** (имеет `dataset`) и прокидывает в прямой `select_features(..., groups=...)`. Иначе
    одиночный `null_importance` на ts/group не получил бы структуру → §5 fail-loud вместо тихого uniform.
- **`scheme` в single-ranker пути (фикс R2-F-R2-5):** `run_slice` **не** несёт строку `scheme` — он выводит
  структуру из **маркеров сплиттера** (`splitter.time_ordered`/`splitter.group_aware`, как уже делает для
  leaderboard-`block_index`). Поэтому `_structure_labels` принимает не `scheme: str`, а уже-выведенный признак
  структуры (group / time-rank / none) — новый сквозной параметр `scheme` в `run_slice` **не** заводится.
- **`FeatureRanker` — не публичный plugin-port в M6d (фикс R2-C8):** entry-point-группы для сторонних ранжеров
  нет (`docs/plugin-contract.md` покрывает только `Estimator`/`honestml.models`). ⇒ «плагинные ранжеры» = форк/
  внутренний код; контракт-change `rank(+groups)` **не** ломает публично-поддерживаемое расширение. R-PORT для
  внешних плагинов ≈ 0; формулировка operational §1 уточнена (back-compat — для конфигов/встроенных).

### 4. Снятие гарда + конфиг (`composition/build.py`, `core/config.py`)
- `_guard_null_importance(strategy, scheme)` (`build.py:181-187`) **снимается** для `null_importance`: ts/group
  больше **не** `ConfigError`, а идут structure-aware-путём. (Гард остаётся актуальным только если структура
  недоступна — см. §5.)
- `FeatureSelectionConfig` +`null_block_size: int = Field(50, ge=2)` (размер временного блока для null;
  рекомендация — §SPIKE). Аддитивно (frozen/extra=forbid), внутри `FeatureSelectionConfig` → fingerprint-
  нейтрально при `fs=None`. Для `group` блок = группа (поле игнорируется).

### 5. Граница доверия и fail-loud остаток (NFR-FSH-6)
- **timeseries без объявленной `time`-колонки** или **group без `groups`** → структуру построить нельзя →
  `null_importance` остаётся `ConfigError` (нет якоря, как M6c). Структурная перестановка требует **объявленной**
  роли TIME/GROUP.
- **Вырожденный блок/группа (фикс R2-C3):** не только «1 строка», но и **константный таргет в блоке**
  (`nunique(y[block]) < 2`) → перестановка тождественна, null-сигнал нулевой. **Это типично для group-классификации**
  (группа = пациент/сессия с одним лейблом). Определение «вырожденного блока» = `len(block) < 2 OR
  nunique(y[block]) < 2`; такие блоки дают нулевой вклад в null (не падают). **Наблюдаемость (фикс R2-C6):**
  `run_report` несёт `null_block_stats` (`n_blocks`, `mean_block_size`, `degenerate_blocks`) — аддитивный ключ;
  WARNING при высокой доле вырожденных блоков (null-сигнал ненадёжен).
- **Допущение** (документируется, без overclaim): валидность — в пределах «обмениваемость **внутри** блока/группы».
  Сильная внутриблочная автокорреляция при крупном `null_block_size` частично нарушает её → рекомендация размера
  блока из SPIKE-M6d-validity; вне допущения метод — приближение, не гарантия.
- **Неравные интервалы времени (R1-F2):** `null_block_size` в **строках** даёт фиксированное **временно́е** окно
  лишь при ~равномерной частоте; при нерегулярных рядах горизонт блока плавает — это явная граница (блок-по-окну
  → M6e).
- **Сила подтверждения по scheme (R1-F3, честно):** **group**-валидность подтверждена SPIKE-M6d-validity
  **однозначно** (100%→0%); **timeseries** — подтверждён лишь **направленно** (v1: 5%→0%, статистически слабо;
  v2-свип φ — см. §SPIKE). Поэтому различитель FR-FSH-1 опирается на **group как решающий**, а timeseries —
  направленный (магнитуда масштабируется с автокорреляцией; эмпирика на реальных данных — Day-2/M6e).

## SPIKE-M6d-validity (структурная валидность → §5)
Симуляция (`spike_m6d_validity_sim.py`, детерминированная sklearn/numpy): синтетика с **автокоррелированным**
таргетом (timeseries) и **групповыми интерсептами** (group); спуриозный структурный признак (независимый AR(1) /
идентификатор группы) vs генуинный i.i.d.-предиктор. Сравниваются `null_importance`-маржины под **uniform** и
**structure-aware** перестановкой; метрика — доля прогонов, где спуриозный признак ошибочно **сохранён**
(margin>0).

**Результаты** (v1, trials=80, trees=60, n_runs=30, p=95; полная сводка — `SPIKE-M6d-validity.md`):

| Сценарий | Перестановка | margin genuine | margin spurious | spurious KEPT |
|---|---|---|---|---|
| **group** 30×40 | uniform | +0.2239 | **+0.2216** | **100%** |
| **group** 30×40 | within-group | +0.2975 | **−0.2159** | **0%** |
| timeseries block=50 | uniform | +0.2577 | −0.0368 | 5% |
| timeseries block=50 | block | +0.2590 | −0.0513 | 0% |

**Структурный вывод (подтверждён на group, направленный на ts):** **group-кейс однозначен** — uniform-перестановка
даёт **невалидный** null (спуриозный признак-идентификатор группы ошибочно сохраняется в **100%** прогонов, margin
сравним с генуинным), within-group — корректно **отвергает** его (**0%**, margin −0.22) и **усиливает** генуинный.
**timeseries** — направление верное (block строго лучше: 5%→0%), но магнитуда скромная и статистически слабая
(независимый AR(1) почти не ложно-важен); φ-свип v2 **не доведён** (прогон остановлен) → ts закрыт **направленно**,
магнитуда — Day-2/M6e. Решение опирается на **group-кейс + структурный вывод** (within-structure null валиден), а
**не** на ts-магнитуду (как SPIKE-M6c-1: структурный вывод, не число). Различитель FR-FSH-1 — **group решающий**.

## Последствия
- (+) `null_importance` работает на ts/group честно (structure-aware null), снимая M6c-`ConfigError`; контракт-
  change минимален и обратносовместим; анти-ликедж/детерминизм сохранены; спайн чист (Dataset не протёк).
- (−/компромисс) Контракт-change публичного порта (опц. kwarg, обратносовместим); `null_block_size` — новый
  параметр с эвристическим дефолтом; остаточный риск внутриблочной зависимости (документирован, не устранён —
  honest boundary); требует объявленной роли TIME/GROUP (иначе `ConfigError`).
- **Влияние на слои:** kwarg порта/`null_block_size` — `core`; биннинг/проводка меток — `application`;
  структурная перестановка — `adapters` (Humble); снятие гарда/резолв — `composition`. `import-linter` 3/3 KEPT.

## Проверки
- `scheme=group`+`null_importance` и `scheme=timeseries`+`null_importance` (с объявленными ролями) проходят
  **без** `ConfigError`; без роли TIME/GROUP → `ConfigError` (§5).
- `importance`/`random_probe` с `groups=None` дают **тот же** subset, что M6c (back-compat, FR-FSH-2).
- **Различитель валидности** (SPIKE): на синтетике uniform-перестановка сохраняет спуриозный структурный признак,
  structure-aware — отвергает (питается `spike_m6d_validity_sim.py`).
- Перестановка только в `fit⊕es`-части (property: перестановка `test`-таргета фолда не меняет вклад фолда);
  детерминизм при seed; вырожденный блок → WARNING.
