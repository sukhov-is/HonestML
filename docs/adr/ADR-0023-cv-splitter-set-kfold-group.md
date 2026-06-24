# ADR-0023 — Набор CVSplitter: KFold (plain) + GroupKFold; снятие M2 group-rejection

- **Статус:** Accepted (реализован 2026-06-08, M3c)
- **Дата:** 2026-06-07
- **Драйверы:** DM3-3 (обобщение), DM3-4 (корректность group-leakage); FR-4. north-star: D2/D3; FR-4.
- **Воркстрим:** M3, дельта к ADR-0013 (сплиттеры). **Частично supersede** ADR-0010 §5/F5 (group-rejection).

## Контекст

Реализованы только `HoldoutSplitter`/`StratifiedKFoldSplitter` (`splitters.py`). `build.py` роутит
`scheme="kfold"/"group"` в `UnsupportedSchemeError("planned: M3")`. `run_slice:130-131` **жёстко
отвергает** группы (`if schema.group is not None: raise ConfigError("... M2 (see M4)")`), хотя
`validate_fold(groups=...)` (проверка group-leakage) **уже реализована** в `core/ports/splitter.py`.
roadmap §M3 включает Group-сплиттер; TimeSeries (purge/embargo/time-order) — остаётся M4.

## Рассмотренные варианты

1. **Только KFold в M3, Group → M4** — меньше объём, но расходится с roadmap §M3 («KFold/Stratified/
   Group/Holdout») и оставляет готовую `validate_fold(groups)` неиспользуемой.
2. **KFold + GroupKFold в M3; TimeSeries в M4** — закрывает roadmap-список, задействует уже готовую
   проверку утечки групп; снимает M2-заглушку на группы. **Выбран.**

## Решение

### 1. `KFoldSplitter` (plain)
`sklearn.model_selection.KFold(n_splits, shuffle, random_state)`. `es_idx` пуст (как M2). Дефолт для
регрессии (`Task.default_cv_scheme == "kfold"`). `validate_fold(fold)` (без групп). **Имя-дисамбигуация:**
`KFoldSplitter` ≠ `StratifiedKFoldSplitter` (последний — отдельный сплиттер для классификации; M2-ревью
зарезервировало это разведение имён — здесь оно соблюдено: plain-KFold не стратифицирует).

### 2. Источник групп: `Dataset.groups()` (фикс R1-M1/2a) + group-aware сплиттеры
**Контракт-change ядра (аддитивно):** в порт `Dataset` добавляется метод `groups() -> np.ndarray | None`
(значения group-колонки `schema.group` **в порядке строк датасета**), реализуется в polars-backend. Это
**единый** источник групп: и сплиттер, и `validate_fold` берут массив из `dataset.groups()` — индексно
выровненный с `x_full`/`design_matrix` (исключает рассинхрон порядка, фикс R1-2a).

- **Классификация** group-aware: `sklearn.model_selection.StratifiedGroupKFold(n_splits, shuffle,
  random_state)` — **стратифицирует** по классу при непересечении групп (фикс R1-2b: устраняет
  одноклассовые/смещённые фолды без стратификации).
- **Регрессия** group-aware: `GroupKFold(n_splits)` (стратификации по непрерывному таргету нет).
- Для каждого фолда `validate_fold(fold, groups=dataset.groups())` — **обязательно** (ни одна группа не
  пересекает fit/test; закрывает RM3-6). Детерминированы → порядок воспроизводим (NFR-4).
- **Инвариант индексов (фикс R2-M1/2a):** `Fold.fit_idx/es_idx/test_idx` — позиции строк в **исходном**
  `dataset` (том же, что отдаёт `dataset.groups()` и `design_matrix(dataset)`). Сплиттер строит фолды на
  `dataset.n_rows` и НЕ переупорядочивает; `run_slice` индексирует `x_full = design_matrix(dataset)` теми же
  индексами (как в M2). Это делает выравнивание `groups()`↔фолды↔`x_full` гарантией дизайна, а не совпадением
  реализаций.

### 3. Снятие M2 group-rejection (supersede ADR-0010 §5/F5 в части отказа)
`run_slice` больше не падает на `schema.group is not None` (`slice.py:130-131`). Группы консумируются
**только** group-aware сплиттером; `run_slice` передаёт `dataset.groups()` в `validate_fold` для group-схемы.
**Два места правки в `slice.py`** (фикс R1-M2): (а) снять отказ `130-131`; (б) **цикл валидации
`149-152`** — сейчас `validate_fold(fold)` без групп с устаревшим комментарием «groups are rejected upstream
in M2»; для group-схемы передать `groups=dataset.groups()`, комментарий обновить.
- **`scheme="group"` без group-колонки → `ConfigError`** (нужна группа).
- **group-колонка присутствует, но схема перемешивающая** (`stratified`/`holdout`/`kfold`, не group): группа
  **не используется для разбиения** (роль `GROUP` исключена из признаков схемой), но это сигнал риска утечки
  → **WARNING** (по аналогии с datetime-лукэхед-WARNING, ADR-0016 §5; фикс R1-2c): «есть group-колонка, но
  CV не group-aware — возможна утечка группы; используйте scheme='group'».

### 4. `build.py` роутинг
`_resolve_splitter`: `kfold → KFoldSplitter(n_splits, shuffle, seed)`; `group →` **по `task.kind`**:
классификация → `StratifiedGroupKFoldSplitter(n_splits, shuffle, seed)`, регрессия → `GroupKFoldSplitter(n_splits)`
(оба требуют `schema.group`, иначе `ConfigError`). `auto` для регрессии → `kfold`. `timeseries`/`purge`/`embargo`
**остаются** `UnsupportedSchemeError` (M4). Инвариант `n_splits >= 2` для k-fold-семейства сохраняется.
Предупреждение о лукэхеде (ADR-0016 §5) расширяется: `kfold` тоже шафлит → при datetime-колонках WARNING;
плюс group-колонка вне group-схемы → WARNING (§3).

## Последствия

- **Положительные:** roadmap-список CV закрыт; group-leakage-проверка задействована; регрессия получает
  естественный `kfold`-дефолт; M2-заглушка на группы снята корректно (с проверкой утечки, не «просто
  разрешили»); классификация group-aware **стратифицируется** (нет смещённого/одноклассового OOF, фикс R1-2b);
  единый `dataset.groups()` исключает рассинхрон источника групп (фикс R1-2a).
- **Отрицательные / компромиссы:** `GroupKFold` (регрессия) не стратифицирует (для непрерывного таргета это
  и не определено); group-колонка при не-group схеме не используется для разбиения, но **сопровождается
  WARNING** (фикс R1-2c). ADR-0010 §5/F5 помечается частично замещённым (back-annotation в ADR-0010).
- **Влияние на слои:** новые сплиттеры — `adapters/splitters`; снятие отказа + проброс групп —
  `application/slice`; **новый метод `Dataset.groups()` — `core`-порт (аддитивно) + polars-backend**;
  `validate_fold(groups)` — уже в `core`. import-linter не нарушен.

## Проверки

`KFoldSplitter`/`GroupKFoldSplitter`/`StratifiedGroupKFoldSplitter` проходят контракт-тест (`validate_fold`:
disjoint; для group — отсутствие общей группы в fit и test); `Dataset.groups()` возвращает значения в порядке
строк (тест выравнивания индексов); `CVConfig(scheme="kfold")`/`("group")` резолвятся в нужный сплиттер (group
→ stratified для классификации, plain для регрессии); `scheme="group"` без group-колонки → `ConfigError`;
group-колонка + перемешивающая схема → WARNING; `run_slice` с group-колонкой и `scheme="group"` завершается
(нет M2-отказа); `timeseries`/`purge`/`embargo` остаются `UnsupportedSchemeError`; датасет с группами и
group-CV — нет строки, где группа в fit и test одновременно (property-тест анти-ликедж).
