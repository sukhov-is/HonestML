# ADR-0086: Наблюдаемость in-sequential band и инвариантность сериализации/инференса

- **Статус:** Proposed
- **Дата:** 2026-06-19
- **Драйверы:** D-1 (наблюдаемая честность), D-5 (back-compat артефактов).
  Источник: FR-5, NFR-5, NFR-6. Зависит от ADR-0083; наследует ADR-0045/0053 §4.

## Контекст
Band-выбор меняет исход «по умолчанию» (R-DEFAULT), поэтому он обязан быть **виден**
(FR-5, NFR-6). При этом он не должен трогать путь сериализации/инференса (D-5): band
влияет только на **состав** выбранных индексов. Тонкость: у sequential **внутри**
`compare` сосуществуют два band — внутренний (по числу признаков) и арбитражный (между
стратегиями, ADR-0053 §4, поля `winner_rule`/`band_members` в `CompareOutcome`). Их
наблюдаемость нельзя смешивать.

## Решение

### 1. Отдельный канал наблюдаемости для in-sequential band
`equivalence_band` уже возвращает `BandResult(member_ids, winner, unstable, width,
winner_by_tiebreak)` ([selection_policy.py:61-76](../../../../src/honestml/core/selection_policy.py#L61)).
Для in-sequential band он прокидывается **отдельным** полем, не переиспользуя
арбитражные `winner_rule`/`band_members`:
- `_select_one`/`ctx.select` для wrapper-ветки возвращает не только субсет, но и
  `BandResult | None` (None при `significance="off"`);
- `CompareOutcome` ([feature_compare.py:51-74](../../../../src/honestml/application/feature_compare.py#L51))
  получает аддитивное поле `seq_band` (например `dict` с `width`, `winner_by_tiebreak`,
  `members`, `rule ∈ {argmax, band_tiebreak}`), заполняемое для **победившей** стратегии,
  если она wrapper и band активен; иначе `None`/`{}` (дефолт сохраняет текущую форму);
- `_feature_selection_report` ([composition/run_report.py:179-231](../../../../src/honestml/composition/run_report.py#L179))
  аддитивно распаковывает `seq_band` в блок `feature_selection`. Старые парсеры
  игнорируют новый ключ; версии отчёта/манифеста не бампаются.

Правило выбора видно как `seq_band.rule`: `argmax` (band пуст / significance off) или
`band_tiebreak` (выбран компактнейший из неотличимых) — симметрично ADR-0053 §4.

**Точный dataflow (фикс OBS-DATAFLOW).** `ctx.select`/`_select_one` для wrapper-ветки
меняют возврат на `tuple[tuple[int, ...], BandResult | None]`; для ranker-ветки — пара
`(subset, None)`. Все вызовы `ctx.select` в `_compare_single`/`_compare_holdout`/
`_compare_nested`/`_compare_per_fold` распаковывают пару и прокидывают `BandResult`
победителя в `_outcome(..., seq_band=band_result)` ([feature_compare.py:644-671](../../../../src/honestml/application/feature_compare.py#L644)),
который кладёт его в `CompareOutcome.seq_band`. Поскольку субсет остаётся первым элементом
пары, downstream-логика арбитража не меняется (ADR-0083 §2).

**Точная форма `seq_band` (фикс MISSING-002 / TESTABILITY-004).** JSON-структура:
`{"width": int, "winner_by_tiebreak": bool, "members": list[str], "rule": "argmax"|"band_tiebreak"}`.
Присутствует **iff** победившая стратегия — wrapper-селектор **и** `significance != "off"`;
иначе ключ **отсутствует** (не `null`) — старые парсеры читают через `.get`. При
`significance="off"` band отсутствует ⇒ поведение наблюдается как чистый argmax (ключа нет).

### 2. Сериализация и инференс — без изменений
Победитель band → `selected_features: tuple[str,...]` (имена в порядке схемы) →
единственный choke-point `design_matrix` ([slice.py:177-204](../../../../src/honestml/application/slice.py#L177)).
Band меняет лишь **какие** имена попадут в `selected_features` — путь
`schema.with_selected_features` → `schema.json` → `design_matrix.predict` **нетронут**
(ADR-0045). `ARTIFACT_VERSION` **не меняется**; старый артефакт без `seq_band` грузится
(наблюдаемость — не часть predict-пути, [artifact.py](../../../../src/honestml/composition/artifact.py)).

### 3. Кэш — авто-инвалидация уже есть (фикс ARTIFACT-EVOLUTION-001 / MISSING-005)
`run_fingerprint` = дамп `RunConfig`, а `significance` — поле `RunConfig`
([config.py:292](../../../../src/honestml/core/config.py#L292)), как и весь `fs`-конфиг
(`seq_min_features`/`seq_patience`/`compare`/`arbitration`, ADR-0045 §5) ⇒ включение/
выключение band и любой FS-параметр, влияющий на траекторию, инвалидируют кэш кандидатов
автоматически; новых ключей кэша не нужно. **Про `alpha`:** ширина band `alpha`
**не** конфигурируема — `build.py` строит `SelectionPolicy(greater_is_better=…)` с дефолтом
`alpha=0.05` ([build.py:173](../../../../src/honestml/composition/build.py#L173)), поля
`RunConfig.alpha` нет ⇒ пользователь не может менять её между прогонами ⇒ stale-hit по
ширине band невозможен. Инвариант для будущего: если `alpha`/`SelectionPolicy` станут
конфигурируемыми, они обязаны войти в `run_fingerprint`.

## Последствия
- **Положительные:** contract-change исхода (R-DEFAULT) наблюдаем; два уровня band не
  смешиваются; сериализация/инференс и версии артефакта инвариантны; кэш корректен.
- **Отрицательные / компромиссы:** `_select_one`/`ctx.select` возвращают пару
  `(subset, BandResult|None)` — правка ripple по call-site'ам всех режимов compare
  (механическая, не меняет логику); `seq_band` хранится в отчёте, не в frozen
  `LeaderboardEntry` (как и band стратегий в ADR-0026 §6 — чтобы старая сборка читала).
- **Влияние на слои:** `application` (возврат `BandResult`, поле `CompareOutcome`);
  `composition/run_report` (распаковка) — аддитивно; `core`/`adapters`/артефакт-формат
  — без изменений.

## Проверки
- FR-5: при `significance != "off"` отчёт sequential несёт `seq_band` с `width`/
  `winner_by_tiebreak`/`rule`; при `off` — `rule="argmax"`/пусто.
- NFR-5: round-trip артефакта (save→load→predict) идентичен; старый артефакт без
  `seq_band` читается; смена `significance` меняет `run_fingerprint` (тест).
- Разделение уровней: в compose-режиме с `sequential` арбитражные `band_members`
  (ADR-0053) и `seq_band` присутствуют независимо (тест на оба ключа).
