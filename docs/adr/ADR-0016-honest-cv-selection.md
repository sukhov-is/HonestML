# ADR-0016 — Честный выбор схемы CV (без тихой подмены) + fail-fast

- **Статус:** Proposed
- **Дата:** 2026-06-06
- **Драйверы:** D3 (корректность/анти-ликедж), D6 (инженерия), D1 (расширяемость);
  FR-1..4; NFR-1/2/3.
- **Воркстрим:** C4 (honest CV-selection), дельта к M2.

## Контекст

`build_default_components` всегда строит `StratifiedKFoldSplitter(shuffle=True)`
(`build.py:55-57`), игнорируя `RunConfig.cv` целиком; дефолт `CVConfig.scheme=
"timeseries"` (`config.py:26`) и `purge`/`embargo` не читает никто; `Task.
default_cv_scheme` противоречит дефолту и не вызывается; `validate_fold` зовётся без
`time_ordered` (`slice.py:152`). Итог — **тихий лукэхед**: запрос временной валидации
молча превращается в перемешанный KFold. Это прямой подрыв дифференциатора (D3).
Ограничение: реализованы только `stratified`/`holdout`; полный TimeSeriesCV с
purge/embargo — M4 (R4: здесь не строим).

## Рассмотренные варианты

1. **Реализовать TimeSeriesCV сейчас** — закрыло бы по существу, но это M4 (purge/
   embargo, es-carve-out, дата-контракт), большой объём, нарушает R4.
2. **Тихо чинить дефолт** (сменить `timeseries`→`stratified`) — уберёт landmine, но
   оставит главную болезнь: запрос недоступной схемы по-прежнему молча подменяется.
3. **Честный резолвер `scheme→splitter` + fail-fast** — выбор управляется `CVConfig`,
   нереализованное явно отвергается с указателем на воркстрим; новых сплиттеров не
   добавляем. Минимально и закрывает корень (тихую подмену).

## Решение

**Вариант 3** (уточнён по ревью R1/R2).

1. **`"auto"`-дефолт.** В `CVScheme` добавляется литерал `"auto"`; `CVConfig.scheme`
   дефолт меняется `"timeseries"` → `"auto"`; `CVConfig.n_splits` дефолт меняется
   `2` → `5` (5 = текущий дефолт build, сохраняет поведение).
   `"auto"` резолвится через `Task.default_cv_scheme` (классификация→`stratified`,
   регрессия→`kfold`) — снимает landmine `timeseries` и противоречие `default_cv_scheme`.
2. **Резолвер в composition.** `build_default_components` принимает `CVConfig` (а не
   только `cv: int`) и сопоставляет схему адаптеру: `auto → Task.default_cv_scheme`;
   `stratified → StratifiedKFoldSplitter(n_splits, shuffle, seed)`; `holdout →
   HoldoutSplitter`. Инвариант: `n_splits<2` для k-fold-семейства → ошибка (holdout
   `n_splits` игнорирует). **Нереализованное → fail-fast** (без StratifiedKFold-фолбэка):
   `kfold(plain)`/`group`/`timeseries`, `purge>0`/`embargo>0`.
   **Порядок отказов:** фильтр способностей/метрики (`_filter_by_capability`, build.py:68-70)
   срабатывает **до** CV-резолва — регрессия падает с «no estimator supports task», а не
   с вводящим в заблуждение CV-сообщением (фикс R2-major: `auto`+regression→`kfold`).
   **Разрешённую конкретную схему резолвер записывает обратно в `RunConfig.cv`** —
   manifest хранит фактически использованную схему, не `"auto"` (фикс «manifest правдив»).
3. **Тип ошибки — новый лист `UnsupportedSchemeError(ConfigError)`** (вводим сразу:
   уже 5 случаев; машиночитаемо; ловится и как `ConfigError`; аддитивно к ADR-0008,
   не расширение базы). Отличает «валидно, но ещё не реализовано» от «невалидно».
4. **Фасад.** `AutoML.__init__(cv: int | CVConfig | None)` — аддитивно: `int` =
   число фолдов (`CVConfig(scheme="auto", n_splits=cv)`), `CVConfig` = полная спека,
   `None` = дефолт. **Дефолт = 5 фолдов сохраняется** (см. п.1). `CVConfig` —
   `frozen` pydantic → хранение verbatim в `__init__` совместимо с sklearn
   `clone`/`get_params`/`set_params` (нет мутаций при reuse).
5. **Предупреждение о лукэхеде (вместо инертного `time_ordered`-хука, фикс R2).**
   Резолвер получает признак наличия datetime-роли (фасад передаёт
   `bool(ds.schema.datetime)`); если он истинен, а схема перемешивает
   (`stratified`/`holdout`/`auto`→stratified, `shuffle=True`) → **`logger.warning`** о
   риске лукэхеда и о том, что корректная временная валидация (TimeSeriesCV+purge/embargo)
   — M4. Это закрывает самый частый кейс (datetime-данные+дефолт), который fail-fast по
   схеме не ловит. **`time_ordered`-валидацию в C4 НЕ подключаем** — она дала бы ложную
   уверенность (проверяет порядок индексов, не дат; гейтится на непустом `es`, которого
   нет в M2/M3). Честная временна́я-порядок проверка — требование **M4** (по значениям
   datetime; непустой `es`).

## Последствия

- **Положительные:** тихий лукэхед закрыт и для явного запроса схемы (fail-fast), и для
  частого кейса datetime+дефолт (WARNING); `RunConfig.cv` правдив (разрешённая схема);
  дефолт по эффекту не меняется (`auto`→`stratified`, 5 фолдов); типизированная ошибка.
- **Отрицательные / компромиссы:** литерал/дефолты `CVScheme`/`n_splits` меняются —
  аддитивно, но затрагивает сериализацию (RF1); manifest со `scheme="timeseries"` теперь
  fail-fast при использовании/replay (намеренно, RF2); WARNING (не fail-fast) на
  datetime+shuffle — компромисс «не блокировать временные данные до M4» (RF5).
- **Влияние на слои:** резолвер/WARNING — в `composition` (ADR-0009); `core` чист;
  `CVScheme`/`default_cv_scheme`/`UnsupportedSchemeError` — домен. import-linter не нарушен.

## Проверки

Тесты: схема управляет типом сплиттера (`holdout`/`stratified`/`auto`); `n_splits<2`
для k-fold → ошибка; недоступная схема и `purge>0`/`embargo>0` →
`UnsupportedSchemeError` (никогда не сплиттер); регрессия → ошибка про эстиматор (не CV);
разрешённая схема записана в `RunConfig.cv`; фасад `cv: int|CVConfig` (вкл. fail-fast на
`timeseries`); back-compat `cv: int`/`cv=None`→5 фолдов; `clone/get_params` с `CVConfig`;
datetime+shuffle → WARNING присутствует, без datetime — нет; round-trip `CVConfig`.
`import-linter` в CI. Честная `time_ordered`-валидация — DoD **M4**, не здесь.
