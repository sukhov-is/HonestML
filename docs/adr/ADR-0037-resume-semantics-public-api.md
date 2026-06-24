# ADR-0037 — Resume-семантика, публичный `cache`-API и контракт детерминизма

- **Статус:** Accepted (реализован 2026-06-09: публичный `cache: str|Path|None` в `AutoML.__init__`
  (verbatim) + `_build_cache`/проводка `cache=` в `run_slice`; `build_run_report` += аддитивные
  `run_fingerprint`/`cache{enabled,reused,computed}` (`RUN_MANIFEST_VERSION` не бампнут); ослаблен
  `_V1_KEYS` до супермножества; тесты RC-d1/d2 в `test_facade.py`/`test_run_report.py`. Без отклонений.)
- **Дата:** 2026-06-09
- **Драйверы:** DM-2 (resume), DM-4 (поверхность/совместимость/наблюдаемость); FR-RC-4/6, NFR-RC-2/6.
  Наследует sklearn-инвариант `__init__` (ADR-0011), budget-gate/детерминизм (ADR-0032 §6), run-report
  (ADR-0033, schema v1).
- **Воркстрим:** M5-resume.

## Контекст
Нужен один публичный способ включить кэш/resume, не ломая sklearn-контракт и дефолтное поведение M5, и
**честный** контракт: что resume гарантирует под каждым режимом бюджета, и как исход кэша виден пользователю.

## Рассмотренные варианты
1. **Флаг `resume: bool` + отдельный `cache_dir`.** Две ручки на одну способность; рассинхрон (resume=True,
   cache_dir=None). **Отвергнут.**
2. **Единый `cache: str | Path | None`** (контракт «каталог = состояние», планка MLJAR `results_path`/
   AutoGluon `path`): `None` — без кэша (M5); `<dir>` — включить переиспользование **и** resume (тот же dir +
   тот же fingerprint → hit; mismatch → новый subdir → авто-инвалидация). **Выбран** (одна ручка, без
   рассинхрона, авто-инвалидация без отдельного `force_rerun`-флага).
3. **Всегда кэшировать во временный каталог.** Сюрпризный диск-рост, неявно; нарушает «дефолт не меняет
   поведение». **Отвергнут.**

## Решение

### 1. Публичный параметр фасада (verbatim, sklearn-инвариант)
`AutoML(..., cache: str | Path | None = None)` — хранится в `__init__` **как есть** (ADR-0011), участвует в
`get_params/set_params/clone`/`Pipeline`. **`None` (дефолт) — поведение M5 без изменений** (ничего не
персистится, fingerprint вычисляется для отчёта, но кэш не строится). Резолв — в `fit`:
- вычислить `run_fingerprint` (ADR-0035);
- при `cache is not None` — построить `JoblibCandidateCache(Path(cache), fingerprint)` (ADR-0036) и передать
  `cache=` в `run_slice`; при `None` — `cache=None` (skip-ветка инертна).
- **Resume неявен:** тот же `cache`-каталог при повторном `fit` → те же `<fingerprint>/<id>/` → hit. Смена
  данных/конфига/версии → другой `<fingerprint>` → miss (свежий пересчёт; старый subdir остаётся для возможного
  возврата к прежнему конфигу).

### 2. Контракт детерминизма resume (NFR-RC-2, фикс взаимодействия с ADR-0032 §6)
**Предусловие `consume` (фикс R1-clean-arch/ADV-minor — делает контракт самодостаточным):** `run_slice`
зовёт `budget.consume(cand.train_time)` одинаково для hit и miss, но семантика — режим-зависимая
(`RunBudget`, `run_budget.py`): под **`mode="trials"`** `consume` считает **один завершённый трайл независимо
от значения `train_time`** → cache-hit (с восстановленным `train_time`) **эквивалентен** свежему завершению;
под **`mode="time"`/`"none"`** `consume` — **no-op** (единственный источник `exhausted` — часы/режим),
поэтому **cached `train_time` НЕ вычитается** из time-бюджета (иначе сломалось бы best-effort-ускорение).
- **`mode="trials"`/`"none"` — точный resume:** cache-hit проходит budget-gate и (под trials) **консьюмит
  трайл** → набор завершённых **тот же**, что в непрерывном прогоне → идентичные leaderboard/winner/артефакт
  (наблюдаемые: entries+best_model_id+`predict`).
- **`mode="time"` — best-effort:** cache-hit near-instant (обучение пропущено, `consume` no-op) → за тот же
  wall-clock завершится **больше** кандидатов → набор может **отличаться**. Это **ускорение, не ошибка**:
  каждый переиспользованный OOF валиден (ADR-0035 §2; ADR-0032 §6 — под time воспроизводимость артефакта и
  так не гарантирована). **Наблюдаемость честна (фикс R1-completeness):** `skipped_by_budget`/`budget_exhausted`
  в `SliceResult`/run-report отражают **фактический** resumed-прогон (под time+cache меньше пропущенных, т.к.
  hit'ы успевают) — отчёт правдив о том, **как прошёл этот прогон**, а не о гипотетическом непрерывном; это
  осознанно (truthful = «как реально прошло»), не дефект.
- **Падение не кэшируется** (ADR-0036 §3): повторный прогон честно ретраит упавшего (транзиентность).

### 3. Наблюдаемость в run-report (аддитивно к ADR-0033 schema v1)
`build_run_report` получает `run_fingerprint` и cache-исход и добавляет **аддитивные** top-level ключи:
- `run_fingerprint: str` (hex digest) — присутствует всегда (даже при `cache=None`: вычислен, не использован).
- `cache: {enabled: bool, reused: list[str], computed: list[str]}` — `enabled=False`/пустые списки при
  `cache=None`; иначе перечисляет переиспользованные/пересчитанные id (truthful, FR-RC-6).
- **Эволюция — аддитивная** (operational ADR-0033 §1): `RUN_MANIFEST_VERSION` **не бампается** (новые ключи,
  не смена семантики существующих); потребители — через `.get`. **Строгий тест `set(report)==_V1_KEYS`**
  (M5b `test_report_v1_schema_keys`) **ослабляется до супермножества** `_V1_KEYS <= set(report)` — пиннинг
  точного набора противоречит политике аддитивной эволюции; это и есть «контракт-изменение run-report» под
  этот ADR (architecture-invariant: смена формата данных → ADR). **Точечно (R1-asis):** меняется **только**
  строка `assert set(report)==_V1_KEYS` → `assert _V1_KEYS <= set(report)`; прочие ассерты теста
  (`winner`/`leaderboard[0]`/`isinstance config`) **сохраняются**.
- **Связь с NFR-M5-6 (R1-cons):** NFR-RC-6 (наблюдаемость кэша) **продолжает** NFR-M5-6 (truthful-провенанс
  прогона); существующая truthful-семантика run-report не регрессирует, новые ключи `run_fingerprint`/`cache`
  тестируются под NFR-RC-6. Докстринги кода, ссылающиеся на `NFR-M5-6`, не переписываются.
- Источник `reused`/`computed` — новые поля `SliceResult` (ADR-0036 §3); `run_fingerprint` прокидывается из
  facade в ассемблер.

## Последствия
- **Положительные:** одна публичная ручка (как MLJAR/AutoGluon); авто-инвалидация без флага; дефолт сохраняет
  M5; честный контракт детерминизма (точный для trials/none, явно best-effort для time); исход кэша виден.
- **Отрицательные/компромиссы:** под time resume не воспроизводим (принято, наследует ADR-0032 §6); накопление
  `<fingerprint>`-subdir при частой смене конфига (R-GC, GC→future); ослабление строгого v1-теста (осознанно,
  по политике аддитивности).
- **Влияние на слои:** `cache`-параметр + резолв/сборка адаптера — `composition/facade`; `run_fingerprint`+
  cache-ключи — `application/run_report` (аддитивно); `SliceResult.reused/computed` — `application`. `core`
  не трогается. `ARTIFACT_VERSION`/`RUN_MANIFEST_VERSION` не меняются.

## Проверки
- `cache=None` → существующий сьют без изменений исхода; `run_report_["cache"]["enabled"]==False`,
  `run_fingerprint` присутствует.
- `cache=<dir>`: второй `fit` → 0 переобучений, `reused`=все id, идентичные leaderboard/`predict` (trials).
- `clone`/`get_params`/`Pipeline` сохраняют `cache`; `str` и `Path` принимаются.
- trials+общий кэш, один seed → идентичные наблюдаемые (NFR-RC-2(1)); cache-hit консьюмит трайл (тест на
  фейках).
- run-report: `run_fingerprint`+`cache`-блок truthful; старое чтение толерантно (`.get`); `_V1_KEYS` —
  супермножество; `RUN_MANIFEST_VERSION` не изменён.
