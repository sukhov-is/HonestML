# ADR-0035 — Run-fingerprint: полный, версионированный, fail-closed ключ прогона

- **Статус:** Accepted (реализован 2026-06-09: `application/run_report.py`
  `compute_run_fingerprint`/`dataset_signature`/`collect_lib_versions`/`FINGERPRINT_VERSION`; сбор
  `lib_versions`+вызов после carve в `composition/facade.py` `_run_fingerprint`/`_packages_for`;
  тесты `tests/unit/test_run_fingerprint.py`/`test_dataset_signature.py`. Без отклонений от дизайна.)
- **Дата:** 2026-06-09
- **Драйверы:** DM-1 (корректность пропуска превыше скорости); FR-RC-1/5, NFR-RC-1/3/7. Наследует
  `RunConfig.model_dump` (M5b basis, `run_report.py`), `honestml_version` (уже эмитится), наивный schema-hash fingerprint как анти-паттерн (нет версии кода).
- **Воркстрим:** M5-resume (resume/stage-cache).

## Контекст
Для корректного переиспользования (ADR-0036) нужен **ключ**, по которому два прогона эквивалентны. Наивный подход
хэшировал схему+n_rows+config, но **НЕ** версию кода/библиотек → stale-hit при апгрейде. Ключ должен быть
**полным** (всё, что влияет на результат) и **fail-closed** (любая неопределённость → miss). Вычисление —
**чистое** (без I/O, синхронно тестируемо, слой `application`/`core`).

## Рассмотренные варианты
1. **Schema-only (схема+n_rows+config).** Дёшево, но: ложный hit при изменившихся **значениях**
   данных (тот же n_rows/схема) и при апгрейде кода/lib. **Отвергнут** (R-STALE — неверный результат).
2. **Полный fingerprint** = резолвнутый `RunConfig` + data-signature (схема + n_rows + **content-digest**) +
   `honestml_version` + **версии модельных библиотек** (по резолвнутому набору estimator'ов) + резолвнутый
   набор estimator-имён + `FINGERPRINT_VERSION`; SHA-256 над **canonical JSON** (`sort_keys`). **Выбран**
   (планка: Optuna study-identity, MLJAR results_path-fingerprint, но строже — с версией кода).
3. **Content-hash данных без config.** Неверно (смена cv/seed/significance меняет результат). **Отвергнут.**

## Решение

### 1. Состав ключа (canonical, детерминированный)
`compute_run_fingerprint(...) -> str` (hex SHA-256) над `json.dumps(parts, sort_keys=True)`. **Ключ обязан
включать ВСЁ, что влияет на per-candidate OOF/score, включая параметры фасада ВНЕ `RunConfig`** (фикс
R1-ADV-blocker: `metric` и `task` — отдельные параметры `AutoML`, в `RunConfig` их нет):
- **`config`** = `RunConfig.model_dump(mode="json")` — **резолвнутый**: `seed`, `budget`, `significance`,
  `model_types` (это **запрошенный** набор — всегда дефолт, см. ниже) и вложенный `cv:CVConfig`
  (scheme после `auto`, n_splits/n_test/n_es/purge/embargo, `outer_holdout`, `selection`, `calibrate`,
  `refinement_min_oof`). [Уточнение R1-asis: `outer_holdout`/`selection`/`calibrate` — поля **вложенного
  `cv`**, не верхнего уровня; `model_dump` их сериализует через `cv`.]
- **`task`** = `Task.model_dump(mode="json")` — **резолвнутый** `kind` (binary/multiclass/regression) +
  `positive_label` (фикс R1-ADV-blocker-2: меняет proba-канал/`oof_class.dtype`/выравнивание → результат).
- **`metric`** = резолвнутая `components.metric`: `name` + `average`/`labels` (если есть) + `greater_is_better`
  (фикс R1-ADV-blocker-1: метрика задаёт `Candidate.score` И какой OOF-канал заполняется — proba vs class;
  два прогона `roc_auc` vs `log_loss` при том же `RunConfig`/данных дают разный результат).
- **`data_signature`** (см. §2).
- **`estimators`** = `sorted(resolved_estimator_names)` (**фактический** набор из `components.estimators`,
  не запрошенный `model_types`) + **`lib_versions`** = `{pkg: importlib.metadata.version(pkg)}` по **всему
  реально используемому compute-стеку**: версии пакетов резолвнутых estimator-дескрипторов (catboost/lightgbm
  **и scikit-learn** для Linear/Baseline) **плюс `numpy`** (фикс R1-ADV-major-1/R-FOLD: sklearn даёт и
  сплиттеры, и линейные модели → его версия влияет на OOF И на fold-границы; апгрейд без version-в-ключе =
  stale-hit). `PackageNotFoundError` → `"<pkg>": null` (детерминированно, не падать).
- **`honestml_version`** (`importlib.metadata.version("honestml")`, как в run-report).
- **`fingerprint_version`** = `FINGERPRINT_VERSION` (бамп при смене состава/семантики ключа). **Роль
  (фикс R2-ADV-minor):** в обычном релизе смена состава ключа идёт вместе с бампом `honestml_version`, поэтому
  `FINGERPRINT_VERSION` — **аварийный ручной инвалидатор** для случаев, когда состав/семантика ключа меняется
  БЕЗ релиза (editable/dev-install, где `honestml_version` = `"0+unknown"`), а не дублирующая ось общего назначения.
> **`model_types` в `config` vs `estimators` (R1-cons-minor):** `RunConfig.model_types` — **запрошенный**
> набор (фасад его не прокидывает → всегда дефолт `("catboost","lightgbm")`, R1-asis); `estimators` —
> **резолвнутый** (источник истины). Оба намеренно в ключе (резолв — каноничен, запрос — для полноты);
> пересечение безвредно для fail-closed.
> **Fold-схема покрыта неявно (R1-cons-minor, расширено):** границы фолдов детерминированы
> `CVConfig`(scheme/n_splits/purge/embargo) + `seed` + соответствующими **data-signature**-входами:
> **target** (стратификация), **groups** (group-aware), **time** (timeseries) — все §2 — **плюс версия
> sklearn** в `lib_versions` (реализация сплиттера). Всё уже в ключе → отдельный хэш фолдов не нужен
> (R-FOLD закрыт без невыписанных допущений).

### 2. Data-signature — чистая, без polars, детерминированная, без утечки
`dataset_signature(dataset) -> str` — **чистая функция `application`** (numpy + `hashlib`, **без polars**,
без I/O). **Считается над ТЕМ ЖЕ `dataset`, что подаётся в `run_slice` — т.е. над DEV после carve**
(`ds`, == `ds_full` при `outer_holdout=0`), а **не** над `ds_full` (фикс R2-COMP-blocker): кэшируемый OOF
получен на dev, поэтому сигнатура обязана покрывать **именно dev**. Так смена `outer_holdout`/`seed`/`stratify`
меняет carve → другой dev → другой digest → нет ложного hit на OOF от другого подмножества строк
(carve-индексы отдельно хэшировать не нужно — dev-контент их уже отражает). SHA-256 над:
- **`design_matrix(dataset)`** — **тот же** материализованный модельный вход (числовой блок ⊕ codes→float64 в
  `schema.features`-порядке), что видят `_run_candidate`/`refit_best`/`FittedModel` (фикс R1-clean-arch-DRY:
  переиспользуем единственную абстракцию `design_matrix`, не параллельные `to_numpy()`+`categorical_codes()`).
  Всегда `float64` → `np.ascontiguousarray(m).tobytes()` детерминирован.
- **`target()`** и (если присутствуют) **`sample_weight()`/`groups()`/`time()`/`label_time()`** — через
  **поэлементную канонизацию, НЕ требующую взаимной сравнимости** (фикс R2-ADV-major: `np.unique`/сортировка
  падает `TypeError` на mixed-object и object-datetime-с-None — типовой pandas-столбец; сырой `object.tobytes()`
  хэширует адреса): числовой/bool → `np.ascontiguousarray(a, dtype=<fixed>).tobytes()`; **не-числовой
  (object/str/datetime/mixed)** → стабильные **UTF-8-байты `repr(v)` каждого элемента в позиционном порядке**
  (без сортировки/дедупликации; по образцу `CategoryTable.fit` `str(v)`, schema.py), с явной фиксированной
  меткой для `None`/`NaN` (например `"\x00NA"`), чтобы пропуски детерминированно влияли на digest и не роняли
  функцию. Порядок строк сохраняется (позиционно), значения входят как байты `repr`.
- **`FeatureSchema.model_dump_json()`** (роли + category-таблицы + dtype-токены) и **`n_rows`**.

В ключ/отчёт идёт только **digest** (хэш), сырые данные не сериализуются (анти-утечка, R-LEAK). Polars не
нужен → `core`/`application` без polars (NFR-RC-3).
> Цена (NFR-RC-7, фикс R1-completeness — формулировка без polars): один проход `tobytes()`/`repr` join+hash
> по уже-материализованным numpy-блокам — `O(n·d)`, **≪** обучения (`O(n·d · folds · models · iters)`).
> `design_matrix` и так строится в `run_slice` (повторный его расчёт для сигнатуры в facade — один лишний
> `hstack`, всё ≪ одного `_run_candidate`). **Smoke-замер — информативный, НЕ gating** (фикс R2-Day2: голое
> сравнение wall-clock флаки в CI); gating-проверка — детерминированный инвариант «ровно один проход»
> (NFR-RC-7).

### 3. Слои и чистота
- `compute_run_fingerprint` и `dataset_signature` — **чистые** функции `application` (hashlib/numpy stdlib),
  тестируемы без I/O. `importlib.metadata` (stdlib) — допустимо в `application`.
- **`Dataset`-порт НЕ меняется** (сигнатура считается над уже существующими port-методами через
  `design_matrix`) — меньше поверхности, чем гипотетический `Dataset.content_hash()` из research.
- Fingerprint вычисляется в **composition/facade**, **ПОСЛЕ carve** (data-signature над dev `ds`, §2,
  фикс R2-COMP-blocker), где **уже известны все резолвнутые входы**: `components.metric`,
  `components.estimators` (имена), резолвнутый `task` (`_resolve_task()`), и собираются `lib_versions` по
  пакетам резолвнутых estimator-дескрипторов + sklearn + numpy (через `importlib.metadata` в facade — **sklearn
  НЕ импортируется** в `application`, лишь читается версия). `metric`-часть читает `average`/`labels` через
  `getattr` (они есть лишь у конкретных multiclass-метрик, не в `Metric`-порте — фикс верификатора). Facade
  передаёт fingerprint вниз **строкой** (в адаптер кэша, ADR-0036); `run_slice` ключ не вычисляет, берёт порт.

### 4. Что НЕ входит в per-candidate OOF-ключ (важно для дешёвого reuse)
Кэшируется **per-candidate OOF** (ADR-0036), а band/significance/calibrate/refinement/`refit_best`
пересчитываются дёшево поверх. Поэтому фактический reuse кандидата зависит лишь от того, что влияет **на сам
OOF**: данные + fold-схема + estimator + features. `significance`/`selection(refinement)`/`calibrate`/`budget`
**меняют пост-OOF обработку, но не OOF**. Решение для простоты и безопасности: **единый run-fingerprint
включает весь `RunConfig`** (в т.ч. significance/budget) — это **строже необходимого** (лишние miss при
смене significance), но **никогда не ложный hit**, и проще рассуждать. Оптимизацию «OOF-ключ уже run-ключа»
(переиспользовать OOF при смене только significance) — **отложить** (future, если профиль покажет нужду);
зафиксировано как осознанный trade-off (простота > доля hit).
> **Trade-off набора моделей (фикс R2-COMP-major):** `estimators` (набор) — в **едином** run-ключе, поэтому
> **добавление/удаление любой модели инвалидирует reuse ВСЕХ кандидатов**, хотя per-candidate OOF
> немодифицированных моделей не изменился (расширил `("catboost","lightgbm")` → `(...,"xgboost")` → новый
> `<fingerprint>`, catboost/lightgbm считаются заново). Это **осознанный** компромисс ради простоты ключа;
> кандидат на ту же будущую оптимизацию «вынести estimators-набор из per-candidate ключа» (per-candidate
> OOF-ключ — лишь data+fold+конкретный estimator+features). Сейчас фиксируется одной строкой, чтобы потом
> не выглядело необъяснённой регрессией UX resume.

## Последствия
- **Положительные:** fail-closed корректность (R-STALE/R-FOLD закрыты); чистый тестируемый ассемблер; нет
  расширения `Dataset`-порта; нет polars в `core`/`application`; digest-only (нет утечки данных).
- **Отрицательные/компромиссы:** строже-чем-нужно ключ (смена significance/budget = miss, хотя OOF тот же) —
  принято (простота); content-hash — оверхед одного прохода (≪ обучения, NFR-RC-7); кросс-машинная
  стабильность numpy-байт зависит от endianness — кэш **локальный/одномашинный** в этом проходе (shared/
  кросс-машинный — future), для same-box resume детерминизм гарантирован.
- **Влияние на слои:** `compute_run_fingerprint`/`dataset_signature` — `application` (чисты); вызов и сбор
  lib-версий — `composition/facade`; `core` не трогается. import-linter не нарушен.

## Проверки
- Тот же `X/y`+params → тот же fingerprint (детерминизм, повторяемо); canonical (`sort_keys`) — устойчив к
  порядку ключей.
- Смена каждого из {seed, cv-параметр, models, **metric**, **task/positive_label**, **значения** данных,
  n_rows, schema, honestml_version, lib-версия (мок `importlib.metadata`)} → **другой** fingerprint (отдельный
  тест на ось, NFR-RC-1). Метаданные `groups`/`time`/`sample_weight`/`label_time` — часть data-signature §2;
  их смена покрыта осью «значения данных».
- `dataset_signature` чист: тест на фейковом `Dataset` без I/O; одинаковые данные → одинаковый digest;
  изменённое значение → другой digest; **строковый/object `target` → стабильный digest между процессами**
  (фикс R1-ADV-major-2 — канонизация, не сырой `object.tobytes()`); сырые данные в digest-вход не попадают.
- `lib_versions` с отсутствующим пакетом (`PackageNotFoundError`) → `null`, не падает; включает sklearn+numpy.
- `FINGERPRINT_VERSION` бамп → другой fingerprint (старые кэши инвалидируются).
