# ADR-0052 — Nested-CV арбитраж (= expanding-window под timeseries): опциональное снятие inner-оптимизма

- **Статус:** Принят (реализован в M6d, 2026-06-10; `make_oof_vector_scorer`+общий `_oof_fold_loop`, nested-ветка
  `compare_features`, C5-деградация к holdout+WARNING, `arbitration_splitter` резолв). Питается **SPIKE-M6d-cost**.
- **Дата:** 2026-06-10
- **Драйверы:** DM-H2 (единый механизм, специализируемый схемой), DM-H3 (снятие inner-оптимизма арбитра —
  best-of-N + home-advantage скоринга, opt-in). FR-FSH-3/6/7, NFR-FSH-1/2/3/5. Наследует арбитраж/`make_oof_scorer`/locus (ADR-0048), scheme-aware
  `splitter.split`/`Fold`/`validate_fold` (ADR-0010/0027), expanding-window `TimeSeriesSplitter` (M4).
- **Воркстрим:** M6d.

## Контекст
ADR-0048 (M6c) выбрал арбитраж на **одиночном** DEV-внутреннем `sel_holdout` (single fit → score), явно отложив
**nested-CV арбитраж** («полное снятие inner-оптимизма ценой DEV-данных») и **expanding-window** арбитраж для
timeseries («фикс-окно недооценивает OOD-дрейф») в M6d. SPIKE-M6c-1 показал: best-of-N inflation на одиночном
holdout мал для N=2–3, но **существует**; для полной честности нужен арбитраж, усредняющий по нескольким
независимым поверхностям.

**Ключевое наблюдение (research §0):** expanding-window арбитраж — это **частный случай** nested-CV под
timeseries-схемой. nested-арбитраж строит K фолдов на DEV scheme-aware сплиттером; для `timeseries` тот же
сплиттер (`TimeSeriesSplitter`) **уже** даёт value-based expanding-window с purge/embargo. → **один** механизм,
а не два.

## Рассмотренные варианты
1. **Оставить single sel_holdout (M6c).** Дёшево (1 fit/стратегия), но inner-оптимизм снят лишь частично, а band
   значимости неприменим (один скаляр, не вектор). **Недостаточно для DM-H3/H4.**
2. **Отдельная expanding-window-конструкция для timeseries + отдельный nested для остальных схем.** Дублирование;
   две поверхности арбитража со своими инвариантами. **Отвергнут** (DM-H2: один механизм).
3. **Единый nested-CV арбитраж, специализируемый scheme-aware сплиттером (для timeseries = expanding-window),
   opt-in.** Переиспользует `make_oof_scorer`. **Выбран.**

## Решение

### 1. Конфиг — аддитивная развилка локуса (`core/config.py`)
```python
class FeatureSelectionConfig(BaseModel, frozen, extra="forbid"):
    ...
    selection_holdout: float = Field(0.25, gt=0, lt=1)             # M6c: holdout-режим
    arbitration: Literal["holdout", "nested"] = "holdout"          # M6d; "holdout" = M6c-дефолт
    arbitration_n_splits: int = Field(5, ge=2)                     # K для nested (SPIKE: дефолт 5)
```
- Дефолт `"holdout"` ⇒ **тождественно M6c**. Поля **внутри** `FeatureSelectionConfig` → при `fs=None` в дамп не
  попадают → fingerprint **идентичен M6b** (ADR-0049 §4; NFR-FSH-5). FS-включённый прогон → новый fingerprint
  (уже принятый компромисс M6c).
- **Валидатор** `_check_config` (→ `ConfigError`): `arbitration_n_splits ≥ 2` (Field-constraint). При
  `arbitration="nested"` поле `selection_holdout` **игнорируется** — не ошибка, а **WARNING** в рантайме (FR-FSH-9,
  R-DEADCFG): «selection_holdout не используется в nested-режиме».
- **`nested` осмыслен только при сравнении ≥2 стратегий** (R1-M1): арбитраж не запускается при одной стратегии
  (N=1 short-circuit, §2). Если `arbitration="nested"` задан без `compare` (одиночный путь `strategy=...`), поле
  **игнорируется** с **WARNING** «arbitration=nested без compare ни на что не влияет».
  - **Почему WARNING, а не `ConfigError` в `_check_config`** (фикс R2-F-R2-7): `nested`+`compare=None` —
    **валидная, но бессмысленная** комбинация (как `selection_holdout` в nested, R-DEADCFG); жёсткая ошибка
    нарушила бы аддитивность (пользователь мог задать дефолтный `arbitration` глобально). Консистентно с
    остальными dead-config-полями M6c.
  - **Единственное место WARNING:** `build._resolve_strategies` (composition), где известны и `arbitration`, и
    `compare` — **не** дублировать в рантайме `compare_features`.

### 2. Механика nested-арбитража (`application/feature_compare.py`)
Развилка в `compare_features` (после per-strategy отбора subset'ов на DEV):
- **`arbitration="holdout"` (дефолт):** как M6c — `carve(sel_holdout)`, `_arbitrate_score` (single fit). Без
  изменений.
- **`arbitration="nested"`:** построить **K арбитражных фолдов на DEV** через инъектированный scheme-aware
  сплиттер (`arb_folds = arbitration_splitter.split(dataset)`); оценить каждый subset **pooled-OOF** по этим
  фолдам. Winner — argmax усреднённого скора (или band, ADR-0053).
  - Для `scheme="timeseries"` `arbitration_splitter` = `TimeSeriesSplitter` ⇒ арбитраж = **expanding-window**
    (purge/embargo соблюдены) — FR-FSH-3 покрыт **этим же** путём.
- **Контракт-change `make_oof_scorer` (фикс R1-B1/M3 — НЕ «переиспользование как есть»):** текущий
  `make_oof_scorer` (`feature_compare.py:81-138`) собирает `oof_pred`/`oof_proba`/`mask` **внутри** замыкания, но
  возвращает **только `float`** (`return sign*metric.score(...)`), т.к. его потребитель — `FeatureSubsetSelector.
  score_subset: Callable[...]->float`. Для nested-арбитража нужен **сам OOF-вектор + маска** (для усреднённого
  скора **и** для band ADR-0053). → вводится **sibling-функция** `make_oof_vector_scorer(...) ->
  Callable[[Sequence[int]], tuple[float, np.ndarray, np.ndarray]]` (возвращает `(score, oof_pred, oof_mask)`);
  **существующий** `make_oof_scorer->float` остаётся для `SequentialSelector` (контракт `score_subset->float` не
  меняется). Это **новый код в `application`**, не нулевая дельта.
  - **Единый источник фолд-цикла (фикс R2-F-R2-2 — против двух копий):** выделить приватный
    `_oof_fold_loop(...) -> (oof_pred, oof_proba, oof_mask)` (фолд-цикл + `_fold_proba`/`project_for_metric`); над
    ним `make_oof_scorer = lambda idx: sign*metric.score(project(_oof_fold_loop(idx)))` (→float) и
    `make_oof_vector_scorer = lambda idx: (score, metric_ready_oof, mask)`. **Оба** скорят на **одной и той же**
    metric-ready проекции (`_fold_proba`/`project_for_metric`) — band ADR-0053 и score ADR-0052 не разъезжаются.
- **Граничные случаи nested (фикс R2-C5 — самый опасный):** при classification арбитражный K-fold может дать
  фолд, где `arb_train` не содержит класса (редкий класс / `arbitration_n_splits=5` на малых данных) → proba не
  выровнять, OOF-строки фолда невалидны. Решение: (1) для classification арбитражный сплиттер — **stratified**
  (как leaderboard, реестр по scheme уже даёт `StratifiedKFold`/`StratifiedGroupKFold`); (2) если стратификация
  невозможна (класс с < K объектов) — **деградация к holdout-арбитражу** с **WARNING** «недостаточно объектов
  класса для nested K=…, арбитраж — holdout»; (3) деген-фолд (`arb_test` без класса) исключается из pooled-OOF
  через `oof_mask` (как leaderboard, ADR-0010 §7) — `global_classes=np.unique(y)` выравнивает proba (наследует
  M6c-фикс). Не молчаливый NaN.
- **Форма `per_strategy` под nested (фикс R2-C1 — публичный run_report-контракт):** `arb_score` в
  `per_strategy` = **среднее по K** арбитражным фолдам (pooled-OOF метрика). Аддитивно: при nested добавляются
  `arb_score_std` (разброс по фолдам) и `n_arb_folds`. M6c-форма `(name, n_selected, arb_score)` сохранена; новые
  ключи — опциональны (старые парсеры игнорируют). Версии не бампаются.

### 3. Что именно снимает nested-арбитраж — честная граница (DM-H3, R-HOME, фикс R1-F5)
Арбитраж **переобучает subset с нуля** на своих `arb_train`-фолдах (`fit_predict`) и скорит `arb_test`, **не**
переиспользуя внутренние OOF-скоры стратегии. ⇒ снимается **best-of-N inflation арбитра** и **home-advantage
скоринга** wrapper'а (его внутренняя оптимизация OOF на `dev_folds` не совпадает с поверхностью арбитража —
арбитраж судит на **другом** фолд-разбиении с переобучением).
- **Чего nested НЕ снимает (честно, без overclaim):** subset каждой стратегии **отбирается один раз на всём
  DEV** (`dev_fit`), затем арбитрируется на фолдах, **строки** которых пересекаются с DEV. Поэтому **оптимизм
  самого отбора** (wrapper-feature-selection-bias: subset подогнан под строки DEV, часть которых попадает в
  `arb_test`) **остаётся** — он не устраняется переобучением, т.к. переобучается *модель*, а *subset* фиксирован.
  Полностью честный nested потребовал бы **переотбора subset внутри каждого внешнего фолда** (per-fold
  re-selection) — это дороже (×K к стоимости отбора, не только скоринга) и **отложено в M6e**.
- Итог формулировки: nested снимает **best-of-N inflation арбитра** (это и был объект SPIKE-M6c-1), а **не**
  «полностью» feature-selection-оптимизм wrapper'а. Честность держится на **переобучении модели** на независимых
  фолд-разбиениях (не на disjoint-строках, как single-holdout M6c). Property-тест R-HOME (§Проверки) покрывает
  обратное направление (арбитраж не дотягивается до отбора), но **не** прямой перенос subset↔строки — это
  отмечено явно.
- **Внешний holdout (ADR-0029) не затрагивается** — арбитраж целиком на DEV (NFR-FSH-1).

### 4. Резолв сплиттера арбитража (`composition/build.py`) — полный объём проводки (фикс R1-M2)
Инъекция нового `arbitration_splitter` в `compare_features` из composition (как `splitter`/`carve`, ADR-0016) —
по `cv.scheme` тем же реестром, что leaderboard-сплиттер, но с `n_splits = fs.arbitration_n_splits` и **отдельным
seed** (ниже). `application` **не** именует конкретный адаптер (`layers`/`usecases-independent-of-adapters` KEPT).
Для `group` — group-aware K-fold; для `timeseries` — `TimeSeriesSplitter` (expanding-window).
- **Это новый сквозной параметр, не «как уже есть»** — он тянется через цепочку: `Components` (+поле
  `feature_arbitration_splitter`) → `build_default_components` (резолв) → `facade`/`run_slice` (прокидывает в
  FS-bundle) → `compare_features(+arbitration_splitter)`. Перечислено в operational §1 (эволюция контрактов) и в
  плане реализации; объём — `composition`+`application`-сигнатуры (без правки `core`).
- **Seed-источник (фикс R1-F10, детерминизм NFR-FSH-3):** `arbitration_splitter` и `BootstrapSignificanceTest`
  (ADR-0053) seed'ятся **детерминированно от run `random_state`** (как leaderboard-сплиттер/тест в M4-composition),
  **не** per-strategy (арбитраж — cross-strategy шаг, один на все кандидаты). Тот же `(random_state)` ⇒ те же
  `arb_folds` и тот же bootstrap ⇒ воспроизводимый winner. Per-strategy seed-изоляция (`_strategy_seed`) остаётся
  **только** на этапе отбора subset'ов, не на арбитраже.

## SPIKE-M6d-cost (квантование R-COST → дефолт K, opt-in)
Замер цены арбитражной единицы (ExtraTrees[100] fit_predict), медиана 5 повторов:

| rows×feats | N | K | holdout(s) | nested(s) | added(s) | ratio |
|---|---|---|---|---|---|---|
| 2000×30 | 3 | 5 | 0.51 | 2.80 | +2.29 | 5.5× |
| 5000×60 | 3 | 5 | 1.43 | 7.56 | +6.13 | 5.3× |
| 20000×120 | 3 | 5 | 9.34 | 49.2 | +39.9 | 5.3× |
| 20000×120 | 3 | 10 | 9.34 | 98.5 | +89.1 | 10.5× |

**Вывод:** nested ≈ **K×** цены holdout-арбитража (ровно как ожидается: N→N×K фитов). При K=5 и N=3 на средних
данных добавка единицы–десятки секунд, на крупных (20k×120) — ~40с; K=10 удваивает. ⇒ (а) nested — **opt-in**
(дефолт `holdout`); (б) дефолт `arbitration_n_splits=5` (баланс снятия оптимизма и цены; K=10 — пользователь
по желанию); (в) включение nested логирует **WARNING** с оценкой `N×K` фитов (NFR-FSH-2). Арбитраж — вне budget
trials (пре-процессинг, как M6c).

## Последствия
- (+) Снятие **best-of-N inflation арбитра** (SPIKE-M6c-1) + home-advantage скоринга wrapper'а как opt-in;
  expanding-window для timeseries «бесплатно» из того же механизма (DM-H2); OOF-вектор (sibling-scorer §2)
  открывает честный тай-брейк (ADR-0053); дефолт не меняет M6c.
- (−/компромисс) Цена ×K (квантовано, opt-in, WARNING); `selection_holdout` мёртв в nested (WARNING); честность
  держится на переобучении модели, а не disjoint-строках (§3); **оптимизм самого отбора subset остаётся** (subset
  фиксирован на всём DEV) — полностью честный per-fold re-selection отложен в **M6e** (§3); требуется **новый**
  sibling-scorer (OOF-вектор), а не нулевая дельта (§2).
- **Влияние на слои:** конфиг/спека — `core`; nested-механика + sibling OOF-scorer — `application`; резолв
  сплиттера арбитража — `composition`; адаптеры сплиттеров — `adapters` (переиспользуются). `import-linter` 3/3 KEPT.

## Проверки
- `arbitration="holdout"` (дефолт) → тот же winner, что M6c (тест эквивалентности); `fs=None` → fingerprint M6b.
- `arbitration="nested"` → winner = argmax усреднённого по K скора, **детерминирован** при seed (seeded
  сплиттер арбитража); N=1 → арбитраж не запускается (как M6c); WARNING с оценкой `N×K` фитов.
- `scheme="timeseries"`+`nested` → expanding-window (purge соблюдён), внешний holdout нетронут.
- **Property независимости (R-HOME, обратное направление):** перестановка таргета в арбитражном `test`-фрагменте
  меняет `arb_score`, но не меняет subset стратегии; `sequential` не получает преимущества скоринга над
  ранжерами на одних данных (тест на фейк-стратегиях). **Граница (R1-F5/F6):** тест покрывает только обратное
  направление (арбитраж не дотягивается до отбора); прямой перенос «subset↔строки DEV» он **не** ловит — это
  задокументированный остаточный оптимизм отбора (§3), снимаемый лишь per-fold re-selection (M6e).
- `arbitration="nested"` без `compare` → WARNING (мёртвый конфиг, §1); `arbitration_n_splits < 2` → `ConfigError`;
  `nested`+`selection_holdout` → WARNING, прогон проходит.
