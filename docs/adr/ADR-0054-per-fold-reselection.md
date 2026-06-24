# ADR-0054 — Per-fold re-selection: полностью честный nested арбитраж (третий режим)

- **Статус:** Принят (дизайн M6e; реализация — скил implementation). Питается SPIKE-M6e-cost.
  **Расширяет ADR-0052** (nested арбитраж) — снимает остаточный оптимизм отбора, явно отложенный там в M6e (§3).
- **Дата:** 2026-06-10
- **Драйверы:** DM-E1 (снять оптимизм самого отбора, не только скоринга). FR-FSE-1/2/3/4,
  NFR-FSE-1/2/3. Наследует `_select_one`/`select_features`/`_oof_fold_loop`/`equivalence_band`/`Candidate`
  (ADR-0044/0046/0052/0053), scheme-aware `splitter.split`/`Fold` (ADR-0010/0027), expanding-window
  `TimeSeriesSplitter` (M4).
- **Воркстрим:** M6e.

## Контекст
M6d nested (ADR-0052) **отбирает subset один раз на всём DEV**, затем пере-обучает *модель* на K арбитражных
фолдах и скорит pooled-OOF. Это снимает **best-of-N inflation арбитра** и **home-advantage скоринга**, но
**не оптимизм самого отбора**: subset подогнан под строки DEV, часть которых попадает в `arb_test`
(`feature_compare.py:422-426`, фиксированный `idx` в `scorer(idx)`). ADR-0052 §3 дословно отложил исправление в
M6e: *«Полностью честный nested потребовал бы переотбора subset внутри каждого внешнего фолда (per-fold
re-selection) — ×K к стоимости отбора, не только скоринга»*. Дифференциатор «честно лучшая модель» требует
честной оценки **обобщающей способности FS-процедуры**, а не зафиксированного subset'а.

**Ключевое наблюдение:** это классический **nested CV для отбора модели** — inner-CV **отбирает** признаки,
outer-CV **оценивает процедуру**, финальный refit на всех данных даёт поставляемый subset. Спайн уже устроен
под это: `select_features(x, y, folds, ...)` принимает фолды; `_select_one` диспетчеризует ранкер/wrapper;
M6c-carve уже демонстрирует рестрикцию данных (`dataset.take` + `splitter.split`).

## Рассмотренные варианты
1. **Оставить M6d nested.** Дёшево (N отборов), но оптимизм отбора остаётся. Недостаточно для DM-E1.
2. **Per-candidate per-fold (переотбор + полный поиск гиперпараметров отбора внутри фолда).** Максимально
   честно, но кратно дороже и вне объёма (per-candidate FS отложен). Отвергнут.
3. **Per-fold re-selection как третий режим арбитража (`nested_per_fold`), opt-in, переиспользуя спайн.**
   Истинный nested: inner отбирает на outer-train, outer оценивает процедуру; финальный subset — отбор
   победителя на всём DEV. **Выбран.**

## Решение

### 1. Конфиг — третий локус арбитража (`core/config.py`)
```python
FSArbitration = Literal["holdout", "nested", "nested_per_fold"]   # M6e добавляет третий
arbitration: FSArbitration = "holdout"                            # дефолт = M6c
```
- Дефолт `"holdout"` ⇒ тождественно M6c; `"nested"` ⇒ M6d. `"nested_per_fold"` — opt-in, **только** при
  `compare` (≥2 стратегии). Поле внутри `FeatureSelectionConfig` ⇒ `fs=None` → fingerprint M6b (NFR-FSE-7).
- **R-DEADCFG-E (FR-FSE-1):** `nested_per_fold` без `compare` → **WARNING** «ни на что не влияет» в
  `build._resolve_strategies` (как M6d nested), не `ConfigError`. `arbitration_n_splits` переиспользуется как
  **число outer-фолдов** K.
- **«per_fold без nested» структурно невозможно (фикс R1-completeness):** `nested_per_fold` — **третье
  значение enum** `arbitration`, а **не** флаг поверх `nested`; комбинация «per_fold+nested» не существует по
  построению — отдельный валидатор не нужен (закрывает открытый вопрос 00-research).
- **Composition-гард литералов (фикс R1-clean-arch-m3):** существующая проводка гейтит арбитраж-сплиттер на
  **строковом литерале** `arbitration == "nested"` в **трёх** местах (`build.py:164,171`,
  `feature_compare.py:399`). Их нужно превратить в проверку членства (`arbitration in {"nested",
  "nested_per_fold"}`), иначе `nested_per_fold` молча получит `feature_arbitration_splitter=None` и
  `compare_features` **тихо** свалится в holdout-ветку (silent wrong-mode, не fail-fast). ⇒ реализация
  обязана: (а) заменить три литерала на множество; (б) добавить **assert** `arbitration_splitter is not None`
  при `nested_per_fold` (громкий отказ вместо тихой деградации).

### 2. Механика — новый strategy-level скорер (`application/feature_compare.py`)
M6d `make_oof_vector_scorer(idx)` параметризован **фиксированным subset** — недостаточен (research §3). Вводится
**sibling** — per-fold-reselection скорер `_score_procedure(strategy, name, seed)`:
```python
# outer = arb_folds (arbitration_splitter); inner строится над outer-train тем же splitter
def _score_procedure(strat, name, seed) -> tuple[float, np.ndarray, np.ndarray, float]:  # (score, OOF, mask, mean_size)
    oof, mask, sizes = np.full(n, np.nan), np.zeros(n, bool), []
    for fold_id, outer in enumerate(arb_folds):
        tr = outer.fit_idx if outer.es_idx.size == 0 else np.concatenate([outer.fit_idx, outer.es_idx])
        seed_f = _strategy_fold_seed(name, random_state, fold_id)             # blake2b, НЕ hash() (§4)
        inner = list(splitter.split(dataset.take(tr)))                       # inner-CV над outer-train
        idx_f = _select_one(strat, x[tr], y[tr], inner, categorical=categorical, config=config,
                            seed=seed_f, sample_weight=sw_slice(tr), fit_predict=fit_predict, metric=metric,
                            task=task, global_classes=global_classes, groups=grp_slice(tr))   # переотбор!
        proba, pred, cls = fit_predict(x[tr][:, idx_f], y[tr], x[outer.test_idx][:, idx_f], sw_slice(tr), random_state)
        oof[outer.test_idx] = _project_fold_oof(proba, pred, cls, global_classes, ...)  # ОБЩИЙ band-fill (см. ниже)
        mask[outer.test_idx] = True
        sizes.append(len(idx_f))                                             # для Occam-ключа (mean per-fold)
    return metric_over(oof, mask), oof, mask, float(np.mean(sizes))
```
- **Тип-контракт (фикс R1-clean-arch-m1):** `_score_procedure` берёт **те же инъектированные** `splitter:
  object` / `arbitration_splitter: object` (duck-typed `.split()`, как `compare_features:352,359`) и
  `fit_predict: FitPredict`; `_select_one` переиспользуется **дословно**. **Никакого** импорта адаптера и
  **никакого** нового порта core (`FeatureRanker.rank`/`CVSplitter.split` достаточны). Реализация **не** должна
  аннотировать сплиттер конкретным адаптером (сломает `usecases-independent-of-adapters`).
- **Точный seam извлечения (фикс R2-adversarial/verifier — НЕ из `_oof_fold_loop`):** `_oof_fold_loop`
  (`feature_compare.py:87-131`) fixed-idx → не переиспользуем. **Единственный безопасно-общий** юнит — **band-vector
  fill** `oof_ready[mask]=proj` (сейчас инлайн в `make_oof_vector_scorer.score_vector`,
  `feature_compare.py:205-211`), используемый **только** vector-scorer'ом. Его выносим в `_project_fold_oof(...)`,
  потребляемый vector-scorer'ом **и** `_score_procedure`. **НЕ** трогать float-scorer `make_oof_scorer`
  (`:158-166`) — он per-row fill **не делает** (проецирует pooled `oof_pred[mask]` один раз), и per-fold
  class-align в `_oof_fold_loop:126` (`_fold_proba`) — **иной** концерн. Приёмка: equivalence-тест, пиннящий
  выход M6d float/vector-scorer'ов **до/после** рефактора (нет регресса).
- **Band/winner:** `_score_procedure` отдаёт `(score, oof_vec, mask, mean_size)`; первые три — та же форма, что
  потребляет `Candidate`/`equivalence_band` (ADR-0053); четвёртая — Occam-ключ. Winner — компактнейшая стратегия
  в band; `winner_rule` (`band_tiebreak`/`argmax_band_empty`) как M6d.
- **`n_features` — int-ключ из round(mean), raw float — в наблюдаемость (фикс R2-adversarial/completeness):**
  per-fold subset'ы различны; Occam должен сравнивать компактность **скорившегося** объекта ⇒ ключ — средний
  размер per-fold subset'ов. Но `Candidate.n_features` объявлен **`int`** (`selection_policy.py:41`, первый
  лексикографический tie-break-ключ) — нельзя писать float. ⇒ `Candidate.n_features = round(mean(sizes))` (**int**,
  для band-ключа), а **сырой** `mean(sizes)` (float) уходит в наблюдаемость (`per_strategy`/report) как
  `mean_n_features` — отдельным полем. Это (а) убирает category-error (компактность скорившегося объекта),
  (б) **не ломает** int-контракт `Candidate`/`LeaderboardEntry`, (в) снимает противоречие с §3 (full-DEV subset —
  **только для победителя**).
- **`significance="off"` (фикс R2-completeness):** при `NoSignificanceTest` band схлопывается в anchor (как
  M6d-nested) ⇒ winner = argmax pooled-OOF процедуры, `winner_rule="argmax_band_empty"`, Occam-tie-break **инертен**
  (mean-per-fold n_features значим **только** при активном тесте). Документировано; проверка эквивалентности с
  M6d-nested-off.

### 3. Поставляемый subset (FR-FSE-3, R-PFSHIP)
OOF выбирает **стратегию**-победителя (band). Финальный поставляемый `winner_idx`/`winner_subset` =
**отбор победившей стратегии на всём DEV** (`_select_one(winner_strat, x_full, y, sel_folds, ..., groups)`) —
детерминированный refit, train==inference (ADR-0045). Это тот же full-DEV subset, что M6d вычисляет для всех
стратегий; в `nested_per_fold` он считается **только для победителя** (экономия), а per-fold OOF — лишь для
честной оценки процедуры. ⇒ артефакт-формат неизменен (хранится один subset).
- **Честная граница (фикс R1-E2 — без overclaim):** per-fold снимает оптимизм из **арбитража/оценки** (выбор
  стратегии больше не основан на оптимистичном скоре зафиксированного-на-DEV subset'а — это и был объект DM-E1).
  Но OOF честно оценивает **процедуру**, а **не** конкретный поставляемый subset: финальный subset отбирается на
  всём DEV и несёт **собственный остаточный selection-оптимизм** (тот же, что у M6d full-DEV subset). Это
  **стандартное ограничение nested-CV** (честная оценка процедуры → выбор → refit на всех данных), **не**
  устранённое и не устранимое без отказа от единого subset (train==inference требует один subset). Формулировка:
  «per-fold делает выбор стратегии честным; обобщающая способность поставляемого subset не ограничена OOF-оценкой».

### 4. Анти-ликедж inner/outer (FR-FSE-4, NFR-FSE-1)
- Inner-фолды строятся **только** над outer-train (`dataset.take(tr)` + scheme-aware `splitter`); `groups`/
  `sample_weight` срезаются к `tr` **до** inner-отбора (как спайн срезает `groups[train_idx]`,
  `feature_selection.py:91`) ⇒ ранкер ни в одном фолде не видит outer-test строк.
- **timeseries — purge УСЛОВЕН, не безусловен (фикс R1-E1 blocker):** границу outer-train↔outer-test пёрджит
  **сам outer (`arbitration_splitter`)** — `TimeSeriesSplitter` применяет purge/embargo **только** при `purge>0`
  или заданном `label_time` (`splitters.py:302,310`). Если outer-сплиттер сконфигурирован с purge/label_time
  (он наследует `cv`-конфиг через `_resolve_splitter`), то `tr` **уже** отделён от outer-test → ранкер не видит
  label-leak. Inner-CV пёрджит **только** свои inner-test-окна, **не** перепёрджит `tr` против outer-test —
  это **зона ответственности outer-сплиттера**. При `purge=0`/без `label_time` пёрджа нет (как у leaderboard-CV
  и M6d nested) — это **предусловие пользователя для timeseries**, а **не** дефект, специфичный для per-fold.
  ⇒ claim переформулирован: «purge границы соблюдён **при сконфигурированном** purge/label_time outer-сплиттера».
  - **Предусловие — наблюдаемо, не молча (фикс R2-day2/adversarial):** при `arbitration∈{nested,
    nested_per_fold}` под `timeseries` с `purge=0` **и** без `label_time` — **WARNING** в composition
    «inner-переотбор не пёрджит границу; задайте purge/label_time для leak-safe per-fold арбитража» (зеркалит
    существующие look-ahead WARNING'и `build.py`). Приёмка: тест с `purge>0` проверяет, что строка с `t1`,
    заходящим в outer-test, **отсутствует** в `tr`-индексах, переданных в `_select_one` (assert по per-fold
    `tr`, не по выходу сплиттера — иначе vacuous-pass); `purge=0` тестируется **явно** как no-purge-by-precondition.
  Внешний holdout (ADR-0029) **не затрагивается** — арбитраж целиком на DEV (NFR-FSE-1).
- **Seed — per-outer-fold blake2b, НЕ Python `hash()` (фикс R2-adversarial/verifier):** один и тот же seed во
  **всех** outer-фолдах коррелирует null_importance-перестановки → занижает дисперсию (`fold_subset_jaccard`).
  ⇒ inner-отбор получает **per-outer-fold** seed через **новый `_strategy_fold_seed(name, random_state,
  fold_id)` на `hashlib.blake2b`** (расширяет существующий `_strategy_seed`, `feature_compare.py:63-71`). **НЕ
  `hash()`** — встроенный Python `hash()` солится `PYTHONHASHSEED` и **недетерминирован между процессами**
  (именно поэтому `_strategy_seed` использует blake2b). Независимые розыгрыши **при сохранении детерминизма**
  (фиксированная функция run-seed) ⇒ два прогона с одним `random_state` → идентичный winner (NFR-FSE-3).
  `arbitration_splitter`/`splitter`/significance seeded от run-seed (cross-strategy, ADR-0052 §4).

### 5. Стоимость (R-PFCOST, NFR-FSE-2) — SPIKE-M6e-cost
Per-fold = **N×K_outer отборов** (каждый — inner per-fold ранкер-цикл над K_inner фолдами) поверх N×K_outer
refit'ов модели. Для `null_importance` отбор сам ×(1+n_runs) фитов ⇒ доминирует. **SPIKE-M6e-cost (измерено):**
множитель = ×K_outer точно; абсолют — `importance`/`shap` per-fold умеренны (десятки–сотни сек), а
**`null_importance` per-fold — до часов на крупных данных** (20000×120, K=5 ≈ 3.8ч). ⇒ дефолт
`arbitration_n_splits=5`, **строго opt-in**, **WARNING** в `_warn_fs_cost`.
- **WARNING несёт ПРОЕКТИРУЕМУЮ оценку (фикс R2-day2):** не голый «N×K_outer отборов», а грубая оценка
  числа фитов `N × K_outer × (1+n_runs для null_importance, иначе 1) × K_inner` — чтобы footgun (часы) был виден
  **до** запуска, а не «молчит часами». Рекомендация: per-fold практичен с `importance`/`shap`;
  `null_importance`+per-fold — малые/средние данные.
- **Hard-budget ceiling — опционально (Day-2):** жёсткий `ConfigError` при превышении проектируемого бюджета
  (как C5-деградация служит «потолком» для структурной невозможности) — **отложен**: в M6e достаточно
  WARNING-с-оценкой + opt-in (не плодить config-knobs); решение на design-gate. Кэш ограничен (outer-train
  дизъюнктны). Арбитраж — вне budget trials.

### 6. C5-граница — outer (глобально) И inner (fold-local, фикс R2-adversarial: без переусердствования)
- **Outer (наследует M6d):** classification + редкий класс < K_outer на **всём DEV** ⇒ редкий класс **глобален**
  ⇒ **деградация всего арбитража к holdout** с WARNING (тот же гард `feature_compare.py:400-410`). Это корректно
  именно потому, что проблема глобальна.
- **Inner — fold-local, НЕ глобальный fallback (фикс R1-E4 + R2-adversarial):** даже при ≥K_outer строк класса
  на DEV, на конкретном outer-train `tr` редкий класс может иметь **< inner n_splits** строк →
  `StratifiedKFoldSplitter` в `_select_one` бросит `ValueError` → широкий `except` → `FeatureSelectionError` →
  жёсткий отказ. Проблема **fold-локальна** (одно неудачное разбиение), поэтому глобальный fallback к holdout —
  **переусердствование** (тихо сбрасывает opt-in честность для всех стратегий из-за одного фолда). ⇒ **до**
  inner-фолдов в `_score_procedure` проверять per-outer-fold `min` класс в `tr` ≥ inner n_splits; если нет —
  **деградировать ТОЛЬКО этот outer-фолд** (фолд исключается из pooled-OOF через `mask` — pooled-OOF уже
  толерантен к непокрытым строкам) с WARNING; **остальные фолды остаются nested**. Глобальный fallback — только
  когда **ни один** outer-фолд не выжил (per-fold невыполним) или для outer-кейса. Приёмка: тест с редким классом
  в одном outer-test фолде → деградирует **один** фолд, арбитраж остаётся per-fold.
- **Наблюдаемость деградации (фикс R2-completeness; 4-е значение — фикс impl-review):** `CompareOutcome`/
  `SliceResult` несут `arbitration_effective ∈ {nested_per_fold, per_fold_partial_c5_inner,
  holdout_degraded_c5_outer, holdout_degraded_c5_inner}` — из манифеста видно, **отработала** ли честная
  per-fold процедура или деградировала и **почему**:
  - `nested_per_fold` — per-fold прошёл полностью;
  - `per_fold_partial_c5_inner` — часть outer-фолдов выпала по inner-C5, остальные nested (`per_fold_reselection
    = true`);
  - `holdout_degraded_c5_outer` — глобально редкий класс < K_outer → весь арбитраж holdout (`= false`);
  - `holdout_degraded_c5_inner` — **ни один** outer-фолд не выжил по inner-C5 (per-fold невыполним) → весь
    арбитраж holdout (`= false`). Отдельное значение, т.к. **причина** иная, чем outer-кейс (правдивее, чем
    переиспользовать `holdout_degraded_c5_outer`). Аддитивно, версии не бампаются.
- Дополнительно inner-отбор на малом outer-train может упереться в floor cutoff'а — спайн уже гарантирует ≥1
  признак (`apply_cutoff` floor).

## Последствия
- (+) Снят **оптимизм самого отбора** (честная оценка FS-процедуры) — закрывает ADR-0052 §3, прямо служит
  дифференциатору; переиспользует спайн/`_select_one`/band (core без изменений); полный back-compat (дефолт
  holdout); expanding-window inner для timeseries «бесплатно».
- (−/компромисс) Цена ×K к **отбору** (квантовано, opt-in, WARNING) — дороже M6d nested; per-fold subset'ы
  различны ⇒ поставляется отбор победителя на DEV (не один из per-fold subset'ов) — задокументировано;
  C5-деградация при редком классе.
- **Влияние на слои:** конфиг — `core`; per-fold скорер + проводка — `application`; сплиттеры — `adapters`
  (переиспользуются). **Порты core не меняются** (`FeatureRanker.rank`/`splitter.split` достаточны).
  `import-linter` 3/3 KEPT.

## Проверки
- **Property (FR-FSE-2, NFR-FSE-1):** перестановка таргета в outer-train **меняет** per-fold subset (в отличие
  от M6d-фиксированного); перестановка в outer-test НЕ меняет subset; ранкер не видит outer-test (фейк-стратегии,
  без обучения модели).
- `arbitration="nested_per_fold"` + 2 неотличимые стратегии → побеждает компактнейшая (band, ADR-0053);
  `winner_subset` = full-DEV отбор победителя (FR-FSE-3).
- `timeseries`+`nested_per_fold` → inner/outer expanding-window (purge **при сконфигурированном**
  purge/label_time outer-сплиттера, фикс R1-E1); внешний holdout нетронут (FR-FSE-4).
- Детерминизм при seed (два прогона `_strategy_fold_seed`/blake2b → идентичный winner, NFR-FSE-3, фикс
  R2-verifier); `nested_per_fold` без compare → WARNING; **C5 outer** (глобальный редкий класс) → деградация
  всего арбитража к holdout + WARNING; **C5 inner** (редкий класс на outer-train) → деградация **только этого
  фолда** (остальные nested) + WARNING, не fail-fast (фикс R2-adversarial).
- **sw / multiclass-проводка через `_score_procedure` (фикс R2-completeness — новый edge):** (а) multiclass с
  классом, **отсутствующим** на одном outer-train (вероятнее при дизъюнктных фолдах, чем в M6d) → pooled-OOF
  proba выровнена к **whole-DEV `global_classes`** (`np.unique(y)`), метрика без column-drift; (б) weighted
  (`sample_weight≠None`) → `sw` срезан к `tr` для inner-ранкера, outer-`fit_predict` **и** pooled-OOF метрики
  (как non-per-fold путь). Явные тесты на оба.
- **Комбинированный путь `nested_per_fold`+`time_window`+interventional (фикс R2-completeness):** whole-DEV
  densified `time_window`-метка идёт как band `block_index`; **она же** срезается к outer-train для inner-null
  (срез глобально-уплотнённой метки — намеренно; per-fold фрагментация блоков ожидаема, §ADR-0055). Интеграционный
  тест на все три knobs разом.
- **Наблюдаемость:** `_feature_selection_report` несёт `per_fold_reselection` (true/false при деградации),
  `arbitration_effective` (§6), `fold_subset_jaccard`/`n_distinct_subsets` (стабильность процедуры) и
  `mean_n_features` (raw float, §2). Аддитивно, версии не бампаются.
- `fs=None` → fingerprint M6b; версии не бампаются; `lint-imports` 3/3.
