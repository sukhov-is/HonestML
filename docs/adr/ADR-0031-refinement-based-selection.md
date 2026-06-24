# ADR-0031 — Refinement-based отбор: ранжирование по cross-fitted калиброванному лоссу

- **Статус:** Accepted (реализован 2026-06-08, M4d; refinement при TS отключён — отдельный future-ADR)
- **Дата:** 2026-06-08
- **Драйверы:** DM4-1 (честный выбор по умолчанию) + DM4-4 (доверенные вероятности); FR-M4-12;
  NFR-M4-2/3/5. **Источник:** Berta, Holzmüller, Jordan, Bach «Rethinking Early Stopping: Refine,
  Then Calibrate» (arXiv:2501.19195v2, 2025). Дельта к ADR-0026 (significance-band) и ADR-0030
  (калибратор + cross-fit gate).
- **Воркстрим:** M4d (реализуется вместе с ADR-0030 — общий порт `Calibrator`).
- **Жёсткая зависимость:** поле `CVConfig.calibrate` вводится ADR-0030 §1; `CVConfig.selection` —
  этим ADR §5. Оба — аддитивные frozen-поля, мёржатся **одним PR** M4d (порядок однозначен).

## Контекст
Дифференциатор библиотеки — «честно лучшая модель». Отбор (ADR-0026: `equivalence_band` →
`select_best`) ранжирует кандидатов по **сырому** OOF-лоссу (anchor = argmax по `Candidate.score`,
band — по сырому `oof_pred`).

Berta et al. (2025) показывают: для proper-loss (logloss/Brier) `Risk = Calibration error +
Refinement error` (Thm 3.4: `Refinement = min_g Risk(g∘f)` — лосс после оптимальной пост-калибровки).
Минимумы этих ошибок НЕ совпадают (§4–7): отбор по сырому лоссу даёт компромисс, субоптимальный по
обоим. Поскольку победителя мы калибруем пост-хок (ADR-0030), **отбирать надо по refinement = лоссу
ПОСЛЕ калибровки** (eq.4: `R̂ = min_{g∈G} (1/n)Σ ℓ(g∘f)`). Табличные эксперименты (XGBoost/MLP/
RealMLP, отбор гиперпараметров — наш кейс, §5/Apx F): отбор по refinement даёт меньший тестовый
logloss, часто лучше accuracy/AUC. **Класс G у статьи — temperature scaling** (выпуклый, 1 параметр,
не переобучается на малой val, не трогает accuracy/AUC). **Границы статьи:** выгода заметна с ~10K
сэмплов (val ~1600), на малых — шумно; isotonic переобучается на малой val (Apx D); refinement-стоп
при CV/time-series-обучении статья **не исследовала** (только holdout) — это влияет на наши границы (§3–5).

`Candidate.oof_pred`/`score` — единственный вход band (ADR-0026 §6 метрика-агностична). Значит
refinement-отбор = подмена этих полей на **cross-fitted калиброванные** до вызова band; сама band не
меняется. Калибратор/cross-fit — те же, что строит ADR-0030 §3 (gate), → один воркстрим.

## Рассмотренные варианты
1. **Сырой отбор, калибровать только победителя (ADR-0030 как есть).** Не использует знание статьи:
   победитель = компромисс-точка, субоптимален по refinement. Отвергнут как неполная «честно лучшая».
2. **In-sample калиброванный отбор** (калибратор на всей OOF + лосс на ней же). Отвергнут (утечка/
   оптимизм): isotonic (высокий DOF) переобучается in-sample → выигрывает шумом (Apx D статьи).
3. **Cross-fitted refinement-отбор:** out-of-fold калиброванная OOF-proba (калибратор фита на блоках-
   кроме-текущего, предсказывает текущий — без in-sample смещения), на ней пересчитываются `score`/
   `oof_pred`; band/тай-брейк как есть. **Выбран.** Opt-in (дефолт `raw`), низкая дисперсия (класс G =
   sigmoid, §5), no-op вне proper-proba (§2).

## Решение

### 1. Порт `Calibrator` + чистая cross-fit-функция (общие с ADR-0030)
Калибровка — sklearn-деталь, `application/slice` не зависит от adapters (import-linter
`usecases-independent-of-adapters`). Поэтому:
- **`core/ports/calibration.py` — `Calibrator` (Protocol)** и `CalibratorFactory = Callable[[],
  Calibrator]` (зеркало `EstimatorFactory`): `fit(proba, y, sample_weight=None) -> None`,
  `transform(proba) -> np.ndarray`. **Контракт `transform`:** вход/выход одной формы — `(n,)` `P(pos)`
  (binary) или `(n,K)` (multiclass); выход **строго положителен и (для multiclass) построчно нормирован
  на 1** (клип в `[ε,1−ε]`, ε как `align_proba` `1e-6`, аналог Laplace-smoothing статьи Apx D) —
  иначе `log_loss`/`brier` на калиброванной OOF и в bootstrap дают `inf`.
- **`application` — чистая `crossfit_calibrate(oof_proba, y, calib_blocks, factory, *, sample_weight) ->
  np.ndarray`** (Humble Object, numpy, тест на фейк-`Calibrator` без sklearn): для каждого блока `b`
  фитит `factory()` на строках `calib_blocks != b` (валидных), `transform` строк `calib_blocks == b`.
  Возвращает построчно калиброванную OOF-proba. **Единый ИСХОДНИК** и для refinement-отбора (§3), и для
  M4d-gate (ADR-0030 §3); под `time_ordered` cross-fit-семантика — expanding (ADR-0030 §3), но
  refinement-отбор при TS **отключён** (§3), поэтому здесь — симметричный K-fold (не-TS).
- **Вырожденный train-блок** (`<2` классов / строк меньше per-block-порога §4) ⇒ калибратор не
  обучаем ⇒ кандидат **невиабелен** → run-level fallback (§4); НЕ смешивать сырые/калиброванные строки
  внутри кандидата.
- **Детерминизм (механизм):** калибраторы беспараметрически детерминированы (нет `random_state`);
  порядок блоков = отсортированный `np.unique(calib_blocks)`; binary isotonic фиксирует `increasing=
  True` явно (не `'auto'` — иначе data-зависимый флип направления). ⇒ cal-OOF побитово воспроизводима.
- **adapters/calibration.py (M4d)** реализует порт (`SigmoidCalibrator`/`IsotonicCalibrator` над
  sklearn `_SigmoidCalibration`/`IsotonicRegression` + клип); composition инжектит `CalibratorFactory`
  **параметром** в `run_slice` (как `estimators`/`splitter`/`significance_test`) — slice.py НЕ импортит
  adapters. import-linter 3/3 KEPT.

### 2. Область применения = ТОЛЬКО proper-proba-losses (гейт, НЕ монотонность)
Refinement-отбор активен ⟺ `selection=="refinement" ∧ metric.proper_proba ∧ task.is_classification`.
- **Причина no-op для ранжирующих/argmax-метрик — ГЕЙТ `proper_proba`, а не монотонность:** `roc_auc`/
  `pr_auc`/`accuracy` имеют `proper_proba=False` ⇒ refinement не активируется вовсе (no-op, нулевая
  цена). Это не опирается на «калибровка ранг-сохраняюща»: **multiclass per-class OvR + ренорм
  (ADR-0030 §1) НЕ покомпонентно монотонен** и МОЖЕТ менять argmax/OvR-ранг (ADR-0030 §1; статья §3.4),
  поэтому обосновывать no-op монотонностью неверно. 1-D монотонность (sigmoid/isotonic) ранг-сохраняюща
  только для **binary** — но и там no-op гарантирует гейт, а не она.
- **Детекция:** атрибут `proper_proba: bool` **объявляется в Protocol `Metric`** (core/ports/metric.py,
  рядом с `greater_is_better`; дефолт-док `False`) и **выставляется** `True` на `LogLoss`/`Brier`,
  `False` — на остальных (наследуется от `_ScorerBase`) в adapters/metrics.py — паттерн `average`
  (ADR-0021). `getattr(metric,"proper_proba",False)` — ТОЛЬКО защита для внешних метрик, не замена
  объявлению (иначе log_loss/brier не задетектятся → refinement тихо станет no-op, провал FR-M4-12).

### 3. Подмена кандидата на cross-fitted калиброванный (в `run_slice`, до band)
Когда refinement-отбор активен, `_run_candidate` помимо сырой `oof_proba` строит
`cal = crossfit_calibrate(oof_proba, y, calib_blocks, calibrator_factory, sample_weight=sw)` и **на ней**
считает `score = metric.score(y[mask], cal[mask], sw[mask])`; `oof_pred = cal`. Band/`rank`/тай-брейк
работают без изменений — anchor = argmax по калиброванному `score`, членство — по калиброванному
`oof_pred`. (Клип §1 гарантирует конечность скалярного `score` — он НЕ обёрнут в try/except, в отличие
от bootstrap significance.py:108, поэтому `inf` недопустим.)
- **`calib_blocks` ≠ band `block_index` (фикс B1):** для cross-fit калибровки вводится **отдельный**
  `calib_blocks` = id CV-фолда строки (строится **всегда при активном refinement**, не-TS — естественное
  K-fold-разбиение OOF). Band-`block_index` (ADR-0026 §2, fold-block bootstrap) передаётся в
  `equivalence_band` **ровно по прежнему правилу** (не-None только при `time_ordered`) — refinement НЕ
  меняет bootstrap-схему band на не-TS прогоне. Два разных смысла блоков разведены.
- **dev-OOF only (фикс holdout):** refinement cross-fit идёт строго по **dev-OOF внутри `run_slice`(dev)**
  (ADR-0029 §3), НИКОГДА не на outer-holdout (иначе нарушится «тронут ровно один раз», NFR-M4-3). В
  отличие от прод-gate ADR-0030 §3 (у него иная роль — несмещённая оценка).
- **time-series — refinement ОТКЛЮЧЁН в M4 (фикс B2):** при `time_ordered` честный TS-calib-cross-fit
  требует expanding-окна, и «первый фолд без прошлых калибровочных данных» вынуждал бы смешивать сырую
  proba фолда-0 с калиброванной остальных в одном proper-loss score — несопоставимая шкала **внутри**
  кандидата (тот же дефект, что §4 запрещает между кандидатами). Поэтому `selection="refinement" ∧
  time_ordered` → **fallback на сырой отбор + WARNING**; purge-aware TS-refinement (expanding,
  per-block) — **отдельный future-ADR** (статья этот режим не исследовала). База-модель не утекает в
  любом случае (OOF leakage-free, ADR-0022).
- **1 кандидат:** отбора нет → cross-fit пропускается, `score`/`oof_pred` остаются сырыми
  (бессмысленно калибровать leaderboard без выбора); `selection_mode` отражает no-op.

### 4. Виабельность и честный fallback — «всё или ничего» на прогон
Сравнивать кандидатов можно только в одном пространстве score. Refinement-отбор **отключается на весь
прогон** (→ сырой отбор + **WARNING**, `selection_mode="raw"`) если выполнено хотя бы одно:
- **(a) Виабельность калибратора — по PER-BLOCK фит-множеству (фикс «n полной OOF vs per-block»):**
  `min` по блокам числа калибровочных строк (`calib_blocks != b`, валидных) `< ` порога ADR-0030 §3
  (isotonic — выше sigmoid); или у блока `<2` классов; или кандидат не `ProbabilisticEstimator`. Это
  честная защита от per-fold недо-обучения (а не от полной OOF, которая «пройдёт» при достаточном
  суммарном n, но per-block isotonic переобучится — Apx D).
- **(b) Порог СИГНАЛА refinement (фикс «папер: выгода с ~10K»):** `CVConfig.refinement_min_oof:
  int = 2000` (настраиваемый рычаг, как `n_boot*alpha>=50` ADR-0026 §7). Если валидных OOF-строк меньше
  — refinement-ранжирование шумно и может флипнуть победителя без пользы → raw + WARNING («недостаточно
  данных для надёжного refinement-отбора; ориентир статьи ~10K сэмплов / val ~1600»).
Решение симметрично консервативной деградации band (ADR-0026 §7) и калибровки (ADR-0030 §1). НЕ «часть
калибруем, часть нет».

### 5. Конфиг, дефолт, класс G, связь с `calibrate`
- **`CVConfig.selection: Literal["raw","refinement"] = "raw"`** (аддитивно). **Дефолт `raw`** —
  сознательно консервативный: ADR-0026 §4 уже сделал significance-ON дефолтом (RM4-6); второй молчаливый
  сдвиг дефолта отбора рискован. `refinement` — opt-in, обратимо. Путь «дефолт для proba-задач» —
  отдельный ADR после валидации на CV-сетапе (статья тестила holdout). Тумблеры `selection` и
  significance-ON **ортогональны** (ADR-0026 §4 не затронут).
- **Класс G для ОТБОРА = sigmoid по умолчанию (фикс «isotonic шумный»):** refinement-метрика отбора
  использует **low-DOF монотонный** калибратор. Дефолт — **`sigmoid` (Platt)**: для binary близок к
  paper-классу TS, не переобучается на средней val (в отличие от высоко-DOF isotonic, который статья
  Apx D маркирует шумным для отбора). isotonic-как-метрика-отбора **не рекомендуется** (доступен явно).
  **`temperature` (1-параметр, выпуклый — точный класс G статьи) — paper-ideal, future-аддитив** (как и
  multiclass-temperature в ADR-0030 §1). Метод **детерминирован** (NFR-M4-2): при `calibrate="off"`
  метод отбора = `sigmoid` (НЕ `auto` — auto сам ветвится по n). Прод-калибратор победителя по-прежнему
  управляется `calibrate` + gate (ADR-0030 §3) — **независимо** (refinement-отбор работает и при
  `calibrate="off"`: статья разводит «refinement как метрику отбора» и «калибровку-пост-шаг»).
- **Регрессия/не-proper-метрика + `selection="refinement"`:** no-op (§2), отбор сырой (не ошибка).

### 6. Наблюдаемость (паттерн ADR-0026 §6, аддитивно)
`SliceResult`/манифест/фасад получают `selection_mode: Literal["raw","refinement"]` (фактический режим
после fallback §3/§4); фасад публикует `selection_mode_`. **Под refinement leaderboard `score` =
калиброванный лосс** — чтобы пара `(metric, score)` не была немой (поле `LeaderboardEntry.metric`
остаётся `"log_loss"`), в **манифест** кладётся плоский ключ **`score_space:
Literal["raw_oof","calibrated_oof"]`** (`.get`-fallback, forward/backward-симметрично) — читатель
`leaderboard.json` трактует score однозначно без устной договорённости. **`LeaderboardEntry` (frozen,
extra=forbid) НЕ трогается.** anchor/`unstable` под refinement считаются по калиброванному `score`
(ожидаемо, ADR-0026 §1).

## Последствия
- **Положительные:** «честно лучшая модель» доведена до отбора; переиспользует band/significance без
  изменений; калибратор/cross-fit единый с ADR-0030; opt-in + обратимо; no-op для AUC/accuracy/регрессии;
  low-DOF sigmoid-класс отбора + per-block-виабельность + сигнальный порог → защита от шумного refinement.
- **Отрицательные/компромиссы:** доп. стоимость = `N_candidates × K` дешёвых 1-D фитов на OOF (постоянный
  множитель, **без рефита базы** — NFR-M4-8); **TS-refinement отложен** (отдельный ADR — статья режим не
  исследовала); класс G отбора = sigmoid, не paper-точный TS (temperature — future); область сужена до
  proper-losses сознательно; **score под `raw` и `refinement` — разные величины:** тренд-сравнение
  `leaderboard.score` между прогонами валидно только при равном `selection_mode`/`score_space`.
- **Влияние на слои (аддитивно):** `core` — порт `Calibrator` + атрибут `Metric.proper_proba` +
  `CVConfig.{selection,refinement_min_oof}`; `application` — `crossfit_calibrate` + refinement-ветка/
  fallback в `run_slice`/`_run_candidate`; `adapters` — калибраторы + клип (M4d); `composition` — инжект
  фабрики + `selection_mode_`/`score_space`. import-linter 3/3 KEPT. `ARTIFACT_VERSION` не меняется
  (selection_mode/score_space — plain-ключи, NFR-M4-5).

## Проверки
- **Refinement выигрывает, где должен:** синтетика, A лучше по сырому logloss, но хуже по refinement,
  чем B (разные confidence-level при равном refinement, §6 статьи) → `selection="refinement"` выбирает
  B, `raw` — A (FR-M4-12).
- **Нет in-sample оптимизма:** cross-fitted score кандидата с шумной in-sample-калибровкой не лучше
  сырого (assert: cross-fit ≠ in-sample на синтетике).
- **B1 — band-схема не меняется на не-TS:** `selection="refinement"` на K-fold log_loss даёт **тот же
  bootstrap-band** (i.i.d. row, не fold-block), что `raw`, при идентичной OOF (calib_blocks изолирован
  от band block_index).
- **B2 — TS отключает refinement:** `selection="refinement" ∧ time_ordered` → `selection_mode=="raw"`
  + WARNING; band на сырых score.
- **Монотонная инвариантность через гейт:** `selection="refinement"` с `roc_auc`/`accuracy`
  (`proper_proba=False`) даёт тот же band/победителя, что `raw` (no-op); `resolve_metric("log_loss")
  .proper_proba is True`, `resolve_metric("roc_auc").proper_proba is False`.
- **Клип/конечность:** cross-fit isotonic на краевой валидации (proba→0/1) не даёт `inf`-score;
  refinement-score конечен для всех кандидатов; multiclass cal-OOF строго положительна и нормирована.
- **Виабельность/fallback:** вырожденный per-block (или `min` per-block < порог; `n_oof < refinement_
  min_oof`) → `selection_mode=="raw"` + WARNING на весь прогон (§4).
- **dev-OOF only:** refinement-cross-fit-индексы дизъюнктны с outer-holdout (ADR-0029); cross-fit фита
  блока `b` дизъюнктна со строками `b` (NFR-M4-3).
- **Детерминизм:** фикс seed/calib_blocks ⇒ побитово равная cal-OOF, band, победитель (NFR-M4-2);
  инвариантность band к добавлению заведомо-худшего кандидата сохраняется (cal считается покандидатно,
  ADR-0026 §2).
- **Дефолт `raw`:** поведение M4a не меняется; `selection_mode_=="raw"`, `score_space=="raw_oof"`.
- **Слои/чистота:** `crossfit_calibrate` тестируется на numpy + фейк-`Calibrator` без sklearn (NFR-M4-6);
  import-linter 3/3; `selection_mode`/`score_space` в манифесте, `.get`-fallback грузит старые артефакты.
