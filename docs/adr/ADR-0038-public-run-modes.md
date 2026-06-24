# ADR-0038 — Публичные run-режимы: `run_mode` как stage-gate (selection / full)

- **Статус:** Accepted (реализован 2026-06-09)
- **Дата:** 2026-06-09
- **Драйверы:** DM-1 (публичная управляемость фаз без новой механики); FR-RM-1/2/3, NFR-RM-1/3. Наследует
  sklearn-инвариант `__init__` (ADR-0011), раздельность `run_slice`/`refit_best` (ADR-0010), outer-holdout
  evaluation (ADR-0029), отвергнутый «две ручки на одну способность» (ADR-0037).
- **Воркстрим:** M5 (run-modes дельта).

## Контекст
Триада selection (light) / evaluation (full) / final-fit — полезная семантика бюджета.
В текущем коде все три фазы **уже раздельны**: selection = `run_slice` (CV-OOF leaderboard + band), evaluation =
`outer_holdout`-score (ADR-0029), final-fit = `refit_best`. Нужна **публичная управляемость**, какие фазы
исполнять, без переписывания оркестрации и без дублирования уже существующих ручек.

## Рассмотренные варианты
1. **`run_mode ∈ {selection, evaluation, full}` (явная триада).** «evaluation» как mode **дублирует**
   `outer_holdout` (ADR-0029) — две ручки на одну способность (анти-паттерн, отвергнутый ADR-0037: рассинхрон
   `run_mode="evaluation"` vs `outer_holdout=0`). **Отвергнут** (R-MODE-DUP).
2. **`run_mode` как профиль бюджета (light/full параметры моделей, И2).** Требует второго набора параметров
   моделей → HPO-территория, scope-creep. **Отвергнут/отложен** в M7 (R-MODE-SCOPE).
3. **`run_mode ∈ {selection, full}` (stage-gate, И1+И3) + evaluation = ортогональный `outer_holdout`.**
   Минимальная механика (фазы уже раздельны), без дублирования; триада полностью учтена. **Выбран.**
4. **Отдельные методы (`fit_selection()`).** Не sklearn-идиома (ломает `clone`/`get_params`/`Pipeline`).
   **Отвергнут** (параметр идиоматичнее, как `budget`/`significance`/`cache`).

## Решение

### 1. Публичный параметр (verbatim, sklearn-инвариант)
`AutoML(..., run_mode: RunMode = "full")`, где `RunMode = Literal["selection", "full"]` (в `core/config.py`,
рядом с `BudgetMode`/`SignificanceMode`). Хранится в `__init__` **как есть** (ADR-0011), участвует в
`get_params`/`set_params`/`clone`/`Pipeline`. Резолв — в `fit`. Для truthful-провенанса добавляется
`RunConfig.run_mode: RunMode = "full"`, и **`fit` прокидывает `self.run_mode` в `RunConfig(...)`**
(`facade.py` сборка `RunConfig` сейчас не передаёт его — это часть milestone RMM-d) → mode попадает в
run-report через существующий `config`-дамп (`report["config"]["run_mode"]`, без отдельного ключа отчёта).

**Валидация — в `fit`, тип `ConfigError` (фикс m6):** `__init__` хранит verbatim (ADR-0011, без валидации,
иначе ломается `clone`). `fit` строит `RunConfig(...)` **прямым** конструктором (не `.parse()`), поэтому
невалидный `run_mode` дал бы голый `pydantic.ValidationError`, а не `ConfigError` (конвенция проекта). Решение:
**явный guard в `fit`** перед сборкой (`if self.run_mode not in ("selection","full"): raise ConfigError(...)`),
по образцу `cv<2 → ConfigError` (`build._normalize_cv`). Так FR-RM-1(3) («невалидное → ConfigError») выполним.

### 2. Семантика (stage-gate)
- **`"full"` (дефолт) — поведение M5 без изменений:** `run_slice` (selection) → `refit_best` (final-fit) →
  `_calibrate_winner` → holdout-score (если `outer_holdout>0`). Выставляются все атрибуты, включая
  `best_estimator_`/`fitted_`; `predict`/`save_artifact` работают.
- **`"selection"` — только leaderboard:** исполняется **только** `run_slice`; final-fit / calibration /
  holdout-score **пропускаются**.
  - **Выставляются** (фикс M1-review «судьба sklearn-атрибутов»): описывающие вход sklearn-атрибуты
    `n_features_in_`, `feature_names_in_` (если `X` — DataFrame), `classes_` (классификация — из `np.unique(y)`,
    доступно без refit), плюс наблюдаемость selection: `leaderboard_`/`best_model_id_`/`band_member_ids_`/
    `band_unstable_`/`band_width_`/`winner_by_tiebreak_`/`selection_mode_`/`run_report_`. Они дёшевы, безвредны
    и помогают интроспекции.
  - **Не выставляются** (требуют refit/пост-отбора): `fitted_`, `best_estimator_`, `calibration_`,
    `reliability_curve_`, `holdout_score_`.
  - `predict`/`predict_proba`/`score` → `NotFittedError` (через существующий guard `_require_fitted`, который
    проверяет `fitted_`) с подсказкой («run_mode='selection' built a leaderboard but no fitted model; use
    run_mode='full' to ship a model»).
  - **Отгрузка артефакта недоступна** (фикс M5-review формулировки): артефакт строится из `FittedModel`
    (`self.fitted_`) функцией `save_artifact(model, dir)` (`composition/artifact.py`); при `selection` `fitted_`
    отсутствует → артефакт собрать нельзя. Это валидное документированное состояние «инспекция кандидатов без
    отгрузки» (у `AutoML` нет собственного метода `save_artifact` — отгрузка идёт через `m.fitted_`).

### 3. Триада без дублирования (evaluation = outer_holdout)
«Evaluation» (несмещённая оценка) **остаётся** ортогональным `cv=CVConfig(outer_holdout=…)` (ADR-0029),
**не** становится третьим mode. `run_mode="full"` + `outer_holdout>0` = полная триада. Маппинг
документируется: **selection** → `run_mode="selection"`; **evaluation** → `outer_holdout` (ADR-0029);
**final-fit** → `refit_best` внутри `full`. Так избегается анти-паттерн ADR-0037 (одна способность — одна
ручка), и `holdout`-семантика остаётся downstream final-fit (score над refit-моделью), как в коде.

### 4. Слои
`run_mode` — резолв в `composition/facade.fit` (гейтит вызовы `refit_best`/`_calibrate_winner`/holdout). Ни
`application`, ни `core` не меняют поведения (`run_slice` не знает о режиме — он всегда selection). Аддитивное
поле `RunConfig.run_mode` — `core/config` (чистый pydantic).

## Последствия
- **Положительные:** публичная триада на существующих швах; дефолт неизменен; `selection` экономит финальный
  refit на **всех** данных (самый дорогой одиночный fit) для инспекции; truthful-провенанс через
  `RunConfig.run_mode`; нет дублирования ручек.
- **Отрицательные/компромиссы:** `selection`-режим не отгружает модель → `predict` бросает (валидное
  sklearn-состояние, opt-in, документировано; дефолт `full` полностью совместим, R-SELECT-PREDICT); экономия
  refit маргинальна при малых данных (refit = 1 fit из `K×models+1`) — честно отмечено.
- **Влияние на слои:** `run_mode`-параметр + гейтинг фаз — `composition/facade`; `RunMode`/`RunConfig.run_mode`
  — `core/config`; `application`/`adapters` не трогаются. `ARTIFACT_VERSION`/`RUN_MANIFEST_VERSION` не меняются.

## Проверки
- `run_mode="full"`/без параметра → существующий сьют без изменений; `clone`/`Pipeline`/`get_params` сохраняют.
- Невалидный `run_mode` (напр. `"eval"`) → `ConfigError` (guard в `fit`, не голый `ValidationError`).
- `run_mode="selection"` → `leaderboard_`/`band_*`/`run_report_`/`n_features_in_`/`classes_` есть;
  `refit_best`/`_calibrate_winner` **не** зван (spy); `fitted_`/`best_estimator_`/`holdout_score_` отсутствуют;
  `predict`/`score` → `NotFittedError` с подсказкой; `selection` детерминирован (общий seed → тот же leaderboard).
- `run_report_["config"]["run_mode"]` правдиво отражает режим (фасад прокинул `run_mode` в `RunConfig`).
- `full` + `outer_holdout>0` → `holdout_score_` выставлен (ADR-0029 не регрессирует); нет публичного mode
  «evaluation».

## Impl-notes (2026-06-09)
- Реализовано как спроектировано (`composition/facade.py` гейт `ship_model = run_mode=="full"`;
  `core/config.py` `RunMode`/`RunConfig.run_mode`). Проверки: `tests/unit/test_facade.py`
  (`test_run_mode_full_default_unchanged`/`test_clone_preserves_run_mode`/`test_invalid_run_mode_raises_configerror`/
  `test_selection_no_refit`/`test_selection_sets_describing_attrs_not_model`/`test_selection_predict_raises_with_hint`/
  `test_selection_report_truthful`/`test_selection_deterministic`/`test_full_run_mode_in_report`),
  `tests/unit/test_core_config.py` (run_mode round-trip/Literal).
- **Уточнение неспецифицированного края `selection` × `outer_holdout>0`:** carve остаётся гейтнутым по
  `outer_holdout` (как в M5), а по `run_mode` гейтится **только holdout-scoring** (с refit/calibration). Так
  `selection` идёт на том же DEV-сплите, что и `full` → leaderboard напрямую сопоставим; неиспользованный
  holdout при `selection` безвреден. Минимальный диф (один `if ship_model`-блок вокруг
  refit/calibrate/holdout-score).
- **Selection-подсказка `NotFittedError`** определяется по `hasattr(self, "leaderboard_")` при отсутствии
  `fitted_` (нового публичного атрибута `run_mode_` не вводили — минимальный диф).
