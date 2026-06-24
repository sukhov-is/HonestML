# ADR-0062 — Честная интеграция HPO (inner-CV objective, внешняя селекция, бюджет, `HPOConfig`)

- **Статус:** Accepted (M7, design-gate pending)
- **Драйвер:** DM-71 (анти-ликедж), DM-74 (стоимость) — FR-HPO-4/5/6; NFR-M7-3/4/5/7
- **Связано:** ADR-0061 (порт Tuner); ADR-0016/0029 (honest CV / outer_holdout); ADR-0032 (Budget/graceful);
  ADR-0035 (fingerprint); ADR-0057/0058 (паттерн resolve-провенанса/cost-оценки FS); SPIKE-M7-hpo (Q3 cost).

## Контекст
Тюнинг на **тех же** CV-фолдах, что и оценка, → leaderboard-скор тюненой модели
оптимистично смещён (R-HPOLEAK). Дифференциатор «честно лучшая модель» требует, чтобы выбор гиперпараметров
**не смещал** внешнюю селекцию. As-is даёт честную внешнюю машину: `run_slice` (OOF/band/significance) +
`outer_holdout` (ADR-0029, несмещённое финальное число). Нужно встроить HPO так, чтобы тюнинг происходил
**внутри**, а честная селекция оставалась **снаружи**.

## Рассмотренные варианты
1. **HPO внутри `run_slice` per-fold (nested-per-fold)** — полностью честный (ре-тюнинг в каждом внешнем
   фолде), но `× K_outer` сверху к стоимости (SPIKE Q3) и глубокая хирургия в `run_slice`. effort L.
2. **HPO на тех же внешних фолдах** — ❌ ликедж селекции; нечестно.
3. **HPO на inner-CV DEV → тюненая фабрика как кандидат внешней честной селекции (flat)** — ✅ дёшево,
   `run_slice` остаётся estimator-blind, тюненая модель **конкурирует в band**; остаточный optimism мягкий и
   раскрываемый, `outer_holdout` его снимает.

## Решение (Вариант 3; nested-per-fold — Day-2)

### §1 `HPOConfig` (core/config.py, opt-in)
`RunConfig.hpo: HPOConfig | None = None` (дефолт None → OFF, fingerprint M6 сохранён, как `fs`). Поля
(`frozen`, `extra='forbid'`):
```
backend: Literal["optuna"] = "optuna"
n_trials: int = Field(50, gt=0)            # per-model HP-поиск (НЕ run-budget, §6)
timeout_s: float | None = Field(None, gt=0)# опц. per-model потолок; задан -> недетерминизм (§5)
inner_cv: int = Field(3, ge=2)             # inner-CV фолдов для objective на DEV
models: tuple[str, ...] | None = None      # какие типы тюнить; None -> все с непустым search_space
keep_baseline: bool = False                # True -> базовая фабрика остаётся кандидатом рядом с тюненой
random_state: int | None = None            # None -> наследует RunConfig.seed
```

### §2 Стадия тюнинга (facade.fit, post-carve, до run_slice)
Точка: после `outer_holdout`-carve (`ds` = DEV) и **до** `run_slice`. Application-spine
`tune_estimators(ds_dev, task, *, tunable, make_factory, tuner, metric, policy, splitter, hpo,
sample_weight, budget) -> dict[str, TuneOutcome]`:
1. Для каждого типа из `hpo.models ?? все-с-непустым-space`: построить **inner-CV** на DEV тем же
   splitter-scheme (`inner_cv` фолдов, не пересекая `outer_holdout`).
2. Замкнуть `score(params)` = mean по inner-фолдам. **Граница переиспользования (уточнено по ревью R2 —
   два РАЗДЕЛЬНЫХ таргета, не один):**
   - **(a) per-fold движок:** переиспользовать **только `_run_candidate`** (fit на `fit⊕es`, score на `test`
     через `Metric`) — НЕ вызывать тело `run_slice` целиком.
   - **(b) OOF-TE — отдельный шаг spine'а:** `_run_candidate` **сам TE не делает** — в `run_slice`
     `_augment_oof_te` стоит в теле, **выше** candidate-цикла и привязан к **внешнему** `oof_fold_index`
     (slice.py). Поэтому `tune_estimators` **сам** вызывает `_augment_oof_te` (вынести в общий
     application-хелпер) против **inner**-fold-index перед/внутри каждого inner-objective, иначе при
     `fe.target_encoding` тюнинг оптимизировался бы против TE-колонок, построенных на full-DEV (leak таргета
     — отдельный, сильнее HP-optimism §4). `_run_candidate` **в одиночку для TE-честности недостаточен**.
   - **НЕ вызывать FS-блок** `run_slice` (`compare_features`/`select_features`) — иначе тюнинг ушёл бы на
     FS-subset и молча сломал бы раскрытие §2a. Inner-объектив видит **полную** ширину признаков DEV.
   - **sample_weight (R2):** inner-fit И inner-score **взвешиваются** `sample_weight` (как `_run_candidate`:
     `est.fit(...,sw_train)`, `metric.score(...,sw_valid)`) — иначе взвешенный ран тюнился бы на невзвешенных
     скорах, расходясь со взвешенным leaderboard.

   Итог objective: fit на `fit⊕es` inner-фолда, score на inner-`test` (взвешенно), mean по inner-фолдам,
   ориентация higher-is-better по `policy.greater_is_better`; внешние selection-фолды и `outer_holdout` не
   видны (NFR-M7-3).
3. `tuner.tune(parse_search_space(spec.search_space), score, max_trials=…, timeout_s=…, …)` (§5 бюджет).
4. Построить **тюненую фабрику** `make_factory(name, outcome.best_params)`.

`make_factory: Callable[[str, Mapping], EstimatorFactory]` инжектируется composition (замыкает
`registry`+`task`+`seed`) — **двухарг, возвращает zero-arg фабрику** (R2-fix arity):
`make_factory = lambda name, params: (lambda: registry.build(name, task=task, random_state=seed, **params))`
→ результат типа `EstimatorFactory = Callable[[], Estimator]` (контракт `run_slice`). Требует проброса
`**params` в build-цепочку (ADR-0061 §4). Tuner — Humble Object: видит только `score(params)->float`.

### §2a Порядок HPO × feature-selection (правка по ревью R1 — был не специфицирован)
FS вычисляется **внутри** `run_slice`, итоговый `selected_features` известен лишь **после** него; стадия HPO
идёт **до** `run_slice`. ⇒ при заданном `fs` HPO тюнит на **полном** DEV-пространстве признаков, а leaderboard/
отгрузка используют FS-subset. Это **не ликедж и не нечестность** (тюненая модель честно оценивается
`run_slice` на subset'е — её leaderboard-скор корректен для пары «тюненые-гиперы × subset»), а **мягкая
субоптимальность**: гиперы выбраны под полный набор, применяются к subset'у. Бустинговые ключевые гиперы
(`depth`/`learning_rate`/регуляризация/`subsample`) переносятся между размерами набора; чувствительны лишь
`n_estimators`/`colsample`. **Решение M7:** тюнить на полном пространстве, **раскрыть** mismatch в report
(`hpo.tuned_on_full_feature_space=true` при `fs!=None`); **tune-on-FS-subset — Day-2** (§Day-2). **Fence
(R2):** inner-objective использует только `_run_candidate`+TE-шаг (§2), **не** FS-блок — проверка
`test_tune_estimators::test_inner_objective_sees_full_feature_width` (ширина = full-DEV, не post-FS subset)
запирает §2a от случайного FS-coupling. Плюс `test_facade::test_hpo_with_fs_tunes_on_full_space_documented`.

### §2b HPO под `run_mode='selection'` (правка по ревью R1; цитата уточнена R2)
HPO выполняется в **обоих** режимах (`selection` и `full`), чтобы selection-leaderboard отражал тех же тюненых
кандидатов, что и `full`. Основание ADR-0038 (точная формулировка): `run_slice` **режим-слепой**, selection
идёт на **том же DEV-сплите** → leaderboard напрямую сопоставим; за `ship_model=='full'` гейтятся **только**
refit/калибровка/holdout. Стадия HPO — до `run_slice`, поэтому идёт в обоих режимах. Следствие:
`run_mode='selection'` несёт полную стоимость тюнинга (`Σ n_trials × inner_cv`) — **документируется** (cost),
budget-gating §5 применяется одинаково. Проверки — `test_facade::test_selection_mode_runs_hpo`,
`::test_selection_mode_hpo_honors_budget_stop_axis`.

### §3 Write-back в кандидаты (replace, дефолт)
Тюненые фабрики **заменяют** одноимённые в `components.estimators` (id остаётся `name`; провенанс — в report).
При `keep_baseline=True` тюненая добавляется отдельным id `f"{name}__tuned"` рядом с базовой → честный band
рассудит, помог ли тюнинг (страховка от inner-overfit, ценой ×2 кандидатов на тип). `run_slice` далее
**не меняется** — оценивает тюненые фабрики на внешних фолдах честной машиной (leaderboard/band/significance).

### §4 Остаточный optimism (раскрытие, NFR-M7-3/7)
Гиперпараметры выбраны на inner-CV DEV, чьи строки пересекаются с внешними selection-`test`-фолдами →
внешний OOF тюненой модели **слегка** оптимистичен (величина — выбор в K-мерном HP-пространстве, малая;
band-гард M4 её частично поглощает). Это **раскрывается** в report `hpo`-блоке (`selection_oof_is_post_tuning:
true`) и в §Последствия. Несмещённое число — `outer_holdout` (ADR-0029), рекомендуется в паре с HPO.
**Полностью-честный nested-per-fold HPO — Day-2** (§Day-2), не молча отсутствует.

### §5 Бюджет (DM-74; SPIKE Q3)
HPO кооперативен под run-`Budget` (ADR-0032). Приложение вычисляет `max_trials`/`timeout_s` из `HPOConfig` и
остатка `Budget`: trials-mode → `max_trials=hpo.n_trials`; time-mode → дополнительно `timeout_s=min(hpo.timeout_s
?? ∞, budget.time_left()/n_models_remaining)`. Graceful degradation: Tuner возвращает best-so-far (≥1 trial);
**0 завершённых trial → fallback на базовую фабрику** (без падения). Стоимость детерминированно оценима
**до** запуска: `Σ_models n_trials × inner_cv` фитов (SPIKE Q3) — логируется (NFR-M7-7). Refit **не**
бюджетируется (ADR-0032 §1). time-mode помечает недетерминизм в report (NFR-M7-2).

**Error-таксономия (R2; переиспользовать существующие типы, новых классов нет):** 0 trials **и** базовая
фабрика не материализуется → существующая `MissingDependencyError`/`FitFailedError` пробрасывается (это не
graceful-кейс — компонент в принципе недоступен). `hpo` задан, но resolved-набор тюнящихся типов **пуст**
(все `search_space={}` — только baselines) → **не молчать**: HPO-стадия — no-op c report-нотой `hpo:
{note:"no tunable models"}`, а не отсутствующий блок. Отдельной `ConfigError` это не требует (валидный конфиг).

**Seed-resolution ordering (R2):** `HPOConfig.random_state=None`/`EnsembleConfig.random_state=None` резолвятся
в `seed` **до** сборки `RunConfig` (зеркало `_resolve_fs`, facade.py) — тогда дамп/fingerprint несут
**эффективный** seed (два рана с разным унаследованным seed → разные хэши). Проверка —
`test_fingerprint::test_resolved_hpo_seed_is_stable_and_seed_sensitive`.

### §6 `HPOConfig.n_trials` vs `BudgetConfig.n_trials` (разведение, R-BUDGETMEAN)
Разные оси: **`HPOConfig.n_trials`** — число конфигов в per-model HP-поиске (вход `tuner.tune`);
**`BudgetConfig.n_trials`** (`RunConfig.budget`, M5) — лимит candidate-loop `run_slice` (число кандидатов/
итераций). Сосуществуют: HP-поиск идёт под общий wall-clock run-`Budget`, но его глубину задаёт `HPOConfig`.
Не дублируют семантику.

### §7 Fingerprint / провенанс (уточнено по ревью R1)
`HPOConfig` в `RunConfig.model_dump(mode='json')` → в `compute_run_fingerprint` (изменён HPO-конфиг → другой
cache-ключ). **Уточнение fingerprint:** `model_dump` эмитит и `None`-поля, поэтому добавление `hpo`/`ensemble`
вписывает `"hpo": null, "ensemble": null` в канонический JSON и **сдвигает хэш для ВСЕХ ранов, включая
off** — это **тот же** разовый cross-milestone сдвиг scope, что уже сделали `fe`/`fs` при M5→M6 (и принят).
Поэтому критерий — НЕ «байт-в-байт как M6», а: **off → стабильный хэш между M7-ранами + семантически
эквивалентный M6-ран; `FINGERPRINT_VERSION` остаётся 1** (ручная инвалидация не нужна, форма аддитивна).
run-report — аддитивный `hpo`-блок (per-type `chosen_params`/`inner_best_score`/`n_trials_run`/`backend`/
`inner_cv` + `selection_oof_is_post_tuning`; при `fs!=None` — `tuned_on_full_feature_space=true` §2a; при
time-mode — `deterministic=false` §5). `RUN_MANIFEST_VERSION` не меняется (NFR-M7-4).

## Последствия
- **+** Тюнинг честен: тюненая модель проходит ту же band/significance, что и базовые; `run_slice` не тронут
  (estimator-blind) — минимальная хирургия.
- **+** Стоимость предсказуема и budget-ограничена; graceful degradation всегда отгружает модель.
- **−/R-HPOLEAK (остаточный):** flat inner-CV даёт мягкий optimism внешнего OOF — **раскрыт**, снимается
  `outer_holdout`; полная честность (nested) отложена осознанно.
- **−/R-HPOCOST:** `× inner_cv × n_trials × n_models` фитов — ограничено бюджетом, flat-дефолт (не nested),
  оценка логируется.

## Day-2 (committed → M7-future)
- **nested-per-fold HPO** (`hpo_arbitration="nested_per_fold"`): ре-тюнинг внутри каждого внешнего фолда,
  нулевой optimism, ценой `× K_outer`. Зеркалит лестницу арбитража FS (ADR-0052/0057). effort L.
- **`hpo_arbitration="auto"`**: выбор flat/nested по форме данных+бюджету (паттерн ADR-0057).
- **tune-on-FS-subset** (§2a): прогнать FS-селекцию до стадии HPO, тюнить на FS-subset'е → устранить мягкую
  субоптимальность «тюнинг на полном, оценка на subset». Требует вынести FS-селекцию из `run_slice` в
  пред-стадию (хирургия M6-интеграции) — отложено.
