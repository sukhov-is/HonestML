# ADR-0058 — Жёсткий cost-budget арбитража (`cost_budget_refits`)

- **Статус:** Accepted (M6f, design-gate pending)
- **Драйвер:** DM-F4 (FR-FSF-5; NFR-FSF-4/6), делит механизм с DM-F1
- **Связано:** ADR-0057 (лестница арбитража + cost-оценщик), ADR-0052/0054 (стоимость nested/per-fold),
  M6e `_warn_fs_cost` (источник формул).

## Контекст
Честные режимы дороги: `nested_per_fold` = SELECTION × K_outer; `null_importance` = ×(1+n_runs) рефитов;
`sequential` = O(n_features²). Сейчас стоимость только **логируется WARNING** (`_warn_fs_cost`,
`compare_features`), гейта нет — прод может неожиданно уйти в часы счёта. Нужен **жёсткий, детерминированный,
портируемый** потолок. Мера в **секундах** зависит от железа; **число рефитов ранкера/кандидата**
детерминированно и переносимо — это та же величина, что уже в WARNING-формулах.

## Рассмотренные варианты
1. **Бюджет в секундах/памяти.** — ❌ непортируемо, недетерминированно; пересекается с run-level
   `BudgetConfig` (candidate-loop, не selection). Отложено в future.
2. **Только WARNING (как сейчас).** — ❌ не «жёсткий»; не закрывает FR-FSF-5.
3. **Hard-ceiling в рефитах с downgrade арбитража, fail-loud на полу.** — ✅ детерминированно, портируемо,
   консистентно с существующими WARNING-формулами, переиспользует лестницу ADR-0057.

## Решение (Вариант 3)
Новое поле `cost_budget_refits: int | None = None` (gt=0; **дефолт None = без гейта, текущее поведение**).

### §1 Детерминированный оценщик стоимости (канонический)
Чистая функция `application/feature_selection.py::estimate_fs_refits(fs, *, n_strategies, n_features,
inner_n_splits) -> int` — **верхняя граница** числа фитов ранкер-модели за selection. Сверена с **реальной**
runtime-формулой `compare_features` (feature_compare.py:617-622), а не с качественным `_warn_fs_cost`:
- `inner_n_splits` = **`cv.n_splits`** главного selection-сплиттера. **В `FeatureSelectionConfig` поля
  `inner_n_splits` НЕТ** (это была ошибка ранней редакции ADR) — значение берётся из `CVConfig.n_splits` и
  передаётся параметром. `n_strategies` = `len(compare)` или 1. `n_features` нужен для sequential.
- база на стратегию (**max** по сравниваемым — верхняя граница): `null_importance` → `(1 + n_runs)`;
  `sequential` → `n_features²` (O(n_features²) score_subset-фитов, верхняя оценка); `importance`/
  `random_probe`/`shap` → `1`.
- множитель арбитража: `holdout`/`nested` → `inner_n_splits`; `nested_per_fold` →
  `arbitration_n_splits × inner_n_splits`.
- **итог = `n_strategies × base × mult`** = суммарная **SELECTION**-стоимость (что и нужно гейту). Сверка
  по as-is (уточнено после R2):
  - **per_fold** = `n_strat × arbitration_n_splits × inner_n_splits × (1+n_runs)` — **численно тождественна**
    compare:617-622 ⇒ эта WARNING **переписывается** на `estimate_fs_refits`. ✓
  - **holdout+null_importance** = `inner_n_splits × (1+n_runs)` — **согласуется по структуре** (per-strategy)
    с `_warn_fs_cost` «n_folds×(1+n_runs)» (build.py:230); `_warn_fs_cost` НЕ несёт `n_strategies`, поэтому
    абсолютные числа сопоставимы лишь при `n_strategies=1` (не знак «≡», а структурная сверка).
  - **nested** (661-665) печатает `n_strat × arbitration_n_splits` — это **другая величина** (арбитраж
    рефитом ФИКСИРОВАННОГО subset через K фолдов, без inner-selection и без `(1+n_runs)`), **НЕ** suммарная
    selection-стоимость. Поэтому nested-WARNING **НЕ переписывается** на канон (иначе показала бы неверное
    число). Гейт использует канон (selection-cost) — верхнюю границу, мажорирующую обе.
  - **sequential** база `n_features²` — **верхняя граница** (score_subset-evaluations, не ranker-fits;
    единицы намеренно консервативны), runtime-эталона отдельным множителем нет (R-COSTACC).

**DRY (уточнённый, без ложного claim):** единый **числовой** источник — `estimate_fs_refits`. На канон
переписывается **только** точная **per_fold**-WARNING (617-622). holdout — структурная сверка; nested-WARNING
(661-665) и sequential остаются как есть (другая величина / верхняя граница). Качественный `_warn_fs_cost`
(build.py:225-243) — ранний build-хинт (печатает текст «×K»/«O(n²)», не произведение), к нему claim DRY не
относится. Тест `test_estimate_fs_refits_matches_compare_formula` сверяет **per_fold**-ветку (importance/
null_importance), НЕ nested и НЕ sequential.

### §2 Гейт (когда `cost_budget_refits` задан)
Гейт работает на **резолве** (`resolve_fs_defaults`, composition, в `facade.fit` post-read) **до** запуска
`compare_features` — по `estimate_fs_refits`, а не по runtime-состояниям. Поэтому он понижает **запрошенный**
арбитраж ещё до того, как может возникнуть C5-деградация (`per_fold_partial_c5_inner`/`holdout_degraded_*`)
— пересечения cost-downgrade и C5-веток нет.
1. Оценить стоимость текущего/`auto`-арбитража. Если ≤ бюджет → без изменений.
2. Иначе **понижать** арбитраж по лестнице `nested_per_fold → nested → holdout`, пока оценка не уложится.
   Понижение пишется в `effective_fs.arbitration` (write-back) + фиксируется в **resolve-record** (§4) +
   **громкий WARNING** `было→стало` (NFR-FSF-6). **`arbitration_effective` НЕ трогается** (это runtime-поле
   `compare_features`; cost-провенанс — отдельная resolve-наблюдаемость, см. §4).
3. Если даже `holdout`-пол превышает бюджет (дорогой рэнкер: большой `n_runs`/`sequential`) → **fail-loud
   `ConfigError`** на резолве с actionable-сообщением: «оценка X рефитов > бюджет Y даже на holdout;
   поднимите `cost_budget_refits` или удешевите рэнкер (n_runs/strategy)».

### §3 Precedence (явный арбитраж × явный бюджет)
Конфликт «явный `arbitration=nested_per_fold` + явный бюджет, которого не хватает» → **бюджет выигрывает**
(downgrade), потому что `cost_budget_refits` — явная директива «цена важнее честности». Но **только громко**
(WARNING explicit→effective + resolve-record §4), не молча (NFR-FSF-6). Для `arbitration="auto"` (ADR-0057)
бюджет — просто вход резолва, конфликта нет.

### §4 Наблюдаемость через resolve-record (НЕ через `arbitration_effective`) — уточнено после R2
`arbitration_effective` рождается **внутри** `compare_features` из `config.arbitration` (feature_compare.py:
594/644/686) и описывает **runtime**-деградацию (C5). Причина cost-downgrade известна только резолверу
(composition) — её **нельзя** молча восстановить из `arbitration_effective` (после downgrade
`nested_per_fold→nested` compare честно вернёт `"nested"` без следа причины). Поэтому **значения
`*_cost_downgraded` в `arbitration_effective` НЕ вводятся** (M6e-перечень `arbitration_effective` не
расширяется). Вместо этого `resolve_fs_defaults` возвращает **resolve-record**
`dict[str,str]` с (только нетривиальными) ключами:
`arbitration_requested` (исходный сентинел/значение), `arbitration_resolved_from ∈ {explicit, auto,
cost_budget}`, аналогично `block_mode_requested`/`block_mode_resolved_from`. `facade.fit` прокидывает запись
в **run-report аддитивной секцией** `fs_resolution` (NFR-FSF-4). Cost-downgrade наблюдаем как
`arbitration_requested != effective.arbitration` ∧ `arbitration_resolved_from="cost_budget"`. Это тот же
паттерн «явный канал возврата», что ADR-0059 §1a — но для resolve-провенанса, а не runtime-stats.

### §5 `cost_budget_refits` vs run-level `BudgetConfig` (разведение, после R2)
Это **разные оси**, обе могут быть заданы независимо: `cost_budget_refits` (`FeatureSelectionConfig`)
ограничивает **selection-рефиты ДО `compare_features`** (детерминированный refit-count); run-level
`BudgetConfig` (`RunConfig.budget`: time/trials/memory) ограничивает **candidate-loop** в `run_slice`
(wall-clock/итерации, не selection; refit не бюджетируется). Не пересекаются, не дублируют семантику.

## Последствия
- **+** Предсказуемый, детерминированный, портируемый потолок; честность снижается **контролируемо и
  наблюдаемо**, а не падает по таймауту.
- **+** Канон `estimate_fs_refits` — единый числовой источник для гейта и точной **per_fold**-WARNING (§1);
  лестница арбитража делится с ADR-0057.
- **−/риск R-BUDGETHARD:** `ConfigError` на полу может удивить — смягчено: только при **явном** opt-in
  бюджете, сообщение actionable, дефолт None.
- **−/риск R-COSTACC:** оценка — верхняя граница (кэш/ранний выход sequential дают реальную ниже); ложный
  downgrade возможен, но «безопасен» (честность ≤ запрошенной); мера в рефитах, не секундах; пороги
  M9-tunable. Документировано.
