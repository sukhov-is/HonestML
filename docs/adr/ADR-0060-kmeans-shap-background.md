# ADR-0060 — kmeans-фон для interventional Tree SHAP (`shap_background`)

- **Статус:** Accepted (M6f, design-gate pending)
- **Драйвер:** DM-F3 (FR-FSF-4; NFR-FSF-1/2)
- **Связано:** ADR-0056 (interventional SHAP + linspace-фон), ADR-0051 (ShapRanker lazy-extra),
  **SPIKE-M6f-shap-bg** (питает решение).

## Контекст
M6e interventional-SHAP использует **детерминированный linspace-фон** (равномерно по порядку строк, ADR-0056
§2). На мультимодальных данных малый linspace-фон недопокрывает density-моды → смещённая атрибуция. kmeans-
центроиды покрывают моды лучше при том же числе точек. **SPIKE-M6f-shap-bg** подтвердил: seeded kmeans
(`sklearn.cluster.KMeans`) **детерминирован**, ближе к full-background рангу на **малом** фоне (k=20:
Spearman 0.972 vs linspace 0.946; k=50: 0.966 vs 0.938), **цена ≈ linspace** при равном k; на большом фоне
(k=100) преимущество исчезает (оба ≈0.98).

## Рассмотренные варианты
1. **Только linspace (как M6e).** — ❌ недопокрывает моды на малом фоне; FR-FSF-4 не закрыт.
2. **`shap.kmeans` (weighted DenseData).** — ⚠️ детерминизм зависит от внутренностей shap; веса усложняют
   провенанс. Отклонено в пользу явного sklearn KMeans.
3. **sklearn `KMeans.cluster_centers_` как фон, seeded, opt-in.** — ✅ детерминизм под нашим контролем
   (`random_state`), sklearn — hard-dep (доступен и под `uv run`), не полагаемся на shap-внутренности.

## Решение (Вариант 3)
Новое поле `shap_background: Literal["linspace", "kmeans"] = "linspace"` (**дефолт = M6e-поведение**).
- При `"kmeans"` + `shap_perturbation="interventional"`: фон interventional-explainer =
  `KMeans(n_clusters=k, n_init=10, random_state=seed).fit(x).cluster_centers_`, где `k =
  shap_background_samples` (если `None` или `k≥n` → full-`x`, как linspace), `seed` = **rank-time**
  `random_state` (см. §2).
- Логика — **в адаптере** `feature_rankers.py` (новая `_kmeans_background(x, k, seed)` рядом с `_background`),
  внутри interventional-ветки `ShapRanker.rank` (там shap уже обязателен). `core`/`application` не
  затрагиваются (NFR-FSF-2; import-linter 3/3).
- **Провенанс** (NFR-FSF-1/4): метод фона (`linspace`/`kmeans`) — в run-report/манифест (config-дамп
  `shap_background`). См. §2 про честную трактовку seed-провенанса.

### §1 Дефолт и opt-in
Дефолт `linspace` (нулевой прирост поверхности, на большом фоне не хуже). kmeans — **opt-in для малых
фонов на мультимодальных данных** (где SPIKE показал выигрыш). Не претендуем на универсальное превосходство
(на k=100 разницы нет) — честная область применимости задокументирована.

### §2 Детерминизм и seed-контракт (зафиксировано после R1)
`KMeans(random_state=seed, n_init=10)` детерминирован (SPIKE: `deterministic=True` на всех k). **Решение:
использовать rank-time seed** — `ShapRanker.rank(..., random_state)` (feature_rankers.py:279) уже получает
детерминированный seed (single/holdout: `config.random_state`; per-fold: `_strategy_fold_seed(name,
random_state, fold_id)`). `__init__` **НЕ расширяется** (минимальный diff; один источник истины seed —
тот же, что фитит ranker-model на этом фолде; нет рассинхрона ranker↔kmeans).
**Честная трактовка провенанса:** в per-fold пути фактический KMeans-seed варьируется по фолдам
(`_strategy_fold_seed`), поэтому он **не равен** буквально `manifest.random_state`. Воспроизводимость
держится на цепочке `(run-seed → детерминированный _strategy_fold_seed)`, а не на точечном равенстве
`manifest.random_state == KMeans.random_state`. **Критерий NFR-FSF-1 (уточнён после R2):** воспроизводимость
= «тот же **код** + seed + данные → идентичные ранги» (тест гонится В репо, где `_strategy_fold_seed`
доступна), а **не** «манифест самодостаточен для реконструкции фона извне». Тест не обещает лишнего.

### §3 Dead-config
`shap_background="kmeans"` при `shap_perturbation="tree_path_dependent"` (фон не используется) → **WARNING
dead-config** (паттерн M6e `shap_background_samples@tpd`), не ошибка.

## Последствия
- **+** Репрезентативнее на малом фоне (cost-чувствительный режим) при той же цене и детерминизме.
- **+** kmeans изолирован в адаптере за lazy-shap; sklearn уже разрешён в adapters; `import honestml` не тянет
  shap (NFR-FSF-2; `test_top_level_import_is_lightweight`).
- **−:** выигрыш **скромный и только на малом фоне** (k≥100 — нет разницы); поэтому **opt-in, не дефолт**.
- **−/риск R-KMEANSDET:** недетерминизм — снят фиксированным seed + провенансом (SPIKE-подтверждён).
- **−/риск R-KMEANSENV:** новая зависимость/утяжеление импорта — снято (sklearn hard-dep, kmeans только в
  interventional-ветке адаптера, core/application чисты).
- **Не-объём:** stratified-by-target фон, weighted `shap.kmeans`-веса → future.
