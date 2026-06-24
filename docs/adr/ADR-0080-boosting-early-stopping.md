# ADR-0080 — Early stopping бустингов по умолчанию: карв es-хвоста и нативная остановка

- **Статус:** Accepted (2026-06-13, v0.2)
- **Дата:** 2026-06-13
- **Драйверы:** backlog finding #2 (бустинги учатся без ES; адаптерный WARNING горит на 100% реальных прогонов — дефолт предупреждает сам о себе); честность сравнения лидерборда.
  Зависит от ADR-0020 §2 (ES → M4, отложено), ADR-0027 (TimeSeriesSplitter уже карвит es), ADR-0010 §6 (`fit ∪ es`), ADR-0061/0062 (HPO).

## Контекст

Контур ES готов на 80%: `CVConfig.n_es`, `Fold.es_idx`, сигнатура `Estimator.fit(X_val, y_val)`, но не замкнут. Ключевое: es-хвост карвит **только** `TimeSeriesSplitter`; i.i.d.-сплиттеры ставят `es_idx=_EMPTY` ([splitters.py](../../src/honestml/adapters/splitters.py)); [slice.py](../../src/honestml/application/slice.py) сливает `fit∪es` и игнорирует `X_val`. Все четыре showcase-датасета — i.i.d., поэтому self-warning горит всегда, а бустинги используют фикс-`n_estimators=300` без честной остановки.

## Решение

### 1. Capability + per-fold карв (только при наличии бустинга)
`Capabilities.supports_early_stopping` (бустинги=True). `build_default_components` карвит es-хвост **только когда** в зоопарке есть ES-модель (`_any_early_stopping`), иначе `es_fraction=0` → **байт-идентично** прежнему поведению. i.i.d.-сплиттеры (Holdout/KFold/Stratified) карвят `es_fraction=0.1` ИЗ train-строк фолда (`_carve_iid_es`, стратифицированно для классификации, сид-детерминированно), поэтому **`fit∪es` = прежний train**: не-ES модели сливают его обратно (прозрачно), только ES-модели держат как валидацию. **Group-схема** карвит es **целыми группами** из train (`_carve_group_es` через `GroupShuffleSplit`, см. поправку ниже).

### 2. Маршрутизация в run_slice
[slice.py](../../src/honestml/application/slice.py) `_run_candidate`: если `est.supports_early_stopping` и `es_idx` непуст → `fit(fit, X_val=es, y_val=es, sample_weight=sample_weight[fit_idx])` (учимся на fit, валидируемся на es); иначе — слияние `fit∪es` как раньше. `test_idx` не трогается → OOF-честность цела. Валидационные веса не прокидываются (порт без `val_sample_weight`-слота; ES — эвристика остановки, а `sample_weight` на дефолтном пути `None`).

### 3. Нативная остановка по библиотекам
`_BoostingBase._es_fit` ([boosting.py](../../src/honestml/adapters/boosting.py)) поднимает число деревьев до потолка `_N_ESTIMATORS_ES=1000` (ES режет пер-фолд) и вызывает нативный API: lightgbm `callbacks=[early_stopping(50), log_evaluation(0)]`+`eval_set`; xgboost ctor `early_stopping_rounds=50`+`eval_set` (в 2.x параметр убран из fit); catboost ctor `early_stopping_rounds`+`eval_set`. `predict` каждой библиотеки автоматически использует best-итерацию. Без хвоста — фолбэк на фикс-300 + старый WARNING.

### 4. Честный манифест
`Components.early_stopping = es_enabled` пробрасывается фасадом в `FittedModel.early_stopping` → манифест (раньше хардкод `False`). Теперь флаг отражает реальность (ES активен на всех схемах при наличии бустинга).

### Границы scope
ES на **пути отбора кандидатов** (внешние OOF-фолды). Inner-CV HPO es НЕ карвит (отдельный сплиттер, `es_fraction=0`) — tuned `n_estimators` становится **потолком**, ES режет (документировано здесь, ADR-0062).

## Поправка реализации (2026-06-13) — group-disjoint es

Изначально group-схема оставляла es пустым (ES неактивен под `scheme="group"`). Закрыто: `_carve_group_es` ([splitters.py](../../src/honestml/adapters/splitters.py)) карвит es **целыми группами** из train-строк фолда через `GroupShuffleSplit` — ни одна группа не пересекает fit/es, поэтому `validate_fold(groups=...)` (попарная group-дизъюнктность fit/es/test) проходит, а ES-валидация под group-схемой так же честна, как внешний carve (finding #11). `GroupKFoldSplitter` получил `random_state` (сидит только es-карв; сам GroupKFold детерминирован). es пуст, если у фолда <2 групп (фолбэк без ES). `early_stopping` теперь = `es_enabled` для всех схем.

## Поправка (2026-06-13) — ES, а не scale_pos_weight, лечит коллапс на дисбалансе

Ре-прогон 04-credit-card-fraud (0.17% позитивов) + изолирующая диагностика показали: **ES — настоящее
лекарство от коллапса нетюненых бустингов на экстремальном дисбалансе** (lightgbm pr_auc 0.012→0.726,
catboost 0.833, xgboost 0.786 — ES один, без балансировки). Предложенный ранее `scale_pos_weight`
(ADR-0079) под ES катастрофически ломал catboost/lightgbm и был **откатан** (см. ADR-0079). Таким
образом backlog #1 (imbalance-коллапс) закрыт именно этим ADR (ES), а не ADR-0079.

## Последствия

- **Положительные:** бустинги честно останавливаются → сравнение лидерборда не благоволит переобучению; self-warning исчезает на не-HPO прогонах; контур `n_es`/`es_idx` наконец задействован; не-бустинговые прогоны байт-идентичны; **закрывает imbalance-коллапс (backlog #1) — см. поправку выше.**
- **Отрицательные/компромиссы:** бустинг учится на ~90% фолда (10% на es) — честная плата за ES (vs переобучение на 100%); group-схема и inner-HPO без ES (follow-up); детерминизм сохраняется (сид-карв + фикс-rounds).
- **Слои:** `core` (capability), `adapters` (splitters/boosting), `application` (slice), `composition` (build/facade/artifact); import-linter не затронут.

## Проверки

i.i.d.-сплиттер с `es_fraction>0` даёт непустой `es_idx`, `fit∪es`=прежний train (не-ES прозрачно); `es_fraction=0` (по умолчанию / без бустинга) — `es_idx` пуст, всё как раньше; `_any_early_stopping` True только при бустинге; run_slice маршрутизирует es только ES-моделям (не-ES учится на union); реальный lightgbm/xgboost/catboost с es-хвостом останавливается (`best_iteration_ < ceiling`) и предсказывает корректно; `early_stopping` в манифесте True для любого бустинг-прогона (i.i.d./group/timeseries карвят es-хвост), False для не-бустинговых прогонов.
