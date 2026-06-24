# ADR-0013 — Адаптеры моделей и метрик для slice

- **Статус:** Proposed
- **Драйверы:** D1 (расширяемость), D2 (обобщение); FR-2, FR-3; NFR-2; R4.
- **Воркстрим:** M2.

## Контекст

Slice нужно гонять в CI на **лёгком ядре** (unit-job ставит `[dev]`, без бустингов).
Значит M2 даёт минимум адаптеров `Estimator`/`Metric` на core-зависимостях (numpy/
pandas/polars/scikit-learn/pydantic). Зоопарк бустингов с нативной cat-обработкой и
multiclass/regression — M3 (R4: не расширять раньше каркаса).

## Рассмотренные варианты

- Модели: (а) сразу бустинги (catboost/lightgbm/xgboost) — тяжёлая зависимость в
  slice/CI, преждевременно; (б) **sklearn-baseline + линейная** — лёгкие, покрывают
  FR-2 «≥1 baseline и ≥1 линейная», валидируют каркас. → (б).
- Категории для sklearn-моделей: нативной cat-обработки нет → коды как числовые
  признаки (приемлемо для slice; правильная обработка — one-hot/нативная в M3/M6).

## Решение

**Estimator-адаптеры (M2, лёгкое ядро), реализуют порт `Estimator`/`Probabilistic
Estimator` на numpy-границе:**
- `DummyEstimator` — baseline (предсказывает априорную частоту класса; sklearn
  `DummyClassifier(strategy="prior")`). Закрывает «≥1 baseline» (FR-2).
- `LinearEstimator` — sklearn `LogisticRegression` (binary). Закрывает «≥1 линейная»
  (FR-2). `+SupportsFeatureImportance` через `coef_`.
- Оба: numpy-вход = `to_numpy()` (числовой блок) ⊕ `categorical_codes()` (как
  числовые столбцы для M2); `feature_names` из схемы. `fit(X, y, X_val, y_val,
  sample_weight)` — `X_val/y_val` игнорируются (нет early stopping у этих моделей;
  `es_idx` фолда задействуют бустинги в M3). `predict_proba` → P(класса).
- `capabilities` (`ModelSpec`): `tasks=("binary",)` для M2 (multiclass/regression —
  M3); `handles_cat=False`, `handles_missing=False`.

**Metric-адаптеры (реализуют порт `Metric`), обёртки `sklearn.metrics`:**
- `RocAuc` (`needs="proba"`, greater_is_better, optimum=1.0),
- `PrAuc` / average_precision (`needs="proba"`),
- `Accuracy` (`needs="class"`), `LogLoss` (`needs="proba"`, greater_is_better=False).
- Дефолт для `Task(kind="binary")` — `RocAuc` (совпадает с `Task.target_metric`).
- `sample_weight` пробрасывается в sklearn (G2).

**Размещение:** адаптеры в `honestml/adapters/` (estimators.py, metrics.py, splitters.py);
тяжёлые зависимости не импортируются ядром (NFR-2, import-linter). sklearn — core-
зависимость, поэтому эти адаптеры доступны в лёгком ядре.

**CVSplitter-адаптеры:** `HoldoutSplitter` (1 фолд: train→fit+es-хвост, test) и
`KFoldSplitter`/стратифицированный для binary (k фолдов, в каждом es-хвост из train) —
выдают `Fold(fit_idx, es_idx, test_idx)`, проходят `validate_fold`. Авто-выбор по
`Task` (binary → stratified KFold) с ручным переопределением (FR-4 минимум; purge/
embargo/TimeSeries — M4).

## Последствия

- (+) Slice гоняется на лёгком ядре и в CI без бустингов; FR-2/FR-3/FR-4 покрыты
  минимально и расширяемо.
- (+) Реализации портов доказывают, что контракты M1 рабочие (де-риск, R4).
- (−) Качество моделирования slice невысоко (коды-как-числа для линейной модели) —
  это **валидация плумбинга, не качества**; нативная cat-обработка и зоопарк — M3.
  Задокументировано как осознанное ограничение M2.
- (−) `es_idx` у M2-моделей не используется как ES — но **не теряется**: не-ES
  модели обучаются на `fit_idx ∪ es_idx` (ADR-0010 §Уточнения п.6); контракт `Fold`
  несёт `es_idx` для бустингов M3 (без смены сигнатур).

## Уточнения контрактов (ревью, фаза 8)

- **Сборка X и feature_names (F9):** адаптер собирает
  `X = np.hstack([dataset.to_numpy(), dataset.categorical_codes()])` в порядке
  `schema.features` (numeric-блок, затем categorical); `feature_names =
  schema.features` (под `coef_`/`feature_importances`). Guard: ≥1 признак, иначе
  `SchemaValidationError` на границе (а не сбой в недрах sklearn).
- **Seed (F7):** `LinearEstimator` → `LogisticRegression(random_state=seed)`;
  `StratifiedKFoldSplitter` → `StratifiedKFold(shuffle=True, random_state=seed)`;
  `DummyEstimator` детерминирован. Имя сплиттера — **`StratifiedKFoldSplitter`**
  (единое во всех документах).
- **metric→policy (F10):** см. ADR-0009 — направление политики берётся из
  `metric.greater_is_better` (LogLoss=False ⇒ argmin).
- **Группы (F5):** при `ColumnRole.GROUP` в схеме M2-сплиттеры/use-case делают
  fail-fast (`ConfigError`); group-aware split — M4.
- **Критерий M2-2 (F-minor):** «смена метрики меняет победителя» проверяется
  фиксированным синтет-кейсом, где RocAuc и Accuracy дают разных победителей среди
  {Dummy, Linear} (не флапающий тест).
