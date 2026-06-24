# ADR-0021 — Multiclass: выравнивание OOF-proba к глобальным классам + averaging метрик

- **Статус:** Accepted (реализован 2026-06-08, M3b)
- **Дата:** 2026-06-07
- **Драйверы:** DM3-3 (обобщение за пределы binary), DM3-4 (корректность выравнивания классов); FR-5.
  north-star: D2/D3; FR-17/FR-3. Зависит от ADR-0010 (slice/OOF), ADR-0013 (метрики).
- **Воркстрим:** M3, дельта к ADR-0010/0013.

## Контекст

`run_slice` прибит к binary: `positive = ... if kind=="binary" else None` (`slice.py:142`); OOF-proba
`oof_proba = np.full(n, np.nan)` — **1-D** (`slice.py:209`); proba-ветка `proba[:, pos_idx]` берёт один
столбец (`slice.py:230-233`). Метрики (`metrics.py`) бинарные, без `average`. На >2 классов: `predict_proba`
даёт `(n, k)`, sklearn-метрики падают/неверны без `multi_class`/`average`; **порядок столбцов `est.classes_`
может отличаться от глобального**, а фолд мог не видеть часть классов в train → `k_fold < K` (RM3-3).

## Рассмотренные варианты

1. **Хранить proba как список разнородных массивов по фолдам** — гибко, но усложняет агрегацию и метрику;
   теряется единый OOF-массив, нужный для значимости (M4).
2. **Единый OOF `(n, K)` по глобальному порядку классов; per-fold reindex; binary остаётся 1-D** —
   корректно и совместимо с binary-путём; метрика получает выровненный вход. **Выбран.**

## Решение

### 1. Глобальный порядок классов
`classes = np.unique(y)` (сортирован) — единый порядок `K = classes.size`. Для multiclass `positive` не нужен.

### 2. Per-fold reindex proba к глобальным классам (со сглаживанием, не нулём — фикс R1-B2)
После `proba_fold = est.predict_proba(x_test)` (порядок столбцов = `est.classes_`, возможно `< K`):
строится `aligned (n_test, K)`, где столбец глобального класса `c` = соответствующий столбец `proba_fold`,
если `c ∈ est.classes_`, иначе **малая ε-масса** (`eps = 1e-6`, не литеральный 0), после чего **каждая
строка ренормируется к сумме 1**. Это исключает катастрофу `-log(0)` в `log_loss` и вырожденный OvR-столбец
в `roc_auc` (ревью R1-1a/1b): отсутствующий класс получает честно-малую, но не нулевую вероятность, а строка
остаётся валидным распределением. Реализуется чистой функцией
`align_proba(proba_fold, est_classes, global_classes) -> np.ndarray` (`application`, numpy-only, синхронно
тестируема — NFR-7). Накопление в `oof_proba (n, K)`.
**Контекст применимости:** при стратифицированном дефолте классификации (`StratifiedKFold`/
`StratifiedGroupKFold`, ADR-0023) каждый train-фолд содержит все классы → ветка ε-сглаживания почти не
срабатывает; она — корректный фолбэк для пограничных непокрытий, не штатный путь.
**Честная оговорка (фикс R2):** построчная ренормировка корректна для `log_loss` (ε ничтожно влияет), но
для `roc_auc(multi_class="ovr")` она построчная с разным делителем → может **слегка исказить OvR-ранжирование**
на ε-строках. Это **принятый компромисс редкого пути** (срабатывает только при непокрытии класса в train-фолде,
т.е. практически только вне стратификации); не «нулевое смещение». Альтернатива без ренорма дала бы
`-log(0)` в log_loss — хуже.

### 3. Хранилище OOF и проекция
- **binary** (`K == 2`): сохраняется **текущий прямой путь** — `proba[:, pos_idx]` (столбец позитива из
  `predict_proba`), **без** `align_proba` → метрики binary бит-в-бит как до дельты (фикс R2-B4). `align_proba`
  — только multiclass-путь.
- **multiclass** (`K > 2`): OOF-proba `(n, K)` через `align_proba`.
- **regression**: proba нет; OOF = `pred` (value).
- `project_for_metric(metric, *, proba, pred, kind)`:
  - `needs ∈ {proba, threshold}` + binary → 1-D `P(positive)`;
  - `needs ∈ {proba, threshold}` + multiclass → `(n, K)` (выровненный);
  - `needs ∈ {class}` → `pred` (метки);
  - `needs == value` → `pred`.

### 4. `Metric` += `average`; единый источник `labels`; multiclass-ветви в адаптерах
- `Metric`-порт: добавляется поле `average: str | None = None` (аддитивно; `None` = бинарная/по-умолчанию).
  Допустимые: `"macro"|"micro"|"weighted"`.
- **Единый источник `labels` (фикс R1-1c/F6).** Глобальный порядок классов `classes = np.unique(y)`
  определяется **composition** (известен `Task`/данные) и передаётся метрике при конструировании:
  `resolve_metric(name, *, classes=None, average=None)`. Метрика хранит `labels=classes` и использует его
  во всех multiclass-вызовах sklearn. **Альтернатива «вывод `labels` из `y_true`» удаляется** — она давала
  два источника истины и `ValueError` при отсутствии класса в OOF. (Сигнатура `Metric.score(y_true, y_pred,
  sample_weight)` **не меняется** — `labels`/`average` метрика берёт из своих полей.)
- Адаптеры метрик ветвятся по размерности входа:
  - `roc_auc`: binary (1-D) → `roc_auc_score(y, p1d)`; multiclass (2-D) → `roc_auc_score(y, P,
    multi_class="ovr", average=average or "macro", labels=self.labels)`.
  - `log_loss`: multiclass → `log_loss(y, P, labels=self.labels)`; binary — **текущее поведение на 1-D
    сохраняется без `labels`** (фикс R1-1d: для binary `labels`/`average` не передаются → бит-в-бит как до дельты).
  - `accuracy`: `accuracy_score(y, pred)` — kind-агностична.
  - Регрессионные метрики — `needs="value"`, ветвь `pred` (см. §5).
- **Совместимость task↔metric (фикс R1-4b, R2-C2).** `resolve_metric`/composition валидирует: proba/class-метрика
  не сочетается с regression-таском, value-метрика — с классификацией → `ConfigError` (закрывает «тег врёт»
  для misconfig regression+proba-метрика до фильтра/обучения). **Дополнительно `pr_auc` на multiclass**:
  `average_precision_score` не определён на `(n,K)` без `label_binarize` → `pr_auc`+multiclass даёт явный
  `ConfigError("pr_auc not supported for multiclass")` (а не краш в недрах sklearn). Полнота: каждая метрика
  декларирует поддержку multiclass; неподдерживаемая комбинация → `ConfigError` на резолве.
- **Реализационная заметка (фикс R2):** чтобы метрика «хранила `labels`/`average`», адаптеры метрик
  переводятся с **атрибутов класса на `__init__(self, *, classes=None, average=None)`** + инстансные поля;
  `_REGISTRY` инстанцирует `cls(classes=…, average=…)`. `resolve_metric(name)` остаётся вызываемым без kwargs
  (back-compat: `classes=None, average=None` по умолчанию).

### 5. Регрессионная метрика (фикс R1-B3)
Регистрируется ≥1 regression-метрика, иначе `Task(kind="regression").target_metric == "rmse"` не резолвится
(`resolve_metric` бросает `ConfigError`) и FR-3 (regression end-to-end) недостижим. Добавляются `rmse`
(`needs="value"`, `greater_is_better=False`, `optimum=0.0`) и `mae` (аналогично) в `_REGISTRY` метрик.
`project_for_metric(... needs="value")` → `pred` (уже предусмотрено §3).

### 6. `_run_candidate` / `run_slice`
- Снимается binary-гейт `positive ... else None` в части multiclass: для классификации всегда копится
  `oof_proba` (1-D для binary, `(n,K)` для multiclass через reindex). `produced_proba` логика сохраняется.
- Проверка «single-class test split» обобщается: для proba-метрики фолд скипается, если в test < 2 классов
  (как сейчас) — корректно для любого `K`. **Метрика в прогоне одна на всех кандидатов** (`run_slice(metric=...)`),
  решение о skip зависит только от `y[test_idx]` и `metric.needs` → **skip одинаков для всех кандидатов**,
  поэтому OOF-маска успешных совпадает (поддерживает честность сравнения, ADR-0022 §1; фикс R2-B6).

## Последствия

- **Положительные:** корректный multiclass-скоринг с заявленным averaging; выравнивание классов устраняет
  тихую кривизну proba (RM3-3); binary-путь и регрессия не затронуты семантически; единый OOF `(n,K)`
  готов к значимости/ансамблям (M4/M7).
- **Отрицательные / компромиссы:** OOF-память multiclass растёт до `(n,K)`; `align_proba` — новый чистый
  код (покрыт тестами); агрегация `feature_importances` для multiclass-линейного — отдельное решение
  (ADR-0020); ε-сглаживание непокрытого класса (1e-6+ренорм) — компромисс «честно-малая, не нулевая
  вероятность» (фикс R1-1a/1b); `labels` — единый источник из composition (фикс R1-1c).
- **Влияние на слои:** `align_proba`/`project_for_metric` — `application` (numpy, без adapters);
  `Metric.average` — `core` (аддитивно); ветви — в `adapters/metrics`. import-linter не нарушен.

## Проверки

multiclass (`K=3`) end-to-end: leaderboard со скаляром; `align_proba` reindex'ит при перестановке
`est.classes_` и при фолде без класса даёт **ε-сглаженные строки с суммой 1** (не 0-столбец) — unit на numpy;
`log_loss` на таком выравнивании **ограничен** (нет `-log(0)`); `roc_auc` multiclass (`multi_class="ovr"`,
`average`, `labels` из единого источника) и `log_loss(n,K, labels)` дают валидный скаляр; binary-метрики
**бит-в-бит** как до дельты (регрессионный тест, без `labels`/`average`); **регрессия: `resolve_metric("rmse")`
возвращает метрику, regression end-to-end даёт leaderboard** (фикс R1-B3); **proba/class-метрика+regression-таск
(и value-метрика+классификация) → `ConfigError`** (фикс R1-4b); `Metric`/`resolve_metric` со старым набором
полей (без `average`/`classes`) валиден (дефолты).
