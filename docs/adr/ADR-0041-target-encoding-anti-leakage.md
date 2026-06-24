# ADR-0041 — Анти-ликедж target-encoding: OOF cross-fit на единых фолдах + full-train спека

- **Статус:** Accepted (реализован в M6a, 2026-06-09; OOF reserve-маска null/unknown→global_mean добавлена по ревью F1, `09-review.md`)
- **Дата:** 2026-06-09
- **Драйверы:** DM-1 (анти-ликедж — headline); FR-FE-3, NFR-FE-1/3/5. Прямое зеркало cross-fit-прецедента
  `crossfit_calibrate` (ADR-0030 §3 / ADR-0031); единый `oof_fold_index` (ADR-0010); коды категорий
  `CategoryTable` (ADR-0005).
- **Воркстрим:** M6a (FE-дельта).

## Контекст
Target-encoding (smoothed target mean per категория) — самый ценный и самый **ликедж-опасный** FE: глобальный
фит на full-train заносит таргет CV-test в признаки → оптимистичный OOF и нечестный отбор. Нужен честный OOF,
**согласованный** с оценочными фолдами `run_slice`, плюс корректная спека для отгружаемой модели/inference.

## Рассмотренные варианты
1. **Глобальный TE на full-train, тот же массив в CV.** Протекает (R-FE-LEAK). **Отвергнут.**
2. **Кросс-фит по собственному TSCV энкодера**. Фолды TE ≠ оценочные фолды →
   TE «подсматривает» eval-test (R-FE-FOLD-ALIGN). **Отвергнут.**
3. **Кросс-фит OOF на ЕДИНОМ `oof_fold_index` run_slice** (зеркало `crossfit_calibrate`) для оценки + full-train
   спека для refit/inference. **Выбран** — честно, согласовано с оценкой, переиспользует прецедент.
4. **Двойная вложенность (nested CV для TE внутри каждого train-фолда).** Строже устраняет остаточную утечку
   train-признаков, но дорого и сложно; прецедент (`crossfit_calibrate`) использует одиночный cross-fit.
   **Отложен** (R-FE-NEST, честно задокументирован ниже).

## Решение

### 1. OOF cross-fit на единых фолдах (оценка)
Для каждой категориальной колонки `c` строится OOF-TE-колонка над **кодами** `CategoryTable` (`int64`) и
таргетом `y`, на **том же** `oof_fold_index`, что использует `run_slice`:
```
для фолда f:  карта_f = smoothed_mean( codes[fold != f], y[fold != f] )
              OOF_TE[fold == f] = карта_f.lookup( codes[fold == f] )   # global_mean для непокрытого кода
```
— дословно паттерн `crossfit_calibrate` (фит на `blocks != b`, transform `b`). Реализация — **чистая функция
`crossfit_encode` в `application`** (numpy, без polars/sklearn), синхронно тестируемая на массивах (Humble
Object, NFR-FE-2). **Инвариант фолдов (R-FE-FOLD-ALIGN):** `oof_fold_index` — **единственный** источник; TE и
оценка делят его.

**Граница данных и сигнатура (фикс ревью R1-clean-arch/A1):** `run_slice` извлекает коды как **numpy** до
вызова — `codes = ds.categorical_codes()[:, te_cols]` (уже материализованы `CategoryTable`), `y = ds.target()`,
`oof_fold_index` (numpy); polars в `crossfit_encode` **не входит**. Сигнатура:
```python
def crossfit_encode(codes: np.ndarray, y: np.ndarray, oof_fold_index: np.ndarray,
                    *, smoothing: float) -> np.ndarray:  # (n, k_te) OOF-TE колонки
    # для каждого фолда f: global_mean_f = mean(y[fold != f]); карта_f на (codes, y)[fold != f];
    # OOF_TE[fold == f] = карта_f.lookup(codes[fold == f]) с fallback global_mean_f
```
`global_mean` считается **внутри** (per-fold для OOF), функция не требует предрасчёта. Все операции — numpy.

**Гейт `oof_fold_index` (фикс ревью R1-consistency/leakage/A6):** `run_slice` строит `oof_fold_index`
**безусловно при `fe.target_encoding`** (сейчас — только при `capture_proba`/refinement). Условие гейта:
`if capture_proba or selection == "refinement" or fe.target_encoding:`. Без этого TE-кросс-фит не на чем
выполнить → инвариант фолдов нарушится. Проверяется тестом (`oof_fold_index is not None` при TE даже без
калибровки).

### 2. Сглаживание и резервы
`smoothed = (cnt·mean + k·global_mean) / (cnt + k)`, `k = FEConfig.te_smoothing` (дефолт 10.0:
сжимает переобучение редких категорий к `global_mean`, сохраняя сигнал частых; **конфигурируемый**, sensitivity —
тест-проверка R1). Непокрытый/unseen код (включая `null_code`/`unknown_code` `CategoryTable`) → `global_mean`
(train-таргета фолда для OOF; полного train для inference). Знак/масштаб не клипуются (модельное мнение — вне
объёма).

### 3. Full-train спека (refit / inference / holdout) — где какая TE применяется (фикс ревью R1-leakage/A8)
**Чёткое разделение путей** (устраняет двусмысленность OOF-vs-full-train):
- **OOF-TE-аугментация — ТОЛЬКО для оценки** (цикл кандидатов `run_slice`): leaderboard/band считаются на
  `x_full`, аугментированном OOF-TE. **Не** используется ни в refit, ни в inference.
- **refit / inference / holdout — full-train `TargetEncodingSpec`** на границе Reader: `TargetEncodingSpec`
  фитится на **всех DEV-строках** (post-carve; код → smoothed_mean + `global_mean`), сериализуется в
  `FeatureSchema` (ADR-0042); `refit_best` обучает отгружаемую модель на DEV с **этой** спекой, `predict`
  применяет **её же** → train(refit)==inference консистентны (нет train/serve-skew).
- **Holdout-несмещённость сохранена (ADR-0029):** holdout вырезается **до** всего (facade, до `run_slice`);
  `TargetEncodingSpec` фитится только на DEV → holdout-строки кодируются DEV-картой и **не участвуют** в её
  обучении → честно. Holdout-скор отгружаемой (full-DEV-TE) модели не смещён TE.

**Асимметрия «OOF для оценки / full-train для отгрузки» намеренна** — тождественна калибровке на уровне
**решения** (cross-fit-гейт оценивает, full-train-объект отгружается, ADR-0030). Это даёт честный OOF-скор и
лучшую полную кодировку у отгружаемой модели; leaderboard-скор оценивает **обобщение**, не точную метрику
shipped-модели (как и везде в OOF-режиме). **Не** утверждаем тождества OOF-скора и shipped-метрики.

### 4. Объём: classification-only (M6a) — где гейт (фикс ревью R1/A3)
M6a TE — **бинарная классификация** (`mean = P(y = positive)`). Multiclass (нужны K колонок/иная трактовка) и
regression (mean таргета — иная статтрактовка) → **skip + WARNING** (R-FE-REG-TE), полноценно — defer/будущий ADR.
**Локация гейта:** **graceful skip в composition/`facade.fit`** (kind известен после резолва `Task`, до
`run_slice`): если `fe.target_encoding and task.kind != "binary"` → WARNING + **эффективный** FE с выключенным
TE (`fe.model_copy(update={"target_encoding": False})`) прокидывается в `RunConfig`. **Не** `ConfigError`
(FR-FE-3 выбрал graceful деградацию, как gating калибровки/refinement — `__init__` хранит verbatim,
sklearn-инвариант ADR-0011 не нарушается). frequency/intersections при этом работают (target-независимы). Не
overclaim. Проверка: multiclass+TE → WARNING, TE не применён, freq/intersections применены.

### 5. Честная граница точности (R-FE-NEST) — переформулировано (фикс ревью R1-leakage/A7)
Одиночный cross-fit устраняет **прямую** утечку: OOF-TE строки `r` **не зависит** от таргета `r` (и любого
таргета её фолда `f` — карта `f` обучена на `fold != f`). Это доказуемо (property §Проверки).

**Остаётся TE-специфичный остаточный эффект (честно, без ложной аналогии):** модель кандидата обучается на
train-строках, чьи OOF-TE-признаки построены carт, видевшими таргеты **других** фолдов → между фолдами идёт
косвенный обмен распределением **через сам вектор признаков**. **Важно:** аналогия с `crossfit_calibrate`
**неполна** — калибровка преобразует **выход** (post-prediction, не меняет вход модели), а TE меняет **входной
признак**, поэтому эффект у TE сильнее и в принципе может слегка оптимизировать отбор под межфолдовую структуру.
Это **НЕ** «тождественно калибровке» (прежняя формулировка ошибочна).

**Диспозиция — принятый известный компромисс, эскалирован на design-gate:** одиночный cross-fit — широко
принятый практический стандарт (Micci-Barreca; out-of-fold TE); полное устранение требует **двойной
вложенности** (nested-CV TE внутри каждого train-фолда) — дорого, отложено (future ADR). Эффект **материален
тем сильнее**, чем: меньше фолдов, выше кардинальность категории, сильнее межфолдовый дрейф распределения;
сглаживание `k` его притупляет (сжатие к `global_mean`). Holdout (ADR-0029, §3) — **независимая** несмещённая
проверка поверх (full-DEV-TE, не OOF) → ловит остаточный оптимизм OOF-скора. **Не выдаётся за zero-leakage.**

## Последствия
- (+) Честный OOF target-encoding, согласованный с отбором; holdout-несмещённость (ADR-0029) не регрессирует;
  переиспользование `crossfit_calibrate`-паттерна и `oof_fold_index`; чистый numpy (тест на фейках).
- (−/компромисс) Остаточный train-feature-эффект одиночного cross-fit (принято, документировано); TE только
  бинарная в M6a; OOF/full-train асимметрия кодировки (намеренна, как калибровка).
- **Влияние на слои:** `crossfit_encode` — `application` (чистый numpy); full-train `TargetEncodingSpec` —
  `core`(данные)/`adapters`(фит); гейт `oof_fold_index` — `application/slice`.

## Проверки
- **Property (анти-ликедж, NFR-FE-1):** перестановка таргета **внутри** фолда `f` **не меняет** OOF-TE строк
  фолда `f` (их кодировка — из фолдов `!= f`); зависимость OOF-TE строки от собственного таргета отсутствует.
- Фолды TE тождественны оценочным (`oof_fold_index` один) — тест.
- full-train `TargetEncodingSpec` round-trip; inference применяет её детерминированно; unseen → `global_mean`.
- `crossfit_encode` на фейк-массивах без polars/обучения = ручной расчёт (golden).
- regression/multiclass + TE → skip+WARNING (не падение, не молча).
- FE-TE прогон с общим seed детерминирован; число фитов TE = `folds × cat_cols` (счётчик, NFR-FE-5).
