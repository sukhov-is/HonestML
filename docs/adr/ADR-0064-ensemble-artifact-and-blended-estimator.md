# ADR-0064 — Артефакт ансамбля: `BlendedEstimator` (протокол `Estimator`) + аддитивный manifest

- **Статус:** Accepted (M7, design-gate pending)
- **Драйвер:** DM-73 (не ломать контракт), DM-74 (наблюдаемость) — FR-ENS-4/5/6; NFR-M7-4/6
- **Связано:** ADR-0012/0024 (artifact-контракт, ARTIFACT_VERSION); ADR-0006 (Estimator-протокол);
  ADR-0030 (аддитивные manifest-ключи `.get`-паттерн); ADR-0033 (run-report); ADR-0063 (рецепт ансамбля).

## Контекст
`FittedModel` держит **один** `estimator: Estimator`; `save_artifact` делает `joblib.dump(model.estimator)`;
inference-путь (`predict`/`predict_proba`/`_aligned_proba`/`_calibrate`) обращается к `estimator.predict`/
`predict_proba`/`classes_`. Manifest аддитивен (band/calibration-ключи через `.get` на load, legacy грузится).
Ансамбль = несколько fitted-моделей + веса. Нужно отгрузить ансамбль, **не ломая** single-estimator
inference-путь и forward/backward-совместимость артефакта (R-ENSARTIFACT).

## Рассмотренные варианты
1. **Композитный `FittedModel` (список estimators + веса в самой модели)** — ❌ ломает inference-путь
   (`self.estimator` повсюду), требует bump ARTIFACT_VERSION, ветвления в каждом методе.
2. **Отдельные файлы `model_0..N.joblib` + список в manifest** — ⚠️ меняет структуру каталога/загрузку,
   больше поверхности; нужно для lazy-load (M8), избыточно для M7.
3. **`BlendedEstimator`, реализующий протокол `Estimator`+`ProbabilisticEstimator`** — ✅ ансамбль **opaque**
   для inference; `self.estimator` = `BlendedEstimator`; один `model.joblib`; manifest — аддитивный
   `ensemble`-блок (провенанс); **ARTIFACT_VERSION неизменен**.

## Решение (Вариант 3)

### §1 `BlendedEstimator` (adapters/ensembling.py)
Реализует `Estimator` (+`ProbabilisticEstimator` для классификации):
```python
class BlendedEstimator:
    feature_names: list[str]
    classes_: np.ndarray         # общий порядок классов (классификация)
    def fit(self, X, y, X_val=None, y_val=None, sample_weight=None) -> BlendedEstimator
    def predict(self, X) -> np.ndarray
    def predict_proba(self, X) -> np.ndarray   # классификация
```
Держит `members: list[Estimator]` + `weights: np.ndarray` (по `member_ids`). `predict_proba`: для каждого
члена выровнять proba к общему `classes_` (`align_proba`, как `FittedModel._aligned_proba`), взвешенно
усреднить, ренорм. `predict`: классификация — argmax усреднённой proba (регрессия — взвешенное среднее
`predict`). `fit` — refit каждого члена на переданных данных (full-DEV) на едином selected-subset
(design_matrix); члены — независимы. Реализует протокол **структурно** (как остальные адаптеры) — `FittedModel`
не отличает его от одиночного эстиматора.

**Инвариант (правка R1):** `BlendedEstimator.classes_` **обязан** равняться глобальному порядку классов
`FittedModel.classes` (`sorted(np.unique(y))`). Тогда внешний `FittedModel._aligned_proba`/`_positive_index`
(`np.where(estimator.classes_==positive)`, artifact.py) остаются **identity-reindex** (двойное выравнивание
безвредно). **Форма proba (правка R2):** `BlendedEstimator.predict_proba` возвращает **полную `(n,K)`**
матрицу (binary → **`(n,2)`**, оба столбца, `∑=1`), потому что `FittedModel._calibrate`/`_positive_index`/
`_score_dataset` индексируют `proba[:,pos]`/`out[:,1-pos]` (нужна 2-колоночная). 1-D `P(pos)` из ADR-0063 §2
— **только** внутренняя математика блендинга. Проверки — round-trip на члене с отличным `classes_`-порядком +
`test_artifact::test_binary_ensemble_calibrate_roundtrip` ((n,2) калиброванная proba воспроизводится).

### §2 Сериализация (один `model.joblib`)
`save_artifact` без изменений: `joblib.dump(model.estimator)` сериализует `BlendedEstimator` целиком
(`members` — атрибуты, рекурсивно). **Новых файлов нет.** load — `joblib.load(model.joblib)` отдаёт
`BlendedEstimator`; `FittedModel.classes`/`estimator.classes_` доступны как раньше. (joblib/pickle — принятая
trust-модель, подпись отложена в M8; ADR-0012 §security — не флагать.)

### §3 Аддитивный `ensemble`-блок manifest (провенанс, FR-ENS-6)
`save_artifact` добавляет (по образцу band-ключей, ADR-0026 §6):
```
"ensemble": {                     # null/absent для одиночной модели (legacy)
    "applied": bool, "method": str, "member_ids": [...], "weights": {id: w, ...},
    "gate_reason": str
}
```
load — `manifest.get("ensemble")` (None для pre-M7 артефактов → одиночная модель, без ветвления inference).
`best_model_id` для ансамбля = синтетический `"ensemble"` (члены — в `member_ids` и leaderboard). **ARTIFACT_VERSION
остаётся 1** (чисто аддитивно, forward/backward-симметрично через `.get`).

### §4 Интеграция facade (FR-ENS-4)
После `run_slice`, если `ensemble` задан и `run_mode='full'`: `ensemble_selection` (ADR-0063) → `EnsembleRecipe`
+ `applied`. Если `applied`: refit каждого члена на full-DEV, обернуть в `BlendedEstimator(members, weights,
classes)`; `FittedModel.estimator = BlendedEstimator`; калибровка (если включена) применяется к `predict_proba`
ансамбля как к обычному `ProbabilisticEstimator`. Если **не** `applied` → обычная отгрузка лучшего сингла
(M6-путь). `run_mode='selection'`: рецепт считается+репортится, модель **не** отгружается (ADR-0038).

**Sequencing id/refit (правка R2 — критично, иначе KeyError):**
- `run_slice.best_model_id` **остаётся реальным** id кандидата сквозь refit/калибровку. Существующие
  `refit_best(... factory=components.estimators[result.best_model_id])` и lookup калибровки
  (`c.id == result.best_model_id`, facade.py) работают как в M6.
- Синтетический `best_model_id="ensemble"` пишется **только** в manifest/report (`ensemble`-блок + поле
  `best_model_id` **на save**), не в рантайм-`SliceResult`.
- **Члены refit'ятся по `components.estimators[mid]` для каждого `mid ∈ recipe.member_ids`** (тот же
  post-HPO-write-back map, что использовал `run_slice` — `mid` есть базовая или тюненая фабрика по id).
- **Failure-таксономия (R2):** если refit члена падает на full-DEV (в отличие от per-fold изоляции
  `_CandidateFailed` в `run_slice`, у member-refit её нет) → **drop-and-renormalize**: член исключается,
  веса ренормируются, **WARNING**; если остаётся <2 членов → fallback на лучший сингл (`gate_reason`-аналог
  в report). Тип ошибки переиспользует `FitFailedError`. Проверка —
  `test_artifact::test_ensemble_member_refit_failure_drops_and_renormalizes`,
  `::test_refit_calibration_resolve_real_candidate_when_applied`.

### §5 run-report (FR-ENS-6, NFR-M7-6)
Аддитивная секция `ensemble` (тот же словарь, что manifest §3) — `applied`/`method`/`member_ids`/`weights`/
`oof_delta`/`gate_reason`. `RUN_MANIFEST_VERSION` неизменен. Симметрично HPO-блоку (ADR-0062 §7).

## Последствия
- **+** Артефакт-контракт **не ломается**: inference-путь не знает про ансамбль (opaque); ARTIFACT_VERSION=1;
  legacy single-model-артефакты грузятся без изменений (R-ENSARTIFACT закрыт).
- **+** Один `model.joblib` — нулевая дельта структуры каталога; manifest/report аддитивны и наблюдаемы.
- **+** Калибровка/holdout-скоринг ансамбля идут existing-путём (`ProbabilisticEstimator`).
- **−:** один joblib-блоб ансамбля крупнее (N моделей) — приемлемо для M7; lazy-load отдельных членов и
  ONNX-экспорт `BlendedEstimator` — **M8**.
- **−:** `members` refit на full-DEV (N фитов) при отгрузке — не бюджетируется (graceful, ADR-0032 §1).
