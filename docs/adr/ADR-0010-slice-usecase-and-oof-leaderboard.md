# ADR-0010 — Use-case прогона slice и OOF-leaderboard

- **Статус:** Proposed · **частично superseded by ADR-0023** (§5/F5 group-rejection: M2-отказ от групп снят
  в M3, заменён group-aware CV с проверкой утечки) · **расширен ADR-0021** (multiclass OOF `(n,K)`/проекция)
  и **ADR-0022** (per-candidate изоляция падения).
- **Драйверы:** D2 (обобщение), D3 (корректность ML); FR-8/9, NFR-3/4; R4, R6.
- **Воркстрим:** M2.

## Контекст

Нужен сквозной сценарий: для каждой модели — CV по `CVSplitter`, сбор предсказаний,
расчёт `Metric`, ранжирование через `select_best`. Ранее оркестрация CV была размазана по нескольким модулям с глобальным состоянием. Решаем, где живёт цикл CV и
как формируется leaderboard, не таща сюда тюнинг/ансамбли/бюджет (M5/M7).

## Рассмотренные варианты

1. **Логика CV в доменном ABC `Estimator`** — нарушает Humble Object: домен должен
   быть чисто/синхронно тестируемым без оркестрации (R-16). Отказ.
2. **Цикл CV в фасаде** — фасад раздувается, логика не переиспользуется и плохо
   тестируется в отрыве от sklearn-обвязки.
3. **Отдельный use-case `run_slice` в `application/`** поверх портов; домен остаётся
   чистым; OOF — универсальная валюта для leaderboard и (позже) ансамблей.

## Решение

Вариант **3**. Use-case `run_slice(dataset, task, *, estimators, splitter, metric,
policy, significance_test, ctx) -> SliceResult`:

1. `splitter.split(dataset)` → фолды `Fold(fit_idx, es_idx, test_idx)`; каждый
   валидируется `validate_fold` (анти-ликедж — механизм, R-6).
2. Для каждой модели × фолд: `dataset.take(fit/es/test)` → numpy+коды; `estimator.fit
   (X_fit, y_fit, X_es, y_es)`; предсказание на `test` собирается в **OOF-вектор**
   (по test-индексам фолдов).
3. **Проекция выхода под `metric.needs`** — helper в use-case (`proba` →
   `predict_proba` срез P(класс); `class` → `predict`; `value` → `predict`). Снимает
   нужду в `predict_proba_positive` (ADR-0006).
4. Метрика считается на OOF (и опц. per-fold) → `Candidate(id, score=oof_metric,
   n_features, train_time, oof_pred)`.
5. `select_best(candidates, policy, significance_test)` — **абсолютный argmax**; в M2
   `NoSignificanceTest` ⇒ тай-брейк инертен (ADR-0007/N-3).
6. Результат `SliceResult`: leaderboard (отсортированные `Candidate` + метрики),
   `best_id`, обученные на полном train модели-кандидаты (для рефита/артефакта).

- **Финальная модель**: победитель дообучается на всём train (es-хвост из train)
  для предсказания/артефакта — отдельный шаг use-case (`refit_best`), через порт.
- **Бюджет/таймаут/resume** — НЕ здесь (M5); use-case принимает `Budget`
  опционально, но в M2 не прерывается (заглушка-проверка `exhausted`).

## Последствия

- (+) Домен чист и синхронно тестируем (NFR-3); оркестрация изолирована (Humble
  Object, R-16).
- (+) OOF как валюта — переиспользуется ансамблями (M7) и стат-значимостью (M4).
- (+) Воспроизводимость: при фиксированном seed и детерминированных адаптерах
  leaderboard стабилен (NFR-4); абсолютный score (C1).
- (−) Рефит финалиста — доп. обучение; оправдано (несмещённый OOF для выбора, полный
  train для деплоя).
- Границы: tuning/ensembling/honest-significance/бюджет-degradation — последующие
  воркстримы; M2 валидирует каркас (R4).

## Уточнения контрактов (ревью, фаза 8)

Зафиксировано до реализации (контракты «дёшево сейчас / breaking потом»):

1. **Форма proba и positive-класс (F1):** `predict_proba` → `(n, n_classes)` по
   порядку `classes_`. `project_for_metric(needs="proba")` берёт столбец
   положительного класса по `idx = np.where(classes_ == positive)[0]`, где
   `positive = Task.positive_label` (дефолт `1`). `class` → `predict`; `value` →
   `predict`. Это снимает хардкод `[:, 1]` и риск инверсии классов.
2. **Контракт `oof_pred` (F2):** хранит **сырой `P(positive)`** (для будущей
   значимости M4), НЕ проекцию под метрику. Score кандидата считается отдельно
   через `project_for_metric` от сырого выхода. Так включение `SignificanceTest`
   в M4 не ломает контракт.
3. **needs↔capability (F3):** совместимость (модель, метрика) проверяется в
   composition по capabilities; несовместимые пары отсеиваются, при отсутствии
   совместимых — `ConfigError`. `project_for_metric` при `needs="proba"` и
   не-`ProbabilisticEstimator` бросает `ConfigError` (а не `AttributeError`).
4. **`validate_fold` — обязателен в use-case (F4):** `run_slice` вызывает
   `validate_fold(fold, groups=...)` на **каждом** `Fold` (механизм анти-ликеджа,
   не доверие стороннему `CVSplitter`).
5. **Группы (F5):** если схема несёт `ColumnRole.GROUP`, use-case извлекает
   `groups` и передаёт в `validate_fold`; в M2 group-aware split не реализован →
   при наличии групп `run_slice` падает `ConfigError` (отсылка к M4). Fail-fast,
   не тихий ликедж.
6. **es-хвост и вырожденные фолды (F6):** не-ES-эстиматоры (Dummy/Linear)
   обучаются на `fit_idx ∪ es_idx` — **es-хвост не теряется**; ES-эстиматоры
   (бустинги, M3) используют `es_idx` для early stopping. Фолд, где test для
   proba-метрики не содержит обоих классов, **пропускается с warning**; если
   полный OOF не содержит обоих классов → `SchemaValidationError`. (Min-class
   guard — на границе `Reader`/`run_slice`.)
7. **Семантика OOF для holdout (F-minor):** holdout даёт частичный OOF (только
   test-часть) → score = holdout-test-метрика; `Candidate` несёт маску валидных
   индексов. Полный OOF (KFold) — без дыр.
8. **`SliceResult`/leaderboard (F8):** `SliceResult` = `leaderboard:
   list[LeaderboardEntry]` + `best_model_id` + обученные кандидаты;
   `LeaderboardEntry{model_id, score, metric, n_features, train_time, rank}`. Та же
   структура — в `leaderboard.json` (ADR-0012) и атрибуте фасада `leaderboard_`
   (публичная поверхность, SemVer).
9. **`Candidate.id` (F-minor):** = имя модели, уникально в прогоне (стабильный
   детерминированный тай-брейк `select_best`).
