# ADR-0030 — Калибровка вероятностей на OOF + метрики Brier/ECE

- **Статус:** Accepted (реализован 2026-06-08, M4d)
- **Дата:** 2026-06-08
- **Драйверы:** DM4-4 (доверенные вероятности); FR-M4-10/11, NFR-M4-3/5. Аддитивный артефакт
  (ADR-0024 §4). Метрики через реестр (ADR-0021 паттерн).
- **Воркстрим:** M4d.

## Контекст
Вероятности из бустингов/линейных моделей часто плохо калиброваны. Калибровка ДОЛЖНА фититься на
held-out/OOF, НИКОГДА на train базовой модели (иначе оптимистична). Сейчас нет ни калибратора, ни
метрик калибровки (Brier/ECE отсутствуют). Артефакт несёт аддитивные ключи (ADR-0024 §4) —
калибратор ложится туда без бампа `ARTIFACT_VERSION`. SPIKE-0002: `FrozenEstimator`/`CalibratedClassifierCV`
доступны (sklearn 1.7.2).

## Рассмотренные варианты
1. **`CalibratedClassifierCV(cv=...)`** — рефитит базу k раз внутри; дорого, и его `cv` не purge-aware
   для time-series. Тяжелее нужного.
2. **1-D калибратор на OOF победителя** (Platt sigmoid / isotonic) — фитится на уже leakage-free
   OOF-предсказаниях победителя `(oof_score, y)`; без рефита базы; дёшево; применяется к финальной
   proba на inference. **Выбран** (для prefit-сценария — `FrozenEstimator`, если нужен внешний calib-сплит).
3. Калибровать на train базы. Отвергнут (утечка, оптимистично).

## Решение

### 1. Калибратор = 1-D отображение на OOF (opt-in)
`CVConfig.calibrate: Literal["off","sigmoid","isotonic","auto"] = "off"` (аддитивно; `auto` →
sigmoid при calib-n < ~1000, иначе isotonic, SPIKE/исслед.). Только классификация (регрессия — вне).
- **Отдельный proba-OOF канал победителя (фикс R1-as-is/adv):** калибровка осмысленна ТОЛЬКО над
  вероятностями, а `Candidate.oof_pred` после ADR-0026 §3 — «метрика-готовый» вектор (для accuracy/
  rmse это класс/значение, не proba). Поэтому калибровка **форсит захват `oof_proba`** победителя
  (`need_proba=True` независимо от selection-метрики) и читает **этот канал** `(oof_proba, y)`, а не
  `oof_pred`. (`oof_proba` уже есть в `_run_candidate:281-323` при `want_proba`.)
- Фит — leakage-free по построению OOF (модель не видела свои OOF-строки): `sigmoid` =
  `_SigmoidCalibration` (Platt), `isotonic` = `IsotonicRegression` (sklearn). Multiclass — per-class
  (OvR) калибровка над `(n,K)`-proba + ренорм (temperature — future-аддитив).
- На inference: `calibrated = calibrator.transform(raw_proba)` в `FittedModel.predict_proba` (после
  align_proba, перед возвратом).
- **`predict()` при калибровке (фикс R2-completeness):** `FittedModel.predict` остаётся **сырым
  `estimator.predict`** (класс/argmax базовой модели), калибровка меняет только УВЕРЕННОСТЬ, не решение.
  Для binary с монотонным sigmoid/isotonic argmax совпадает; для multiclass per-class+ренорм argmax
  калиброванной proba МОЖЕТ отличаться от `predict()` — это задокументированный контракт (threshold-
  tuning → M7); `predict()` и `predict_proba()` могут давать разный класс на multiclass, без молчания.
- **Деградация (фикс R2-completeness):** если proba-OOF победителя недоступен/вырожден (все-NaN,
  `<2` классов, не `ProbabilisticEstimator`) при `calibrate!="off"` — калибратор НЕ прикрепляется,
  `calibration_applied=False` + WARNING (не падение; симметрично консервативности band).

### 2. Анти-ликедж как проверяемый (NFR-M4-3)
Калибратор фитится ТОЛЬКО на OOF (или, при `outer_holdout`, на dev-OOF) — индексы фита **дизъюнктны**
с любым train базовой модели по построению OOF. При prefit-внешнем calib-сплите — `FrozenEstimator`
(не рефитить базу). Assert дизъюнктности в тестах.

### 3. Cross-fitted gate, прод-калибратор на ПОЛНОЙ OOF (фикс R2-major «hard split хуже отсутствия»)
Hard 50/50 split (раунд 1) переусердствовал: на малых данных режет OOF, isotonic переобучается на
половине, gate шумен. **Решение (стандарт sklearn `CalibratedClassifierCV(ensemble=False)`):**
- **Gate — cross-fitted, без потери данных:** внутренний K-fold по proba-OOF; на каждом фолде калибратор
  фитится на K−1, предсказывает отложенный → честные out-of-fold калиброванные proba **без** in-sample
  смещения; на них считаются Brier/ECE vs сырые. Если `outer_holdout>0` (ADR-0029) — gate на
  outer-holdout (ещё чище).
- **Прод-калибратор — на ВСЕЙ OOF** (а не на половине): после прохождения gate финальный калибратор
  фитится на полной proba-OOF победителя и кладётся в артефакт. Калибровочные данные не теряются.
- **Min-n авто-отключение:** при `n_calib < ` порога (напр. 50; isotonic — выше) калибровка
  авто-отключается с WARNING (`calibration_applied=False`), а не применяет недо-обученный калибратор.
- Если не лучше на gate — НЕ прикрепляется, `calibration_applied=False` (NFR-M4-7). ECE — не
  единственный гейт (не proper); ведущий — Brier. **Anti-leakage:** OOF leakage-free по построению
  (модель не видела свои OOF-строки); assert индексы OOF ∩ train базы = ∅; cross-fit gate не пересекает
  fit/eval внутри K-fold (NFR-M4-3).
- **Переиспользование (ADR-0031):** порт `Calibrator` + чистая `crossfit_calibrate` (cross-fit-цикл)
  — **единый источник** и для этого gate, и для refinement-отбора (ADR-0031 §1). Реализуются в одном
  воркстриме M4d; калибратор не строится дважды.
> **Реализация M4d (отклонение, обоснованное):** прод-калибровка **отключена при `time_ordered`**
> (как refinement-отбор, ADR-0031 §3 B2): симметричный cross-fit gate на TS-OOF заглядывал бы в будущее
> (калибровка фолда `k` на фолдах `>k`) → оптимистичный Brier/ECE. Вместо implementing expanding-окна в
> M4 калибровка под TS не прикрепляется (`applied=False`, WARNING, `reason="time-series"`), а
> **expanding TS-gate вынесен в тот же future-ADR, что и TS-refinement**. Не-TS gate — симметричный
> K-fold (корректен). Per-block-виабельность gate проверяется **тем же** `viable_blocks`, что и
> refinement (общая предпосылка `crossfit_calibrate`). Единый `MIN_CALIB_N=50` для sigmoid/isotonic;
> метод-зависимый порог (isotonic выше) отложен.

### 4. Аддитивное хранение в артефакте (NFR-M4-5)
`FittedModel.calibrator: object | None = None` (аддитивное поле). `save_artifact`: ключ манифеста
`calibration` (метод/применён + Brier/ECE до-после) + опц. `calibrator.joblib`. `load_artifact`:
читает `getattr/.get` fallback → старые артефакты (без калибратора) грузятся. **`ARTIFACT_VERSION` не
меняется.** **Pickle-trust (фикс R2-completeness):** `calibrator.joblib` — **та же** pickle/joblib
trust-граница, что и `model.joblib` (ADR-0024: load только из доверенного источника); SECURITY-докстринг
`load_artifact` расширяется на ОБА файла. Нового механизма не требуется — наследование риска.

### 5. Метрики Brier/ECE (FR-M4-11)
В `adapters/metrics.py` `_REGISTRY` (паттерн ADR-0021):
- `brier` — `needs='proba'`, `greater_is_better=False`, `optimum=0.0`, обёртка `brier_score_loss`
  (binary; multiclass — sum по классам / OvR), `sample_weight`-aware.
- `ece` — `needs='proba'`, `greater_is_better=False`, `optimum=0.0`; **binned**, своя реализация
  (в sklearn нет). **Спецификация (фикс R2-completeness):** (а) **пустые бины** — вес `|Bm|/n=0`,
  пропускаются; при `quantile`-стратегии вырожденные (дубль-proba) границы коллапсируют в один бин;
  (б) **multiclass ECE = confidence/top-label** (по `max`-proba предсказанного класса — планка
  AutoGluon; classwise — future); (в) **weighted:** вес бина `Σw/Σw_total`, `acc` и `conf` —
  взвешенные. Дефолт бинов 10, strategy uniform. Reliability-кривая (`calibration_curve`) — в отчёт.
- Доступны как диагностики и target-метрики; ECE как ЕДИНСТВЕННАЯ selection-метрика **не рекомендуется**
  (не proper) — документируется. `sample_weight` делится вместе с OOF в gate (§3).

## Последствия
- **Положительные:** доверенные вероятности (AutoGluon-планка `calibrate=True`); leakage-safe и
  дёшево (1-D на OOF, без рефита базы); аддитивно (без бампа версии); Brier/ECE + reliability видимы.
- **Отрицательные/компромиссы:** multiclass-калибровка per-class+ренорм (не temperature — future);
  improvement-gate тратит часть OOF на eval; isotonic переобучается при calib-n<~1000 (→ `auto`
  выбирает sigmoid).
- **Влияние на слои:** калибратор-логика и Brier/ECE — `adapters`; применение — `composition/artifact`
  (`FittedModel`); gate-оркестрация — `composition`. `core` — без изменений (метрики через реестр,
  калибратор — деталь адаптера/артефакта). sklearn — только в адаптере (NFR-M4-4/6). import-linter не
  нарушен.

## Проверки
- Индексы фита калибратора дизъюнктны с train базы (NFR-M4-3); калиброванные Brier/ECE ≤ сырых на
  held-out, иначе не применяется (FR-M4-10).
- `brier`/`ece` резолвятся и считают; ECE binned, число бинов настраиваемо, `sample_weight`-aware;
  ECE сопровождается Brier (FR-M4-11).
- Артефакт с калибратором save→load→predict_proba даёт калиброванные вероятности; старый артефакт
  (без калибратора) грузится (NFR-M4-5).
- Регрессия + `calibrate!="off"` → `ConfigError` (калибровка только для классификации).
- `calibrate="off"` (дефолт) — поведение M3 не меняется.
