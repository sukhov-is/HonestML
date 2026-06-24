# ADR-0068 — `finalize` / `refit_full`: shipped-модель на всех данных

- **Статус:** Accepted (M8, design-gate pending)
- **Драйвер:** DM-83 (честность shipped-модели) — FR-SRV-4; NFR-SRV-4/5
- **Связано:** ADR-0029 (honest outer-holdout, baseline), ADR-0010 (`refit_best`, baseline), ADR-0064 §4
  (`refit_members` drop-and-renormalize, в `application.ensemble`), ADR-0030 (калибратор на DEV-OOF),
  ADR-0035 (состав run-fingerprint).

## Контекст
При `outer_holdout>0` (honest-режим, ADR-0029) каверится `ds=dev`/`holdout_ds`; селекция/refit/калибровка —
на DEV; `holdout_ds` трогается **один раз** для несмещённого `holdout_score` (facade.py:421), затем **не
возвращается в обучение**. → shipped-модель (`refit_best`/`refit_members` на DEV) **недообучена на доле
данных**, ушедшей в holdout. «Финального фита на всех данных» нет. При дефолте `outer_holdout=0` DEV==все →
пробела нет (no-op). Это стандартный паттерн «оценить по holdout, обучить финал на всех данных» — но вторая
половина отсутствует.

## Рассмотренные варианты
1. **Шиппить DEV-модель (статус-кво)** — ❌ при honest-holdout выбрасывает holdout-долю из shipped-модели;
   пользователь, включивший holdout ради честной оценки, получает **худшую** продакшн-модель.
2. **Всегда рефитить на всех данных, молча** — ❌ при `outer_holdout>0` shipped ≠ scored молча: репортируемая
   оценка перестанет соответствовать отгруженному без раскрытия (R-SRVFINAL).
3. **`finalize` (default-on в honest-режиме): рефит shipped-победителя на DEV+holdout; репортируемая оценка
   остаётся оценкой DEV-модели, `shipped_on` явно фиксирует объём** — ✅ best-модель в прод, честная оценка
   сохранена и раскрыта.

## Решение (Вариант 3)

### §1 `finalize`-параметр + размещение (НЕ в fingerprint — правка ревью R1)
`AutoML(..., finalize: bool = True)` — публичный `__init__`-параметр, хранится **verbatim** (sklearn-инвариант,
`clone`, FR-SRV-5). Эффект только при `ship_model and outer_holdout>0`. **`finalize` НЕ кладётся в `RunConfig`
и НЕ входит в run-fingerprint:** это **пост-селекционное** shipping-решение — оно **не меняет per-candidate
OOF/leaderboard/селекцию**, поэтому по смыслу fingerprint (идентичность вычисления селекции, ADR-0035) ему там
не место (точнее: «не влияет на per-candidate OOF», а не «как несуществующий refit-параметр» — уточнение R2).
Иначе любое новое hashed-поле в `RunConfig` ломает «off≡M7 fingerprint» даже при `outer_holdout=0` (R1-blocker:
`compute_run_fingerprint` хеширует `run_config` целиком). Провенанс идёт по **наблюдаемым каналам** (§5):
артефакт-манифест (`shipped_on`/`finalize`) и run-report (явный `serving=` параметр `build_run_report`), а **не**
через fingerprint. При `outer_holdout=0` стадия — **no-op** (поведение, fingerprint и артефакт-модель идентичны
M7). `finalize=False` → старое DEV-refit-поведение (escape hatch).

### §2 Целостность honest-оценки (R-SRVFINAL — раскрытие, не смешение)
`holdout_score`/`leaderboard_` **остаются** оценкой модели, обученной на DEV. После finalize отгружается
модель, дообученная на всех данных, поэтому `holdout_score` — **консервативная (пессимистичная) нижняя
оценка** shipped-модели: реальный шиппинг обучен на большем объёме, разрыв растёт с `outer_holdout` (раскрытие
R1, одностороннесть устранена). Манифест/run-report несут `shipped_on ∈ {"dev","all"}`, `finalize`-флаг и
`outer_holdout` → честно: «оценка относится к DEV-модели (нижняя граница); отгружена модель на всех данных».
Второго скоринга all-data-модели нет (нет held-out, на чём честно мерить — в этом и смысл оценки до finalize).

### §3 Калибратор под finalize (R-SRVCAL + classes-инвариант, правка R1)
Калибратор фитнут на DEV-OOF (ADR-0030); после all-data refit свежего held-out OOF нет → **сохраняется
DEV-калибратор** как раскрытая аппроксимация. **Инвариант классов (R1-адверсариально):** калибратор
привязан к каналу `P(pos)` через `_positive_index` = `np.where(estimator.classes_ == positive)` (порядок-
независимо), а `BlendedEstimator.classes_` по контракту = **глобальный** порядок (ADR-0064 §1, не выводится
из членов) → после finalize-refit класс-канал стабилен. **Защита (уточнено R2):** триггер отцепа — не
«изменилось множество классов estimator» (для `BlendedEstimator` оно всегда глобально и не меняется), а
**«в глобальных классах есть класс, отсутствовавший в DEV-OOF, на котором фитился калибратор»** (редкий класс,
попавший только в holdout): для такого класса per-class калибровка невалидна. В этом случае калибратор
**отцепляется** (`applied=False` + WARNING), а не применяется к необученному столбцу. Для регрессии
(`classes=None`, калибратора нет) §3 — **no-op** (правка R2). Полная рекалибровка на свежем inner-split → Day-2.
Тест: `dev_unseen_class_detaches_calibrator`.

### §4 finalize для ансамбля (граничный случай, правка R1)
Если на DEV был отгружен `BlendedEstimator`, finalize **повторно** вызывает `refit_members(ds_full, …)`
(`application.ensemble`) на **тех же** `kept`-членах с той же drop-and-renormalize-семантикой (ADR-0064 §4):
- ≥2 члена выжили на `ds_full` → новый `BlendedEstimator` (глобальные `classes_`, перенормированные веса);
  `shipped_on="all"`.
- <2 выжили на `ds_full` → откат к синглу-победителю, refit на `ds_full`; `ensemble.applied=False`,
  `gate_reason="insufficient_members_after_refit"`; провенанс честно показывает расхождение «scored=ensemble,
  shipped=single». (DEV-ансамбль остаётся в leaderboard/оценке; отгружается то, что выжило.)
`shipped_on="all"` проставляется в **обоих** путях (сингл/ансамбль). **Одно-модельный/вырожденный прогон**
(нет `ensemble.applied`) — тривиальный сингл-путь: `refit_best(ds_full)` победителя, без ансамбль-ветки
(правка R2).

### §5 Механика вставки + каналы провенанса (правка R1 — путь данных)
Текущий порядок фасада: `_ship_estimator`(DEV) → `_calibrate_winner`(DEV-OOF) → сборка `self.fitted_` →
скоринг holdout (`fitted_._score_dataset`, 421). **finalize-стадия вставляется ПОСЛЕ скоринга holdout** и
**заменяет** `self.fitted_.estimator` и `self.best_estimator_` на `ds_full`-refit (через `refit_best`/
`refit_members`), **сохраняя** DEV-калибратор (§3), `holdout_score` (§2) и `leaderboard_`. Провенанс:
- **манифест** (`save_artifact`): аддитивные `shipped_on`, `finalize` (ADR-0065 §1);
- **run-report**: `build_run_report` получает **новый явный канал** (параметр `serving=`/поле, симметрично
  `hpo=`/`ensemble=`) с `{finalize, shipped_on, outer_holdout}` — иначе провенанс наблюдаем по требованию, но
  пути данных нет (R1-major). `finalize`-флаг отчёта берётся из facade-параметра, `shipped_on` — из стадии.
  **Под `run_mode=selection`** (`ship_model=False`) finalize-стадия **не выполняется** → `serving=None` (не
  `finalize=True` для неотгруженной модели), симметрично not-applied у hpo/ensemble (правка R2). Тест:
  `serving_absent_when_selection`.

### §6 `refit_full` vs `finalize` vs `distill`
`refit_full` — низкоуровневый примитив «refit на всех данных» (переиспользует `refit_best`
(`application.slice`) / `refit_members` (`application.ensemble`) с `ds_full`, без новой машины). `finalize` —
фасад-стадия, производящая отгружаемый артефакт. **`distill` — НЕ в этом проходе** (компрессия/дистилляция —
M8c/позже).

## Последствия
- **+** При honest-holdout shipped-модель использует все данные — нет выброшенной доли; честная оценка
  сохранена, раскрыта как нижняя граница (`shipped_on`/`outer_holdout`).
- **+** Дефолт (`outer_holdout=0`) — no-op; `finalize` вне fingerprint → off≡M7 точно (fingerprint/поведение/
  артефакт-модель стабильны).
- **+** Переиспользует `refit_best`/`refit_members` — нулевая новая refit-машина (NFR-SRV-6).
- **−:** shipped ≠ scored при holdout — раскрыто (`shipped_on`), стандартная практика; для ансамбля возможен
  откат к синглу на ds_full (§4) — провенанс честный.
- **−/R-SRVCAL:** калибратор — DEV-аппроксимация после finalize (раскрыто, classes-инвариант защищён §3);
  полная рекалибровка — Day-2.
- **−:** +1 refit-стоимость при `outer_holdout>0, finalize=True` (не бюджетируется — graceful, как refit).

## Day-2 (committed)
- Рекалибровка после finalize на свежем inner-split (нулевая аппроксимация калибратора).
- `distill` (квантизация/компрессия/teacher-student) → M8c/позже.
