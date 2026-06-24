# ADR-0029 — Честный режим: holdout-leaderboard (несмещённая финальная оценка); nested-CV → M7

- **Статус:** Accepted (реализован 2026-06-08, M4c)
- **Дата:** 2026-06-08
- **Драйверы:** DM4-3 (несмещённая финальная оценка); FR-M4-9, NFR-M4-3/7. Решение:
  band+holdout сейчас, nested → M7. Наследует ADR-0010 (slice/OOF), ADR-0016 (резолвер схем).
- **Воркстрим:** M4c.

## Контекст
Сейчас `run_slice` делает плоскую per-model CV → OOF → абсолютный `select_best` на ВСЕХ данных; та
же CV служит и отбором, и оценкой ⇒ оценка победителя оптимистически смещена (Cawley-Talbot 2010:
max шумных оценок смещён вверх). Полная nested-CV убирает это, но её ценность — защита внутреннего
HPO/FE, которых в M4 нет (HPO=M7, FE=M6). Решение: дать **дешёвую несмещённую финальную оценку**
победителя через once-touched outer-holdout; полную nested-CV отложить в M7.

## Рассмотренные варианты
1. **Полная nested-CV сейчас** (outer k_outer × inner k_inner). Цена мультипликативна; защищать
   нечего (нет inner HPO/FE). Отложен → M7.
2. **Holdout-leaderboard:** carve outer-holdout ОДИН раз; отбор (CV+band) на dev; победитель
   скорится на нетронутом outer-holdout → несмещённая финальная оценка. Цена O(1). **Выбран.**
3. Ничего (статус-кво). Отвергнут: оценка победителя остаётся смещённой — бьёт по north-star.

## Решение

### 1. Opt-in `CVConfig.outer_holdout: float` (аддитивно)
`outer_holdout: float = Field(default=0.0, ge=0.0, lt=1.0)` — доля данных, удерживаемая ОДИН раз для
несмещённой финальной оценки. `0.0` = выкл (дефолт; band из ADR-0026 — основной честный механизм,
holdout — опциональная добавка-оценка). `CVConfig` frozen/extra=forbid ⇒ поле аддитивно.
**Граничные guard'ы (фикс R2-completeness):** holdout слишком мал (`outer_holdout*n < min_rows`, напр.
`<2*n_classes` для proba-метрики) → `ConfigError` (иначе single-class holdout → `ValueError` в метрике);
для `timeseries` после выреза позднего окна dev должен вместить `n_splits` фолдов (+purge/embargo),
иначе `ConfigError` («недостаточно dev после holdout-выреза для n_splits»).

### 2. Scheme-aware carve (OOF-инвариант)
При `outer_holdout>0` composition вырезает outer-holdout **согласованно со схемой CV** (один фолд):
- `timeseries` → holdout = **последнее** окно по времени (+ purge/embargo до dev), без утечки будущего;
- `stratified`/`kfold` → стратифицированный/обычный одиночный split (seed-детерминирован);
- `group` → group-дизъюнктный holdout.
**Нарезка — за портом `CVSplitter`** (фикс R1-clean: leakage-sensitive carve не в composition root):
reuse `HoldoutSplitter` (классы/группы) и **первого/последнего окна `TimeSeriesSplitter`** (для
timeseries holdout = поздние времена с purge/embargo до dev — тот же проверяемый `validate_fold`
инвариант). Composition лишь оркеструет последовательность carve→`run_slice(dev)`→refit→score(holdout),
сам carve-инвариант живёт за портом. Dev = остаток.

### 3. Поток (OOF-инвариант, NFR-M4-3)
(1) carve индексов (dev, holdout) scheme-aware (§2) + `ds.take(dev_idx)`/`ds.take(holdout_idx)`;
(2) полный отбор — `run_slice` на **dev** (CV+OOF+band+тай-брейк) → победитель; (3) `refit_best`
победителя на **dev** → `Estimator`; (4) **composition строит временный** `FittedModel(estimator=
refit, schema, task, metric, classes, ...)` (как `facade.fit:96-104`) и зовёт `.score` на
**holdout**-`Dataset` ОДИН раз → `holdout_score` (`refit_best` возвращает `Estimator`, не score —
фикс R1-as-is; FittedModel — путь оценки). Outer-holdout НЕ участвует в CV/отборе/band/калибровке.
- **Семантика финальной модели (фикс R1-adv):** `holdout_score` — несмещённая оценка **процедуры**
  на dev-обученной модели. По умолчанию **отгружается dev-обученная модель** (holdout не «возвращается»
  в обучение). Опц. финальный refit на dev∪holdout для прода — отдельный явный шаг (тогда
  `holdout_score` относится к процедуре, не к отгружаемой модели); документируется.
- **TS-оговорка (фикс R1-adv):** для `timeseries` holdout = позднее окно, dev = ранние времена ⇒
  `holdout_score` смешивает selection-bias и **temporal drift**; интерпретировать как оценку «на
  будущем», не как чистую меру selection-смещения. Документируется.
`holdout_score` ложится в `SliceResult` + artifact-манифест (НЕ в frozen `LeaderboardEntry`; см.
ADR-0026 §5 о контейнере наблюдаемости), NFR-M4-7.
> **Реализация M4c (детали):** carve — функция `outer_holdout_carve` в `adapters/splitters.py` (за
> границей splitter-порта, фикс R1-CA-4): не-TS/не-group переиспользует `HoldoutSplitter`, group —
> `GroupShuffleSplit`, timeseries — позднее окно по `dataset.time()` с purge до dev. Скоринг — новый шов
> `FittedModel._score_dataset(ds)` (вынесённое ядро `score`, отдаёт **сырую** метрику в её ориентации,
> прямо сравнимую с `leaderboard_`); публичный `score(X,y)` остаётся обёрткой с sklearn-flip. Оркестрация
> (guard'ы min-rows/TS-остаток → `ConfigError`, `ds.take(dev)/take(holdout)`, единственный
> `_score_dataset(holdout)`) — в `facade._carve_holdout`/`fit`. `holdout_score` проводится фасадом в
> `SliceResult.holdout_score` (контейнер наблюдаемости), `FittedModel.holdout_score` (аддитивный
> манифест-ключ, `ARTIFACT_VERSION`=1) и публичный `holdout_score_` (None при `outer_holdout=0`).

### 4. Авто-выбор по размеру — отложен
Порог авто-переключения cv↔holdout по размеру данных/бюджету (FLAML `eval_method='auto'`) — НЕ в
M4 (явный opt-in достаточно; над-инженерия запрещена). Future-аддитив.

## Последствия
- **Положительные:** несмещённая финальная оценка победителя дёшево (один доп. split); OOF-инвариант
  соблюдён (holdout тронут один раз); честность видна (CV-score vs holdout-score). Готовит почву для
  M7 nested-CV.
- **Отрицательные/компромиссы:** один holdout — выше дисперсия оценки, чем nested-CV (на малых данных
  шумно — документируется; пользователь сам включает); «тратит» данные на holdout. Полная nested-CV —
  M7.
- **Влияние на слои:** carve + последовательность — `composition` (composition root оркеструет); скор —
  переиспользует `FittedModel.score`/`refit_best` (`application`). `core` — только аддитивное поле
  `CVConfig`. import-linter не нарушен.

## Проверки
- **«Тронут РОВНО один раз» — операционализировано (фикс R2-major):** (а) структурно — `holdout_idx`
  дизъюнктны со ВСЕМИ dev-фолдами И с refit-train (assert пересечений = ∅); (б) поведенчески — **spy/mock**
  на holdout-`Dataset`: `FittedModel.score` вызван ровно 1 раз, `ds.take(holdout_idx)` — 1 раз; composition
  НЕ передаёт holdout в `run_slice`/band/calibrate (статически) (FR-M4-9, NFR-M4-3).
- **Несмещённость — заданный генератор (фикс R2-major):** N кандидатов = общий базовый сигнал + i.i.d.
  фолд-шум; `anchor = argmax(CV)`; при фикс. seed `CV_score(anchor) − holdout_score(anchor) = δ > 0`
  (CV оптимистичен, holdout — нет). Порог δ зафиксирован seed'ом.
- `timeseries`+`outer_holdout`: holdout = поздние времена, dev = ранние, с purge/embargo (нет
  утечки будущего).
- `outer_holdout=0.0` (дефолт) — поведение M3 не меняется; `holdout_score` отсутствует.
- Воспроизводимость carve при фикс. seed (NFR-M4-2).
