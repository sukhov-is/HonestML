# ADR-0103: Non-inferiority margin в порт SignificanceTest и правило выбора размера каскада

- **Статус:** Accepted (реализован — 2026-07-07)
- **Дата:** 2026-07-07
- **Драйверы:** D-1 (адаптивность по мощности), D-3 (estimator-agnostic), D-5 (детерминизм),
  D-6 (слои). Источник: FR-1, FR-2, FR-6, NFR-1, NFR-2, NFR-4. Amendment к ADR-0100/0085.

## Контекст

Каскад (`_band_over_trajectory`) и `no_selection_gate` выбирают размер через **двустороннее**
членство в equivalence-band: точка допустима, если `(1−alpha)`-CI разницы Δ(точка−якорь)
**включает 0**. При низкой мощности (мало данных и/или нечувствительный estimator-agnostic
прокси ExtraTrees) CI широк, весь трек «неотличим», Occam берёт минимум — **чем ниже мощность,
тем агрессивнее прунинг** (бремя доказательства перевёрнуто). Эмпирика: titanic 44→1
(`seq_band.width=13`), adult 27→5 (−2.2 п.п. holdout). Нужен **односторонний** статвопрос:
«точка не хуже якоря более чем на margin». Порт `SignificanceTest` умеет только двустороннее
`equivalent(...) -> bool` (значимо: `runtime_checkable`, isinstance-контракт).

## Рассмотренные варианты

1. **Новый метод порта `noninferior(...) -> bool`.** Плюсы: два разных статвопроса — два
   явных метода; читаемо; ориентация по метрике инкапсулирована в адаптере. Минусы: расширение
   `runtime_checkable` Protocol ⇒ `NoSignificanceTest` + 6+ тест-фейков реализуют метод (R-2).
2. **Опциональный `margin` в существующем `equivalent(..., margin=None)`.** Плюсы: меньше
   поверхности (фейки добавляют kwarg). Минусы: имя `equivalent` конфлатит двустороннюю и
   одностороннюю семантику — вводит в заблуждение на statкорректном пути.
3. **Порт отдаёт сырую нижнюю границу `delta_lower_bound(...) -> float`, решение — в приложении.**
   Плюсы: композируемо. Минусы: float-контракт, вырожденные случаи и `NoSignificanceTest`
   (`n_boot=0`) отдают `−inf` — больше краёв; решение размазано.

## Решение

**Вариант 1** — новый метод порта, ориентация в адаптере.

**Контракт порта** (`core/ports/significance.py`):
```
def noninferior(self, pred_a, pred_b, y_true, *, alpha, margin,
                block_index=None, sample_weight=None) -> bool: ...
```
Семантика: `pred_a` (кандидат) **non-inferior** `pred_b` (референс) в пределах абсолютного
`margin ≥ 0` — нижняя односторонняя `(1−alpha)`-граница **ориентированного улучшения**
`improvement = orient·(score(pred_a) − score(pred_b))` ≥ `−margin`, где
`orient = +1 if metric.greater_is_better else −1`.

**Адаптер** `BootstrapSignificanceTest.noninferior` (`adapters/significance.py`):
переиспользует `_delta_distribution`/`_period_delta_distribution` (тот же сид, тот же массив
`finite`, что и `equivalent` — **новый RNG-проход не нужен**, NFR-1/NFR-2). As-is
`_delta = score(pred_b) − score(pred_a)`, значит `improvement_dist = −orient · finite`;
`LB = percentile(improvement_dist, 100·alpha)`; вернуть `LB ≥ −margin`. Вырожденные случаи
(консервативно к «не прунить», R-1): `finite.size == 0` → `False`; `ptp(finite) == 0` →
единственная ориентированная дельта ≥ `−margin`.

**`NoSignificanceTest.noninferior` → `False`** всегда (off-path ⇒ non-inferiority не применяется
⇒ чистый argmax, FR-6).

**Правило в band** (`core/selection_policy.py`): `SelectionPolicy` получает
`margin_frac: float | None = None`. `_band_members`: при `margin_frac is None` — как сейчас,
`test.equivalent(c, anchor)` (двусторонне — лидерборд/ensemble не трогаются, FR out-of-scope);
при `margin_frac is not None` — `margin = margin_frac · abs(anchor.score)` и
`test.noninferior(c.oof_pred[mask], anchor_pred, yt, alpha=policy.alpha, margin=margin, ...)`.
Якорь и Occam-тай-брейк не меняются (anchor = argmax по score).

**Off-значение `refine_tol` = текущее поведение (Major-1 ревью):** маппинг
`margin_frac = config.refine_tol if config.refine_tol > 0 else None`. ⇒ **`refine_tol = 0`
(домен `ge=0`) ⇒ `margin_frac=None` ⇒ двусторонний band = ТЕКУЩЕЕ поведение (opt-in-нейтраль)**;
`refine_tol > 0` ⇒ non-inferiority. Так «выкл» выразимо и совместимо.

**FS-каскад/гейт — область только ранкер-каскад (Major/minor ревью):** `_select_one` имеет и
`config`, и `policy`, поэтому **деривит FS-политику именно там** — `pol_fs =
policy.model_copy(update={"margin_frac": (config.refine_tol or None)})` — и передаёт её **только в
ранкер-каскадный вызов** `_band_over_trajectory` (`feature_compare.py:399`). **Wrapper-sequential
вызов (:361) остаётся `margin_frac=None`** (двусторонний — вне scope FR-1). `no_selection_gate`
получает `refine_tol` из `config` (новый параметр внутри `application`, слой не нарушается) и
деривит margin-политику так же. Глобальный проброс `policy` в composition не меняется.

**Ориентация (R-1)** проверяется явным knife-edge тестом на loss-метрике (rmse/log_loss) И
score-метрике (roc_auc): при заведомо худшем кандидате non-inferiority отклоняет на ОБЕИХ, при
эквивалентном — принимает; зеркальность знака ловится.

## Последствия

- **Положительные:** выбор размера адаптивен к мощности по построению — низкая мощность ⇒
  нижняя граница глубоко отрицательна ⇒ прунинг останавливается рано (titanic-режим лечится);
  высокая мощность/узкий CI ⇒ уверенный спуск (ieee/home-credit не ломаются). Ручка `refine_tol`
  осмысленна: «сколько метрики готов отдать за компактность». Ноль доп. фитов, детерминизм.
- **Отрицательные / компромиссы:**
  - **R-2:** расширение `runtime_checkable` порта — все реализации+фейки реализуют `noninferior`
    (внутренний порт, нет внешних; blast radius в 00-research).
  - **R-5 (titanic-тензия):** при **истинно низкой мощности** non-inferiority держит БОЛЬШЕ фич,
    чем прежний argmax-в-минимум. Это **желаемая консервативная сторона**: если данные не
    подтверждают компактность — честно не прунить. Легитимная «1 фича» достижима через
    mass-доминанту (ADR-0104: `Sex_freq` покрывает массу ⇒ низкий пол) и/или явный конфиг; но
    автоматический спуск-в-1 на шумных данных теперь не происходит — это и есть фикс.
  - **R-7:** Same-OOF residual (ADR-0085 §5) уже смещает band в «больше фич»; non-inferiority
    поверх — тот же консервативный вектор, взаимодействие документируется в amendment ADR-0100.
  - **Прокси-слепота НЕ лечится этим ADR** (узкий CI вокруг нуля ⇒ margin-тест проходит) —
    страхуется (эвристически) ADR-0104 (mass-floor); полное лекарство — F145 (out-of-scope).
    Обе части обязательны.
  - Fingerprint-чурн (`refine_tol` в конфиге) — NFR-6, раскрыть в CHANGELOG.
- **Влияние на слои/границы:** метод — в `core.ports`; реализация статистики — в `adapters`;
  правило — в `core.selection_policy` + `application`. Направление зависимостей не меняется,
  новых модулей нет ⇒ import-linter контракты KEPT (NFR-4).

## Проверки
- FR-1: тест — на широком CI компактная точка отклонена (спуск рано), на узком — принята (red-green).
- FR-2: тест — `no_selection_gate` ветирует subset хуже более чем на margin.
- FR-6: тест — `significance="off"` ⇒ выбор размера = прежний argmax.
- R-1: knife-edge тест ориентации на loss- и score-метрике.
- NFR-2: тест детерминизма double-run; массив дельт non-inferiority == тот же `finite`.
- NFR-4: `lint-imports` 3/3 kept; `isinstance(NoSignificanceTest(), SignificanceTest)`.

---
> Amendment к ADR-0100 (правило выбора размера каскада) и ADR-0085 §5 (Same-OOF band): при
> реализации — пометка в docs/adr/ADR-0100 и ADR-0085 со ссылкой на этот ADR.
