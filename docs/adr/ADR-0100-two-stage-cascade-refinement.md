# ADR-0100: Двухступенчатый каскад отбора признаков (ранкер-срез → backward-добивка)

- **Статус:** Accepted (реализован — 2026-07-06)
- **Дата:** 2026-07-06
- **Драйверы:** эксплуатация (churn, 1256 фич): одиночный ранкер-срез `importance`+`auto`
  оставил 499/1256 признаков, ручной референс (грубый фильтр ⊕ sequential BE) — 46 с лучшим
  holdout. Legacy-сценарии S01/S03/S07 (M6b 00-research) документировали именно двухступенчатые
  пайплайны, но продуктизация сделала стратегии взаимоисключающими.

## Контекст
Ранкер-срез (`select_features`: OOF-ранжирование → `apply_cutoff`) — грубый: `auto` = «выше
среднего» (1/n), `top_frac` — фиксированная доля. Обёртка `sequential` (ADR-0047) существует, но
это ОТДЕЛЬНАЯ стратегия с exhaustive-greedy O(n²) score_subset-вызовов — неподъёмна от сотен
признаков и не компонуется с ранкером. Честный выбор размера по траектории уже есть:
`_band_over_trajectory` (ADR-0083/0085/0086, Same-OOF + band + Occam).

## Рассмотренные варианты
1. **Оставить как есть** (стратегии взаимоисключающие). Не решает 499→46. Отвергнут.
2. **Каскад с пере-ранжированием на каждом шаге** (как ручной BE: важности из скорящего фита).
   Контракт `FitPredict`/`score_subset` не отдаёт важностей; расширение порта тянет через
   `oof_scorer`, адаптеры и все фейки. Отложен как Day-2.
3. **Каскад со статическим порядком стадии 1 + chunked-спуск + band-выбор.** Генерация траектории —
   чистая арифметика над agg-вектором (0 score_subset-вызовов); каждая точка честно скорится один
   раз в `_band_over_trajectory`. Кост O(длина траектории × n_splits). Риск статического порядка
   ограничен консервативностью band (якорь — более ранняя/широкая точка: «оставить больше», не
   over-prune). **Выбран.**

## Решение
1. **Точка врезки — одна:** ранкер-ветка `_select_one` (`application/feature_compare.py`):
   `aggregate_scores` (ранжируем ОДИН раз; реюз для среза и порядка спуска) → `apply_cutoff` →
   при `refine` и `len(subset) > floor` — `refine_trajectory` → `_band_over_trajectory` →
   `(subset, band, refine_meta)`. Все режимы (single/holdout/nested/per-fold/`_score_procedure`)
   наследуют каскад автоматически; per-fold честно меряет весь каскад внутри outer-фолда.
2. **M6b-путь перероутен:** `build.py` направляет одиночный ранкер в compare-путь при
   `feature_selection.refine`; `refine=False` — legacy `feature_ranker`-путь бит-в-бит.
3. **`refine_trajectory(subset, agg, *, max_features, drop_frac, min_features)`**
   (`application/feature_selection.py`): порядок — убывающий `agg` со stable-тай-семантикой
   `apply_cutoff`; `trajectory[0]` — НЕобрезанный набор выживших (band может ветировать вредную
   обрезку `max_features`; обрезанный top-k — следующая точка); далее шаги
   `max(1, ceil(len·drop_frac))` слабейших, кламп к полу `max(1, min_features)`; размеры строго
   убывают, каждая точка отсортирована по позиции колонки (FR-FS-7).
4. **Конфиг** (`FeatureSelectionConfig`): `refine: bool = True` (норма — «добивка всегда»),
   `refine_max_features: int | None = 200`, `refine_drop_frac: float = 0.05`; пол — общий
   `seq_min_features`; `seq_patience` не участвует (full-descent + band). Валидатор: явные
   `refine_*` при `refine=False` → ValueError (dead-config). Composition-WARNING: явный
   `refine=True` при чистом `strategy='sequential'` (обёртка — свой спуск).
5. **Cost-модель:** `_refine_steps` (детерминированная верхняя оценка длины траектории, зеркало
   `refine_trajectory`) добавляется к per-strategy `base` в `estimate_fs_refits`; budget-гейт
   (`resolve_fs_defaults`, ADR-0058) получает новую ступень: при провале даже holdout-пола сперва
   `refine=False` (WARNING + `refine_resolved_from="cost_budget"`), затем прежний floor-fail.
6. **Наблюдаемость:** `CompareOutcome.refine` / `FeatureSelectionReport.refine` / run-report
   `feature_selection.refine` = `{n_after_rank, n_after_refine, trajectory_len, capped}`; band
   рефайнмента едет по существующему каналу `seq_band`.

## Последствия
- **Положительные:** дефолтный FS даёт компактные наборы уровня ручного BE при O(k) OOF-оценок;
  честность — та же band+Occam машинерия; обрезка cap никогда не молчалива (якорь-вето).
- **Отрицательные/компромиссы:** смена результатов всех fs-ранов по умолчанию (откат
  `refine=False` бит-в-бит; guardrails: якорь `trajectory[0]` + `no_selection_gate`);
  fingerprint-чурн всех fs-конфигов (новые поля в дампе) — cold recompute, не stale-hit;
  статический порядок может оставить чуть больше фич, чем пере-ранжирование (Day-2).
- **Слои:** core (поля), application (траектория+каскад+cost), composition (роутинг, budget-ступень,
  WARNING). Контракты import-linter не затронуты.

## Проверки
- Траектория: строго убывающие сортированные подмножества; `trajectory[0]` = необрезанные выжившие;
  cap-точка; кламп к полу без overshoot; порядок = stable argsort как в `apply_cutoff`; детерминизм.
- Каскад достижим из single/holdout-режимов на фейках; band-argmax выбирает компактную точку при
  «уже — лучше» и ветирует cap при «шире — лучше»; wrapper-стратегии не затронуты.
- `refine=False`: роутинг, подмножества и `estimate_fs_refits` — байт-в-байт как до фичи.
- Budget-ступень: refine сбрасывается до floor-fail, провенанс в `fs_resolution`.
