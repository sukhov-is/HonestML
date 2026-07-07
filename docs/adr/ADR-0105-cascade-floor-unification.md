# ADR-0105: Унификация пола каскада (`max(min_features, seq_min_features, mass_floor)`)

- **Статус:** Accepted (реализован — 2026-07-07)
- **Дата:** 2026-07-07
- **Драйверы:** D-2, D-4. Источник: FR-4, NFR-3. Amendment к ADR-0100; включает F126/F127.

## Контекст

Сейчас **два разных пола** (research Q6): `apply_cutoff` использует `min_features`
(`feature_selection.py:285`), а спуск каскада `_select_one`/`refine_trajectory` — только
`max(1, seq_min_features)` (`feature_compare.py:389`, `feature_selection.py:250`). Документированный
пол ranker-пути `min_features` **в каскаде игнорируется**: конфиг с `min_features > seq_min_features`
может отгрузить subset ниже `min_features`. ADR-0104 добавляет третий источник — `mass_floor`.
В рабочем дереве уже применены F126 (cap клампится `max(max_features, floor)`) и F127
(`_refine_steps` поднимает `k0` к `min_features`) — этот ADR их поглощает и завершает унификацию.

## Рассмотренные варианты

1. **Единый descent-пол `max(min_features, seq_min_features, mass_floor)` в `_select_one`,
   `apply_cutoff` не трогать.** Плюсы: `min_features` учтён в каскаде; single-cut путь
   (`select_features`) без изменений (R-4); один источник истины для descent. Минусы: описания
   полей надо дописать (роль в каскаде).
2. **Слить `min_features` и `seq_min_features` в одно поле.** Отвергнут: ломает публичный конфиг
   и single-cut семантику; избыточный breaking-change.
3. **Кросс-валидатор `refine_max_features >= seq_min_features` вместо клампа.** Частично (F126
   уже клампит); валидатор не покрывает `min_features`/`mass_floor` — недостаточно.

## Решение

**Вариант 1.** В `_select_one` (`feature_compare.py:389`):
```
mass_floor = _mass_floor(agg, config.refine_min_mass)          # ADR-0104
floor = max(1, config.min_features, config.seq_min_features, mass_floor)
```
Этот `floor` пробрасывается в `refine_trajectory(min_features=floor)` (проброс уже есть) и в
short-circuit `if not config.refine or len(subset1) <= floor`. `apply_cutoff` (single-cut,
`select_features`) **не меняется** — продолжает использовать `min_features` (изоляция R-4).

**Cost-оценка (`_refine_steps`, NFR-3):** остаётся a-priori — использует
`max(1, min_features, seq_min_features)` (F127 уже поднял `k0`/floor к `min_features`), БЕЗ
`mass_floor` (data-dependent, только поднимает пол ⇒ оценка остаётся upper bound, ADR-0104).
F126-кламп cap к полу — согласован с новым floor.

**Description полей:** `min_features` и `seq_min_features` дописать — оба участвуют в поле
каскада как `max(...)`; `seq_min_features` остаётся полом спуска wrapper-sequential.

## Последствия

- **Положительные:** один источник истины для descent-пола; `min_features` больше не игнорируется
  каскадом; F126/F127 завершены; `mass_floor` вплетён согласованно.
- **Отрицательные / компромиссы:**
  - Изменение результата каскадных ранов с `min_features > seq_min_features` (раньше спускались
    ниже `min_features`) — часть общего fingerprint-чурна (NFR-6).
  - **R-4:** `apply_cutoff` вызывается и вне каскада (`select_features:203-229`) — НЕ трогаем,
    чтобы не сдвинуть single-cut путь; тест на изоляцию.
- **Влияние на слои/границы:** только `application` (feature_compare/feature_selection); чистая
  арифметика над полями конфига + `agg`. Порты/adapters не трогаются. Import-linter KEPT.

## Проверки
- FR-4: тест — `min_features > seq_min_features` ⇒ каскад не отгружает ниже `min_features`;
  пол = `max` из трёх источников.
- R-4: тест — single-cut путь (`refine=False`/`select_features`) байт-идентичен (не задет).
- NFR-3: тест — `_refine_steps`/`estimate_fs_refits` a-priori upper bound при активном mass-floor.

---
> Поглощает F126/F127 (уже в рабочем дереве) и завершает пол-унификацию для F142. Amendment к
> ADR-0100 §3-4 (пол descent) при реализации.
