# ADR-0078 — Импьютер в Pipeline линейных/baseline: NaN не выбивает простые модели из зоопарка

- **Статус:** Accepted (2026-06-13, v0.2)
- **Дата:** 2026-06-13
- **Драйверы:** backlog finding #6 (NaN в numeric тихо сжимает зоопарк до бустингов); FR-2 (baseline+linear как честный референс), NFR-9 (аддитивность).
  Зависит от ADR-0013 (адаптеры на numpy-границе), ADR-0020 §2 (`handles_missing` per-дескриптор), дельта к #5 (`StandardScaler` в Pipeline).

## Контекст

`build._select_estimators` ([build.py:773](../../src/honestml/composition/build.py)) при `has_missing=True` выкидывает все модели с `handles_missing=False`. У встроенных это `baseline`/`linear` (`_LIGHT_CAPS`), и на самом классическом датасете (titanic) лидерборд остаётся «только бустинги»: значимостная полоса теряет простейшего члена-тайбрейк (north-star честности). ADR-0020 §2 отложил импьютер в M6; настало время.

Finding #5 уже ввёл паттерн `Pipeline([StandardScaler, model])` внутри `fit` линейных адаптеров — масштабирование живёт **в модели**, поэтому joblib/ONNX/parity-гейт его несут. Импьютация — такой же per-fold boundary-шаг, который обязан выполняться **до** масштабирования (NaN ломает `StandardScaler`).

## Рассмотренные варианты

1. **Глобальный импьютер на границе Reader** (новый `ImputationSpec` в схеме, FEConfig-флаг, версионирование артефактов). Отвергнут: (а) **менее честно** — глобальная медиана по всему train считается по строкам, которые в каждом фолде попадают в OOF-test → утечка статистики в OOF (та самая, ради избегания которой TE кросс-фитится, ADR-0041); (б) больше кода/риска (схема, сериализация, совместимость артефактов); (в) пришлось бы импьютировать и бустинги либо делать исключение, ломая их нативную NaN-обработку (ADR-0020 §2, нет train/serve skew).
2. **`SimpleImputer` в Pipeline линейных/baseline** (продолжение #5). **Выбран.**

## Решение

### 1. Импьютер впереди Pipeline линейных и baseline
`Linear{Classifier,Regressor}` и `Baseline{Classifier,Regressor}` ([estimators.py](../../src/honestml/adapters/estimators.py)) строят
`Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), (model)])`
(baseline — `Pipeline([SimpleImputer, Dummy])`, без scaler). Импьютер **фитится по фолду** на `fit_idx` (Pipeline строится в `factory()` и фитится на срезе train) → **leak-free**, как уже работает scaler. На данных без NaN — no-op (медианы посчитаны, но не применяются).
- `median` (робастна, sklearn-стандарт); индикатор пропусков **не добавляется** (минимализм; меняет ширину матрицы/схему).
- `feature_importances` читает `_fitted()[-1].coef_` — последний шаг по-прежнему модель, без изменений.

### 2. Флип capability `handles_missing=True`
Линейные/baseline теперь честно обрабатывают NaN → `_LIGHT_CAPS` ([registry.py](../../src/honestml/composition/registry.py)) и `_CLF_CAPS`/`_REG_CAPS` ([estimators.py](../../src/honestml/adapters/estimators.py)) ставят `handles_missing=True`. Гейт [build.py:773](../../src/honestml/composition/build.py) их больше не роняет.

### 3. Гейт остаётся для плагинов
Логика гейта и сообщение-подсказка ("impute the data or add a NaN-capable model") сохраняются: они корректны для сторонних/будущих компонентов, объявляющих `handles_missing=False`. Среди встроенных таких больше нет.

## Последствия

- **Положительные:** baseline/linear участвуют на NaN-данных; значимостная полоса сохраняет простого члена; нулевая конфигурация (как #5); ONNX/joblib несут импьютер в Pipeline; бустинги не тронуты.
- **Отрицательные/компромиссы:** на данных без NaN импьютер — лишний (дешёвый) шаг; ветка ошибки «все модели выбиты» теперь достижима только плагинами (не мёртвая — контракт плагинов).
- **Слои:** изменения только в `adapters`/`composition`; import-linter не затронут.

## Проверки

`build_default_components(has_missing=True)` **сохраняет** `linear`/`baseline` и не логирует NaN-варнинг; на NaN-данных линейный/baseline обучаются и предсказывают; импьютация — пер-фолд (медианы из train-среза, не из всего датасета); `models=("linear",)` + `has_missing=True` больше не падает; per-kind `capabilities.handles_missing` — `True`.
