# ADR-0089: Материализация нативного категориального входа в адаптере

- **Статус:** Proposed
- **Дата:** 2026-06-22
- **Драйверы:** D-4 / FR-1, FR-4; опирается на SPIKE-0004

## Контекст
Граница slice↔adapter передаёт `X: np.ndarray` float64 (ADR-0005). Из research:
- **LightGBM** принимает float64 + `categorical_feature=indices`; значения
  кастятся в `int32` (округление к 0), коды неотрицательны и целочисленны → каст
  без потерь.
- **CatBoost** в cat-колонках **не принимает float** (требует int/string) →
  передать float64-матрицу с `cat_features` нельзя; нужен int-каст cat-блока.

Менять float64-контракт границы для всех нельзя (ADR-0005, регрессия не-нативных).
Значит материализация нативного входа — **внутри адаптера**, в точке инъекции
`_make`/`fit` (`boosting.py:167-186, :220-283`), где сейчас собираются kwargs и
вызывается `model.fit(X, …)` без `cat_features`.

## Рассмотренные варианты
1. **Менять `design_matrix` на mixed-dtype / DataFrame** — ломает ADR-0005 и
   numpy-границу для всех моделей; отвергнуто.
2. **CatBoost через `Pool(data, cat_features=…)`** — официальный путь; принимает
   int-колонки; чисто инкапсулирует материализацию.
3. **CatBoost через numpy object-массив с int в cat-колонках** — работает, но
   копия всей матрицы в object дороже Pool.
4. **(выбрано) Per-backend материализация в адаптере, минимальная:**
   - **LightGBM:** передать `categorical_feature=self.categorical_indices` в `fit`
     (и `_es_fit`) поверх существующего float64 `X` — без копий/кастов.
   - **CatBoost:** построить вход с **int** cat-колонками. Базовый план — собрать
     `catboost.Pool(X, label=y, cat_features=self.categorical_indices, weight=…)`,
     приведя cat-колонки к int (численно равны кодам, т.к. неотрицательные целые
     во float64). Способ (Pool vs object-array) подтверждается SPIKE-0004 по
     паритету/скорости; fallback — object-array, если Pool неудобен с ES.

## Решение
- В `_BoostingBase` добавить per-backend материализацию категориального входа,
  активную только когда `self.categorical_indices` непуст (т.е. для native-capable
  обёртки с проставленными индексами, ADR-0088):
  - **LightGBM:** `model.fit(X, y, categorical_feature=idx, eval_set=…, …)`.
  - **CatBoost:** `model.fit(Pool(X_int, y, cat_features=idx, weight=sw), eval_set=Pool(X_val_int, y_val, cat_features=idx))`,
    где `X_int` — копия с cat-колонками, приведёнными к int (numeric-блок остаётся
    float). Предпочтительно `Pool`; финальный способ — за SPIKE-0004.
- Регуляризаторы категорий (ADR-0090) добавляются теми же kwargs `_make`.
- Когда `categorical_indices` пуст или модель не native-capable (xgboost, linear,
  baseline) — путь **без изменений** (текущий `model.fit(X, …)`), что гарантирует
  NFR-3.
- Та же материализация применяется на inference (ADR-0091): predict-вход
  пересобирается тем же способом по сохранённым индексам.

## Краевые случаи и инварианты входа
- **NaN в категориях не возникает.** Категориальный блок `design_matrix` — это
  целочисленные коды `CategoryTable`, где null уже отображён в `null_code` (валидный
  неотрицательный int), а unknown — в `unknown_code`; NaN там нет. Поэтому int-каст
  для CatBoost и `categorical_feature` для LightGBM безопасны (нет NaN→int UB).
  NaN возможны только в numeric-блоке, но те колонки не категориальны.
- **Пустые блоки.** `categorical_indices == []` (нет категорий на датасете/фолде) →
  материализация — **no-op**: CatBoost обучается как обычно (без `cat_features`/Pool
  по категориям), LightGBM — `categorical_feature=[]`. Пустой numeric-блок (все фичи
  категориальны) допустим — Pool/категориальная разметка строится по индексам.
- **`sample_weight`.** В CatBoost Pool передаётся `weight=sample_weight`, когда он
  не `None` (иначе опускается). **es-валидационный Pool — без весов** (как
  `eval_set` у LightGBM сейчас): es-хвост — стоп-эвристика, не взвешиваемая метрика.
- **Согласование importances/SHAP.** Нативная обработка не меняет позиционное
  соответствие `feature_importances_`↔`feature_names` (1:1 по столбцам), т.к. набор
  и порядок колонок прежние; `SupportsFeatureImportance`/`SupportsShap` работают над
  той же обёрткой. Выравнивание подтверждается тестом в реализации.

## Последствия
- **Положительные:** float64-граница slice↔adapter не меняется (ADR-0005 цел);
  материализация инкапсулирована в адаптере, где ей место; LightGBM-путь почти
  бесплатен; CatBoost-каст локален и детерминирован.
- **Отрицательные / компромиссы:** для CatBoost — дополнительная копия cat-блока в
  int на fit/predict (память/время; контролируется в SPIKE-0004, NFR-8); две
  ветки материализации (catboost/lightgbm) в адаптере.
- **Влияние на слои/границы:** целиком в `adapters/boosting.py`; портов и слоёв не
  затрагивает; `core`/`application` не знают о Pool/int-касте.

## Проверки
- SPIKE-0004: паритет train↔inference и round-trip ≤ 1e-6 (FR-4); CatBoost
  принимает выбранное представление; время в бюджете фолда (NFR-8).
- Тест: native-ветка вызывает `fit` с `cat_features`/`categorical_feature`,
  не-native — без них (FR-1, NFR-3).
- ES-путь (`_es_fit`) с нативным `eval_set` (Pool/categorical_feature) сходится (NFR-8).
