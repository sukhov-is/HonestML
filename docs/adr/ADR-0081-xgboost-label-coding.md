# ADR-0081 — Кодирование меток для бустингов, требующих 0..K-1 (xgboost)

- **Статус:** Accepted (2026-06-14, v0.2)
- **Дата:** 2026-06-14
- **Драйверы:** баг, вскрытый сборкой multiclass-витрины (Otto, метки `Class_1..Class_9`): xgboost тихо (с WARNING) выпадал из честного лидерборда. Зависит от ADR-0020 (зоопарк бустингов), ADR-0022 (изоляция упавшего кандидата), ADR-0070 §2 (нативный round-trip + `_check_classes`).

## Контекст

`XGBClassifier` (xgboost ≥ 2.x, проверено на 3.2.0) убрал внутренний label-encoder и **отвергает любой таргет классификации, который не является непрерывным `0..K-1`**: строки, `{1,2}`, неконтинуальные int → `ValueError: Invalid classes inferred from unique values of \`y\``. `catboost`/`lightgbm` принимают произвольные метки нативно.

Адаптер ([boosting.py](../../src/honestml/adapters/boosting.py)) метки не кодировал, поэтому на любом классификационном таргете с метками ≠ `0..K-1` xgboost падал на `fit`, изолировался (ADR-0022 §1) и **исчезал из лидерборда** — с WARNING при настроенном логировании, но «честное сравнение» становилось неполным на широком классе задач. Баг прятался, потому что все три существующих binary-датасета (titanic/adult/fraud) и весь набор тестов нативного round-trip кормили xgboost метками `0..K-1` (`make_classification`); единственный тест на не-`{0,1}` метки (`test_lgbm_clf_full_path_non01_labels`) покрывал **только** lightgbm.

## Решение

### 1. Флаг бэкенда `requires_int_labels`
`_Backend.requires_int_labels: bool` (по умолчанию `False`; `XGBOOST=True`). Декларативно отмечает бэкенды, нативный классификатор которых принимает только коды `0..K-1`.

### 2. Кодирование/декодирование в обёртке классификатора
`_BoostingClassifier` для `requires_int_labels`-бэкендов:
- `fit`: `self._label_index = np.unique(y)` (сортированные исходные метки == порядок классов sklearn), таргет → `np.searchsorted(self._label_index, y)` (коды `0..K-1`); es-хвост (`y_val`) кодируется **тем же** отображением (`_encode_targets_apply`);
- `classes_` отдаёт **исходные** метки (`_label_index`), а не нативные коды;
- `predict` декодирует коды обратно: `self._label_index[pred]`;
- `predict_proba` без изменений — нативные столбцы уже в порядке `0..K-1` == сортированные исходные классы, поэтому `align_proba`/`_positive_index` ([slice.py](../../src/honestml/application/slice.py), [artifact.py](../../src/honestml/composition/artifact.py)) работают симметрично с catboost/lightgbm.

Не-`requires_int_labels` бэкенды (catboost/lightgbm) и все регрессоры — `_label_index=None`, метки проходят насквозь (байт-идентично прежнему поведению).

### 3. Нативный round-trip
Нативное тело xgboost (UBJSON) хранит только коды `0..K-1`. На загрузке `_NativeBoostingSerializer.load` ([serializers.py](../../src/honestml/adapters/serializers.py)) передаёт `manifest["classes"]` (глобальный порядок меток) в `from_native(classes=...)`, который для `requires_int_labels`-бэкенда восстанавливает `_label_index` → `classes_` совпадает с manifest, проходит `_check_classes` (ADR-0070 §2), `predict` декодирует. Это та же стратегия, что у нативного LightGBM-классификатора (`_NativeLgbmClassifier` берёт `classes_` из manifest). joblib-формат (дефолт) round-трипит обёртку целиком (включая `_label_index`) и так.

## Границы scope
Только классификация бустингов с нативным требованием `0..K-1` (сейчас — только xgboost). Не трогает регрессию, не-бустинговые модели, и нативные бэкенды (catboost/lightgbm). Изоляция кандидата (ADR-0022) остаётся — она по-прежнему ловит **настоящие** падения; этот ADR убирает один ложный класс падений (просто неудобные метки), а не саму изоляцию.

## Последствия

- **Положительные:** xgboost честно участвует в лидерборде на любых метках (multiclass-строки, binary-строки, `{1,2}`, неконтинуальные int) — «честное сравнение» больше не теряет модель из-за кодировки меток; multiclass-витрина показывает полный зоопарк; закрыт латентный баг во всех binary-кейсах с нестандартным таргетом.
- **Отрицательные/компромиссы:** одно дополнительное `searchsorted`-кодирование на fit (дёшево); нативный round-trip xgboost с нестандартными метками теперь зависит от `manifest["classes"]` (он там всегда есть для классификации).
- **Слои:** `adapters` (boosting/serializers); `core`/`application`/`composition` не изменены; import-linter не затронут.

## Проверки

Fake-бэкенд `requires_int_labels=True` со strict-int нативом (отвергает не-`0..K-1`): обёртка кодирует строки/`{1,2}`/неконтинуальные → `fit` не падает, `classes_`=исходные, `predict` декодирован, es-хвост кодируется тем же отображением; не-`requires_int_labels` бэкенд оставляет метки насквозь (`_label_index is None`). Реальный xgboost: участвует в multiclass-лидерборде на строковых метках и `y+1`; нативный round-trip (save→load) bit-идентичен на не-`{0,1}` метках для **всех** трёх семейств (xgboost/catboost/lightgbm) — покрытие, которого не было.
