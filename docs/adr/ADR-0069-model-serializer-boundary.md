# ADR-0069 — Граница `ModelSerializer` и реестр `model_type`

- **Статус:** Accepted (M8b-design, design-gate). Драйверы: DM-B1, DM-B5, DM-B6. Питается SPIKE-0003.
- **Контекст:** M8-фундамент (ADR-0065 §3) объявил `model_type` несущей точкой диспетчеризации, но реализовал
  только `joblib`: `load_artifact` отбивает прочее жёстким `if model_type != "joblib": raise` (artifact.py:296),
  а `save_artifact` хардкодит `"model_type": "joblib"`. M8b добавляет нативные форматы (xgb/cat/lgbm) и
  ONNX-канал. Растить `if/elif` из конкретных библиотек в composition root — нарушение OCP и риск затащить
  тяжёлые/опциональные импорты в `load_artifact`-ядро (а значит и в slim-конус). Нужна абстракция.

## Рассмотренные варианты
1. **`if/elif model_type` прямо в `load_artifact`/`save_artifact`.** Просто, но каждый новый формат правит ядро,
   тащит lib-импорты в модуль артефакта, ломает OCP и грозит slim-конусу. ❌
2. **Сериализатор как поле `ComponentDescriptor`.** Дескриптор знает, как строить эстиматор, — но на `save` у нас
   фитнутый объект, а не дескриптор; маппинг «объект → дескриптор» хрупок. ❌
3. **Порт `ModelSerializer` + реестр `model_type → адаптер`** (выбран). Граница, обобщающая seam: ядро артефакта
   не знает конкретных форматов, адаптеры лениво тянут библиотеки, новый формат — новый адаптер без правки ядра.

## Решение

### §1. Порт `ModelSerializer` (core/ports) — чистый Protocol
```
class ModelSerializer(Protocol):
    model_type: str                                  # "joblib" | "xgboost" | "catboost" | "lightgbm"
    def can_serialize(self, estimator: Estimator) -> bool: ...
    def save(self, estimator: Estimator, directory: Path) -> ModelFiles: ...   # пишет файл(ы), вернёт basenames
    def load(self, directory: Path, manifest: Mapping[str, Any]) -> Estimator: ...
```
Живёт в `honestml/core/ports/` — **без** импортов адаптеров/библиотек (core-independence KEPT). `ModelFiles` —
лёгкий value (basename(s) + опц. требуемый рантайм-extra), кладётся в манифест.

### §2. Реестр и оркестрация (composition)
Реестр — упорядоченный список сериализаторов; **`JoblibSerializer` — catch-all дефолт** (`can_serialize`≡True,
последний). `save_artifact(..., model_format="joblib")` → выбирается joblib (M8-поведение **байт-в-байт**);
`model_format="native"` → первый сериализатор с `can_serialize(estimator)==True`, иначе прозрачный фолбэк на
joblib (sklearn-модели). `load_artifact` диспетчеризует по `manifest["model_type"]` → `registry[model_type]`;
**неизвестный тип → `SchemaValidationError`** (сохраняем M8-таксономию и тест `test_unknown_model_type_rejected`;
новый `ArtifactFormatError` рассмотрен и отклонён как YAGNI-расширение публичной поверхности). **Правка R2:**
поскольку `SchemaValidationError` теперь покрывает и форматные случаи (неизвестный `model_type`/`model_format`,
«baseline/ensemble не ONNX-exportable», «unexpected ONNX output schema»), её docstring в `core/exceptions.py`
**расширяется** перечислением этих случаев — публичный контракт исключения должен соответствовать фактическому
использованию (нового класса не вводим, back-compat сохранён).

### §3. Доступ к нативной модели — role-interface `SupportsNativeModel` (core/ports)
```
class SupportsNativeModel(Protocol):
    native_format: str            # "xgboost" | "catboost" | "lightgbm"
    def native_model(self) -> Any: ...   # нижележащий sklearn-API booster
```
— объявляется как отдельный `runtime_checkable` Protocol в `core/ports` (рядом с `SupportsShap`), возвращает
`typing.Any`. Реализуют его **только** бустинг-обёртки (`_BoostingBase`; минимальная добавка — публичный
read-only аксессор к тому, что сейчас приватный `_model`); `Baseline`/`Linear` — **нет**. `can_serialize`
нативного сериализатора — это `isinstance(estimator, SupportsNativeModel)`; не-матч → joblib-fallback. Порт
`Estimator` **не меняется** и `native_model` в его `__all__` не добавляется — это опциональная capability
(как существующие `SupportsFeatureImportance`/`SupportsShap`).

### §4. Контракт-инварианты (наследуются от ADR-0065/0067, обязаны сохраниться)
- `ARTIFACT_VERSION=1`; новые манифест-ключи (`required_extra` + формат-метаданные) — **аддитивны**,
  читаются `manifest.get(key, default)`.
- Любой файл, записанный сериализатором, попадает в `checksums.files` (по basename) и проходит существующий
  `_verify_integrity` **до** десериализации; `model_file` на загрузке конфайнится `Path(...).name` (anti-traversal).
- Порядок на загрузке неизменен: version-gate → integrity-verify → **`model_type`-dispatch** → `serializer.load`.

### §5. Слои (import-linter, 3 контракта KEPT)
Порт+role-interface — в `core`; адаптеры-сериализаторы (лениво импортят библиотеки) — в `adapters`; реестр и
оркестрация (`save_artifact`/`load_artifact`) — в `composition`. `application` не затрагивается
(`usecases-independent-of-adapters` KEPT). Зависимости строго внутрь.

## Последствия
- **+** OCP для форматов: новый `model_type` — новый адаптер, ядро артефакта не меняется; slim-конус защищён
  (joblib-путь не тянет ничего нового; нативные lib-импорты — лениво в адаптере).
- **+** Дефолт `joblib` неизменен → M8-артефакты и тесты зелёные без правок ожиданий (NFR-SER-1).
- **−** Новый порт + role-interface + публичный read-only аксессор на бустинг-обёртках (минимальная добавка).
- **Day-2:** реестр сериализаторов — точка для будущих форматов (PMML/ONNX-load-back, если понадобится); версия
  рантайма в манифесте — основа для понятного отказа при отсутствии библиотеки (ADR-0070 §6).
