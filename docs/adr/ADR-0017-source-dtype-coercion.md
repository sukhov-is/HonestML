# ADR-0017 — Source-dtype в `CategoryTable` и value-preserving coercion на inference

- **Статус:** Proposed
- **Дата:** 2026-06-06
- **Драйверы:** D1 (корректность без форка), D6 (закалка); FR-1..5, NFR-1..5; RF-1..6.
- **Воркстрим:** FR-F (дельта к M1a). Расширяет **ADR-0005**; закрывает committed
  follow-up **ADR-0014** §Детерминизм п.4.

## Контекст

`CategoryTable` строит ключи как `str(v)` на train и `cast(Utf8)` на inference
(`schema.py:54`, `polars_dataset.py:102`), не фиксируя dtype источника. Когда train
прочитан как `Int64` (parquet), а inference — как `Float64` (csv эвристикой polars),
ключи расходятся (`"1"` vs `"1.0"`) → коды схлопываются в `unknown_code` → **тихий
train≠inference дрейф** (F1 review B-1). ADR-0014 понизил гарантию до best-effort и
маршрутизировал полный фикс сюда: это **изменение M1a-контракта** (сериализуемая
`CategoryTable`), поэтому оформляется отдельным ADR.

Прод-кодирование выполняется `_encode_expr` (адаптер polars); `CategoryTable.encode`
в проде не вызывается — это reference-оракул (см. 00-research §5).

## Рассмотренные варианты

1. **Канонический строковый ключ, dtype-независимый** — на fit и encode нормализовать
   представление (целочисленный float `1.0`→`"1"`). (−) меняет уже существующие ключи
   (для float-категорий было `"1.0"`); (−) эвристика (что с `1.5`, экспонентой, локалью);
   (−) ломает round-trip старых схем и доменно конфликтует со строковыми категориями
   вида `"1.0"`. Отклонён: меняет контракт там, где не нужно, и хрупок.
2. **Параллельный `dict[str,str]` source-dtypes в `FeatureSchema`** — не трогает
   `CategoryTable`. (−) второй источник истины, который надо держать синхронным с
   `categories`; легко рассинхронизировать при `select`/`with_categories`. Отклонён:
   нарушает когезию (dtype — свойство именно той таблицы, что владеет кодами).
3. **Source-dtype в `CategoryTable` + value-preserving coercion в адаптере** —
   таблица, владеющая кодами, владеет и dtype, под который коды зафиксированы;
   приведение представления — на границе (адаптер). **Выбран.**

## Решение

Вариант **3**.

### 1. Контракт `core.CategoryTable` (M1a contract change, аддитивный)
Добавляется поле:
```python
source_dtype: str | None = None   # непрозрачный канонический токен train-dtype; None = best-effort
```
- **Аддитивно-опционально:** старые `schema.json` без поля грузятся (`None`), поведение
  идентично прежнему (FR-4); `ARTIFACT_VERSION` не меняется (см. operational).
- **Forward-compat — явно:** `model_config = ConfigDict(frozen=True, extra="ignore")`
  у `CategoryTable` (раньше `extra` не задавался и держался на дефолте pydantic `ignore`).
  Делаем инвариант явным, чтобы старый код, читая новый JSON с `source_dtype`, гарантированно
  игнорировал ключ (а не зависел от версии pydantic). Подтверждено эмпирически (pydantic 2.11.4).
- `core` **не интерпретирует** токен — лишь хранит/сериализует строку (NFR-1). Парсинг
  токена и каст — забота адаптера (polars-зона).
- `fit` получает токен от вызывающего адаптера (он знает polars-dtype):
  `CategoryTable.fit(values, *, source_dtype: str | None = None)`. Без аргумента →
  `None` → ветка coercion не активируется (эквивалентность-тест не затрагивается, RF-5).
- **Токен берётся ТОЛЬКО из канонического словаря** (§2), не из сырого `str(dtype)`.
  ⚠️ Не путать с `TypingDecision.source_dtype` (`reader.py:38`) — то поле хранит сырой
  `str(dtype)` для диагностики авто-типизации и имеет иную семантику; `_fit_categories`
  должен использовать словарный токен, иначе теряется изоляция от repr-дрейфа (RF-3).

### 2. Канонический словарь токенов (адаптер)
В адаптере — фиксированный двусторонний словарь `polars dtype ↔ токен` из стабильного
вокабуляра: `"int8".."int64"`, `"uint8".."uint64"`, `"float32"/"float64"`,
`"string"`, `"categorical"`, `"boolean"`. Reader пишет канонический токен (а **не**
`str(pl.Int64)`), encode читает его обратно. Токен вне словаря (новый/незнакомый) →
coercion-skip (best-effort, без падения) — изолирует контракт от repr-дрейфа polars
между версиями (RF-3).

### 3. Value-preserving coercion (адаптер `_encode_expr`)
Coercion применяется **только когда токен — целочисленный** (единственный класс
дрейфа: float-категорий авто-инференс не делает, строки уже стабильны). Для строковых/
категориальных/float токенов и `None` — текущее поведение (`cast(Utf8)`), регресса нет
(FR-5).

**Строковый ключ строится в каждой ветке отдельно** — нельзя смешивать `Int64` и
`Float64` в одном `when/then/otherwise`: polars приводит ветки к общему супертипу, и
для Float-входа (ровно csv-сценарий) `casted=1` повысился бы обратно до `1.0` → `"1.0"`
→ unknown, отменяя весь coercion (blocker R-1, проверено на polars 1.41.2). Поэтому
обе ветки стрингуются **до** `when`:
```python
target = _INT_TOKEN_TO_DTYPE.get(table.source_dtype)   # None → ветка пропускается
if target is not None:
    col = pl.col(col_name)
    casted = col.cast(target, strict=False)            # нерепрезентабельное → null
    # round-trip-гвард: значение целочисленно представимо в target ⇒ ключ из casted,
    # иначе оригинальный ключ (→ не совпадёт с train-ключом → unknown_code).
    # strict=False на сравнении: нечисловой Utf8-вход → null → preserved=false (без throw).
    preserved = casted.is_not_null() & (casted.cast(pl.Float64) == col.cast(pl.Float64, strict=False))
    as_str = pl.when(preserved).then(casted.cast(pl.Utf8)).otherwise(col.cast(pl.Utf8))
else:
    as_str = pl.col(col_name).cast(pl.Utf8)
# далее без изменений: null→null_code (as_str.is_null()), replace_strict(mapping, default=unknown_code)
```
- Целочисленный дрейф `1.0`→`1`→`"1"` совпадает с train (FR-2). Обе ветки уже `Utf8`
  ⇒ супертип-промоции нет (закрывает R-1).
- Дробное `1.5`: `casted=null` → `preserved=false` → `col.cast(Utf8)="1.5"` ∉ mapping →
  `unknown_code` (FR-3) — **не** обрезается до `"1"` (закрывает RF-2). NaN/inf → `null`
  при `cast(target, strict=False)` → тоже `unknown` (как и в текущем polars-пути; null
  входа по-прежнему → `null_code`).
- Ветка рассчитана на **числовой/строковый** фактический `col` (Int/Float/Utf8); прочие
  фактические dtype при целочисленном токене не ожидаются — деградация безопасна (→ unknown).
- Известное ограничение: целые > 2^53 теряют точность в round-trip через Float64
  (RF-4) — допустимо для low-cardinality категориальных целых; задокументировано.

### 4. Слой (NFR-1)
`CategoryTable.source_dtype` — строка в `core`; интерпретация токена, polars-каст и
round-trip-гвард — в адаптере (`polars_dataset._encode_expr`, токен-словарь там же или в
соседнем adapter-модуле). `Reader._fit_categories` проставляет токен через словарь.
`CategoryTable.encode` (reference) остаётся в string-домене — coercion-семантика
принадлежит представлению (границе), а не ядру; новый property-тест проверяет
coercion напрямую (int-фрейм ≡ float-фрейм целых → равные коды), а не через оракул.

## Последствия

- (+) Гарантированные стабильные коды train↔inference при int↔float дрейфе (csv↔parquet)
  — закрыт тихий скоринг-баг; xfail-регресс снимается и становится зелёным.
- (+) Безопасность: нерепрезентабельные значения → `unknown_code`, не чужой код.
- (+) Полная обратная совместимость: старые артефакты грузятся и работают как прежде;
  `ARTIFACT_VERSION` стабилен; контракт расширен аддитивно.
- (+) Слои чисты: `core` без polars; coercion в адаптере; токен изолирован словарём.
- (−) `CategoryTable` усложняется одним опциональным полем + ветка в `_encode_expr`.
- (−) RF-4 (целые > 2^53) — принятый known-limitation.
- Связь: расширяет ADR-0005 (CategoryTable теперь несёт train-dtype); закрывает
  ADR-0014 follow-up; ADR-0005/0014 получают кросс-ссылки.

## Проверки

- `test_csv_int_float_dtype_drift_known_limitation` снят с `xfail` и зелёный (FR-2).
- Unit: дробное против целочисленного train → `unknown_code` (FR-3).
- Unit/round-trip: `CategoryTable(source_dtype=...)` JSON round-trip; схема без поля →
  `source_dtype is None` и идентичное кодирование (FR-1, FR-4).
- **Forward-compat (R-1 ревью):** старый `CategoryTable` (без поля) десериализует JSON
  **с** `source_dtype` и игнорирует ключ → регресс-тест (FR-4, явный `extra="ignore"`).
- Property: **Float64-фрейм** целых значений ≡ Int64-фрейм → равные коды (а не Utf8,
  чтобы тест ловил supertype-промоцию R-1) (NFR-2, FR-2); существующий
  `test_polars_encoding_equivalence` зелёный без правок (FR-5, RF-5).
- `lint-imports` 3/3 KEPT; `core/schema.py` без polars (NFR-1).
