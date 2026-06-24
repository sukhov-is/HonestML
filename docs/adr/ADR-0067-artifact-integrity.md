# ADR-0067 — Целостность артефакта (checksum + опц. подпись)

- **Статус:** Accepted (M8, design-gate pending)
- **Драйвер:** DM-81 (доверие к артефакту) — FR-SRV-3; NFR-SRV-3 (← NFR-16, G-S1)
- **Связано:** ADR-0012/0024 (манифест), ADR-0030 (calibrator-файл в той же trust-границе), ADR-0065
  (`checksums` — аддитивный ключ манифеста; порядок version-gate → integrity → dispatch). Закрывает
  carry-in G-S1 → NFR-16.

## Контекст
`load_artifact` десериализует `model.joblib`/`calibrator.joblib` через joblib/pickle = **произвольное
исполнение кода** при загрузке недоверенного артефакта (G-S1/NFR-16). Сейчас защиты нет: только гейт
`artifact_version` (forward-compat, не integrity) + docstring-предупреждение «trusted source». Нет детекта
порчи (битый трансфер) или подмены-в-транзите. Нужен дешёвый детект целостности и **честное** описание
границы доверия (checksum ≠ authenticity).

## Рассмотренные варианты
1. **Только docstring-предупреждение (статус-кво)** — ❌ ни порча, ни подмена не детектятся.
2. **Полная асимметричная подпись по умолчанию** — ⚠️ ценно для authenticity, но тащит управление ключами
   (распространение/ротация) в OSS-библиотеку по умолчанию; неуместно как дефолт. Оставить **опциональным**.
3. **Integrity-манифест (SHA-256 всех файлов), проверка до `joblib.load` + опц. detached-подпись (hook)**
   — ✅ stdlib `hashlib`, нулевые новые зависимости; детектит **случайную порчу** и **наивную подмену** (без
   пересчёта сумм); **не** меняет pickle-trust.

## Решение (Вариант 3)

### §1 Контрольные суммы (обязательно, default) — ровно по присутствующим файлам
`save_artifact` пишет в манифест `checksums = {"algo": "sha256", "files": {name: hexdigest}, "manifest":
hexdigest}` по **фактически записанным** файлам (`schema.json`, `model.joblib`, `leaderboard.json`, и
`calibrator.joblib` **⇔** `calibrator_file != None` — правка R1: при `calibrator=None` ключа нет).
`checksums.manifest` — дайджест manifest-данных **без** самого блока `checksums` (корень целостности).
`hashlib` — stdlib, **нет новых зависимостей**.

### §2 Проверка на загрузке — порядок version-gate → integrity → dispatch (правка R1/R2)
`load_artifact`: (1) прочитать манифест; (2) **version-gate** (ADR-0065 §2, неизвестный major → отказ);
(3) **verify checksums** для всех заявленных файлов; (4) только потом **`model_type`-dispatch** + `joblib.load`.
Решения по **модели** (`model_type`/`model_file`) принимаются **после** verify. **Оговорка R2:** version-gate
(шаг 2) читает `artifact_version` из ещё непроверенного манифеста — это допустимо, т.к. он **compatibility-only,
не trust** (отвергает несовместимый формат, как и `model_type`-dispatch не доверяет — оба под той же
не-authenticity-границей §5); читать версию надо до verify, чтобы вообще понимать формат блока `checksums`.
**Anti-traversal (расширено R2):** не только `model_file`/`calibrator_file` (`.name`), но и **все ключи
`checksums.files`** — verify итерирует строго по `basename` (`.name`), отвергая имена с разделителями/`..`/
абсолютные/симлинки → `ArtifactIntegrityError(missing_file)`; читает только из каталога артефакта (path-traversal
на read закрыт). Несоответствие → **`ArtifactIntegrityError`** (новое boundary-исключение в
`honestml.core.exceptions`, реэкспорт через `honestml`, рядом с `SchemaValidationError`). Тест:
`checksum_file_outside_dir_rejected`.

### §3 Бэк-совместимость + таксономия отказа (правка R1) + сигнатуры (правка R2)
Артефакт без блока `checksums` (M2..M7) грузится с **WARNING** при `require_integrity=False` (default —
NFR-SRV-2/3); `load_artifact(dir, require_integrity=True)` → отказ при отсутствии сумм (строгий serving).
`ArtifactIntegrityError` различает причины (для NFR-SRV-5 и действенного сообщения, всегда с именем файла):
- `missing_checksums` — блока нет (legacy, под `require_integrity=True`);
- `missing_file` — файл заявлен в `checksums`, но отсутствует на диске;
- `digest_mismatch` — сумма не совпала (порча/наивная подмена).

WARNING — через библиотечный логгер (`get_logger`), фиксированный префикс/уровень `WARNING`, чтобы тест ловил
его по логам стабильно (правка R2); строгий тест предпочтительно ассертит `missing_checksums`-исключение, а не
текст лога. **Сигнатуры (back-compat, правка R2):** все новые параметры **keyword-only с дефолтами старого
поведения** — `load_artifact(directory, *, require_integrity=False, verify=None)`,
`save_artifact(model, directory, *, honestml_version=None, sign=None)`. Старые позиционные вызовы
`load_artifact(dir)`/`save_artifact(model, dir)` работают без изменений (тест back-compat).

### §4 Подпись — опциональный hook (authenticity)
Detached-подпись — задокументированный seam: `save_artifact(..., sign=callable)` пишет `signature` (по
`checksums.manifest`); `load_artifact(..., verify=callable)` проверяет. **По умолчанию off** — управление
ключами на стороне деплоера. Библиотека даёт механизм, не политику ключей.

### §5 Честная trust-граница (раскрытие — расширено R1)
Docstring/доки явно: sha256-самопроверка детектит **случайную порчу** и **наивную подмену** (которая не
пересчитала суммы). Она **НЕ аутентифицирует**: (а) злонамеренный автор embed-кода + валидную сумму; (б)
изощрённую подмену-в-транзите, которая **пересчитает** `checksums.manifest` под подменённый файл — самопроверка
пройдёт. Authenticity даёт **только подпись (§4)** над `checksums.manifest` доверенным ключом. Поэтому «load
only from a trusted source» **остаётся в силе**; pickle-RCE сокращает только не-pickle native/ONNX (M8b).
Никакого ложного чувства безопасности (R-SRVINT).

## Последствия
- **+** Случайная порча и наивная подмена детектятся по умолчанию; нулевые новые зависимости (hashlib).
- **+** Trust-модель сужена и **честно** описана (что детектит / чего нет — §5).
- **+** Аддитивно: `ARTIFACT_VERSION` остаётся 1; legacy грузится (WARNING). Таксономия отказа закрывает
  NFR-SRV-5 (причина в исключении).
- **−:** не устраняет pickle-RCE и authenticated-подмену без подписи (только native/ONNX в M8b + подпись) —
  раскрыто.
- **−:** небольшой оверхед хеширования на save/load (один проход по файлам) — приемлемо.

## Day-2 (committed)
- Встроенная асимметричная схема подписи + политика/ротация ключей (если потребуется из коробки).
- Сокращение RCE-поверхности нативными/ONNX-форматами → **M8b**.
