# ADR-0073 — MLflow-адаптер: клиентская дисциплина и маппинг run_report

- **Статус:** Accepted
- **Драйверы:** DM-C3, DM-C4, DM-C5 (02-drivers.md)
- **Зависит от:** ADR-0072 (порт и payload)

## Контекст

Адаптер живёт в библиотеке, встраиваемой в чужой процесс: fluent-API mlflow
(`set_tracking_uri`/`set_experiment`/`start_run`) мутирует процесс-глобальное состояние
(active-run-stack стал thread-local только с 2.18) — для библиотеки это побочные эффекты
на пользовательский код (DM-C4). Client-side validation mlflow жёсткая: ключ ≤250,
param value ≤6000, `log_batch` ≤100 params за вызов, param immutable в рамках run —
превышение → `MlflowException` (00-research §3).

## Варианты

1. **`MlflowClient(tracking_uri=...)` + явный `run_id`, ни одного fluent-вызова.** ✅
2. Fluent `mlflow.start_run(...)` — проще, но мутирует глобальное состояние и
   active-run пользователя; floor пришлось бы поднять до 2.18. Отклонено.
3. Гибрид «если есть `mlflow.active_run()` — логировать в него» — удобство ценой
   недетерминированного владельца рана и param-immutability-конфликтов с пользовательскими
   параметрами. Отклонено (Day-2 при спросе).

## Решение

### §1. Класс и гейт

`src/honestml/adapters/tracking.py` → `MlflowTracker(experiment="honestml",
tracking_uri=None, run_name=None, tags=None)` — все поля `TrackerConfig`, **кроме
`backend`** (он — ключ диспетчеризации `_resolve_tracker` и в адаптер не форвардится:
`config.model_dump(exclude={"backend"})`). Дубль дефолтов в конструкторе осознанный —
instance-форма должна работать standalone; источник истины для config-формы —
`TrackerConfig` (NFR-TRK-6). Конструктор: (а) гейтит
`find_spec("mlflow") is None → MissingDependencyError("mlflow")` — fail-fast и для
instance-формы, и для config-формы (composition не дублирует гейт); (б) валидирует
пользовательский вход: tag-ключи (str, ≤250, не из резервированного namespace
`honestml.*`, без коллизий после санитизации), непустой `experiment` (+ `min_length=1`
на `TrackerConfig`), `run_name: str | None` — плохой вход падает
до обучения, а не теряет трекинг после. Модуль import-light: `mlflow` импортируется
только внутри `log_run`. `mlflow-skinny` предоставляет тот же модуль — гейт его
принимает; extra ставит полный `mlflow>=2.9` (floor не меняется: клиентский поднабор
стабилен и старше).

### §2. `log_run(report)` — последовательность

1. `from mlflow.tracking import MlflowClient` (в теле метода);
   `client = MlflowClient(tracking_uri=self._tracking_uri)` — `None` уходит в
   собственное разрешение mlflow (env `MLFLOW_TRACKING_URI` → `file:./mlruns`).
2. Experiment по имени идиомой **create-then-get** (закрывает гонку двух параллельных
   fit с одним именем — штатный sweep-паттерн): `get_experiment_by_name(...)`; если
   None → `create_experiment(...)` в try/except already-exists → повторный
   `get_experiment_by_name(...)`. Клиентский аналог `set_experiment` без глобального
   эффекта; на FileStore гонка двух create неустранима полностью (известное
   ограничение mlflow) — идиома закрывает SQL/REST-сторы.
3. `create_run(experiment_id, run_name=...)` — **новый run на каждый fit**
   (param-immutability не срабатывает по построению); теги уходят батчами на шаге 4
   (единый чанкуемый канал `log_batch`, см. §3 — синхронизировано с реализацией).
4. Маппинг (§3: tags/params/metrics) батчами `log_batch`; полный отчёт —
   `client.log_dict(run_id, report, "run_report.json")`.
5. `set_terminated(run_id)` → FINISHED; INFO-лог `run_id`/experiment/uri.
   Cleanup при сбое: шаги 4–5 в `try/except BaseException` — попытка
   `set_terminated(run_id, "FAILED")` (её собственный сбой подавляется, чтобы не
   подменить первопричину в фасадном WARNING) и **re-raise**; `KeyboardInterrupt`
   при этом доходит до пользователя (фасад его не глушит, ADR-0072 §2).

### §3. Маппинг run_report → MLflow

| MLflow | Из отчёта | Правила |
|---|---|---|
| tags | `honestml.version`, `honestml.fingerprint`, `honestml.winner`, `honestml.run_manifest_version` + пользовательские `tags` из конфига | namespace `honestml.*`; значения усекаются до 8000 с WARNING; чанк ≤100 на `log_batch` |
| params | `config` (полный `RunConfig`-дамп), уплощённый dot-join (`cv.n_splits`, `budget.mode`, …) | значения `str(...)`, пре-усечение до 6000 с WARNING; ключи санитизируются и ≤250 (дамп короче по построению); чанки ≤100 на `log_batch` |
| metrics | `score.<model_id>` (каждая строка лидерборда), `winner_score`, `holdout_score` (если не None), `time.<group>.<stage>` (timings) | ключи санитизируются под алфавит MLflow `[/\w.\- ]` (plugin-model_id может нести произвольные символы — замена на `_`); только конечные float, нефинитные пропускаются с debug-логом; step=0 |
| artifact | весь `report` → `run_report.json` | `log_dict`; единственный полный провенанс — без дублирующих вычислений (DM-C3) |

Адаптер читает только перечисленные ключи и игнорирует незнакомые — аддитивная
эволюция отчёта (ADR-0033) не ломает его; отсутствие опционального блока (`hpo=None`)
просто не порождает записей. **Residual risk (NFR-TRK-6, принято):** schema-level
имена признаков заказчика входят в `run_report.json` (блок `feature_selection`) —
строк данных там нет, но для чувствительных имён колонок пользователю следует выбирать
приватный store или не включать трекер.

### §3a. Run-verified факт: FileStore в maintenance mode (mlflow ≥3.13)

Прогон показал (проверено прогоном, июнь 2026): mlflow 3.13 переводит filesystem-бэкенд
в maintenance mode — `FileStore` поднимает `MlflowException`, пока не выставлен env
`MLFLOW_ALLOW_FILE_STORE=true`; рекомендованный локальный бэкенд — `sqlite:///mlflow.db`.
Следствия: (а) юниты используют hermetic file-store в `tmp_path` с этим env-флагом
(фикстура `file_store`); (б) для пользователя `tracking_uri=None` уходит в собственное
разрешение mlflow — на свежем 3.13+ без настройки оно упадёт с понятным сообщением
самого mlflow (наш fail-soft переведёт его в WARNING с первопричиной); адаптер ничего
не маскирует и не выставляет env за пользователя (DM-C4 — никаких глобальных эффектов).

### §4. Тестовая стратегия (гейтинг как onnx)

mlflow не в dev-наборе → happy-path юниты начинаются с `pytest.importorskip("mlflow")`
и используют file-store во временной директории (`tracking_uri=f"file:{tmp_path}"` +
env `MLFLOW_ALLOW_FILE_STORE=true`, §3a); в plain-suite они skipped. В CI их гоняет
выделенная джоба `extras` (`pip install -e ".[dev,boosting,onnx,mlflow]"` — она же
исполняет onnx-гейтед тесты M8b); локальная верификация — эфемерно
(`uv run --with "mlflow>=2.9" pytest ...`). Mlflow-независимые свойства (стаб-инстанс
через фасад, fail-soft, гейт, формы opt-in, фингерпринт, конусы) — в обычном suite.
Юниты лимитов (усечение/чанкинг/нефинитные) тестируют приватные хелперы маппинга
без стора.

## Последствия

- (+) Ноль глобальных эффектов в процессе пользователя (проверяется тестом
  неизменности `mlflow.get_tracking_uri()`).
- (+) Стабильный клиентский поднабор API → floor `>=2.9` сохранён, 3.x совместим.
- (−) Пользователь, ждущий «авто-подхват» своего `mlflow.start_run()`, должен передать
  `tracking_uri`/experiment явно (гибрид — Day-2).
- (−) Happy-path не гоняется в plain-suite (цена опциональности — как onnx).
