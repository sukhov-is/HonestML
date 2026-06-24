# ADR-0072 — Порт `ExperimentTracker`: post-fit one-shot поверх run_report

- **Статус:** Accepted
- **Драйверы:** DM-C1, DM-C2, DM-C3 (02-drivers.md); north-star FR-12/D5/P12
- **Spike:** не требуется (00-research §4)

## Контекст

Roadmap §M8 требует трекинг-порт + MLflow-адаптер. ADR-0033 (M5) уже сделал run_report
tracker-независимым источником истины (`RUN_MANIFEST_VERSION=1`, JSON primitives,
аддитивная эволюция) — и явно отложил «трекинг — порт M8». В коде нет per-fold метрик и
per-trial HPO-истории (00-research §2): стримить детальнее стадий/кандидатов нечего.
Боль F4.4 — сводка, существующая только в MLflow, — прямое следствие трекера,
вычисляющего собственный отчёт.

## Варианты

1. **Post-fit one-shot: `log_run(report)` после `build_run_report`** — порт-потребитель
   готового отчёта. ✅
2. Live-порт с событиями по стадиям/кандидатам (методы `on_stage`, `on_candidate`) —
   расширяет сигнатуры use-cases (`run_slice` уже ~20 kwargs) ради событий, для которых
   нет данных детальнее уже логируемого `ctx.timed_stage`; провенанс (hpo/ensemble/
   serving/fingerprint) всё равно живёт только в фасаде → live-порт неполон без
   post-fit-вызова. Отклонено как преждевременное (Day-2 при появлении спроса).
3. Трекер на `RunContext` (как `logger`) — service-locator, core получает порт ради
   внешней интеграции. Отклонено.

## Решение

### §1. Контракт порта

`src/honestml/core/ports/tracker.py` — чистый Protocol (по образцу `significance.py`):

```python
@runtime_checkable
class ExperimentTracker(Protocol):
    def log_run(self, report: Mapping[str, Any]) -> None: ...
```

`report` — run_report ADR-0033 (версионирован `run_manifest_version`, JSON primitives
only). Контракт: один вызов на один `fit`; неизвестные ключи реализация игнорирует
(аддитивная эволюция отчёта не ломает адаптеры). Фасад передаёт **глубокую копию**
отчёта (`copy.deepcopy` — дёшево на JSON primitives): `log_run` — системная граница
с чужим кодом, и мутирующий стаб не должен молча портить `run_report_` /
последующий `save_run_report`. Возврата нет — идентификатор рана наблюдаем через
INFO-лог адаптера, не через API. Null-object не заводится: в отличие от
`SignificanceTest` (течёт глубоко в application), точка вызова одна — `None`-as-off,
как `budget`/`cache`.

### §2. Точка врезки и семантика отказов (DM-C2)

Вызов — в `AutoML.fit` сразу после `self.run_report_ = build_run_report(...)`
(facade.py:466-476), перед `return self`:

```python
if tracker is not None:
    try:
        tracker.log_run(copy.deepcopy(self.run_report_))
    except Exception:  # boundary: a tracking failure must not destroy a finished fit
        logger.warning("experiment tracking failed", exc_info=True)
```

Асимметрия осознанная: **резолв** (включая гейт отсутствующего mlflow) — в начале
`fit`, до чтения данных, чтобы невозможный трекинг падал до дорогого обучения
(`MissingDependencyError`); **логирование** — fail-soft, потому что завершённый fit
ценнее записи о нём. Это единственное место, где исключение глушится до WARNING —
системная граница внешнего сервиса. `except Exception` намеренно НЕ ловит
`KeyboardInterrupt`/`SystemExit` — прерывание пользователя пробрасывается (адаптер
делает свой cleanup, ADR-0073 §2). Работает и в `run_mode="selection"` (отчёт там
тоже строится).

### §3. Opt-in фасада и `TrackerConfig`

`AutoML(tracker: ExperimentTracker | TrackerConfig | str | None = None)` — вербатим в
`__init__` (sklearn clone invariant), резолв в `_resolve_tracker()` в начале `fit`:

- `None` → off (дефолт; fit байт-в-байт прежний);
- `"mlflow"` → сахар `TrackerConfig()` (прецедент `task: Task | str`); другая строка →
  `ConfigError`;
- `TrackerConfig` → composition строит `MlflowTracker` (локальный импорт по образцу
  `_build_cache`); гейт — в конструкторе адаптера (единственный источник, §ADR-0073);
- объект, проходящий `isinstance(x, ExperimentTracker)` **и** `callable(x.log_run)`
  (runtime_checkable проверяет лишь наличие атрибута — callable-проверка закрывает
  поле-не-метод) → вербатим: кастомный бэкенд без правки composition (OCP, DM-C1);
  несоответствие *сигнатуры* при этом честно остаётся fail-soft зоной §2 (Protocol
  не проверяем структурно глубже); иное → `ConfigError`.

`TrackerConfig` — pydantic-модель в `core/config.py` (прецедент `HPOConfig`), домашний
стиль конфигов: `model_config = ConfigDict(extra="forbid", frozen=True)`, публичная
(`EXPECTED_PUBLIC` + баррели): `backend: Literal["mlflow"] = "mlflow"`,
`experiment: str = "honestml"`, `tracking_uri: str | None = None` (None → собственное
разрешение бэкенда: env/`file:./mlruns`), `run_name: str | None = None` (None →
нейтральное имя генерирует бэкенд, NFR-TRK-6),
`tags: dict[str, str] = Field(default_factory=dict)`.

### §4. Tracker вне воспроизводимости (NFR-TRK-5)

`tracker` НЕ входит в `RunConfig` и run-fingerprint — он не влияет на модель
(post-selection опция, точный прецедент `finalize`, ADR-0068). Отчёт `config`-блока
его не упоминает.

### §5. Аддитивный ключ `holdout_score` в run_report

`build_run_report` дополняется top-level `"holdout_score": result.holdout_score`
(float | None; None в `run_mode="selection"` и при `outer_holdout=0`). Это закрытие
as-is-пробела (честная финальная оценка жила только в манифесте артефакта — G-O1), а
для трекера — главная honest-метрика. Аддитивно по ADR-0033/0037:
`RUN_MANIFEST_VERSION` остаётся 1, существующие потребители не затронуты.

## Последствия

- (+) Ядро/чистые слои: порт без ML-импортов, application не тронут; OCP instance-формой.
- (+) Один источник провенанса (run_report); трекер не может «знать больше», чем отчёт, —
  расхождение «таблица vs MLflow» (F4.9) исключено по построению.
- (−) Нет live-прогресса в трекере во время fit; падение посреди fit оставляет пустой
  трекер. Принято: прогресс наблюдаем логами; live-порт — Day-2 (operational.md).
- (−) Поверхность: +`TrackerConfig` в публичный API (запиновано тестом).
