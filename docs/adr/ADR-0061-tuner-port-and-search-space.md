# ADR-0061 — Порт `Tuner` + backend-нейтральный `SearchSpace` + Optuna-адаптер

- **Статус:** Accepted (M7, design-gate pending)
- **Драйвер:** DM-72 (FR-HPO-1/2/3; NFR-M7-1/2), делит Humble-Object-механизм с ADR-0046 (FeatureSubsetSelector)
- **Связано:** SPIKE-M7-hpo (Q1/Q2 — детерминизм + транзляция); ADR-0019 (registry/extras-gating);
  ADR-0006 (ModelSpec.search_space — заложенный слот); ADR-0062 (как Tuner вызывается честно).

## Контекст
HPO нужно за портом (backend сменяем, домен чист) с **декларацией поиска компонентом**. As-is уже несёт
`ModelSpec.search_space: dict[str,Any] = {}` («tuner (M7) consumes; here just carried data») — пустой слот,
заложенный M3. Антипаттерн: тюнер, зашитый в Optuna и одну метрику, где per-backend suggest-функции —
callback-style `trial.suggest_*`. `FeatureSubsetSelector` (ADR-0046) даёт образец: адаптер-«поисковик»
получает инжектируемый скаляр-скорер и **не видит сырых строк** → домен свободен от sklearn/optuna,
анти-ликедж by construction.

## Рассмотренные варианты
1. **Optuna-трайл напрямую в домене** (порт принимает `optuna.Trial`). — ❌ протекает backend в core,
   ломает NFR-M7-1; нельзя сменить на Ray/Hyperopt.
2. **Непрозрачный `dict` SearchSpace, валидация в адаптере.** — ⚠️ легче, но сдвигает проверку в Optuna,
   нет composition-time-валидации; имена/типы не контрактны.
3. **Декларативный типизированный `ParamSpec` в core + Humble-Object `Tuner` + адаптер-транслятор.** —
   ✅ backend-нейтрально, валидируемо на composition-time, образец `FeatureSubsetSelector`, домен чист.

## Решение (Вариант 3)

### §1 `SearchSpace` / `ParamSpec` (core, декларативно)
`SearchSpace = Mapping[str, ParamSpec]`. `ParamSpec` — дискриминированный по `type`:
```
{"type": "int",         "low": int,   "high": int,   "step": int=1}
{"type": "float",       "low": float, "high": float, "log": bool=False}
{"type": "categorical", "choices": list[str|int|float|bool]}
```
Чистый core-парсер `parse_search_space(raw: Mapping) -> dict[str, ParamSpec]` валидирует `type`/границы
(`low<high`, непустой `choices`) и отвергает неизвестное → понятная доменная ошибка. **optuna в core не
импортируется.** `ParamSpec` — `Pydantic`-модели (frozen), сериализуемы (входят в fingerprint через
`HPOConfig`/`ModelSpec`).

### §2 Порт `Tuner` (core/ports/tuner.py, Humble Object)
```python
@runtime_checkable
class Tuner(Protocol):
    name: str
    def tune(
        self,
        search_space: Mapping[str, ParamSpec],
        score: Callable[[Mapping[str, Any]], float],   # higher-is-better, инжектируется приложением
        *,
        max_trials: int,
        timeout_s: float | None,
        greater_is_better: bool,
        random_state: int,
    ) -> TuneOutcome: ...
```
`TuneOutcome(best_params: dict[str, Any], n_trials_run: int, best_score: float)` — frozen dataclass.
**Нормализация (R2):** `best_params` нормализуются на границе порта к **python-native** скалярам
(`int`/`float`/`str`/`bool`, не `np.int64`/`np.float64`) — иначе report-эмиссия/fingerprint (`json.dumps
sort_keys`) байт-нестабильны. Тест `test_best_params_are_native_scalars`. Адаптер **владеет петлёй**
поиска (как `study.optimize`), но на каждом trial зовёт инжектируемый `score(params)->float` — **скаляр**,
никогда сырые строки/фолды (Humble Object). `score` ориентирован higher-is-better приложением (флип по
`greater_is_better` делает caller), но `greater_is_better` передаётся и в адаптер для `study.direction`/логов.
**Бюджет передаётся скалярами** (`max_trials`/`timeout_s`), вычисленными приложением из run-`Budget` (ADR-0062
§5) — порт не зависит от `Budget`-семантики, остаётся лёгким.

### §3 Optuna-адаптер (adapters/tuning.py)
`OptunaTuner.name="optuna"`. `tune`: транслирует `ParamSpec → trial.suggest_int/float(log=)/categorical`,
`study = create_study(direction=max if greater_is_better else min, sampler=TPESampler(seed=random_state))`,
`study.optimize(objective, n_trials=max_trials, timeout=timeout_s, n_jobs=1)`. `objective(trial)` строит
`params` из space и возвращает `score(params)`. Возвращает `TuneOutcome(study.best_params, len(study.trials),
study.best_value)`. **Lazy-импорт** `optuna` внутри модуля/функции; registry-дескриптор `requires=('optuna',)`
→ `is_available` через `find_spec` без импорта; явный запрос без экстры → `MissingDependencyError`.
**`n_jobs=1` обязателен** (SPIKE Q1: параллельные trials рушат детерминизм порядка завершения).

### §4 Наполнение `ModelSpec.search_space` + проброс параметров (FR-HPO-3)
Дескрипторы бустингов объявляют `search_space` (типовые диапазоны: `depth/max_depth`,
`learning_rate`(log), `n_estimators`/`iterations`, `l2`/`reg_lambda`, `subsample`, `colsample_bytree`);
линейные — `C`(log)/`alpha`(log); baselines — `{}` (HPO пропускает, 0 trials). **Имена ключей тождественны
build-kwargs** эстиматора.

**ВАЖНО (правка по ревью R1 — это НОВАЯ работа, не no-op):** в as-is `ModelSpec.search_space` — лишь
*декларативный слот* (объявлен M3, нигде не потребляется), но цепочка сборки **не принимает** тюненые
параметры: `ComponentRegistry.build(name, **kwargs)` зовёт `descriptor.build(**kwargs)`, а build-callable'ы
жёстко `(*, task, random_state)` без `**params` (`build_boosting`, `_build_linear`, `_build_baseline`) →
`registry.build(..., depth=6)` сегодня бросает `TypeError`. Дополнительно `_BoostingBase._make()` хардкодит
`_N_ESTIMATORS=300` и игнорирует любые внешние параметры. Поэтому M7 **расширяет build-цепочку**:
- каждый build-callable дескриптора получает `**params` и пробрасывает их в конструктор адаптера;
- `_make()` строит ctor-kwargs как `{n_estimators_kwarg: _N_ESTIMATORS, seed_kwarg: seed, **extra_kwargs,
  **params}` — **`params` последними**, чтобы тюненый tree-count **перекрывал** дефолт 300 (без дубль-ключа).
- **Tree-count ключ привязан к `_Backend.n_estimators_kwarg` per-backend (правка R2 — критично):** catboost
  имеет `n_estimators_kwarg='iterations'`, lightgbm/xgboost — `'n_estimators'`. Если catboost-`search_space`
  объявит ключ `n_estimators`, `params`-last даст `{iterations:300, n_estimators:<tuned>}` — а
  `CatBoostClassifier.__init__` принимает **оба** (`iterations` И `n_estimators`), поэтому `iterations=300`
  **не** перекрывается, тюненое значение игнорируется. ⇒ tree-count ключ каждого `search_space` **обязан**
  равняться `_Backend.n_estimators_kwarg` (catboost → `iterations`); инвариант-тест
  `test_search_space_tree_key_matches_backend_kwarg`. `inspect.signature`-валидация ниже **не ловит** этот
  кейс для catboost (его ctor несёт оба имени) — поэтому пиннинг ключа обязателен, а тест `test_tuned_
  n_estimators_overrides_default` гоняется **на catboost-бэкенде**, не только lightgbm/xgboost.
- линейные пробрасывают `C`/`alpha`/`max_iter` в `LinearClassifier`/`LinearRegressor.__init__` (уже их несут).
- **Composition-time валидация:** ключи `search_space`/`best_params`, отсутствующие в принятых build-kwargs
  компонента (`inspect.signature` конструктора либо явный per-component allowlist), → `ConfigError` на резолве,
  **не** молча отбрасываются (FR-HPO-3 acceptance). NB: для catboost (`**kwargs`-ctor) `inspect.signature`
  слаб — дополняется явным per-backend allowlist канонических ключей.

Затрагиваемые файлы (в матрице FR-HPO-3): `composition/registry.py` (форвардинг+валидация), `adapters/boosting.py`
(`build_boosting`/`_BoostingBase._make` + `**params`), `adapters/estimators.py` (линейные), декларации
`search_space` в дескрипторах.

### §5 Детерминизм (SPIKE Q1 → NFR-M7-2)
trials-mode (`timeout_s=None`, фикс `max_trials`, single-thread): один `random_state` → идентичные
`best_params` (подтверждено). `optuna` пинуется в fingerprint `lib_versions` (как catboost/lightgbm, ADR-0035).
time-mode (`timeout_s` задан) недетерминирован — ADR-0062 §5 документирует и помечает в report.

## Последствия
- **+** Backend-нейтральный, валидируемый, чистый домен; Optuna заменяем; новый компонент объявляет `ParamSpec`.
- **+** Заложенный `ModelSpec.search_space` наконец используется по назначению — нет параллельного механизма
  (NFR-M7-8).
- **+** Humble Object = анти-ликедж и тестируемость: `tune` юнит-тестируется на чисто-числовом `score` без
  обучения моделей (как `FeatureSubsetSelector`).
- **−/R-OPTEXTRA:** новая опц-экстра `optuna` (+sqlalchemy, 14 пакетов) — registry-gated, дефолт-skip.
- **−/R-HPODET:** детерминизм только single-thread trials-mode — задокументировано; параллелизм/time-mode
  явно не воспроизводимы.
