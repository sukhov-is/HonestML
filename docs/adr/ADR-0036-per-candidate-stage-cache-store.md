# ADR-0036 — Per-candidate stage-cache: порт + durable joblib-store + skip-on-hit

- **Статус:** Accepted (реализован 2026-06-09: порт `core/ports/cache.py` `CandidateCache`; адаптер
  `adapters/candidate_cache.py` `JoblibCandidateCache`+`CACHE_VERSION` (atomic `tmp`+`os.replace`,
  `_commit_meta` последним, fail-closed, анти-traversal); skip-on-hit в `application/slice.py`
  `run_slice`+`SliceResult.reused/computed`; тесты `tests/unit/test_candidate_cache.py` +
  RC-c в `test_run_slice.py`. Без отклонений от дизайна.)
- **Дата:** 2026-06-09
- **Драйверы:** DM-2 (durable reuse + resume), DM-3 (слои/порт); FR-RC-2/3, NFR-RC-3/4/5. Наследует
  `Candidate` (`core/selection_policy.py`), per-candidate цикл `run_slice` (`slice.py:263`),
  `RunContext.record_stage_time` (шов «loaded from cache»), joblib trust-модель (`artifact.py:195-202`),
  прецедент атомарной per-candidate записи (tmp+rename).
- **Воркстрим:** M5-resume.

## Контекст
Единица повторно вычисляемой работы — итерация `run_slice` по `estimators.items()`.
Нужно: (а) переиспользовать завершённого кандидата без переобучения; (б) пережить краш (resume досчитывает
оставшихся). Кандидат несёт **numpy-OOF** (`oof_pred`/`oof_mask`/`oof_proba`), который потребляют band/
calibrate/refinement **после** отбора — значит entry обязан хранить OOF, иначе reuse бесполезен. Кэш — I/O
(диск) → должен жить за **портом** (контракт usecases-independent-of-adapters), `run_slice` не именует адаптер.

## Рассмотренные варианты
1. **Кэш как конкретный объект в `run_slice`** (прямой joblib-доступ из use-case). Нарушает слои (адаптер в
   use-case), не тестируется на фейке. **Отвергнут.**
2. **Порт `CandidateCache` в `core` + адаптер `JoblibCandidateCache` в `adapters`.** `run_slice` берёт
   **порт-параметр** (как `Budget` в M5), возвращает/принимает доменный `Candidate` (диск-формат не течёт в
   use-case). Durable per-candidate запись, атомарность, версионирование. **Выбран** (симметрия с
   `Budget`/`RunBudget`).
3. **Batch-запись всех кандидатов в конце.** Не даёт resume-после-краха (краш на 3/5 → теряем 1-2).
   **Отвергнут** (DM-2).

## Решение

### 1. Порт `CandidateCache` (`core/ports`, доменные типы)
```
class CandidateCache(Protocol):
    def get(self, candidate_id: str) -> Candidate | None: ...   # hit -> доменный Candidate, miss/битый -> None
    def put(self, candidate_id: str, candidate: Candidate) -> None: ...  # durable, атомарно, по завершении
```
- Порт оперирует **доменным `Candidate`** (с numpy-OOF) — диск/joblib-формат **не течёт** в `run_slice`.
- **Размещение (R1-clean-arch-minor):** `core/ports/cache.py` + экспорт в `core/ports/__init__.py` и
  `core/__init__.py` (тем же путём, что `Candidate`/`Budget`, чтобы `run_slice` импортировал порт как
  `Budget`). Выбран каталог `core/ports/` по образцу `Budget` (новые порты идут в `ports/`); `core/dataset.py`
  — легаси-исключение, не прецедент.
- Порт **скоупится фингерпринтом адаптером** (конструируется с `cache_dir`+`fingerprint`), поэтому ключ в
  сигнатуре — лишь `candidate_id` (минимальная поверхность; fingerprint — забота composition, ADR-0035 §3).
- `get` **fail-closed**: отсутствие/несовместимый `CACHE_VERSION`/нечитаемость/битый entry → `None` (miss),
  **не исключение наружу** (NFR-RC-1/4).
- **Порт намеренно остаётся `get`/`put` (фикс R2-COMP-major):** `run_slice` (use-case) во время прогона
  только читает/пишет per-candidate — `list/exists/clear/evict` ему **не нужны**. GC/очистка — **вне порта**,
  на уровне **файловой системы** `cache_dir` (удаление `cache_dir` или конкретного `<fingerprint>/`-subdir;
  будущая composition-утилита `clear`, оперирующая каталогом). Это **не** нарушает usecases-independent-of-
  adapters (use-case порт не зовёт). Поэтому «адаптер-метод без смены порта» (operational §7) корректен:
  расширение касается store/composition, а **контракт порта зафиксирован минимальным навсегда**.

### 2. Адаптер `JoblibCandidateCache` (`adapters`)
- **Раскладка:** `cache_dir/<fingerprint>/<candidate_id>/` с двумя файлами:
  - `entry.joblib` — весь `Candidate` (скаляры + numpy-OOF; joblib нативно сериализует numpy).
  - `meta.json` — `{cache_version: CACHE_VERSION, candidate_id, fingerprint, written_at}` — **commit-маркер,
    пишется ПОСЛЕДНИМ**.
- **Запись (`put`) атомарна** (NFR-RC-4, R-CRASH): `entry.joblib` → `tmp`+`os.replace`; затем `meta.json` →
  `tmp`+`os.replace` **последним**. Краш между ними оставляет `entry.joblib` **без** `meta.json` →
  `get` трактует как miss (нет commit-маркера) → пересчёт. Полу-entry никогда не ложно-валиден.
  - **Tmp-имена уникализированы по процессу** (`tmp`-суффикс = PID + uuid, фикс R2-COMP-major/R-CONCURRENT):
    два конкурентных процесса в один `<id>/` не коллидируют tmp-файлами (контракт — один процесс на
    `cache_dir`, но уникализация дёшева и страхует от порчи при его нарушении).
  - **`put` всегда перезаписывает ОБА файла** (`entry.joblib` И `meta.json`, оба `tmp`+`os.replace`, meta
    последним; фикс R2-COMP-minor): после бампа `CACHE_VERSION` и пересчёта в `<id>/` нет рассинхрона версий
    между `entry.joblib` и `meta.json`.
  - **Тестируемый шов краха (фикс R2-Day2):** запись `meta.json` — отдельный приватный шаг (напр.
    `_commit_meta`), чтобы тест через `monkeypatch` уронил `put` **после** `entry.joblib`, но **до**
    `meta.json`, и проверил `get`→`None` реальным код-путём (а не ручной раскладкой файлов).
- **Чтение (`get`)**: нет `meta.json` → `None`; `cache_version != CACHE_VERSION` → `None`; иначе
  `joblib.load(entry.joblib)` → `Candidate`. Любая ошибка десериализации → `None` (fail-closed) + WARNING.
- **`CACHE_VERSION = 1`** — отдельная константа; бамп при смене формата entry (старые → miss).
- **Trust (R-TRUST, NFR-RC-5):** `joblib.load` наследует trust-границу `load_artifact` — SECURITY-docstring
  «грузить кэш только из доверенного каталога; version-check = forward-compat, не integrity; signing→M8».
  `candidate_id` и `fingerprint` как имена каталогов санируются через `Path(...).name` / hex-digest
  (анти-traversal).

### 3. Интеграция в `run_slice` (skip-on-hit, durable)
`run_slice(..., cache: CandidateCache | None = None)`. Внутри per-candidate цикла, **после** budget-gate:
```
for name, factory in estimators.items():
    if budget and budget.exhausted: skipped.append(name); budget_exhausted=True; continue
    cand = cache.get(name) if cache is not None else None
    if cand is not None:
        reused.append(name)                  # ОСНОВНОЙ канал наблюдаемости hit
    else:
        try: cand = _run_candidate(...)
        except _CandidateFailed as exc: failed.append(...); continue
        if cache is not None: cache.put(name, cand)   # durable СРАЗУ по завершении (resume-готово)
        computed.append(name)
    candidates.append(cand)
    if budget is not None: budget.consume(cand.train_time)   # см. ADR-0037 §2 (trials: +1 трайл; time/none: no-op)
```
- **cache-hit засчитывается как завершённый** (consume бюджета) → trials/none-детерминизм (ADR-0037,
  NFR-RC-2). **Падение кандидата НЕ кэшируется** (`put` только на успехе) — повторный прогон честно
  ретраит (падение могло быть транзиентным; и `_CandidateFailed` не несёт OOF).
- **Наблюдаемость hit — через `reused`/`computed` (списки id) в `SliceResult`** (ADR-0037 §3), **НЕ** через
  `RunContext.record_stage_time` (фикс R1-asis-major): текущая модель таймингов **не per-candidate** —
  `run_slice` вообще не пишет в `ctx.timings`, selection таймится одним блоком в facade
  (`timed_stage("run","selection")`). `ctx` к тому же **опционален** (`ctx: RunContext|None`). Per-candidate
  тайминг кэша — **вне объёма**; если позже понадобится, вызов `ctx.record_stage_time` обязан быть под
  guard `if ctx is not None`. `record_stage_time` остаётся доступным швом, но дизайн **не выдаёт** его за
  готовый per-candidate механизм.
- **`refit_best`/calibrate НЕ кэшируются** (пост-отбор, дёшев относительно CV; пересчитывается поверх
  переиспользованных OOF) — FR-RC границы.

## Последствия
- **Положительные:** durable per-candidate reuse + resume-после-краха; порт сохраняет слои (тест на фейковом
  `CandidateCache` без диска); атомарность исключает ложно-валидный полу-entry; `record_stage_time`-шов
  переиспользован.
- **Отрицательные/компромиссы:** диск-рост (OOF×кандидаты×fingerprint'ы) — GC отложен (R-GC, operational);
  joblib-trust наследуется (R-TRUST, signing→M8); один процесс на каталог (конкурентный доступ — future).
- **Влияние на слои:** порт `CandidateCache` — `core/ports`; адаптер `JoblibCandidateCache` — `adapters`;
  `run_slice` (gate+skip+`reused`/`computed`) — `application`; сборка адаптера со скоупом-fingerprint —
  `composition/facade`. `joblib` только в адаптере. import-linter не нарушен. `ARTIFACT_VERSION` не тронут.

## Проверки
- Hit: предзаполненный фейковый кэш → `_run_candidate` **не** зван для закэшированных id (spy); восстановленный
  `Candidate` несёт OOF; `leaderboard`/band идентичны.
- Resume: `k` entry в кэше → пересчёт только `N−k`; `reused`=k id, `computed`=N−k id.
- Crash-safety: `entry.joblib` без `meta.json` → `get`→miss→пересчёт; несовместимый `CACHE_VERSION` → miss.
  **Инъекция краша (R2-Day2):** `monkeypatch` роняет `put` после `entry.joblib`, до `meta.json` → `get`→`None`
  реальным путём.
- Fail-closed: битый `entry.joblib` → `get`→`None`+WARNING, не исключение.
- Атомарность: `put` использует `tmp`+`os.replace` (tmp уникализирован PID/uuid), `meta.json` пишется
  последним; re-`put` поверх существующего `<id>/` перезаписывает оба файла (нет рассинхрона версий).
- **Граничные (R2-COMP-minor):** N=1 cache-hit → `reused=[id]`, band = lone-anchor, leaderboard идентичен;
  все кандидаты упали → кэш **не** записан (`reused`/`computed` пусты), повторный `fit` ретраит все N
  (падение не кэшируется).
- `lint-imports` 3/3 KEPT; `joblib` только в адаптере; `run_slice` берёт порт, не именует `JoblibCandidateCache`.
