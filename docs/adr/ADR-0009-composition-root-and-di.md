# ADR-0009 — Composition root и внедрение зависимостей

- **Статус:** Proposed
- **Драйверы:** D1 (расширяемость), D6 (инженерия); FR-10; NFR-1/2.
- **Воркстрим:** M2.

## Контекст

Ядро M0/M1 дало порты (`Estimator`/`Metric`/`CVSplitter`/...) и доменные сущности,
но `application/` и `composition/` — пустые каркасы. M2 должен **собрать** сквозной
прогон из портов и конкретных адаптеров, не нарушая правило слоёв и не вводя
преждевременно реестр-плагины (это M3, R4).

## Рассмотренные варианты

1. **DI-фреймворк** (`dependency-injector`, `injector`) — лишняя зависимость и
   рантайм-магия для библиотеки; избыточно.
2. **Глобальный сервис-локатор/синглтоны** — антипаттерн (как раз убран в M0, A2);
   ломает конкурентность.
3. **Ручная конструкторная инъекция + composition root** — use-case принимает порты
   аргументами; `composition/` собирает дефолты и отдаёт фасаду. Стандартно, явно,
   тестируемо подменой портов фейками.

## Решение

Вариант **3**.
- **Use-case** (`application/`) объявляет зависимости как параметры-порты и **не
  называет конкретные адаптеры** (import-linter `usecases-independent-of-adapters`).
- **Composition root** (`composition/`) — единственное место, где импортируются
  конкретные адаптеры и собираются дефолты: фабрика `build_default_components(task)
  -> (estimators, splitter, metric, policy)` (для M2 — хардкод 1–2 моделей, KFold/
  holdout, метрика по `Task`). Реестр на entry-points — **M3**, не здесь.
- **Фасад** (`composition/`) — публичная точка входа `AutoML`, которая внутри зовёт
  composition root и use-case. Зависимости можно переопределить через конструктор
  фасада (инъекция для тестов/продвинутых сценариев).
- Поток зависимостей: `composition → application → core`; `adapters → core`;
  `composition → adapters`. Без стрелок наружу из core (граф — `05-design`).

## Последствия

- (+) Чистая граница: ядро и use-case не знают адаптеров; замена/добавление
  компонента не трогает их (готовность к реестру M3).
- (+) Тестируемость: use-case гоняется на фейк-портах без I/O/моделей.
- (+) Реентерабельность: всё состояние — в передаваемых объектах (`RunContext`,
  результат), без глобалей.
- (−) Немного «ручной проводки» в composition root — приемлемо; реестр уберёт
  хардкод дефолтов в M3.

## Уточнения контрактов (ревью, фаза 8)

- **Seed (F7):** сигнатура `build_default_components(task, *, random_state: int)`;
  seed прокидывается в сплиттер (`StratifiedKFold(shuffle=True,
  random_state=...)`) и в адаптеры (`LogisticRegression(random_state=...)`).
  Источник: `AutoML.random_state` → `RunConfig.seed` → `RunContext`. Без этого
  детерминизм leaderboard (NFR-4) недоказуем.
- **Синхронизация политики (F10):** `build_default_components` строит
  `SelectionPolicy(greater_is_better=metric.greater_is_better)` — иначе метрика с
  `greater_is_better=False` (LogLoss) ранжируется в неверную сторону.
- **Отсев по capabilities (F3):** composition исключает пары (модель, метрика),
  где `metric.needs="proba"`, а модель не `ProbabilisticEstimator`; если совместимых
  не осталось — `ConfigError`.
