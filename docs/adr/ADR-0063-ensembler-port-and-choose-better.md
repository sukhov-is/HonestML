# ADR-0063 — Порт `Ensembler` + Caruana-дефолт + честный гард `choose_better`

- **Статус:** Accepted (M7, design-gate pending)
- **Драйвер:** DM-71 (честный ансамбль), DM-73 (переиспользовать машину) — FR-ENS-1/2/3; NFR-M7-3/6
- **Связано:** ADR-0046 (Humble-Object образец); ADR-0007/0026 (SignificanceTest/band — гард); ADR-0010
  (Candidate/OOF).

## Контекст
As-is несёт всё для ансамбля: `Candidate.oof_pred`(metric-ready)/`oof_proba`(raw)/`oof_mask`, common-mask в
`equivalence_band`, `SignificanceTest.equivalent`, `Metric`. Наивный блендинг оптимизирует веса на CV и
**гардит** «blend>best на OOT», но: (1) зашит в PR-AUC, (2) гард — наивный `>` (не значимость), (3) только
weighted-blend. Нужен порт, где адаптер-«комбинатор» видит лишь инжектируемый скаляр-скорер (Humble Object,
домен без scipy/sklearn.ensemble), а **честность** ансамбля обеспечивается significance-гардом M4.

## Рассмотренные варианты
1. **Ансамбль всегда (если включён)** — ❌ ансамбль ради ансамбля; Caruana hill-climb переобучается на OOF
   (R-ENSOVERFIT), «кажется» лучше.
2. **Наивный гард `blend>best`** — ⚠️ ловит грубые случаи, но не отличает шум от сигнала.
3. **Порт `Ensembler` (Humble Object) + significance `choose_better`-гард** — ✅ ансамбль принимается только
   если **значимо** лучше лучшего сингла (та же M4-машина); иначе — сингл. Honest by construction.

## Решение (Вариант 3)

### §1 Порт `Ensembler` (core/ports/ensembler.py, Humble Object)
```python
@runtime_checkable
class Ensembler(Protocol):
    name: str
    def combine(
        self,
        oof: np.ndarray,                      # (n_models, n_rows[, K]) blend-пространство, common-mask
        y: np.ndarray,                        # (n_rows,) на common-mask
        *,
        score: Callable[[np.ndarray], float], # blended -> higher-is-better (обёртка Metric)
        member_ids: Sequence[str],
        random_state: int,
        sample_weight: np.ndarray | None = None,
    ) -> EnsembleRecipe: ...
```
`EnsembleRecipe(weights: dict[str, float], method: str, member_ids: tuple[str, ...])` — frozen dataclass,
веса-симплекс (`≥0, ∑≈1`), **нормализованы к python-native `float`** (R2: не `np.float64` — для байт-стабильной
report/manifest-эмиссии). Адаптер видит только `score(blended)->float` и `oof`-массив (уже без сырых строк/
фолдов) — Humble Object: метрика-агностичен, не делает проекций/ликеджа. В core нет scipy/sklearn.

### §2 Blend-пространство (per-task; уточнено по ревью R1)
- **Multiclass:** `oof_proba (n,K)` (выровнено `align_proba`), линейная комбинация + **ренорм по строке**;
  `score` проецирует blended в метрику (`project_for_metric`).
- **Binary:** захваченный `oof_proba` — **1-D `P(pos)`** (slice.py хранит `raw_proba[:,pos]`, не `(n,K)`),
  ренорм по строке **не нужен** (no-op); blended `P(pos)` идёт в `project_for_metric` напрямую.
- **Регрессия:** `oof_pred` (value) линейно.
- Линейная комбинация **меток** (`needs='class'` `oof_pred`) некорректна → ансамбль требует proba/value-канала;
  его нет → **skip+WARNING** `gate_reason="no_proba_channel"` (graceful, как калибровка ADR-0030 §1).
Приложение (`ensemble_selection`) собирает `oof`-массив (форма `(n_models, n_rows[, K])`, FR-ENS-1 её
допускает) по common-mask и `score`-замыкание; адаптер — чистая арифметика весов.

### §3 Адаптеры (adapters/ensembling.py)
- **`CaruanaEnsembler` (дефолт, `name="caruana"`):** жадный отбор с заменой из библиотеки кандидатов,
  максимизирует `score` пошагово (Caruana 2004); seeded **bagging** (`n_bags` подвыборок библиотеки) для
  стабильности; веса = частоты выбора, нормированные. Размер шага/библиотеки — `EnsembleConfig.size`.
  **Детерминизм (правка R1):** при равном инкрементальном `score` tie-break **детерминирован** — выбирается
  кандидат с наименьшим `member_id` (first-seen индекс), а не argmax-порядок. Caruana — дефолтный путь, его
  детерминизм нагрузочный; bagging seeded от `random_state`.
- **`WeightedEnsembler` (`name="weighted"`):** SLSQP-симплекс (`bounds=[0,1]`, `∑=1`, `x0=1/n`) — оптимизация весов на симплексе, метрика-агностично через `score`. **Детерминизм:** фикс `x0` фиксирует старт, но SLSQP
  (scipy/BLAS) может давать float-дрейф между окружениями — помечается как time-mode-уровень
  недетерминизма (не нагрузочный путь: дефолт — Caruana).
- **stacking — НЕ в M7 (правка R2):** `StackingEnsembler` как `could`-класс без falsifiable-теста
  оставлять нельзя (named class без проверки → отгрузится непротестированным или молча выпадет). Поэтому
  упрощённый stacking **исключён** из M7-реализации и маршрутизирован в **M7-future** рядом с полным stacking
  (§Day-2). M7 поставляет два метода: `caruana` (дефолт) + `weighted`. `EnsembleConfig.method`-литерал в M7 =
  `Literal["caruana","weighted"]` (без `"stacking"`).

### §4 `EnsembleConfig` (core/config.py, opt-in)
`RunConfig.ensemble: EnsembleConfig | None = None` (дефолт None → OFF, fingerprint M6). Поля (`frozen`):
```
method: Literal["caruana", "weighted"] = "caruana"   # stacking → M7-future (R2)
size: int = Field(50, ge=1)          # шаги Caruana / потолок библиотеки
n_bags: int = Field(20, ge=1)        # bagging-подвыборки Caruana (стабильность); 1 = без bagging
metric: str | None = None            # None -> метрика рана
random_state: int | None = None      # None -> наследует RunConfig.seed
```

### §5 Честный гард `choose_better` (application `ensemble_selection`, FR-ENS-3)
После `combine`: построить blended-OOF по рецепту, сравнить с **лучшим синглом** через **ту же**
`SignificanceTest`/`equivalence_band` на common-mask:
- `significance="bootstrap"` (дефолт): отгружать ансамбль ⟺ blended **не эквивалентен** лучшему синглу **и**
  лучше по метрике (значимо лучше). Иначе — сингл.
- `significance="off"`: fallback на строгий `>` (наивная строго-больше семантика).
Решение наблюдаемо: `applied: bool`, `gate_reason ∈ {significant_improvement, equivalent_to_best,
worse_than_best, no_proba_channel, single_candidate, degenerate_recipe}`, `oof_delta`. (`degenerate_recipe`
(R2) — `combine` вернул вырожденный рецепт: все веса на одном члене / пустой набор после фильтрации → отгружать
сингл.) **Анти-ликедж:** веса на cross-fit OOF
(`oof_mask`), гард — та же машина, что и селекция. **Замечание (R1):** `equivalent()` — **двусторонний**
CI-overlap (significance.py), а claim «ансамбль лучше» односторонний → эффективный уровень `alpha/2`
(консервативнее, в пользу честности). При этом CI считается на **том же** OOF, на котором Caruana подбирал
веса (§6 optimism) — гард его смягчает, но не обнуляет (нулевой — ensemble-validation split, Day-2).

### §6 Остаточный optimism (раскрытие, NFR-M7-3/7)
Caruana hill-climb выбирает веса на том же OOF, на котором затем меряется → мягкий optimism. Снижается:
(а) significance-гардом (консервативен), (б) seeded bagging. **Отдельный ensemble-validation split (нулевой
optimism) — Day-2.** Раскрыто в report и §Последствия.

## Последствия
- **+** Ансамбль честен: принимается только значимо-лучший; иначе отгружается сингл — нет «ансамбля ради
  ансамбля» (R-ENSOVERFIT смягчён).
- **+** Переиспользует `SignificanceTest`/band/`Candidate`/OOF — нулевая новая стат-машина (DM-73, NFR-M7-8);
  SLSQP-оптимизация весов добавляется точечно.
- **+** Humble Object → юнит-тест на чисто-числовом `score`/`oof` без обучения.
- **−/R-ENSOVERFIT (остаточный):** Caruana-optimism — раскрыт, смягчён гардом/bagging; нулевой — Day-2.
- **−:** stacking (упрощённый и полный) **не в M7** — исключён из объёма (R2) и маршрутизирован в M7-future
  (§3/§Day-2); M7 поставляет caruana+weighted.

## Day-2 (committed → M7-future)
- Отдельный **ensemble-validation split** (carve из DEV) для нулевого hill-climb-optimism.
- Полноценный **stacking** с вложенной CV meta-learner.
- Per-member feature-selection (свой subset на члена).
