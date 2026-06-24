# ADR-0056 — Interventional-SHAP с фоном (opt-in расширение `ShapRanker`)

- **Статус:** Принят (дизайн M6e; реализация — скил implementation). Питается SPIKE-M6e-shap.
- **Дата:** 2026-06-10
- **Драйверы:** DM-E4 (причинно-чище атрибуция). FR-FSE-9/10, NFR-FSE-5/6. **Расширяет ADR-0051**
  (`ShapRanker`, lazy-extra) — не отменяет: `tree_path_dependent` остаётся дефолтом.
- **Воркстрим:** M6e.

## Контекст
ADR-0051 (M6d) реализует `ShapRanker` через `TreeExplainer(model, feature_perturbation=
"tree_path_dependent")` — точный, **без фоновых данных**, детерминированный, дешёвый. ADR-0051 (Последствия,
:88-90) ценит детерминизм **без скрытого сабсэмплинга фона** и отнёс `interventional` (с фоном) в Day-2.
tree_path_dependent — **условная** (tree-path) атрибуция; `interventional` интегрирует по **фоновому
распределению** → причинно-чище attribution (intervenes на фон). Пользователям с интервенционной семантикой
нужен opt-in, без отказа от детерминизма/дешевизны дефолта.

SPIKE-M6e-shap (**измерено**, shap 0.48.0 в dev-.venv; под `uv run` shap отсутствует → поведенческие тесты
`importorskip`, как M6d): `interventional` **требует** `data=background`; cost для глубоких ExtraTrees — **0.2–0.3×
tree_path_dependent** при capped head-slice (аналитический «×|background|» **завышал**; рост по |bg| 50→200
слабый); детерминизм **подтверждён** (`deterministic=True`) при фиксированном фоне.

## Рассмотренные варианты
1. **Оставить tree_path_dependent (M6d).** Дёшево/детерминированно, но только условная атрибуция. Недостаточно
   для DM-E4.
2. **Сделать interventional дефолтом.** Дороже (×|bg|), вносит выбор фона (детерминизм-риск), ломает M6d.
   Отвергнут.
3. **Opt-in `shap_perturbation`, детерминированный head-slice фон по умолчанию, дефолт tree_path_dependent.**
   Back-compat, детерминизм сохранён, cost ограничен `shap_background_samples`. **Выбран.**

## Решение

### 1. Конфиг — аддитивный режим перестановки (`core/config.py`)
```python
class FeatureSelectionConfig(BaseModel, frozen, extra="forbid"):
    ...
    shap_max_samples: int | None = Field(None, gt=0)                                  # M6d: explained rows
    shap_perturbation: Literal["tree_path_dependent", "interventional"] = "tree_path_dependent"  # M6e
    shap_background_samples: int | None = Field(None, gt=0)                            # M6e: |background| cap
```
- Дефолт `"tree_path_dependent"` ⇒ **тождественно M6d**. Поля внутри `FeatureSelectionConfig` ⇒ `fs=None` →
  fingerprint M6b (NFR-FSE-7).
- **Валидатор/WARNING (R-DEADCFG-E, FR-FSE-10):** `shap_background_samples` осмыслен лишь при
  `interventional`; при `tree_path_dependent` — **WARNING** «фон игнорируется» (не `ConfigError`,
  консистентно с dead-config M6d).

### 2. Механика (`adapters/feature_rankers.py::ShapRanker`) — порт неизменен
`__init__` растёт двумя опц. аргументами (`perturbation`, `background_samples`); `rank`:
```python
if self._perturbation == "interventional":
    k = self._background_samples
    if k is None or k >= x.shape[0]:
        bg = x
    else:  # ДЕТЕРМ. равномерно-распределённые индексы (НЕ strided x[::step] — вырождается в head-slice при k>n/2)
        bg = x[np.linspace(0, x.shape[0] - 1, k).astype(np.int64)]
    explainer = shap.TreeExplainer(model, data=bg, feature_perturbation="interventional")
else:  # tree_path_dependent (M6d)
    explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
return _mean_abs_per_feature(explainer.shap_values(x_explain), x.shape[1])
```
- `_mean_abs_per_feature` и `auto_threshold=1/n` **без изменений** (interventional даёт те же list/2-D/3-D
  формы) ⇒ **порт `FeatureRanker` не меняется**, спайн (L1-норма/cutoff) не трогается. Всё внутри
  `ShapRanker`-адаптера.
- **Фон — детерминированные равномерные `linspace`-индексы, не head-slice (фикс R1-adversarial + R2):** `x[:K]`
  (head-slice) детерминирован, но **смещён** — первые K строк неперемешанного/групп-упорядоченного матрикса
  нерепрезентативны, а смещённый фон сдвигает interventional-атрибуцию и через L1-норму спайна **меняет отбор**.
  Наивный `x[::n//K][:K]` **вырождается** в head-slice при `K>n/2` (step=1, фикс R2-minor). ⇒ дефолт —
  `x[np.linspace(0, n−1, K).astype(int)]`: **равномерно по всему диапазону** строк при любом K, детерминирован
  (воспроизводим без seed, NFR-FSE-5). (Опц. `shap.sample`/`kmeans`-фон — Day-2, seeded от `rank.random_state`.)
  Граница: при сильной корреляции порядка строк с признаками даже равномерная выборка несовершенна —
  документируется; полная репрезентативность (стратифицированный/kmeans фон) — Day-2.
- **Детерминизм подтверждён (NFR-FSE-5, измерено):** SPIKE-M6e-shap — `deterministic=True` для фиксированного
  (linspace/head) фона во всех |bg|; interventional-интеграция точна при фиксированном фоне.
- **Оси cost (NFR-FSE-6, измерено):** `shap_max_samples` ограничивает **объясняемые** строки (`x_explain`),
  `shap_background_samples` — **фон** (`bg`). Аналитический «×|background| blowup» **завышал**: для глубоких
  ExtraTrees interventional измерен **0.2–0.3× tree_path_dependent** (рост по |bg| 50→200 слабый, SPIKE-M6e-shap).
  `shap_background_samples` остаётся рычагом/предохранителем (для иных моделей соотношение иное).

### 3. Проводка (`composition/build.py::_make_strategy`)
`ShapRanker(task, max_samples=fs.shap_max_samples)` → `ShapRanker(task, max_samples=fs.shap_max_samples,
perturbation=fs.shap_perturbation, background_samples=fs.shap_background_samples)`. Единая точка маппинга —
single-path и compare в синхроне (как ADR-0051 §3). Cost-WARNING для interventional — в `_warn_fs_cost`/resolve
(NFR-FSE-6). `MissingDependencyError` при отсутствии `shap` — fail-fast в `__init__` (как M6d).

### 4. Честная граница
- Тайминг **замерен** (SPIKE-M6e-shap, shap 0.48.0): interventional не «дороже вообще» — для древесных
  ранкер-моделей M6e 0.2–0.3× tpd при capped фоне; множитель модель-зависим (`shap_background_samples` —
  предохранитель). Под `uv run` shap отсутствует → поведенческие тесты `importorskip` (как M6d).
- interventional — иная **семантика** атрибуции (интервенционная vs условная), не «точнее вообще»; выбор за
  пользователем под задачу. Документируется, без претензии на превосходство по умолчанию.
- Фон (даже strided) — выборка распределения; полная репрезентативность (стратифицированный/kmeans) — Day-2.

## Последствия
- (+) Причинно-чище атрибуция как opt-in; детерминизм сохранён (равномерный linspace детерм. фон, измерено);
  cost ограничен и в практике невелик для древесных моделей (`shap_background_samples`); порт/спайн неизменны;
  полный back-compat (tree_path_dependent дефолт); shap остаётся в `adapters`.
- (−/компромисс) ещё два config-поля; фон — выборка (linspace смягчает смещение, не устраняет полностью);
  множитель cost модель-зависим (замерен для ExtraTrees).
- **Влияние на слои:** конфиг — `core`; механика — `adapters` (`ShapRanker`); проводка — `composition`.
  `core-independence` (shap только в adapters) и `import-linter` 3/3 KEPT.

## Проверки
- (под `importorskip("shap")`) `interventional` строит explainer с `data=`; равномерный (linspace) детерм. фон ⇒ два прогона
  идентичны (детерминизм, NFR-FSE-5); дефолт `tree_path_dependent` ⇒ без `data=`, тождественно M6d.
- `shap_background_samples` при `tree_path_dependent` → WARNING (FR-FSE-10); отсутствие `shap` → fail-fast
  `MissingDependencyError` в `__init__`.
- `fs=None` → fingerprint M6b; `lint-imports` 3/3 (shap не в core/application).
