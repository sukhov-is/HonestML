# ADR-0057 — Авто-резолв FS-дефолтов по форме данных (`arbitration`/`null_block_mode` = `auto`)

- **Статус:** Accepted (M6f, design-gate pending)
- **Драйвер:** DM-F1 (FR-FSF-1, FR-FSF-2; NFR-FSF-1/6)
- **Связано:** ADR-0016 (honest CV-резолвер `scheme="auto"` — образец), ADR-0052/0054 (арбитраж/per-fold),
  ADR-0055 (time_window-блок), ADR-0058 (cost-budget делит лестницу арбитража).

## Контекст
Честные режимы M6d/M6e (`nested`, `nested_per_fold`, `time_window`-блок) мощны, но требуют от пользователя
ручного выбора под форму данных. Нужен безопасный авто-дефолт, который выводит арбитраж и режим блока из
`(n_rows, n_features, n_strategies, наличие/регулярность времени)` — **детерминированно, наблюдаемо,
opt-in, без переопределения явного выбора**. Прецедент уже есть: CV `scheme="auto"` → `Task.default_cv_scheme`,
с записью резолва обратно в конфиг (`build.py:399`).

## Рассмотренные варианты
1. **Сделать `auto` дефолтом** арбитража/блока. — ❌ меняет fs-default fingerprint (M6e-репро ломается),
   нарушает «дефолты тождественны M6e». Отклонено.
2. **`auto` как opt-in значение сентинела, дефолт неизменен.** — ✅ паттерн CV `scheme="auto"`; аддитивно;
   fingerprint сохранён; пользователь осознанно включает.
3. **Авто-выбор ещё и `strategy`/`compare`** (какие рэнкеры). — ❌ эмпирика/мета-обучение, не выводится из
   формы данных без бенчмарка (R-AUTOCLAIM). Отложено в M9-пресеты.

## Решение
Расширяем сентинелами **только арбитраж и режим блока** (Вариант 2):
- `FSArbitration` += `"auto"`; **дефолт остаётся `"holdout"`**.
- `null_block_mode` += `"auto"`; **дефолт остаётся `"rank"`**.

**Где резолвится и write-back (уточнено после R1/R2):** чистая функция
`composition::resolve_fs_defaults(fs, *, n_rows, n_features, inner_n_splits, times, scheme, purge) ->
tuple[FeatureSelectionConfig, dict[str, str]]` → `(effective_fs, resolve_record)`; warnings — через `logger`.
- **Входы доступны в `facade.fit` post-read:** `ds_full.n_rows`, `ds_full.schema` (→ `n_features`),
  `ds_full.time()` (**единый источник Δt** — сырой массив или None), `cv.n_splits` (→ `inner_n_splits`),
  `cv.scheme`/`cv.purge`. **`is_classification` НЕ нужен** (убран после R2): C5-выполнимость уже
  обрабатывается runtime-гейтом `compare_features` (graceful-degrade), поэтому авто может выбрать nested на
  малой классификации — это **безопасно** (рантайм деградирует наблюдаемо), резолверу min-class-count не нужен.
- **Связь с pre-read `_resolve_fs` (R2):** `facade._resolve_fs` (facade.py:506-522, **pre-read**) остаётся
  только для seed (`random_state`). `resolve_fs_defaults` — **отдельный post-read шаг**, принимает уже-seeded
  `fs` как вход. `facade.fit` **переприсваивает** `effective_fs = resolve_fs_defaults(...)[0]` и подаёт его
  И в `build_default_components(feature_selection=effective_fs)`, И **переключает** строку
  `RunConfig(fs=...)` (facade.py:171) с `fs` на `effective_fs` — иначе манифест сохранит сентинел `auto`.
- `resolve_record` (resolve-провенанс: `arbitration_requested`/`arbitration_resolved_from`/
  `block_mode_*`) — для наблюдаемости (ADR-0058 §4); прокидывается в run-report секцией `fs_resolution`.
- Чистая `resolve_fs_defaults` юнит-тестируется напрямую (синтетические входы). Паттерн write-back — как CV
  `scheme="auto"` (build.py:399).

### §1 Лестница арбитража (`arbitration="auto"`)
По возрастанию цены выбирается **самый честный по силам** (затем — понижение под cost-budget ADR-0058):
- `n_rows < N_SMALL` → `nested_per_fold` (полностью честный, по силам на малых данных);
- `N_SMALL ≤ n_rows < N_MED` → `nested`;
- `n_rows ≥ N_MED` → `holdout` (масштаб; nested-стоимость не оправдана).
- **Single-strategy** (`compare` из одной стратегии или `compare=None`) → `holdout` (арбитражу нечего
  разрешать). Покрыто FR-FSF-1 (под-случай) + тест `test_auto_single_strategy_resolves_holdout`.
- **Timeseries + purge=0 анти-ликедж** (после R1): если `scheme="timeseries"` и `cv.purge==0` (и нет
  `label_time`), `auto` **не** выбирает leak-уязвимый `nested`/`nested_per_fold` (inner-переотбор учится на
  строках, смежных с outer-test) → понижает до `holdout`. Это не молчит: существующий boundary-WARNING
  (build.py:179-188) сохраняется. «Безопасный дефолт» не должен молча включать leak-режим.
- Пороги-кандидаты: `N_SMALL=2000`, `N_MED=20000` (консервативно; **M9-tunable**, не претендуют на
  оптимальность — R-AUTOCLAIM). Объявлены константами модуля резолва.

### §2 Режим блока (`null_block_mode="auto"`)
- Нет объявленной time-колонки (`times is None`) → `"rank"`.
- Есть time-колонка **и** ряд **нерегулярен**: коэффициент вариации Δt по **отсортированным** `ds_full.time()`
  (`std(Δt)/mean(Δt)`) > `CV_IRREG` (число, кандидат `0.25`) → `"time_window"`; иначе `"rank"` (регулярный
  ряд rank-блоки покрывает корректно). Δt считается по **тому же** `ds_full.time()`, что потом строит
  structure-блоки в `run_slice` (единый источник; тест-сторож на совпадение режима манифеста и фактического).
- При резолве в `"time_window"` без явного `null_block_window` → **производное окно** (число, не описание):
  `null_block_window = median(Δt) × null_block_size` ⇒ ≈ `null_block_size` точек на окно при регулярной
  плотности. Записывается обратно. (Замена расплывчатого `WINDOW_FACTOR` — детерминированно/тестируемо.)
- **Guard вырожденных Δt (после R2):** если `times` пуст/одноэлементен **или** `median(Δt) ≤ 0` /
  `mean(Δt) ≤ 0` (дубликаты времён — батч событий в одну метку) → fallback на `"rank"`, окно **не
  выводится** (иначе `null_block_window = 0` нарушил бы `Field(gt=0)` → ValidationError на write-back).
  Тест `test_block_auto_zero_dt_falls_back_to_rank`.
- **Оговорка outer_holdout (после R2):** при `outer_holdout>0` блоки в `run_slice` строятся на `dev`-срезе
  (`ds_full.take(dev_idx)`), а Δt-резолв — по `ds_full`. Принимается как допустимое приближение (окно,
  выведенное на full, применяется к dev); тест-сторож совпадения режима гоняется и с `outer_holdout>0`.

### §3 Инварианты резолва
- **Никогда не трогает явный non-auto** (NFR-FSF-6): резолв срабатывает только если поле == `"auto"`.
- **Детерминизм** (NFR-FSF-1): чистая арифметика по данным, без RNG.
- **Наблюдаемость** (NFR-FSF-4): эффективные значения + флаг «resolved from auto» в run-report.

## Последствия
- **+** Снижает порог входа в честные режимы без потери воспроизводимости; явный выбор неприкосновенен.
- **+** Делит лестницу арбитража и точку резолва с cost-budget (ADR-0058) — единый механизм.
- **−/риск:** пороги (`N_SMALL/N_MED/CV_IRREG/WINDOW_FACTOR`) — **эвристические, не оптимальные**; помечены
  M9-tunable; авто **не** претендует на «лучший» выбор, только на «безопасный дефолт» (R-AUTOCLAIM). Не
  спайкуется без бенчмарка — принято как консервативная политика, не как доказанная оптимизация.
- **−:** `auto` — не дефолт, поэтому «из коробки» поведение не меняется (осознанный trade-off ради
  fingerprint/back-compat; out-of-box smart-defaults → M9-пресеты).
