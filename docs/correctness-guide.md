# Correctness guide

Библиотека раскрывает способ выбора модели, вычислительный контекст и границы
оценки качества. Ниже описаны механизмы контроля и ограничения. DEV-оценка после
адаптивного поиска не гарантирует качество на новых данных.

## Honest selection

- **Out-of-fold scoring.** Every candidate is scored on pooled OOF predictions of a
  shared CV split — never on its own training rows. Fold validity (disjointness,
  group integrity, time order) is enforced by `validate_fold` on every fold: a
  checked mechanism, not a convention.
- **Equivalence band, not bare argmax.** With `significance="bootstrap"` (the
  default) the winner is chosen through a seeded paired-bootstrap equivalence band:
  candidates statistically indistinguishable from the top score form a band, and
  the simplest band member wins. Two candidates are equivalent when the two-sided
  confidence interval of their metric difference over the pooled OOF includes 0.
  For time-series CV the bootstrap resamples whole test folds, since i.i.d. row
  resampling understates variance under autocorrelation. The full band is reported
  (`band_member_ids`, tie-break disclosed), not silently dropped; pure argmax is an
  explicit opt-out.
- **Воспроизводимый контекст.** Метрики не нормируются относительно кандидатов.
  Seed, разрешённая конфигурация, данные и версии библиотек входят в fingerprint
  для кеша и возобновления. Таймаут HPO и ресурсные ограничения поиска могут
  изменить выполненную работу и итог; соответствующие решения раскрыты в отчёте.

## Leakage controls

- **Подготовка признаков и адаптивный отбор.** Target encoding использует
  cross-fit; FS сравнивает признаки внутри DEV. После адаптивного выбора subset,
  параметров или семейства DEV-оценку следует читать как `post_search_dev`.
  Широкий контроль проверяет альтернативу полного набора на DEV; он не заменяет
  независимую внешнюю приёмку.
- **Time series.** `cv=CVConfig(scheme="timeseries")` orders folds by time value,
  not row position, and applies `purge`/`embargo` gaps; optional label-end times
  (`label_time`) implement the de Prado purge for labels that span a horizon.
  Group CV keeps each group on one side of every split.
- **Внешний holdout и finalize.** `CVConfig(outer_holdout=...)` отделяет данные
  оценки от DEV, используемого для поиска и калибровки. Оценка относится к
  DEV-модели, включая ансамбль, если он применён. При `finalize=True` модель
  затем обучается на всех данных; сохранённый score не является гарантированной
  границей её качества. Просмотренный holdout не служит независимым тестом
  последующих изменений: для приёмки заранее фиксируют внешний набор/период,
  метрику, допуск и сравниваемые предсказания.

## Small datasets

Honest selection needs enough rows for the statistics to mean anything. The
enforced floors fail fast (`ConfigError` before any model is fit) instead of
crashing mid-CV:

- **Stratified CV**: every class must fit in every fold — the least populated
  class needs at least `n_splits` rows.
- **K-fold**: at least one row per fold (`n_rows >= n_splits`).
- **Time-series CV**: `n_rows >= n_splits * n_test + purge + n_es + 1`, enforced
  by the splitter; the outer-holdout carve additionally requires enough dev rows
  for the folds and at least two classes in the holdout.

Soft guidance is warned about, not blocked: an outer holdout below ~30 rows is
high-variance and its single score should be treated as indicative; refinement
selection falls back to raw scores below its OOF floor
(`refinement_min_oof`, default 2000). Fold counts are not auto-adapted to the
data yet — choose `cv=` against the floors above.

## Artifacts and serving

- The artifact is a versioned directory with a sha256-checksummed manifest;
  corruption and naive substitution (a file swapped without rewriting the
  manifest) are detected at load. The manifest verifies integrity, not
  authenticity.
- **Trust model.** The default body is joblib/pickle — loading an artifact executes
  code, so load only artifacts you trust. Native boosting bodies
  (`model_format="native"`) are structural (no pickle) and round-trip exactly; a
  natively loaded LightGBM body is inference-only (refit raises). The native format
  covers boosting bodies only: a non-boosting winner or a shipped ensemble falls
  back to a joblib (pickle) body, and a fitted calibrator is always stored as
  `calibrator.joblib` (pickle) — so the trust rule above applies to every artifact,
  native or not.
- ONNX export is a parity-gated, export-only channel: parity against the native
  model is validated on your sample before any file is written, so a silently
  diverging graph is never shipped. The graph is the raw pre-calibration estimator
  (see [Known limitations](#known-limitations)).

## Reporting

The run report (`run_report_`) is the tracker-independent source of truth; the
MLflow tracker is a pure consumer of a copy of it. The numbers in your tracker can
never diverge from the local report.

## Known limitations

- **TEXT columns are not auto-detected**: free-text columns are typed as
  categorical (one category per unique string). Declare or drop them yourself.
- **Linear and baseline see categoricals as ordinal codes**: CatBoost and LightGBM
  now split on categoricals natively — CatBoost via per-fold ordered target
  statistics, LightGBM via `categorical_feature` — fit inside each fold so the
  target statistics never leak across folds (the early-stopping validation split is
  unweighted). The `linear`/`baseline` models still consume the integer codes,
  whose arbitrary numeric order can limit them on high-cardinality categoricals.
- **Early stopping внутри HPO.** При доступном ES-разбиении внутренний CV
  выделяет validation внутри train; inner-test не используется для остановки.
  IID-разбиения используют долю train, временные — настроенный ES-хвост.
  Явные `iterations`/`n_estimators` задают потолок ES-fit; итоговый refit
  использует медиану фактических DEV rounds.
- **Адаптивный размер subset.** Refinement выбирает размер по non-inferiority
  относительно лучшей точки той же OOF-траектории с допуском `refine_tol` и
  importance mass floor. Повторное использование этих оценок сохраняет риск
  оптимизма. Широкий контроль ограничивает риск регрессии на DEV, но не
  гарантирует отсутствие ухудшения на внешних данных. При возврате к полному
  набору `feature_selection=None` не означает отсутствия выполненной FS-работы.
- **Preprocessing and probability calibration are not part of the ONNX graph.**
  The graph consumes the numeric design matrix: rebuild it from the bundled
  `schema.json` (the categorical ordinal mapping is in
  `onnx_manifest.json: columns[].ordinal`). Calibrated models are disclosed in
  `onnx_manifest.json`; re-apply the mapping downstream. A boosting model trained
  with native categorical features has no such ordinal graph: `export_onnx` raises
  `NativeCategoricalONNXUnsupportedError` before writing anything, so the mapping
  above only applies to models without native categoricals (linear, or boosting on
  purely numeric data).
- **Privacy of artifacts**: the artifact schema stores category tables (raw
  category values) and feature names; the run report stores selected-feature names
  when feature selection is enabled. Treat artifacts, reports and tracker stores as
  data-bearing.

The configuration surface used here (`CVConfig`, `significance`, `finalize`,
`model_format`, `export_onnx`) is documented in the [API reference](api.md); for an
end-to-end run see the [quickstart](quickstart.md).
