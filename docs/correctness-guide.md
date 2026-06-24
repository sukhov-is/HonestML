# Correctness guide

The library's claim is an **honestly best model**: the score you see is the score
you can expect. This page lists the enforced mechanisms behind that claim — each is
a tested mechanism, not an aspiration — and the known limitations.

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
- **Absolute, reproducible scores.** Metrics are absolute — no candidate-relative
  normalization. The pipeline is seeded end to end, so the same inputs give the
  same selection (the one opt-out is a wall-clock-capped HPO search,
  `HPOConfig.timeout_s`, disclosed in the run report); the run-fingerprint — a hash
  over seeds, resolved config, data signature and library versions — identifies the
  run for caching and resume.

## Leakage controls

- **Feature engineering and selection are OOF-honest.** Target encodings are
  cross-fitted out-of-fold for evaluation, so a row never sees its own fold's
  target; frequency encoding is target-independent, so its full-train fit cannot
  leak the label. Feature selection arbitrates on an internal selection holdout or
  nested DEV folds — never on the selection OOF itself.
- **Time series.** `cv=CVConfig(scheme="timeseries")` orders folds by time value,
  not row position, and applies `purge`/`embargo` gaps; optional label-end times
  (`label_time`) implement the de Prado purge for labels that span a horizon.
  Group CV keeps each group on one side of every split.
- **Outer holdout.** `CVConfig(outer_holdout=...)` carves an untouched share once.
  Selection, refit and calibration see only DEV — the development split, everything
  outside the holdout; the winner is scored on the holdout a single time, and the
  report keeps that DEV-vs-holdout separation. With `finalize=True` the *shipped*
  model is refit on all data AFTER scoring — the reported score remains the holdout
  score of the DEV-trained model, a conservative bound for the finalized all-data
  model; it is never re-measured after the refit.

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
- **Early stopping is not used inside HPO**: boosting models early-stop on the
  selection folds (the `n_es` tail carved from each fold's training rows is held out as
  the validation set on every CV scheme, ADR-0080), but **not** inside inner-CV
  hyperparameter search — there a tuned `n_estimators` is a ceiling that early stopping
  later trims on the selection folds.
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
