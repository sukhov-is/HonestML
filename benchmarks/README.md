# Honesty benchmark suite

Verifies the project's main claim — **the score reported at selection is honest** —
on a fixed, offline corpus, and keeps releases comparable.

## Protocol

For every dataset in `corpus.py` (sklearn built-ins + seeded synthetics, no
network) the runner fits `AutoML(task, random_state=SEED,
cv=CVConfig(outer_holdout=0.2))` in full mode, with no timeout-dependent options —
the run is deterministic.

Recorded per dataset: `selection_score` (the winner's leaderboard score),
`holdout_score` (the untouched outer holdout, scored once) and

```text
optimism = (selection_score - holdout_score) * sign(greater_is_better)
```

The orientation is applied by the runner and taken from the LIBRARY's metric —
`resolve_metric(report["metric"]).greater_is_better` — never hand-declared; report
scores are raw. The no-regress check orients the holdout comparison the same way:
for rmse/log_loss a GROWN holdout score is the degradation. Aggregates are
reported **per metric family** (AUC- and RMSE-scaled values are never averaged
together) as a diagnostic.

## Gate

`python benchmarks/run.py --check` — **no regress** against the committed
`baseline.json` (created at bootstrap, see below): per-dataset `optimism` and
`holdout_score` must not regress by more than the per-dataset `atol` recorded in
the baseline (improvements are unbounded).
An absolute optimism threshold is NOT gated in v1 (an initial run cannot certify
itself); a threshold will be fixed once enough release data has accumulated.

## Baseline provenance

- `results.json` carries no timings: two runs in ONE environment are byte-identical
  (pinned by a slow test on a tiny synthetic corpus).
- Cross-environment variation (OS/BLAS/PyPI drift) is excluded by canonization: the
  baseline is generated and updated ONLY by the CI job (`benchmark.yml`:
  `ubuntu-24.04` + `uv sync --frozen` from the committed lockfile). Do not commit a
  locally generated baseline.
- `atol` (default 0.02) absorbs minor noise; hand-tuned per-dataset values survive
  `--update-baseline` (the runner merges them). Widening an atol is a reviewed
  decision (PR + CHANGELOG), like any baseline refresh.

## Native-categorical cardinality gate (`native_cat_gate.py`)

A separate, single-purpose calibration that pins **`Task.native_cat_max_unique`** — the cap above
which a categorical column is routed to ordinal codes instead of natively into CatBoost/LightGBM
(ADR-0092/0093). It exists so the default is **derived, not arbitrary**.

```text
python benchmarks/native_cat_gate.py
```

For a sweep of category counts (rows fixed, one seeded categorical predictor with a fixed per-level
signal), the runner fits the **untuned** native path of each backend (the unprotected default — the
overfit knobs are tuned only under `preset="best"`) and records per cardinality:

- `native_oof` / `codes_oof` — honest k-fold OOF AUC, native vs the codes fallback (the "does native
  help?" axis — typically near-neutral, the design's honest finding);
- `overfit_gap` — train AUC minus native OOF AUC: the **downside-risk** signal the gate targets;
- `fit_seconds` — native fit wall-clock (diagnostic only; not reproducible, not used by the recommendation).

`recommend_cap()` returns the largest cardinality whose overfit gap stays within a margin of the
low-card baseline — the knee past which native ordered target statistics start memorizing the
thinly-populated levels. The recommendation (deterministic, unit-tested in `test_benchmarks.py`) is
rounded into the "tens" band and pinned as the `Task` default; the cap is a calibrated heuristic with
an opt-out (`native_cat_max_unique=None`), corroborated by the cardinality of genuinely useful
categoricals in real data (district/profession/product — tens) sitting well below id-like / `a__b`
cardinalities (hundreds+). It writes `native_cat_gate_results.json`; this is **not** a CI gate (it
fits models and reports timings) — it is a calibration tool, re-run consciously when the default is
revisited.

## Bootstrap and release

- **First-time bootstrap (once):** dispatch `benchmark.yml` with
  `update_baseline=true`, download the artifact and commit `baseline.json`
  (+ a CHANGELOG line).
- **Every release:** dispatch `benchmark.yml` (check mode) **on the commit being
  tagged**; paste the run URL into the GitHub Release notes (docs/releasing.md).
