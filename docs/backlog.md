# Backlog: findings from the showcase-notebook runs (2026-06-13)

Issues and improvement candidates observed while running the full pipeline on
real Kaggle datasets (`notebooks/01..04`). Each item has the evidence in the
executed notebook outputs. Ordered by severity.

> **Status (2026-06-20): every finding below is resolved for the v1.0.0 release.**
> Quick map (full evidence inline per finding):
>
> | # | Finding | Resolution |
> |---|---------|------------|
> | 1 | Untuned boosting collapse on extreme imbalance | Closed by early stopping (#2). The `scale_pos_weight` attempt (ADR-0079) was implemented then **reverted** — on real fraud data ES is the cure and `scale_pos_weight` broke lgbm/catboost under ES |
> | 2 | Boosting trained without early stopping | **ADR-0080** — ES default-ON for boosting across all CV schemes (i.i.d. + group + time) |
> | 3 | pyarrow required but not installed by any extra | `MissingDependencyError("pyarrow")` raised at the Reader boundary with the install hint |
> | 4 | feature-name mismatch warnings flood logs | silenced via `_quiet_feature_names` around boosting predict (root: lightgbm auto-names numpy cols) |
> | 5 | lbfgs fails to converge on linear | `StandardScaler` inside the linear `Pipeline` (scaling makes max_iter sufficient) |
> | 6 | NaN shrinks the zoo to boosting-only | **ADR-0078** — `SimpleImputer` prepended to linear/baseline pipeline; `handles_missing=True` |
> | 7 | near-unique string columns priced wrong | `string_id_like` role → IGNORE (Task thresholds + Reader); high-cardinality strings flagged in the typing report |
> | 8 | int categoricals break on NaN-float drift | **ADR-0017** — value-preserving dtype coercion; **re-verified 2026-06-20** on the real house-prices CSVs + 4 regression tests |
> | 8a | all-null / constant columns mistyped | role IGNORE (any dtype) detected at read time, which also stops the #6 NaN-gate cascade |
> | 9 | ensemble crash on hard-label metrics | label projection before scoring + facade fault-isolation (a failed ensemble degrades, never kills the fit) |
> | 10 | feature selection has no no-selection gate | `no_selection_gate` — ship the subset only when not significantly worse than the full set |
> | 11 | row-wise outer holdout fooled by groups | group-aware outer carve when `groups=` + TE group-structure WARNING + holdout-optimism diagnostic |

## 1. Untuned LightGBM/XGBoost collapse on extreme imbalance

**Evidence:** `04-credit-card-fraud.ipynb` (284,807 rows, 0.17% positives,
`metric="pr_auc"`): leaderboard PR-AUC — catboost 0.838, linear 0.700,
**lightgbm 0.113, xgboost 0.026** (baseline 0.002). The two collapsed models
score *below logistic regression*; their defaults never learn to rank the
rare class. The honest OOF ranking caught it, but two of three boosting
candidates were wasted compute.

**Fix direction:** imbalance-aware defaults in the boosting adapters when the
positive rate is below a threshold (`scale_pos_weight` / `is_unbalance` from
the observed class ratio), and/or wiring early stopping (see #2), which the
adapters themselves flag as missing.

**Update (level-2 run):** Optuna HPO fully rescues both collapsed models —
xgboost PR-AUC 0.026 -> 0.797, lightgbm 0.113 -> 0.790 (20 trials). So the
collapse is purely a defaults problem, and the fix above would deliver
HPO-grade robustness for free.

## 2. Boosting trained without early stopping (adapter's own warning)

**Evidence:** every fit logs
`WARNING honestml.adapters.boosting: boosting '<id>' trained without early
stopping; leaderboard comparison may favor overfit settings` for all three
boostings, on all four datasets. `CVConfig` already has an `n_es` parameter,
so the contour exists but is evidently not active on the default path.

**Fix direction:** activate the ES split for boosting candidates by default
(or document why not); the warning firing on 100% of real runs means the
default config warns about itself.

## 3. pyarrow is effectively required, but not installed by any training extra

**Evidence:** first run of `01-titanic.ipynb` failed inside
`Reader._to_frame` -> `pl.from_pandas(X)`:
`ImportError: pyarrow is required for converting a pandas dataframe to
Polars, unless each of its columns is a simple numpy-backed one`. Any pandas
DataFrame with a string/object column — i.e. every real CSV — hits this.
The pyproject comment sells pyarrow as an optional accelerator ("the
numpy+codes path does not require it"), which is true only for all-numeric
input.

**Fix direction:** either add pyarrow to the boosting/dev/all install
guidance for training on DataFrames, or catch the ImportError at the Reader
boundary and re-raise as `MissingDependencyError("pyarrow", ...)` with the
install hint (ADR-0008 taxonomy); the raw polars ImportError surfaces from
deep inside the stack.

## 4. Feature-name mismatch warnings between fit and predict paths

**Evidence:** all notebooks spam
`sklearn UserWarning: X does not have valid feature names, but
LGBMClassifier was fitted with feature names` (dozens of repetitions per
fit). The adapters fit on a named frame but predict on a bare ndarray (or
vice versa).

**Fix direction:** materialize the matrix identically on both paths (both
named or both unnamed). Cosmetic, but it floods real-run logs and drowns the
library's own honesty-relevant warnings.

## 5. Linear candidate: lbfgs hits max_iter=1000 without converging

**Evidence:** `03-adult-income.ipynb` and `04-credit-card-fraud.ipynb` log
sklearn `ConvergenceWarning: lbfgs failed to converge (max_iter=1000)` on
every fold for the `linear` candidate. Suggests features reach the solver
unscaled (capabilities declare `needs_scaling=False`) or the cap is too low
for 30k+ rows.

**Fix direction:** verify the scaling step in the linear pipeline; if absent,
add it (or raise max_iter). An unconverged linear baseline understates the
honest reference point the zoo is supposed to provide.

## 6. NaN in numeric features silently shrinks the zoo to boosting-only

**Evidence:** `01-titanic.ipynb`: `WARNING honestml.composition.build: data
contains NaN in numeric features; skipping models that require imputed
input: ['baseline', 'linear']` — the leaderboard had no non-boosting
candidates at all.

**Fix direction:** an imputation step for the models that need it would keep
the simple candidates in the comparison (the significance band loses its
"simplest member" tiebreak candidates otherwise). At minimum, document the
behavior; it surprises on the most classic dataset there is.

## 7. Near-unique string columns become categorical features — internal estimates can't price them

**Evidence:** `01-titanic.ipynb`, two live submissions. `Name` (100% unique)
is auto-typed categorical; at inference the Reader warns `100% of 'Name'
values were unseen at train` (`Ticket` 64%, `Cabin` 11%). With the raw
columns: holdout said 0.7933, real Kaggle test gave **0.75598** (below the
no-model gender baseline 0.76555). WITHOUT the three columns: internal
numbers got consistent (OOF 0.799 / holdout 0.805) but the real test fell
further to **0.73684** — the partially-shared `Ticket`/`Cabin` vocabulary
carries genuine family/deck signal. So neither "keep" nor "drop" is priced
correctly by any internal estimate: vocabulary overlap between an internal
split and the training file is structurally higher than with real inference
traffic. (Public-test noise on ~400 rows is ±0.02-0.03, part of the gap.)

**Fix direction:** ADR-0015 territory, two separable pieces: (a) a
`high_cardinality`/`id` auto-typing rule for ~100%-unique string columns like
`Name` — as a category it is pure noise at inference by construction;
(b) for partially-overlapping ones, the honest contour cannot measure
transfer — at minimum the typing report should flag train-side near-unique
columns at FIT time (not only warn at predict time), so the user decides
before shipping.

## 11. Row-wise outer holdout + target encoding is fooled by group-structured rows

**Evidence:** `01-titanic.ipynb` level-2 (TE on, FS kept `Ticket_te`,
`Sex__Ticket`, `Cabin_freq`...): OOF selection 0.8188, untouched holdout
**0.9609** (optimism −0.142), live Kaggle test **0.74401**. The holdout
overpromised by 22 accuracy points. Mechanism: families share a ticket AND an
outcome; the outer carve is row-wise, so most holdout passengers have
relatives in DEV; target encoding then carries the relatives' survival into
the holdout rows. The encoder itself is honestly cross-fit — what broke is
the split's independence assumption, which no amount of within-file honesty
can repair.

**Fix direction:** three layers:
(a) when `groups=` is passed, the outer-holdout carve must be group-aware
(verify whether it already is — CV folds are, per ADR-0023/0025);
(b) auto-suggest group structure: a categorical column whose values repeat
across rows AND which feeds TE is a red flag worth a WARNING at fit time;
(c) report-side sanity: optimism strongly negative relative to the band width
(here −0.142) is itself diagnostic of split dependence — the report should
say so instead of printing a quietly absurd holdout number.

## 10. Feature selection has no honest gate against the no-selection baseline

**Evidence:** the level-2 runs (2026-06-14) enabled FS on three cases and the
shipped model regressed on the untouched holdout in ALL three, traceable to
the auto cutoff being aggressive:

- House Prices: kept 21 of 188 (post-FE) features -> holdout RMSLE 0.134 ->
  0.171, live LB 0.1297 -> 0.1561 (percentile 58.6% -> 21.3%);
- Adult: kept 27 of 68 -> holdout AUC 0.9218 -> 0.9132;
- Fraud: kept 10 of 30 -> holdout PR-AUC 0.8765 -> 0.793 (while HPO had just
  rescued the candidates, see #1 update — FS then threw the gain away).

The honest contour *reports* the damage (selection/holdout both worse), but
nothing *prevents* it: unlike ensembling — which must significantly beat the
single winner to ship (ADR-0063 gate) — a feature-selection config never has
to beat the no-selection baseline. `compare=` arbitrates strategy-vs-strategy
only; "no selection" is not an arm.

**Fix direction:** add the no-selection baseline as an implicit arm of the FS
arbitration (its OOF candidates exist anyway in a non-FS run shape), with the
same gate semantics as ensembling: ship the selected subset only when it is
not significantly worse (or: only when better). At minimum, WARN when the
arbitration score of the chosen strategy is worse than a cheap full-feature
control fold.

## 9. Ensembling crashes on threshold metrics (accuracy) — and the crash is not isolated

**Evidence:** `01-titanic.ipynb` level-2 run (2026-06-14): `AutoML(task="binary",
metric="accuracy", preset="best", ...)` — selection + HPO completed honestly
(188s), then the ensemble stage raised
`ValueError: Classification metrics can't handle a mix of binary and
continuous targets` from inside `ensemble_selection` (facade.py:429) and the
WHOLE fit died. The Caruana blend mixes candidate probabilities (continuous)
and scores the blend with the selection metric; for hard-label metrics
(accuracy) nothing thresholds the blended probabilities first. Continuous
metrics (roc_auc/pr_auc/rmse) are unaffected — which is why the Adult level-1
ensemble worked.

**Fix direction:** two separable defects:
(a) in `ensemble_selection`, project blended probabilities to labels before
hard-label metrics (the inference path already knows how to do this), or
score the blend on the metric's required input type via the metric adapter's
contract;
(b) fault-isolate the ensemble stage like candidate failures (ADR-0022): it
is optional and post-selection — an honest winner already exists, so the
correct degraded outcome is `ensemble: {applied: false, gate_reason:
"failed: <err>"}` + WARNING, not a dead fit after minutes of valid work.

## 8a. Auto-typing has no rule for all-null and constant columns

**Evidence:** synthetic check through `Reader` (2026-06-13): an all-NaN column
keeps role `numeric` (not dropped) and a constant column keeps role `numeric`.
The all-NaN case cascades: it triggers the "NaN in numeric features" rule and
silently evicts baseline/linear from the candidate zoo (see #6) — one garbage
column impoverishes the whole leaderboard. Related: the id-like drop
(`numeric_id_like`) exists for NUMERIC columns only; string columns get the
baseline `Utf8 -> categorical` with no cardinality check at all (see #7) —
a numeric `PassengerId` is dropped, its string twin is not.

**Fix direction:** ADR-0015 extension: `all_null` and `constant` reasons in
the typing report with role `ignore` (cheap to detect at read time), and the
same composite id-like rule applied to string columns.

## 8. Low-cardinality-int categoricals break on NaN-induced float dtype at inference

**Evidence:** `02-house-prices.ipynb`: train `BsmtFullBath`/`BsmtHalfBath`/
`GarageCars` are int64 -> auto-typed categorical (`low_cardinality_int`). In
`test.csv` those columns contain NaN, so pandas reads them as float64 — and
the Reader reports **"100% of values were unseen at train"**: float `1.0`
does not match the train category `1`. Three real features were silently
nulled to `unknown_code` for every inference row (the run still scored LB
RMSLE 0.12973, i.e. this cost actual accuracy invisibly).

**Fix direction:** dtype coercion at the inference boundary (ADR-0017): cast
the incoming column to the schema's stored source dtype (nullable Int64)
before category-code lookup, so `1.0`/`<NA>` matches the train vocabulary;
the 100%-unseen warning should probably also fail-fast-or-explain when the
dtype differs from the schema's recorded one, since that is schema drift the
schema can actually diagnose.

**Update (level-2 run):** the bug propagates into FE — intersection features
built on the affected columns (`3SsnPorch__GarageCars` etc.) are also 100%
unseen at inference, so each broken source column now silently nulls several
derived features too.

---

# Release plan: remaining steps (target v1.0.0)

All v0.1 findings (#1–#11, #8a) are resolved, and the items once deferred to
v0.2 (NaN imputer #6, boosting early stopping #2, imbalance defaults #1) were
pulled forward — #6 and #2 shipped (ADR-0078/0080), #1 was implemented and
**reverted** (ADR-0079; ES is the real cure). Full suite green (1012 passed /
24 skipped as of 2026-06-20). **No commits yet** — publication is gated on the
docs-readiness sign-off.

## Phase 2 — first publication (blocked on the owner's go)

1. First commit + push to `github.com/sukhov-is/HonestML`.
2. One-time manual setup (owner-only, see `docs/releasing.md` → One-time
   setup): PyPI **Trusted Publisher** (`release.yml` + environment `pypi`),
   **environment protection rules** for `pypi`, **GitHub Pages** source =
   GitHub Actions.
3. Bootstrap `benchmarks/baseline.json`: dispatch `benchmark.yml` with
   `update_baseline: true`, commit the baseline + a CHANGELOG line
   (`docs/releasing.md` → First release).

## Phase 3 — v1.0.0

4. Per-release checklist in `docs/releasing.md`: cut `[Unreleased]` →
   `[1.0.0]` in the CHANGELOG, tag `v1.0.0`, the pipeline does the rest
   (`pyproject.toml`, `honestml.__version__` and the `test_public_api` pin are
   already aligned at 1.0.0).

## Deferred to post-1.0 (decided before the release)

- **Native categorical in boostings** (ADR + ONNX re-spike + parity +
  serialization) — a documented limitation for v1.0.0.
- **Word/regulatory "model development report"** (docx via pandoc over
  `render_report` markdown) — blocked on the required document structure
  from the owner.
- **Small-data policy, full version** — data-driven folds/holdout resolution
  with run-report disclosure (the lite version shipped: fail-fast floors +
  the correctness-guide "Small datasets" section).
- **Independent-OOF significance band for the sequential feature count** —
  the band is currently scored on the selection folds (same-OOF, ADR-0085 §5);
  nested independent-OOF scoring removes the residual optimism.
