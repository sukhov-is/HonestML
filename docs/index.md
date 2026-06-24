# honestml

honestml is a general tabular AutoML library for binary/multiclass
classification and regression. It is built around a clean, extensible core:
pluggable models, metrics, cross-validation, feature engineering/selection,
tuning, ensembling, serving and tracking.

The differentiator is the **honestly best model**: out-of-fold selection with a
bootstrap equivalence band, leakage-controlled feature engineering and
selection, an optional untouched outer holdout, and a tracker-independent run
report — the score you see is the score you can expect in production.

```bash
pip install honestml
```

- [Quickstart](quickstart.md) — fit, presets, reports, artifacts (examples run in CI).
- Guide — every capability with copy-paste examples: [data input](guide/data-input.md),
  [CV and honest selection](guide/cv-selection.md), [presets and budget](guide/presets-budget.md),
  [features](guide/feature-engineering-selection.md), [HPO and ensembling](guide/hpo-ensembling.md),
  [reports and tracking](guide/reports-tracking.md), [artifacts and ONNX](guide/artifacts-serving.md).
- [API reference](api.md) — the pinned public surface.
- [Correctness guide](correctness-guide.md) — why the scores are honest.
- [Plugin contract](plugin-contract.md) — ship your own model via entry points.
- [Versioning policy](versioning-policy.md) — what stays compatible across releases.

Maintainers: [Releasing](releasing.md).
