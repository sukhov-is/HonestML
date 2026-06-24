# Artifacts, serving and ONNX export

How a fitted model leaves the training process: a versioned artifact directory
for serving with honestml, and an optional ONNX bundle for external runtimes.
Every `python` block on this page is **self-contained**: copy any one of them
and it runs as-is, and every block is executed on each CI run, so the examples
cannot rot.

## Saving an artifact

`save_artifact(model.fitted_, directory)` serializes the shipped model
(`fitted_` is the `FittedModel` behind the facade) into a self-contained
versioned directory: `manifest.json` (versioned metadata plus provenance — the
winner, the equivalence band, calibration, holdout score), `schema.json` (the
serialized preprocessing schema, the single source of truth that keeps train
and inference identical), the model body, and `leaderboard.json`. The manifest
carries a `checksums` block — sha256 of every file plus a digest of the
manifest payload — so corruption or naive file substitution is detected before
anything is deserialized. An optional `sign=` hook signs the manifest digest,
enabling an authenticated `verify=` on load.

```python
import json
import tempfile
from pathlib import Path

from sklearn.datasets import make_classification

from honestml import AutoML, save_artifact

X, y = make_classification(n_samples=150, n_features=6, n_informative=4, random_state=0)
model = AutoML(task="binary", models=("baseline", "linear"), random_state=0).fit(X, y)

art = Path(tempfile.mkdtemp())
save_artifact(model.fitted_, art)

print(sorted(p.name for p in art.iterdir()))
manifest = json.loads((art / "manifest.json").read_text(encoding="utf-8"))
print(manifest["best_model_id"], manifest["checksums"]["algo"])
```

## Loading and predicting

`load_artifact(directory)` verifies the checksums first, then returns a
`FittedModel` — the exact inference path the facade itself uses — with
`predict`, `predict_proba` (classification) and `score`. Prediction does not
need the training stack: preprocessing is rebuilt from `schema.json`, and a
standalone load that only predicts never resolves the AutoML metric or any
selection machinery. `require_integrity=True` turns a missing checksums block
(an artifact written by an older build) from a warning into an error.

```python
import tempfile

import numpy as np
from sklearn.datasets import make_classification

from honestml import AutoML, load_artifact, save_artifact

X, y = make_classification(n_samples=150, n_features=6, n_informative=4, random_state=0)
model = AutoML(task="binary", models=("baseline", "linear"), random_state=0).fit(X, y)

art = tempfile.mkdtemp()
save_artifact(model.fitted_, art)
loaded = load_artifact(art)

print(np.array_equal(loaded.predict(X), model.predict(X)))  # True
print(loaded.best_model_id, loaded.predict_proba(X[:3]).shape)
```

## The trust model: pickle vs native

The default body is `model.joblib` — joblib, which is pickle, and unpickling
executes code: **load only artifacts you trust**. The sha256 manifest detects
corruption and naive substitution, not malice — a malicious author can embed
code with matching digests, so authenticity needs the `sign=`/`verify=` hooks
and a trusted source. For boosting models, `model_format="native"` stores the
body in the library's stable structural format instead (XGBoost UBJSON,
CatBoost cbm, LightGBM text): no pickle in the body, and the file survives
library upgrades that break pickle. Two disclosed caveats: anything without a
native format (sklearn models, a shipped ensemble) transparently stays joblib,
and a fitted probability calibrator is always stored as `calibrator.joblib` —
pickle even inside an otherwise native artifact.

```python
import json
import tempfile
from pathlib import Path

import numpy as np
from sklearn.datasets import make_classification

from honestml import AutoML, load_artifact, save_artifact

X, y = make_classification(n_samples=150, n_features=6, n_informative=4, random_state=0)
model = AutoML(task="binary", models=("lightgbm",), random_state=0).fit(X, y)

art = Path(tempfile.mkdtemp())
save_artifact(model.fitted_, art, model_format="native")

manifest = json.loads((art / "manifest.json").read_text(encoding="utf-8"))
print(manifest["model_type"], manifest["model_file"])  # lightgbm model.txt
print(np.array_equal(load_artifact(art).predict(X), model.predict(X)))  # exact round-trip
```

## Slim serving with honestml[inference]

The `inference` extra installs only what loading and predicting needs (numpy,
pandas, polars, scikit-learn, pydantic, joblib) — no optuna, no boosting
libraries, no training-only dependencies. `import honestml` pulls just the pure
core, and `load_artifact(...).predict(...)` resolves lazily, so the serving
process never executes the training stack. A native boosting artifact records
its `required_extra` in the manifest; the serving box then additionally needs
that one library (for example `pip install lightgbm`), and a missing runtime
surfaces as `MissingDependencyError` before any deserialization.

```text
pip install "honestml[inference]"   # slim runtime: load artifacts + predict

from honestml import load_artifact

loaded = load_artifact("artifacts/churn-v3")
predictions = loaded.predict(X_new)
```

## ONNX export

`export_onnx(model.fitted_, directory, sample=...)` converts a supported
estimator (linear, lightgbm, xgboost, catboost) into an export-only bundle for
external runtimes: `model.onnx`, `schema.json`, `onnx_manifest.json` and a
`README.md`. A boosting model trained with native categorical features is the
exception — it cannot be represented as a numeric-design-matrix graph, so
`export_onnx` raises `NativeCategoricalONNXUnsupportedError` before writing
anything; keep such models in joblib or native format. The graph is the raw
estimator over the **numeric design
matrix** — it contains neither preprocessing (the consumer reproduces the
design matrix from the bundled schema: columns in the manifest's
`feature_order`, categoricals fed as ordinal codes) nor calibration (disclosed
in the manifest). `sample` is required because the export is parity-gated:
before any file is written, the converted graph is compared against the native
estimator on that sample, and a divergence raises instead of shipping a
silently wrong graph — only a near-tie label flip within float32 noise is
downgraded to a WARNING and recorded in the manifest. The bundle is one-way:
it is not loadable back into honestml, so round-trips stay with
`save_artifact`.

```text
pip install "honestml[onnx]"   # onnx, onnxruntime, skl2onnx, onnxmltools

from honestml import export_onnx

parity = export_onnx(model.fitted_, "onnx_bundle", sample=X[:100])
print(parity)   # e.g. {'proba_max_abs': 1.2e-07, 'label_verdict': 'ok', ...}
```

Where the model came from and how it was selected is recorded next to it — see
[run reports and experiment tracking](reports-tracking.md).
