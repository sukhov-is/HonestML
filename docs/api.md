# API reference

Everything below is importable from the top-level `honestml` package. Heavy training
dependencies are imported lazily, so `import honestml` stays fast — loading an artifact
for serving never executes the training stack.

## Facade

After `fit`, the estimator exposes `best_model_id_` (the honest winner), `leaderboard_`
(absolute OOF scores), `fitted_` (the `FittedModel` serving handle for `save_artifact`)
and `run_report_` (the JSON-serializable run report).

::: honestml.AutoML
    options:
      members: ["fit", "predict", "predict_proba", "score", "available_models"]

## Artifacts and serving

`save_artifact(model.fitted_, path)` writes the fitted handle that `AutoML.fit` exposes
as the `fitted_` attribute; `load_artifact` returns it back as a `FittedModel` — the
lightweight serving handle.

::: honestml.save_artifact

::: honestml.load_artifact

::: honestml.FittedModel
    options:
      members: ["predict", "predict_proba", "score"]

::: honestml.export_onnx

## Run report

`save_run_report` writes the `run_report_` mapping produced by `AutoML.fit` as JSON;
`render_report` turns it into markdown or self-contained HTML.

::: honestml.save_run_report

::: honestml.render_report

## Configuration

`RunConfig` is the resolved run configuration that `AutoML.fit` records in the run
manifest (`run_report_["config"]`). You configure `AutoML` through its constructor
arguments, which accept the section classes below directly: `cv=CVConfig(...)`,
`budget=BudgetConfig(...)`, `feature_engineering=FEConfig(...)`,
`feature_selection=FeatureSelectionConfig(...)`, `hpo=HPOConfig(...)`,
`ensemble=EnsembleConfig(...)`. `TrackerConfig` stands apart: it configures the
experiment tracker passed through the `tracker` argument of `AutoML`.

::: honestml.RunConfig

::: honestml.CVConfig

::: honestml.BudgetConfig

::: honestml.FEConfig

::: honestml.FeatureSelectionConfig

::: honestml.HPOConfig

::: honestml.EnsembleConfig

::: honestml.TrackerConfig

## Data and selection types

`Task`, `FeatureSchema`, `ColumnRole` and `Dataset` describe the input data;
`SelectionPolicy`, `Candidate` and `select_best` implement final-model selection.

::: honestml.Task

::: honestml.FeatureSchema

::: honestml.ColumnRole

::: honestml.Dataset

::: honestml.SelectionPolicy

::: honestml.Candidate

::: honestml.select_best

## Runtime utilities

`honestml.__version__` — the installed package version.

::: honestml.RunContext

::: honestml.get_logger

## Exceptions

All errors derive from `honestml.AutoMLError`:

::: honestml.AutoMLError

::: honestml.ConfigError

::: honestml.SchemaValidationError

::: honestml.MissingDependencyError

::: honestml.ArtifactIntegrityError

::: honestml.NotFittedError

::: honestml.BudgetExhaustedError

::: honestml.FeatureSelectionError

`fit` may also raise more specific subclasses — notably `FitFailedError` (importable
from `honestml.core`) when every candidate fails; catch `AutoMLError` to cover them all.
