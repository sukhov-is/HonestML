# Showcase notebooks: honestml on real Kaggle cases

Executable, end-to-end case studies on real datasets: fit the full honestml
pipeline, read the honest run report (selection score vs untouched outer
holdout), and compare the result against what Kaggle participants actually
score. They complement `benchmarks/` (the offline, CI-gated honesty suite):
these notebooks are **not** run in CI — they need network, Kaggle credentials
and minutes-long fits.

Notebooks `01`–`04` are i.i.d. tabular cases. `05`–`08` extend the coverage to
**multiclass**, **time-series** and **group-aware** cross-validation, and large
noisy **relational feature stores** — the regimes where the honest contour and
the level-2 machinery (HPO, feature selection, ensembling) have the most to say.

| Notebook | Case | Task / metric | Comparison source |
| --- | --- | --- | --- |
| `01-titanic.ipynb` | [Titanic](https://www.kaggle.com/competitions/titanic) | binary / accuracy | public leaderboard (live submit) |
| `02-house-prices.ipynb` | [House Prices](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) | regression / rmse on log target (= LB RMSLE) | public leaderboard (live submit) |
| `03-adult-income.ipynb` | [Adult Census Income](https://www.kaggle.com/datasets/uciml/adult-census-income) | binary / roc_auc | published benchmark results |
| `04-credit-card-fraud.ipynb` | [Credit Card Fraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) | imbalanced binary / pr_auc | published benchmark results |
| `05-otto-product-classification.ipynb` | [Otto Group](https://www.kaggle.com/competitions/otto-group-product-classification-challenge) | **multiclass** / log_loss | published winner results |
| `06-store-sales.ipynb` | [Store Sales](https://www.kaggle.com/competitions/store-sales-time-series-forecasting) | regression / rmse on log1p (= LB RMSLE), **time-series CV** | public leaderboard |
| `07-ieee-fraud.ipynb` | [IEEE-CIS Fraud](https://www.kaggle.com/competitions/ieee-fraud-detection) | imbalanced binary / roc_auc, **group + time CV** | public leaderboard |
| `08-home-credit.ipynb` | [Home Credit](https://www.kaggle.com/competitions/home-credit-default-risk) | imbalanced binary / roc_auc, relational feature store | public leaderboard |

## Setup

1. The project dev environment (from the repo root):
   `uv sync --extra dev --extra boosting --extra shap --extra optuna --extra pyarrow`
   (pyarrow is required: real CSVs have string columns, and the pandas->polars
   boundary needs it),
   then `uv pip install -r notebooks/requirements.txt` (jupyter tooling).
2. The Kaggle CLI v2+ (needs Python >= 3.11, so it lives OUTSIDE the project
   venv): `uv tool install --python 3.12 kaggle`.
3. A Kaggle API token in the `KAGGLE_API_TOKEN` environment variable (create
   one at kaggle.com -> Settings -> API). The repo root `.env` is gitignored —
   keep it there and export it before running.
4. Data download by notebook:
   - `01`/`02`/`06` pull **official competition** files — accept the competition
     rules once on the Kaggle website (the "Join Competition" button, or "I
     Understand and Accept" on the Rules tab of a finished competition), else the
     download returns 403;
   - `05`/`07`/`08` download from community Kaggle **dataset mirrors** of finished
     competitions (identical files, no rules acceptance needed);
   - `03`/`04` are plain Kaggle datasets (no rules).

## Running

Each notebook downloads its data on first run (`data/`, gitignored) and writes
reports and submissions to `results/` (gitignored). Run interactively, or
headless:

```text
uv run papermill notebooks/01-titanic.ipynb notebooks/01-titanic.ipynb
```

Submission cells are skipped unless `KAGGLE_SUBMIT=1` is set — so the
notebooks stay fully executable without touching the leaderboard.
