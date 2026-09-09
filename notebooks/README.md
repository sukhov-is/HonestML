# Showcase notebooks: honestml on real Kaggle cases

Восемь исполняемых примеров на реальных Kaggle-наборах показывают обучение, отчёт DEV/holdout и контекст опубликованных результатов соревнований. Они дополняют синтетические проверки benchmarks и не входят в CI: для полного выполнения нужны данные, Kaggle-доступ и длительные запуски обучения.

Ноутбуки 01–04 описывают IID-задачи; 05–08 добавляют multiclass, временные и групповые разбиения, а также реляционные признаки. В IEEE групповой и временной сценарии выполняются отдельно. Границы независимой оценки и версии контрольных прогонов указаны ниже.

| Notebook | Case | Task / metric | Comparison source |
| --- | --- | --- | --- |
| `01-titanic.ipynb` | [Titanic](https://www.kaggle.com/competitions/titanic) | binary / accuracy | public leaderboard (live submit) |
| `02-house-prices.ipynb` | [House Prices](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) | regression / rmse on log target (= LB RMSLE) | public leaderboard (live submit) |
| `03-adult-income.ipynb` | [Adult Census Income](https://www.kaggle.com/datasets/uciml/adult-census-income) | binary / roc_auc | published benchmark results |
| `04-credit-card-fraud.ipynb` | [Credit Card Fraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) | imbalanced binary / pr_auc | published benchmark results |
| `05-otto-product-classification.ipynb` | [Otto Group](https://www.kaggle.com/competitions/otto-group-product-classification-challenge) | **multiclass** / log_loss | published winner results |
| `06-store-sales.ipynb` | [Store Sales](https://www.kaggle.com/competitions/store-sales-time-series-forecasting) | regression / rmse on log1p (= LB RMSLE), **time-series CV** | public leaderboard |
| `07-ieee-fraud.ipynb` | [IEEE-CIS Fraud](https://www.kaggle.com/competitions/ieee-fraud-detection) | imbalanced binary / roc_auc, **group / time CV, separate runs** | public leaderboard |
| `08-home-credit.ipynb` | [Home Credit](https://www.kaggle.com/competitions/home-credit-default-risk) | imbalanced binary / roc_auc, relational feature store | public leaderboard |

## Контрольные прогоны и интерпретация

Итоговая проза ноутбуков опирается на сохранённые отчёты 8–9 сентября 2026 года. Встроенные outputs относятся к собственным исполнениям ячеек; итоговые таблицы ссылаются на отдельные контрольные прогоны. Режимы серии:

- **baseline** — обычный поиск по исходному сценарию;
- **fast / frozen fast** — ограниченный поиск с confirmation_rows=16384;
- **final fast** — отдельные завершённые Otto/Home-прогоны с confirmation_rows=65536 и собственными source SHA.

Контрольный runner использовал seed 42, четыре native/BLAS-потока, выключенный кеш, HPO до пяти trials на семейство при его включении и отключённые отправки Kaggle. Код исходных ячеек сохраняет свои бюджеты HPO и не включает SearchConfig; прямой запуск этих ячеек не воспроизводит fast-таблицу автоматически. Фактические параметры, статус, выполненные fits и SHA доступны в связанных отчётах.

Время в таблицах — elapsed_s одного AutoML.fit. Смена семейства, subset, ансамбля, снимка кода и нагрузки ограничивает причинную интерпретацию отношения времён. DEV-пробы выбора семейства и изолированные FS-пробы описаны отдельно от полных AutoML fit. Незавершённые попытки не считаются успешными notebook-прогонами.

**Независимая приёмка качества открыта.** Holdout этой серии уже просмотрен при анализе, поэтому не служит нетронутым тестом последующих изменений. Для приёмки требуется заранее согласовать внешний набор/период, метрику и допустимое ухудшение, зафиксировать модели и предсказания до раскрытия меток. Для ансамбля его holdout нельзя автоматически сопоставлять с DEV score опорной одиночной модели.

Условный прогноз стоимости FS/HPO и протокол фиксации качества описаны в [руководстве по стоимости обучения](../docs/training-performance.md). Их [короткая синтетическая проверка](../docs/training-results.md#forecast) не является повторным прогоном этих ноутбуков и не подтверждает точность прогноза на churn.

Основные источники: [сравнительная серия](../docs/training-results.md#comparison), [final Otto](../docs/training-results.md#final-runs), [final Home](../docs/training-results.md#final-runs), [FS-пробы](../docs/training-results.md#targeted).

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
