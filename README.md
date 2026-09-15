# Telco Customer Churn: from model selection to score analysis

An end-to-end machine-learning project that predicts customer churn and turns
probabilities into a ranked score table for technical review. The analysis goes
beyond model accuracy: it compares imbalance strategies, selects the decision
threshold under hypothetical error-cost weights, freezes score deciles on the
training distribution, and
keeps the final holdout set untouched until every modeling decision is locked.

The notebook is written in Italian; this README and the model card summarize the
method and results in English.

This portfolio project was developed as part of my training as a Machine
Learning/AI Engineer. It demonstrates a reproducible prototype, not a production
deployment or a professional client engagement.

## What this project demonstrates

- A single scikit-learn pipeline for feature engineering, scaling, encoding and
  classification, with deterministic handling of unseen categories.
- A numerical comparison of no correction, class weights and SMOTE across
  Logistic Regression, Random Forest and XGBoost.
- Five-fold out-of-fold predictions for model comparison, threshold selection
  and reference risk deciles.
- A final holdout evaluation performed once, without retuning.
- Global and local interpretability: logistic coefficients, SHAP for XGBoost,
  and customer-level reasons attached to each prediction.
- A serialized model with its threshold, decile boundaries and explanation
  baseline embedded in the same artifact.
- Strict validation of numeric fields and missing or blank categorical values,
  plus an end-to-end test covering Excel input, real model loading, inference
  and CSV output in a new Python process.

## Final holdout results

The selected Logistic Regression uses a threshold of `0.4265`, chosen from
out-of-fold training predictions with a false-negative to false-positive cost
ratio of 5:1.

| Model | Threshold | ROC-AUC | PR-AUC | Precision | Recall | F1 | FN | FP | Cost units |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression, selected | 0.4265 | 0.854 | 0.675 | 0.490 | 0.837 | 0.618 | 61 | 326 | 631 |
| XGBoost, locked benchmark | 0.3775 | 0.854 | 0.670 | 0.481 | 0.877 | 0.621 | 46 | 354 | 584 |

XGBoost achieved the lower cost under the chosen hypothetical cost ratio. The
Logistic Regression was retained because its cross-validated ROC-AUC
(`0.8617 ± 0.0124`) was within the one-standard-error range of XGBoost
(`0.8641 ± 0.0098`), while offering a smaller, directly interpretable model.
This is a modeling choice, not a claim that the logistic model dominates every
business scenario. The notebook reports the full sensitivity analysis.

## Repository contents

```text
.
├── telco_customer_churn_analysis.ipynb  # executed end-to-end analysis
├── churn_inference.py                   # preprocessing and inference contract
├── pipeline_churn.joblib                # fitted Logistic Regression pipeline
├── predict_churn.py                     # Excel-to-CSV command-line example
├── MODEL_CARD.md                        # intended use, metrics and limitations
├── DATASET.md                           # source and data contract
├── examples/
│   └── customers_sample.xlsx            # synthetic inference-only sample
└── tests/
    ├── test_churn_inference.py
    └── test_end_to_end.py
```

## Dataset

The training dataset is not redistributed in this repository. Download
`Telco_customer_churn.xlsx` from the
[IBM Telco Customer Churn dataset on Kaggle](https://www.kaggle.com/datasets/yeanzc/telco-customer-churn-ibm-dataset)
and place it next to the notebook. Alternatively, set `TELCO_CHURN_DATA` to the
file path. On Kaggle, the notebook also searches the attached input datasets.

See [DATASET.md](DATASET.md) for the expected schema and source notes.

## Run inference

Use CPython 3.12. The model is serialized with scikit-learn 1.9.0, so the pinned
environment is part of the artifact contract.

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python predict_churn.py
```

The default command reads the synthetic Excel sample and writes
`outputs/churn_scores_above_threshold.csv`. Custom paths are supported:

```powershell
.venv\Scripts\python predict_churn.py `
  --input path\to\customers.xlsx `
  --model pipeline_churn.joblib `
  --output outputs\churn_scores_above_threshold.csv
```

Never load an untrusted `.joblib` file: Python model serialization can execute
code while loading.

## Reproduce the analysis

```powershell
.venv\Scripts\python -m pip install -r requirements-notebook.txt
.venv\Scripts\python -m ipykernel install --user --name telco-churn
```

Open `telco_customer_churn_analysis.ipynb`, select the `telco-churn` kernel and
run all cells. The notebook uses a single process to keep the run deterministic
across constrained environments. Kaggle users can instead attach the dataset
and run the notebook directly.

## Run the tests

```powershell
.venv\Scripts\python -m pip install -r requirements-dev.txt
.venv\Scripts\python -m pytest
```

The full pinned environment used by continuous integration is available in
`requirements-lock.txt`.

## Scope

This is a portfolio-grade decision-support prototype built on an IBM sample
dataset. It is not a deployed retention system and does not establish causal
effects. Before operational use it would require current company data, a
calibrated cost model, subgroup performance review, monitoring and a retraining
policy.
