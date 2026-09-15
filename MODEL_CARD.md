# Model card

## Model overview

- **Task:** binary classification of customer churn.
- **Selected estimator:** Logistic Regression with balanced class weights.
- **Artifact:** `pipeline_churn.joblib`.
- **Runtime:** CPython 3.12, scikit-learn 1.9.0.
- **Decision threshold:** `0.4265`.
- **Score ranking:** ten fixed score bands derived from out-of-fold training
  probabilities; decile 1 is the highest-score band.

The serialized artifact contains feature engineering, scaling, one-hot
encoding, the classifier, the decision threshold, the reference decile cut
points and the training baseline used by local explanations.

## Selection protocol

The data were split once into stratified training and holdout sets. Within the
training set, five-fold out-of-fold predictions were used to:

1. compare imbalance strategies and candidate estimators;
2. select the decision threshold under hypothetical error-cost weights;
3. define fixed score-decile boundaries.

The holdout set was opened only after these decisions were frozen. The XGBoost
benchmark was evaluated with its own threshold selected under the same training
protocol; it was not tuned against the holdout result.

## Holdout performance

The final holdout contains 1,409 customers.

| Metric | Logistic Regression |
|---|---:|
| ROC-AUC | 0.854 |
| PR-AUC | 0.675 |
| Precision | 0.490 |
| Recall | 0.837 |
| F1 | 0.618 |
| False negatives | 61 |
| False positives | 326 |
| Cost units, FN:FP = 5:1 | 631 |

The cost values are scenario units, not measured currency. They are useful for
threshold sensitivity analysis but should be replaced with validated retention
economics before operational use.

## Input contract

The pipeline requires the 18 fields documented in [DATASET.md](DATASET.md).
Numeric fields are validated before prediction:

- `Tenure Months` must be a non-negative integer;
- `Monthly Charges` must be finite and non-negative;
- `Total Charges` must be finite and non-negative, except that a missing value
  is accepted for a new customer with zero tenure and is then set to zero.

Missing, empty or whitespace-only categorical values are rejected before
prediction. Unseen but non-empty categorical values are encoded
deterministically and raise a warning. They should be monitored because
repeated unknown values indicate data drift or an extraction problem.

## Intended use

- Demonstrating a reproducible scoring workflow on the IBM sample dataset.
- Comparing threshold choices under explicitly hypothetical error-cost weights.
- Retrospective technical analysis of ranking and model contributions.
- Demonstrating reproducible tabular-ML engineering and interpretation.

## Out-of-scope use

- Fully automated adverse decisions about individual customers.
- Causal claims about which intervention will prevent churn.
- Use on another company or period without validation and recalibration.
- Treating the displayed probability as perfectly calibrated risk.

## Limitations and risks

- The source is an IBM sample dataset rather than current operational data.
- The holdout is a random stratified split, not an out-of-time validation.
- The economic loss ratio is hypothetical.
- Probability calibration is assessed descriptively, not guaranteed.
- Subgroup fairness and intervention effects were not evaluated.
- `Senior Citizen` is used as a model input; governance review is required
  before any real customer-level use.
- Local reasons describe model contributions, not causal explanations.

## Validation needed before considering operational use

Any future operational proposal would need current data, validated costs,
temporal and subgroup evaluation, calibrated monitoring criteria, causal tests
of any intervention and an explicit human-governance process. None of those
conditions is established by this prototype.
