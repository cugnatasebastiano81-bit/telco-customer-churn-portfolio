# Dataset and data contract

## Source

The analysis uses the IBM Telco Customer Churn sample dataset. The original IBM
sample files are available from the
[IBM Cognos sample-data directory](https://public.dhe.ibm.com/software/data/sw-library/cognos/mobile/C11/data/).
The exact Excel file used by this notebook is also distributed through the
[Kaggle dataset page](https://www.kaggle.com/datasets/yeanzc/telco-customer-churn-ibm-dataset).

The training file is deliberately not copied into this repository. Download it
from the source and review the source terms before redistributing it.

Expected filename: `Telco_customer_churn.xlsx`.

## Notebook setup

Choose one of these options:

1. place the Excel file next to `telco_customer_churn_analysis.ipynb`;
2. set `TELCO_CHURN_DATA` to its full path;
3. on Kaggle, attach the dataset as a notebook input.

The notebook validates the expected 7,043 rows and 33 source columns before
continuing.

## Inference schema

The exported model accepts these 18 fields:

| Field | Expected type or values |
|---|---|
| Senior Citizen | `Yes`, `No` |
| Partner | `Yes`, `No` |
| Dependents | `Yes`, `No` |
| Tenure Months | non-negative integer |
| Phone Service | `Yes`, `No` |
| Multiple Lines | `Yes`, `No`, `No phone service` |
| Internet Service | `DSL`, `Fiber optic`, `No` |
| Online Security | `Yes`, `No`, `No internet service` |
| Online Backup | `Yes`, `No`, `No internet service` |
| Device Protection | `Yes`, `No`, `No internet service` |
| Tech Support | `Yes`, `No`, `No internet service` |
| Streaming TV | `Yes`, `No`, `No internet service` |
| Streaming Movies | `Yes`, `No`, `No internet service` |
| Contract | `Month-to-month`, `One year`, `Two year` |
| Paperless Billing | `Yes`, `No` |
| Payment Method | source catalog value |
| Monthly Charges | finite, non-negative number |
| Total Charges | finite, non-negative number; may be blank only at zero tenure |

`CustomerID` is required by the command-line script as a unique row identifier,
but it is never passed to the model as a predictive feature.

The workbook in `examples/` contains synthetic records created only to exercise
the inference interface. It is not part of the training dataset.
