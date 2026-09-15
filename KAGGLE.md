# Kaggle publication notes

## Files to upload

- `telco_customer_churn_analysis.ipynb`
- `churn_inference.py` as a supporting notebook file
- `requirements-notebook.txt` as a supporting file for the pinned fallback
- `requirements.txt`, in the same directory as `requirements-notebook.txt`,
  because the notebook requirements include the runtime pins with `-r`

Attach the public IBM Telco Customer Churn dataset as an input. The notebook
first checks the configured, local and canonical Kaggle paths. If none exists,
its fallback searches `/kaggle/input` for `Telco_customer_churn.xlsx` and stops
if more than one matching copy is present.

## Suggested metadata

- **Title:** Telco Churn: Cost-Sensitive Modeling and Explainability
- **Subtitle:** OOF model selection, business thresholding, SHAP and a tested
  inference pipeline
- **Description:** Portfolio project developed during training as a Machine
  Learning/AI Engineer; reproducible prototype, not a production deployment
- **Language:** Italian
- **License:** use the platform setting appropriate for the notebook code;
  respect the dataset page terms separately

## Reproducibility check before publishing

1. Start a fresh Kaggle session.
2. Confirm that `churn_inference.py` is importable and that
   `requirements-notebook.txt` and `requirements.txt` are available together
   from the notebook working directory.
3. Check the Kaggle image versions. If they differ from the pinned environment,
   enable Internet for the session and run this before the analysis:

   ```python
   %pip install -r requirements-notebook.txt
   ```

   Restart the kernel after installation, then start the notebook from the
   first cell. If Internet cannot be enabled, record the Kaggle runtime versions
   and do not describe the run as pinned.
4. Run all cells from the first to the last.
5. Confirm that all 69 code cells complete without errors.
6. Compare the final Logistic Regression and XGBoost metrics with `README.md`
   and the selected Logistic Regression metrics with `MODEL_CARD.md`.
7. Save a notebook version only after the clean run.

Kaggle images may update library versions over time. Do not repair metric drift
by editing published numbers: rerun with the pinned versions or document the
runtime difference and keep one clearly identified canonical run.
