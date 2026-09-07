# Credit Risk Prediction

This notebook follows a practical credit-risk workflow inspired by [A Machine Learning Approach To Credit Risk Assessment](https://towardsdatascience.com/a-machine-learning-approach-to-credit-risk-assessment-ba8eda1cd11f/). The goal is to predict `loan_status`, where 1 represents default and 0 represents no default.

The analysis combines data quality checks, exploratory analysis, categorical encoding, model comparison, probability evaluation, and feature interpretation. Because defaults are the minority class, recall, F1 score, ROC-AUC, and calibration are more informative than accuracy alone.

All analysis, feature engineering, and model decisions in this notebook are my own.

## Live Demo
- 🌐 **App:** https://credit-risk-prediction-jhw3npbvyf3ut2fsd8zn3q.streamlit.app
- 🔧 **API docs:** https://credit-risk-prediction-6tdi.onrender.com/docs

*(The API runs on a free tier and may take 30-60 seconds to wake up on first use after inactivity.)*

## Dataset
Kaggle "Loan Default Prediction" dataset — ~28,600 rows after cleaning, with applicant demographics, loan details, and credit history. Non-default rate is roughly 78%, so the target is imbalanced.

## Approach
1. Data cleaning and missing-value checks
2. Exploratory analysis and statistical comparison of defaulters vs non-defaulters
3. Categorical encoding (one-hot)
4. Model comparison: Logistic Regression, KNN, XGBoost
5. Class imbalance handling via `scale_pos_weight`
6. Evaluation via ROC-AUC, calibration (reliability curve + Brier score), and threshold tuning
7. Feature importance via XGBoost gain
8. Model served through a FastAPI backend, with a Streamlit front-end for interactive use

## Results

| Model | Recall (default class) | ROC-AUC | Brier Score |
|---|---|---|---|
| Logistic Regression | — | 0.811 | — |
| KNN | — | 0.782 | — |
| XGBoost (unweighted) | 0.74 | — | — |
| **XGBoost (weighted)** | **0.80** | **0.951** | **0.056** |

**Final model:** Weighted XGBoost — chosen for the strongest balance of recall (catching actual defaulters) and discrimination ability (ROC-AUC), at some cost to probability calibration compared to the unweighted version.

## Sanity Checks
The deployed model was tested against two contrasting applicant profiles:
- A low-risk profile (stable income, low loan-to-income ratio, no prior default) — predicted a low default probability, as expected.
- A high-risk profile (very low income, high loan-to-income ratio, poor credit grade, prior default on file) — predicted a substantially higher default probability, confirming the model responds sensibly to stacked risk factors.

## Project Structure
Credit-Risk-Prediction/
├── artifacts/ # saved model + feature columns
├── data/ # dataset
├── credit_risk_analysis.ipynb # full analysis notebook
├── main.py # FastAPI backend
├── app.py # Streamlit front-end
├── Dockerfile # container for the API
├── requirements.txt # full dev environment
├── requirements-api.txt # minimal deps for the API container
└── README.md


## Running Locally

**1. Set up the environment:**
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

**2. Run the notebook** (`credit_risk_analysis.ipynb`) to explore the analysis or regenerate the model artifacts.

**3. Run the API:**
```bash
uvicorn main:app --reload
```
Visit `http://127.0.0.1:8000/docs` to test predictions directly.

**4. Run the Streamlit app** (in a separate terminal):
```bash
streamlit run app.py
```

## Limitations & Next Steps
- Model calibration degrades under class weighting — probabilities should be recalibrated (e.g. `CalibratedClassifierCV`) before being used for anything beyond ranking risk.
- No cross-validation or out-of-time validation performed yet — results reflect a single train/test split.
- Feature importance reflects predictive value, not causal relationships.
- Fairness and regulatory compliance checks (e.g. disparate impact across demographic groups) would be required before any real lending use.