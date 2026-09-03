from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Literal
import pickle
import pandas as pd

app = FastAPI(title="Credit Risk Prediction API")

# Load model + expected column order
with open("artifacts/xgb_weighted_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("artifacts/feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)


class LoanApplication(BaseModel):
    person_age: int = Field(..., ge=18, le=100)
    person_income: float = Field(..., gt=0)
    person_emp_length: float = Field(..., ge=0)
    loan_amnt: float = Field(..., gt=0)
    loan_int_rate: float = Field(..., gt=0)
    loan_percent_income: float = Field(..., ge=0, le=1)
    cb_person_cred_hist_length: int = Field(..., ge=0)
    person_home_ownership: Literal["MORTGAGE", "OTHER", "OWN", "RENT"]
    loan_intent: Literal["DEBTCONSOLIDATION", "EDUCATION", "HOMEIMPROVEMENT",
                          "MEDICAL", "PERSONAL", "VENTURE"]
    loan_grade: Literal["A", "B", "C", "D", "E", "F", "G"]
    cb_person_default_on_file: Literal["Y", "N"]


def build_feature_row(app_data: LoanApplication) -> pd.DataFrame:
    # start with all columns at 0
    row = {col: 0 for col in feature_columns}

    # numeric fields — direct assignment
    row["person_age"] = app_data.person_age
    row["person_income"] = app_data.person_income
    row["person_emp_length"] = app_data.person_emp_length
    row["loan_amnt"] = app_data.loan_amnt
    row["loan_int_rate"] = app_data.loan_int_rate
    row["loan_percent_income"] = app_data.loan_percent_income
    row["cb_person_cred_hist_length"] = app_data.cb_person_cred_hist_length

    # one-hot fields — set the matching column to 1
    row[f"person_home_ownership_{app_data.person_home_ownership}"] = 1
    row[f"loan_intent_{app_data.loan_intent}"] = 1
    row[f"loan_grade_{app_data.loan_grade}"] = 1
    row[f"cb_person_default_on_file_{app_data.cb_person_default_on_file}"] = 1

    return pd.DataFrame([row])[feature_columns]  # enforce exact column order


@app.post("/predict")
def predict(application: LoanApplication):
    X = build_feature_row(application)
    probability = model.predict_proba(X)[0][1]
    prediction = int(probability >= 0.5)  # your chosen threshold

    return {
        "default_probability": round(float(probability), 4),
        "prediction": "default" if prediction == 1 else "no default"
    }


@app.get("/")
def root():
    return {"status": "Credit Risk API is running"}