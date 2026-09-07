import streamlit as st
import requests

st.set_page_config(page_title="Credit Risk Predictor", page_icon="💳")
st.title("💳 Credit Risk Predictor")
st.write("Enter applicant details to estimate default risk.")

API_URL = "https://credit-risk-prediction-6tdi.onrender.com/predict"

with st.form("loan_form"):
    col1, col2 = st.columns(2)

    with col1:
        person_age = st.number_input("Age", min_value=18, max_value=100, value=25)
        person_income = st.number_input("Annual Income ($)", min_value=0, value=50000)
        person_emp_length = st.number_input("Employment Length (years)", min_value=0.0, value=3.0)
        loan_amnt = st.number_input("Loan Amount ($)", min_value=0, value=10000)
        loan_int_rate = st.number_input("Interest Rate (%)", min_value=0.0, value=11.5)
        loan_percent_income = st.slider("Loan % of Income", 0.0, 1.0, 0.2)

    with col2:
        cb_person_cred_hist_length = st.number_input("Credit History Length (years)", min_value=0, value=4)
        person_home_ownership = st.selectbox("Home Ownership", ["RENT", "MORTGAGE", "OWN", "OTHER"])
        loan_intent = st.selectbox("Loan Purpose", ["EDUCATION", "MEDICAL", "PERSONAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"])
        loan_grade = st.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])
        cb_person_default_on_file = st.selectbox("Prior Default on File?", ["N", "Y"])

    submitted = st.form_submit_button("Predict Risk")

if submitted:
    payload = {
        "person_age": person_age,
        "person_income": person_income,
        "person_emp_length": person_emp_length,
        "loan_amnt": loan_amnt,
        "loan_int_rate": loan_int_rate,
        "loan_percent_income": loan_percent_income,
        "cb_person_cred_hist_length": cb_person_cred_hist_length,
        "person_home_ownership": person_home_ownership,
        "loan_intent": loan_intent,
        "loan_grade": loan_grade,
        "cb_person_default_on_file": cb_person_default_on_file,
    }

    try:
        response = requests.post(API_URL, json=payload)
        result = response.json()

        prob = result["default_probability"]
        prediction = result["prediction"]

        st.subheader("Result")
        st.metric("Default Probability", f"{prob:.1%}")

        if prediction == "default":
            st.error(f"⚠️ High risk — predicted: {prediction}")
        else:
            st.success(f"✅ Low risk — predicted: {prediction}")

    except requests.exceptions.ConnectionError:
        st.error("Couldn't reach the API. Make sure `uvicorn main:app --reload` is running.")