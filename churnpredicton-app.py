import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import time
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
import google.generativeai as genai

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

st.markdown(
    """
    <style>
        body {background: linear-gradient(to right, #6a11cb, #2575fc);}
        .stButton>button {
            background: linear-gradient(90deg, #6a11cb, #2575fc);
            color: white; font-size: 18px; padding: 12px 24px;
            border-radius: 8px; border: none; cursor: pointer;
            transition: 0.3s;
        }
        .stButton>button:hover {transform: scale(1.05);}
        .prediction-result {
            font-size: 22px; padding: 10px; border-radius: 10px;
            text-align: center; font-weight: bold;
        }
        .churned {background-color: #ff4b4b; color: white;}
        .retained {background-color: #4CAF50; color: white;}
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------ LOAD MODEL & SCALER ------------------
@st.cache_resource
def load_model_and_scaler():
    with open("best_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_model_and_scaler()

# ------------------ GEMINI API ------------------
genai.configure(api_key=st.secrets.get("GOOGLE_API_KEY", "YOUR_FALLBACK_KEY"))

def explain_churn(features, values):
    """Generate an AI-based churn explanation using Gemini."""
    prompt = f"""Based on the following customer details:
    {features}
    With values:
    {values}
    Provide an explanation why the customer is likely to churn or stay."""
    try:
        gmodel = genai.GenerativeModel("gemini-1.5-pro-latest")
        response = gmodel.generate_content(prompt)
        return response.text if hasattr(response, "text") else "Explanation unavailable."
    except Exception as e:
        return f"AI explanation unavailable: {e}"

# ------------------ SIDEBAR INPUTS ------------------
st.sidebar.title("🌟 User Inputs")
st.sidebar.info("Adjust the values to see how customer churn risk changes.")

user_inputs = {
    "CreditScore": st.sidebar.number_input("Credit Score", value=600, step=1),
    "Age": st.sidebar.number_input("Age", value=30, step=1),
    "Tenure": st.sidebar.number_input("Tenure (Years)", value=2, step=1),
    "Balance": st.sidebar.number_input("Balance Amount", value=50000, step=1000),
    "NumOfProducts": st.sidebar.number_input("Number of Products", value=2, step=1),
    "EstimatedSalary": st.sidebar.number_input("Estimated Salary", value=60000, step=1000),
    "Geography_France": int(st.sidebar.checkbox("Geography - France")),
    "Geography_Germany": int(st.sidebar.checkbox("Geography - Germany")),
    "Geography_Spain": int(st.sidebar.checkbox("Geography - Spain")),
    "Gender_Female": int(st.sidebar.checkbox("Gender - Female")),
    "Gender_Male": int(st.sidebar.checkbox("Gender - Male")),
    "HasCrCard_0": int(not st.sidebar.checkbox("Does not have Credit Card")),
    "HasCrCard_1": int(st.sidebar.checkbox("Has Credit Card")),
    "IsActiveMember_0": int(not st.sidebar.checkbox("Not Active Member")),
    "IsActiveMember_1": int(st.sidebar.checkbox("Active Member")),
}

input_df = pd.DataFrame([user_inputs])
scale_cols = ["CreditScore", "EstimatedSalary", "Tenure", "Balance", "Age", "NumOfProducts"]
input_scaled = input_df.copy()
input_scaled[scale_cols] = scaler.transform(input_df[scale_cols])

# ------------------ PREDICTION ------------------
st.title("✨ Customer Churn Prediction ✨")
st.subheader("Will the customer stay or leave?")
st.markdown("---")

if st.button("Predict 🚀"):
    with st.spinner("⏳ Analyzing data..."):
        time.sleep(2)

    # Detect model type
    model_type = type(model).__name__
    st.caption(f"Loaded model type: {model_type}")

    try:
        # --- Handle sklearn XGBClassifier ---
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(input_scaled)[0]
            pred = int(model.predict(input_scaled)[0])

        # --- Handle raw XGBoost Booster model ---
        elif isinstance(model, xgb.Booster):
            dinput = xgb.DMatrix(input_scaled)
            probs = model.predict(dinput)
            pred = int(probs[0] > 0.5)
            probs = [1 - probs[0], probs[0]]

        else:
            raise TypeError("Unsupported model type for prediction.")

        # --- Display prediction ---
        label = "Churned" if pred == 1 else "Retained"
        st.markdown(
            f"<div class='prediction-result {'churned' if pred else 'retained'}'>"
            f"{'⚠️' if pred else '✅'} <b>Predicted Status:</b> {label}</div>",
            unsafe_allow_html=True
        )
        st.write(f"**📊 Probability of Churn:** {probs[1]:.2%}")
        st.write(f"**📊 Probability of Retention:** {probs[0]:.2%}")

        # ---------------- SHAP EXPLANATION ----------------
        try:
            # Use TreeExplainer directly (avoid shap.Explainer recursion bug)
            if isinstance(model, xgb.Booster):
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(xgb.DMatrix(input_scaled))
            else:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(input_scaled)

            shap_array = shap_values[0] if isinstance(shap_values, list) else shap_values
            important = sorted(
                zip(input_df.columns, shap_array),
                key=lambda x: abs(x[1]),
                reverse=True
            )
            top_features = {k: float(v) for k, v in important[:5]}

            ai_explanation = explain_churn(list(top_features.keys()), list(top_features.values()))
            st.markdown("### 🤖 AI Explanation")
            st.write(ai_explanation)

        except Exception as e:
            st.warning(f"SHAP explanation unavailable: {e}")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
