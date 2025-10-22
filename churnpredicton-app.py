from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import pickle
import pandas as pd
import time
import shap
import google.generativeai as genai

# Set Streamlit layout to wide and style the UI
st.set_page_config(layout="wide", page_title="Customer Churn Prediction")

st.markdown(
    """
    <style>
        body {
            background: linear-gradient(to right, #6a11cb, #2575fc);
            font-family: 'Arial', sans-serif;
        }
        .stButton>button {
            background: linear-gradient(90deg, #6a11cb, #2575fc);
            color: white;
            font-size: 18px;
            padding: 12px 24px;
            border-radius: 8px;
            transition: all 0.3s ease-in-out;
            cursor: pointer;
            border: none;
        }
        .stButton>button:hover {
            background: linear-gradient(90deg, #2575fc, #6a11cb);
            transform: scale(1.05);
        }
        .prediction-result {
            font-size: 22px;
            padding: 10px;
            border-radius: 10px;
            text-align: center;
            font-weight: bold;
            transition: all 0.3s ease-in-out;
        }
        .churned {
            background-color: #ff4b4b;
            color: white;
        }
        .retained {
            background-color: #4CAF50;
            color: white;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Load the trained model
with open('best_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the MinMaxScaler
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Set up Gemini API Key
genai.configure(api_key="AIzaSyAUVZBtiki0-vs4sRdrdBYGoBqDb5LYr0Q")

def explain_churn(features, values):
    prompt = f"""Based on the following customer details:
    {features}
    With values:
    {values}
    Provide an explanation why the customer is likely to churn or stay."""
    
    model = genai.GenerativeModel("models/gemini-1.5-pro-latest")
    response = model.generate_content(prompt)
    return response.text

# Sidebar Information
st.sidebar.title("🌟 User Inputs")
st.sidebar.markdown("### ℹ️ About This Tool")
st.sidebar.info("This AI-powered tool predicts whether a customer will churn based on financial and demographic details. Adjust inputs and hit 'Predict'!")

# User Inputs in Sidebar
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
"HasCrCard_0": int(not st.sidebar.checkbox("Does not have Credit Card", key="HasCrCard")),
"HasCrCard_1": int(st.sidebar.checkbox(" Has Credit Card", key="HasCrCard_1")),
"IsActiveMember_0": int(not st.sidebar.checkbox("Not a Active Member", key="IsActiveMember")),
"IsActiveMember_1": int(st.sidebar.checkbox("Active Member", key="IsActiveMember_1"))


}

# Convert inputs to DataFrame
input_data = pd.DataFrame([user_inputs])
input_data_scaled = input_data.copy()
scale_vars = ["CreditScore", "EstimatedSalary", "Tenure", "Balance", "Age", "NumOfProducts"]
input_data_scaled[scale_vars] = scaler.transform(input_data[scale_vars])

# Main Container
st.markdown("<div class='main-container'>", unsafe_allow_html=True)
st.title("✨ Customer Churn Prediction ✨")
st.subheader("Will the customer stay or leave? Let's find out!")

# Prediction Section
st.markdown("---")
st.header("🔍 Prediction")

if st.button("Predict 🚀"):
    with st.spinner("⏳ Analyzing data..."):
        time.sleep(2)
    
    probabilities = model.predict_proba(input_data_scaled)[0]
    prediction = model.predict(input_data_scaled)[0]
    prediction_label = "Churned" if prediction == 1 else "Retained"
    
    st.markdown(f"<div class='prediction-result {'churned' if prediction == 1 else 'retained'}'>"
                f"{'⚠️' if prediction == 1 else '✅'} <b>Predicted Status:</b> {prediction_label}</div>", 
                unsafe_allow_html=True)
    st.write(f"**📌 Probability of Churn:** {probabilities[1]:.2%}")
    st.write(f"**📌 Probability of Retention:** {probabilities[0]:.2%}")
    
    # SHAP Explanation
    explainer = shap.Explainer(model)
    shap_values = explainer(input_data_scaled)
    important_features = sorted(zip(input_data.columns, shap_values.values[0]), key=lambda x: abs(x[1]), reverse=True)
    top_features = {feature: value for feature, value in important_features[:5]}
    
    # AI-based Explanation
    ai_explanation = explain_churn(list(top_features.keys()), list(top_features.values()))
    st.markdown("### 🤖 AI Explanation")
    st.write(ai_explanation)

st.markdown("</div>", unsafe_allow_html=True)
