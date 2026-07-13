#%%
import streamlit as st
import joblib
import numpy as np

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)

st.title("❤️ Heart Disease Prediction System")

st.write("Enter the patient's details and select a machine learning model.")

# -----------------------------
# Load Models
# -----------------------------
models = {
    "Logistic Regression": "logistic_model.joblib",
    "Ensemble(VotingClassifier)": "voting_model.joblib",
    "Random Forest": "rf_model.joblib"
}

selected_model = st.selectbox(
    "Select Machine Learning Model",
    list(models.keys())
)

model = joblib.load(models[selected_model])

st.success(f"{selected_model} Loaded Successfully!")

st.markdown("---")

# -----------------------------
# User Inputs
# -----------------------------

sex = st.selectbox(
    "Gender",
    ["Female", "Male"]
)

sex = 1 if sex == "Male" else 0

age = st.number_input(
    "Age",
    min_value=20,
    max_value=100,
    value=40
)

currentSmoker = st.selectbox(
    "Current Smoker",
    [0, 1]
)

cigsPerDay = st.number_input(
    "Cigarettes Per Day",
    min_value=0,
    max_value=80,
    value=0
)

BPMeds = st.selectbox(
    "BP Medication",
    [0, 1]
)

prevalentStroke = st.selectbox(
    "Previous Stroke",
    [0, 1]
)

prevalentHyp = st.selectbox(
    "Hypertension",
    [0, 1]
)

diabetes = st.selectbox(
    "Diabetes",
    [0, 1]
)

totChol = st.number_input(
    "Total Cholesterol",
    min_value=100,
    max_value=700,
    value=200
)

sysBP = st.number_input(
    "Systolic Blood Pressure",
    min_value=80,
    max_value=300,
    value=120
)

diaBP = st.number_input(
    "Diastolic Blood Pressure",
    min_value=40,
    max_value=200,
    value=80
)

BMI = st.number_input(
    "BMI",
    min_value=10.0,
    max_value=60.0,
    value=25.0,
    format="%.2f"
)

heartRate = st.number_input(
    "Heart Rate",
    min_value=40,
    max_value=200,
    value=72
)

glucose = st.number_input(
    "Glucose",
    min_value=40,
    max_value=400,
    value=80
)

st.markdown("---")

# -----------------------------
# Prediction
# -----------------------------

if st.button("Predict Heart Disease"):

    input_data = np.array([[

        sex,
        age,
        currentSmoker,
        cigsPerDay,
        BPMeds,
        prevalentStroke,
        prevalentHyp,
        diabetes,
        totChol,
        sysBP,
        diaBP,
        BMI,
        heartRate,
        glucose

    ]])

    prediction = model.predict(input_data)

    st.subheader("Prediction Result")

    st.write("Selected Model:", selected_model)

    if prediction[0] == 1:
        st.error("⚠️ Heart Disease Risk Detected")
    else:
        st.success("✅ No Heart Disease Risk Detected")

    # Show probability if supported
    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(input_data)

        st.subheader("Prediction Probability")

        st.write(f"No Heart Disease : {probability[0][0] * 100:.2f}%")
        st.write(f"Heart Disease : {probability[0][1] * 100:.2f}%")