import streamlit as st
import numpy as np
import joblib
import pandas as pd

# Load saved objects
model = joblib.load('titanic_model.pkl')
scaler = joblib.load('scaler.pkl')
pca = joblib.load('pca.pkl')

st.title("🚢 Titanic Survival Prediction App")

# User inputs
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.number_input("Age", min_value=0.0, max_value=100.0, value=25.0)
sibsp = st.number_input("Siblings/Spouses Aboard", 0, 10, 0)
parch = st.number_input("Parents/Children Aboard", 0, 10, 0)
fare = st.number_input("Fare", min_value=0.0, value=30.0)
embarked = st.selectbox("Embarked", ["S", "C", "Q"])

# Encode inputs
sex_male = 1 if sex == "male" else 0
embarked_Q = 1 if embarked == "Q" else 0
embarked_S = 1 if embarked == "S" else 0

input_data = pd.DataFrame([[
    pclass, age, sibsp, parch, fare, sex_male, embarked_Q, embarked_S
]])

# Scale and apply PCA
input_scaled = scaler.transform(input_data)
input_pca = pca.transform(input_scaled)

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_pca)[0]

    if prediction == 1:
        st.success("🎉 Passenger Survived")
    else:
        st.error("❌ Passenger Did Not Survive")
