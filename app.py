import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Load dataset
with open("df.pkl", "rb") as f:

    df = pickle.load(f)

# Train model with balancing
X = df[['Height', 'Weight']]
y = df['Gender']
model = LogisticRegression(class_weight="balanced")
model.fit(X, y)

st.set_page_config(page_title="Gender Prediction", layout="centered")
st.title("👩‍🦰 Gender Prediction App")
st.write("Enter height and weight to predict whether the person is male or female.")

# Inputs
height = st.number_input("Enter Height (cm)", min_value=50.0, max_value=250.0, step=0.1)
weight = st.number_input("Enter Weight (kg)", min_value=10.0, max_value=250.0, step=0.1)

# Prediction
if st.button("Predict"):
    features = np.array([[height, weight]])
    prediction = model.predict(features)[0]
    proba = model.predict_proba(features)[0]

    if prediction == 0:
        st.success(f"Prediction: **Female** 👩 (Confidence: {proba[0]:.2f})")
    else:
        st.success(f"Prediction: **Male** 👨 (Confidence: {proba[1]:.2f})")
