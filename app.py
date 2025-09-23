import streamlit as st
import pickle
import pandas as pd

# Load DataFrame from pickle
with open("df.pkl", "rb") as f:
    df = pickle.load(f)

# Streamlit UI
st.set_page_config(page_title="BMI Index Predictor", layout="centered")

st.title("👤 BMI Index Predictor")
st.write("Enter your details to see your predicted Index (based on dataset).")

# User inputs
gender = st.selectbox("Select Gender", df["Gender"].unique())
height = st.number_input("Enter Height (cm)", min_value=100, max_value=250, step=1)
weight = st.number_input("Enter Weight (kg)", min_value=30, max_value=200, step=1)

# Button for prediction
if st.button("Predict Index"):
    # Find closest match in dataset (simple demo logic)
    df["distance"] = (df["Height"] - height).abs() + (df["Weight"] - weight).abs()
    closest_row = df[df["Gender"] == gender].sort_values("distance").iloc[0]

    predicted_index = int(closest_row["Index"])

    st.success(f"✅ Predicted Index: {predicted_index}")
    st.write("*(Based on closest data point in dataset)*")

# Show dataset preview
st.subheader("📊 Dataset Preview")
st.dataframe(df.head())
