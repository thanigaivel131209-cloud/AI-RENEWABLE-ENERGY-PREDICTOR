import streamlit as st
import pandas as pd
import joblib

# ---------------- SETUP ----------------
st.set_page_config(page_title="Renewable Energy AI", layout="wide")

st.title("⚡ Renewable Energy Output Predictor")
st.write("AI system for predicting Solar and Wind energy output")

# ---------------- LOAD MODELS ----------------
solar_model = joblib.load("solar_model.pkl")
wind_model = joblib.load("wind_model.pkl")

# ---------------- SIDEBAR ----------------
page = st.sidebar.radio("Navigation", ["Home", "Solar Prediction", "Wind Prediction"])

# ---------------- HOME ----------------
if page == "Home":
    st.subheader("🏠 Home")
    st.write("Select a prediction page from sidebar.")

# ---------------- SOLAR PAGE ----------------
elif page == "Solar Prediction":
    st.subheader("☀️ Solar Energy Prediction")

    temp = st.slider("Temperature", 15, 45, 25)
    humidity = st.slider("Humidity", 20, 95, 50)
    sunlight = st.slider("Sunlight Hours", 0, 12, 6)
    cloud = st.slider("Cloud Cover", 0, 100, 30)

    if st.button("Predict Solar Energy"):
        input_data = pd.DataFrame([[temp, humidity, sunlight, cloud]],
                                  columns=["temperature", "humidity", "sunlight_hours", "cloud_cover"])

        result = solar_model.predict(input_data)[0]
        st.success(f"Predicted Solar Energy: {result:.2f}")

# ---------------- WIND PAGE ----------------
elif page == "Wind Prediction":
    st.subheader("🌬️ Wind Energy Prediction")

    wind = st.slider("Wind Speed", 0.5, 25.0, 10.0)
    humidity = st.slider("Humidity", 20, 95, 50)
    temp = st.slider("Temperature", 15, 45, 25)

    if st.button("Predict Wind Energy"):
        input_data = pd.DataFrame([[wind, humidity, temp]],
                                  columns=["wind_speed", "humidity", "temperature"])

        result = wind_model.predict(input_data)[0]
        st.success(f"Predicted Wind Energy: {result:.2f}")