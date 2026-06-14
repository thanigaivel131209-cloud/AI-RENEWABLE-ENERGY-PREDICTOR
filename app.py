import streamlit as st
import pandas as pd
import joblib
from chatbot import get_bot_response

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Renewable Energy AI",
    page_icon="⚡",
    layout="wide"
)

# ---------------- LOAD MODELS ----------------
solar_model = joblib.load("solar_model.pkl")
wind_model = joblib.load("wind_model.pkl")

# ---------------- SIDEBAR NAVIGATION ----------------
st.sidebar.title("⚡ Navigation")

page = st.sidebar.radio(
    "Select Page",
    ["Home", "Solar Prediction", "Wind Prediction", "Chatbot"]
)

# ---------------- HOME ----------------
if page == "Home":
    st.title("⚡ Renewable Energy Output Predictor")
    st.write("AI-powered system for predicting Solar and Wind energy output.")
    st.success("Use the sidebar to navigate the app.")

# ---------------- SOLAR PREDICTION ----------------
elif page == "Solar Prediction":
    st.title("☀️ Solar Energy Prediction")

    temp = st.slider("Temperature (°C)", 15, 45, 25)
    humidity = st.slider("Humidity (%)", 20, 95, 50)
    sunlight = st.slider("Sunlight Hours", 0, 12, 6)
    cloud = st.slider("Cloud Cover (%)", 0, 100, 30)

    if st.button("Predict Solar Energy"):
        input_data = pd.DataFrame([[temp, humidity, sunlight, cloud]],
                                   columns=["temperature", "humidity", "sunlight_hours", "cloud_cover"])

        prediction = solar_model.predict(input_data)[0]

        st.success(f"☀️ Predicted Solar Energy Output: {prediction:.2f}")

# ---------------- WIND PREDICTION ----------------
elif page == "Wind Prediction":
    st.title("🌬️ Wind Energy Prediction")

    wind = st.slider("Wind Speed (m/s)", 0.5, 25.0, 10.0)
    humidity = st.slider("Humidity (%)", 20, 95, 50)
    temp = st.slider("Temperature (°C)", 15, 45, 25)

    if st.button("Predict Wind Energy"):
        input_data = pd.DataFrame([[wind, humidity, temp]],
                                   columns=["wind_speed", "humidity", "temperature"])

        prediction = wind_model.predict(input_data)[0]

        st.success(f"🌬️ Predicted Wind Energy Output: {prediction:.2f}")

# ---------------- CHATBOT ----------------
elif page == "Chatbot":
    st.title("🤖 AI Chatbot Assistant")

    user_input = st.text_input("Ask me anything about renewable energy:")

    if user_input:
        response = get_bot_response(user_input)
        st.info(response)