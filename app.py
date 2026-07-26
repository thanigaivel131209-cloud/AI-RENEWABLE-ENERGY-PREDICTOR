import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.ensemble import RandomForestRegressor

from weather import get_weather
from chatbot import get_bot_response

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Renewable Energy Output Predictor",
    page_icon="⚡",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>

.main{
    background-color:#eef7ff;
}

h1{
    color:#0B5ED7;
}

.stButton>button{
    background:linear-gradient(90deg,#0B5ED7,#00B4D8);
    color:white;
    border-radius:12px;
    height:3em;
    width:100%;
    font-weight:bold;
}

.weather-box{
    background:#ffffff;
    padding:20px;
    border-radius:15px;
    border:2px solid #90caf9;
    margin-top:15px;
}

.prediction-box{
    background:#dff6dd;
    padding:20px;
    border-radius:15px;
    font-size:22px;
    font-weight:bold;
    text-align:center;
}

.footer{
    text-align:center;
    color:gray;
}

</style>
""", unsafe_allow_html=True)

# ---------------- MODEL CREATION ----------------

def create_models():

    np.random.seed(42)

    rows = 3000

    temperature = np.random.randint(15,45,rows)
    humidity = np.random.randint(20,95,rows)
    wind_speed = np.random.uniform(0.5,25,rows)
    sunlight = np.random.uniform(0,12,rows)
    cloud = np.random.randint(0,100,rows)

    solar_output = (
        sunlight*40
        +(100-cloud)*0.4
        -humidity*0.2
        +np.random.normal(0,8,rows)
    )

    wind_output = (
        (wind_speed**2)*2.5
        -humidity*0.15
        +np.random.normal(0,5,rows)
    )

    df = pd.DataFrame({

        "temperature":temperature,
        "humidity":humidity,
        "wind_speed":wind_speed,
        "sunlight_hours":sunlight,
        "cloud_cover":cloud,
        "solar_output":solar_output,
        "wind_output":wind_output

    })

    df.to_csv("dataset.csv",index=False)

    solar = RandomForestRegressor(random_state=42)

    solar.fit(
        df[["temperature","humidity","sunlight_hours","cloud_cover"]],
        df["solar_output"]
    )

    wind = RandomForestRegressor(random_state=42)

    wind.fit(
        df[["wind_speed","humidity","temperature"]],
        df["wind_output"]
    )

    joblib.dump(solar,"solar_model.pkl")
    joblib.dump(wind,"wind_model.pkl")


if not os.path.exists("solar_model.pkl") or not os.path.exists("wind_model.pkl"):
    with st.spinner("Training AI models for first use..."):
        create_models()

solar_model = joblib.load("solar_model.pkl")
wind_model = joblib.load("wind_model.pkl")

# ---------------- SIDEBAR ----------------

st.sidebar.image("https://img.icons8.com/color/96/solar-panel.png", width=80)

st.sidebar.title("⚡ AI Energy Predictor")

page = st.sidebar.radio(

    "Navigation",

    [
        "🏠 Home",
        "☀ Solar Prediction",
        "🌬 Wind Prediction",
        "🤖 AI Chatbot",
        "📊 Analytics"
    ]

)
# ---------------- HOME PAGE ----------------

if page == "🏠 Home":

    st.title("⚡ Renewable Energy Output Predictor")

    st.markdown("""
    ### 🌍 Predict Renewable Energy Using Live Weather

    This AI application automatically fetches live weather
    information from your city and predicts:

    ☀ Solar Energy Output

    🌬 Wind Energy Output

    🤖 AI Powered | 🌦 Live Weather | 📊 Smart Prediction
    """)

    st.divider()

    city = st.text_input(
        "🏙 Enter your City Name",
        placeholder="Example: Chennai"
    )

    if st.button("🌦 Get Live Weather"):

        weather = get_weather(city)

        if weather is None:

            st.error("❌ City not found or API key is invalid.")

        else:

            st.success(f"Live Weather for {weather['city']}")

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "🌡 Temperature",
                    f"{weather['temperature']} °C"
                )

                st.metric(
                    "💧 Humidity",
                    f"{weather['humidity']} %"
                )

            with col2:

                st.metric(
                    "💨 Wind Speed",
                    f"{weather['wind_speed']} m/s"
                )

                st.metric(
                    "☁ Cloud Cover",
                    f"{weather['cloud_cover']} %"
                )

            sunlight_hours = max(
                0,
                12 - (weather["cloud_cover"] / 100) * 12
            )

            solar_input = pd.DataFrame([{
                "temperature": weather["temperature"],
                "humidity": weather["humidity"],
                "sunlight_hours": sunlight_hours,
                "cloud_cover": weather["cloud_cover"]
            }])

            wind_input = pd.DataFrame([{
                "wind_speed": weather["wind_speed"],
                "humidity": weather["humidity"],
                "temperature": weather["temperature"]
            }])

            solar_prediction = solar_model.predict(solar_input)[0]

            wind_prediction = wind_model.predict(wind_input)[0]

            st.divider()

            c1, c2 = st.columns(2)

            with c1:

                st.markdown(
                    f"""
                    <div class="prediction-box">

                    ☀ Solar Output

                    <br><br>

                    {solar_prediction:.2f} kWh

                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with c2:

                st.markdown(
                    f"""
                    <div class="prediction-box">

                    🌬 Wind Output

                    <br><br>

                    {wind_prediction:.2f} kWh

                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.info(
                "💡 Prediction is based on live weather data from your selected city."
            )
# ---------------- SOLAR PREDICTION ----------------

elif page == "☀ Solar Prediction":

    st.title("☀ Solar Energy Prediction")

    city = st.text_input(
        "🏙 Enter City Name",
        key="solar_city"
    )

    if st.button("Predict Solar Energy"):

        weather = get_weather(city)

        if weather is None:

            st.error("❌ Unable to fetch weather data.")

        else:

            sunlight_hours = max(
                0,
                12 - (weather["cloud_cover"] / 100) * 12
            )

            solar_input = pd.DataFrame([{
                "temperature": weather["temperature"],
                "humidity": weather["humidity"],
                "sunlight_hours": sunlight_hours,
                "cloud_cover": weather["cloud_cover"]
            }])

            prediction = solar_model.predict(solar_input)[0]

            st.success(f"☀ Estimated Solar Energy Output: {prediction:.2f} kWh")

            st.write("### Current Weather")

            c1, c2 = st.columns(2)

            with c1:
                st.metric("🌡 Temperature", f"{weather['temperature']} °C")
                st.metric("💧 Humidity", f"{weather['humidity']} %")

            with c2:
                st.metric("☁ Cloud Cover", f"{weather['cloud_cover']} %")
                st.metric("☀ Estimated Sunlight", f"{sunlight_hours:.1f} hrs")

# ---------------- WIND PREDICTION ----------------

elif page == "🌬 Wind Prediction":

    st.title("🌬 Wind Energy Prediction")

    city = st.text_input(
        "🏙 Enter City Name",
        key="wind_city"
    )

    if st.button("Predict Wind Energy"):

        weather = get_weather(city)

        if weather is None:

            st.error("❌ Unable to fetch weather data.")

        else:

            wind_input = pd.DataFrame([{
                "wind_speed": weather["wind_speed"],
                "humidity": weather["humidity"],
                "temperature": weather["temperature"]
            }])

            prediction = wind_model.predict(wind_input)[0]

            st.success(f"🌬 Estimated Wind Energy Output: {prediction:.2f} kWh")

            st.write("### Current Weather")

            c1, c2 = st.columns(2)

            with c1:
                st.metric("💨 Wind Speed", f"{weather['wind_speed']} m/s")
                st.metric("🌡 Temperature", f"{weather['temperature']} °C")

            with c2:
                st.metric("💧 Humidity", f"{weather['humidity']} %")
                st.metric("☁

# ---------------- CHATBOT ----------------

elif page == "🤖 AI Chatbot":

    st.title("🤖 Renewable Energy AI Assistant")

    st.write("Ask me anything about renewable energy.")

    question = st.text_input(
        "Your Question",
        placeholder="Example: What is solar energy?"
    )

    if st.button("Ask AI"):

        if question.strip() == "":
            st.warning("Please enter a question.")
        else:
            answer = get_bot_response(question)
            st.success(answer)

# ---------------- ANALYTICS ----------------

elif page == "📊 Analytics":

    st.title("📊 Renewable Energy Analytics")

    if os.path.exists("dataset.csv"):

        df = pd.read_csv("dataset.csv")

        st.subheader("Dataset Preview")

        st.dataframe(df.head())

        st.subheader("Solar Energy Distribution")

        st.bar_chart(df["solar_output"])

        st.subheader("Wind Energy Distribution")

        st.bar_chart(df["wind_output"])

        st.subheader("Weather Parameters")

        st.line_chart(
            df[
                [
                    "temperature",
                    "humidity",
                    "wind_speed"
                ]
            ].head(100)
        )

    else:

        st.warning("Dataset not found.")

# ---------------- FOOTER ----------------

st.markdown("---")

st.markdown(
    """

<div class="footer">

⚡ Renewable Energy Output Predictor

Built using ❤️ Streamlit • Machine Learning • OpenWeatherMap API

</div>
""",
unsafe_allow_html=True
)
