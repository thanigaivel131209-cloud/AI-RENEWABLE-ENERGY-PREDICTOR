import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.ensemble import RandomForestRegressor
from chatbot import get_bot_response
from weather import get_weather
import base64


# ==============================
# PAGE CONFIGURATION
# ==============================

st.set_page_config(
    page_title="Renewable Energy AI",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==============================
# CUSTOM CSS
# ==============================

st.markdown("""
<style>

.main {
    background: linear-gradient(135deg, #e0f7fa, #ffffff);
}

h1 {
    color: #0b5394;
    font-weight: 800;
}

h2 {
    color: #166534;
    font-weight: 700;
}

h3 {
    color: #075985;
}

.stButton > button {
    background-color: #0284c7;
    color: white;
    border-radius: 12px;
    border: none;
    padding: 10px 20px;
    font-weight: bold;
}

.stButton > button:hover {
    background-color: #0369a1;
    color: white;
}

.card {
    padding: 20px;
    border-radius: 18px;
    background-color: white;
    box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.10);
    margin-bottom: 20px;
}

.solar-card {
    padding: 25px;
    border-radius: 20px;
    background: linear-gradient(135deg, #fff7ed, #ffedd5);
    box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.10);
}

.wind-card {
    padding: 25px;
    border-radius: 20px;
    background: linear-gradient(135deg, #ecfeff, #cffafe);
    box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.10);
}

.result-card {
    padding: 25px;
    border-radius: 20px;
    background: linear-gradient(135deg, #dcfce7, #bbf7d0);
    text-align: center;
    margin-top: 20px;
}

.footer {
    text-align: center;
    padding: 20px;
    color: #64748b;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)


# ==============================
# MODEL CREATION
# ==============================

def create_models():

    np.random.seed(42)

    rows = 4000

    temperature = np.random.uniform(15, 45, rows)
    humidity = np.random.uniform(20, 95, rows)
    wind_speed = np.random.uniform(0, 20, rows)
    sunlight_hours = np.random.uniform(2, 12, rows)
    cloud_cover = np.random.uniform(0, 100, rows)

    solar_output = (
        sunlight_hours * 8
        + (100 - cloud_cover) * 0.08
        + temperature * 0.15
        - humidity * 0.03
        + np.random.normal(0, 2, rows)
    )

    wind_output = (
        wind_speed ** 2 * 0.4
        + humidity * 0.05
        + temperature * 0.1
        + np.random.normal(0, 2, rows)
    )

    solar_output = np.maximum(solar_output, 0)
    wind_output = np.maximum(wind_output, 0)

    dataset = pd.DataFrame({
        "temperature": temperature,
        "humidity": humidity,
        "wind_speed": wind_speed,
        "sunlight_hours": sunlight_hours,
        "cloud_cover": cloud_cover,
        "solar_output": solar_output,
        "wind_output": wind_output
    })

    dataset.to_csv("dataset.csv", index=False)

    solar_features = dataset[
        [
            "temperature",
            "humidity",
            "sunlight_hours",
            "cloud_cover"
        ]
    ]

    wind_features = dataset[
        [
            "wind_speed",
            "humidity",
            "temperature"
        ]
    ]

    solar_target = dataset["solar_output"]
    wind_target = dataset["wind_output"]

    solar_model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    wind_model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    solar_model.fit(solar_features, solar_target)
    wind_model.fit(wind_features, wind_target)

    joblib.dump(solar_model, "solar_model.pkl")
    joblib.dump(wind_model, "wind_model.pkl")


# ==============================
# LOAD MODELS
# ==============================

@st.cache_resource
def load_models():

    if (
        not os.path.exists("solar_model.pkl")
        or not os.path.exists("wind_model.pkl")
        or not os.path.exists("dataset.csv")
    ):
        create_models()

    solar_model = joblib.load("solar_model.pkl")
    wind_model = joblib.load("wind_model.pkl")

    return solar_model, wind_model


solar_model, wind_model = load_models()


# ==============================
# HELPER FUNCTIONS
# ==============================

def calculate_sunlight(cloud_cover):

    sunlight = 12 - ((cloud_cover / 100) * 12)

    return max(sunlight, 0)


def calculate_total(output, quantity):

    return output * quantity


def show_weather_metrics(weather):

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "🌡️ Temperature",
        f"{weather['temperature']} °C"
    )

    col2.metric(
        "💧 Humidity",
        f"{weather['humidity']} %"
    )

    col3.metric(
        "🌬️ Wind Speed",
        f"{weather['wind_speed']} m/s"
    )

    col4.metric(
        "☁️ Cloud Cover",
        f"{weather['cloud_cover']} %"
    )


def get_solar_prediction(weather):

    sunlight_hours = calculate_sunlight(
        weather["cloud_cover"]
    )

    input_data = pd.DataFrame({
        "temperature": [weather["temperature"]],
        "humidity": [weather["humidity"]],
        "sunlight_hours": [sunlight_hours],
        "cloud_cover": [weather["cloud_cover"]]
    })

    prediction = solar_model.predict(input_data)[0]

    return max(float(prediction), 0)


def get_wind_prediction(weather):

    input_data = pd.DataFrame({
        "wind_speed": [weather["wind_speed"]],
        "humidity": [weather["humidity"]],
        "temperature": [weather["temperature"]]
    })

    prediction = wind_model.predict(input_data)[0]

    return max(float(prediction), 0)

# ==============================
# SIDEBAR NAVIGATION
# ==============================

st.sidebar.markdown(
    """
    <h1 style="text-align:center;">☀️ RE AI</h1>
    <p style="text-align:center;">
    Renewable Energy Intelligence
    </p>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "📌 Navigate",
    [
        "🏠 Home",
        "☀️ Solar Prediction",
        "🌬️ Wind Prediction",
        "📊 Analytics",
        "🤖 AI Chatbot",
        "ℹ️ About Project"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info(
    "This application uses Machine Learning "
    "and weather information to estimate "
    "renewable energy production."
)


# ==============================
# HOME PAGE
# ==============================

if page == "🏠 Home":

    st.markdown(
        "<h1>☀️ Renewable Energy AI Predictor</h1>",
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="card">
        <h3>🌱 Smart Energy for a Sustainable Future</h3>
        This application uses Artificial Intelligence and
        weather conditions to predict solar and wind energy output.
        </div>
        """,
        unsafe_allow_html=True
    )

    if os.path.exists("images/banner.jpg"):

        st.image(
            "images/banner.jpg",
            use_container_width=True
        )

    st.subheader("📍 Check Your City's Weather")

    city = st.text_input(
        "Enter your city",
        placeholder="Example: Chennai"
    )

    if st.button("🔍 Get Weather Information"):

        if city.strip() == "":

            st.warning("Please enter a city name.")

        else:

            with st.spinner("Fetching weather data..."):

                weather = get_weather(city)

            if weather is None:

                st.error(
                    "Unable to find weather information. "
                    "Please check the city name."
                )

            else:

                st.success(
                    f"Weather data received for {weather['city']}"
                )

                show_weather_metrics(weather)

                st.markdown("---")

                solar_output = get_solar_prediction(weather)
                wind_output = get_wind_prediction(weather)

                col1, col2 = st.columns(2)

                with col1:

                    st.markdown(
                        f"""
                        <div class="solar-card">
                        <h2>☀️ Solar Energy</h2>
                        <h3>{solar_output:.2f} units</h3>
                        <p>Estimated solar energy output</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                with col2:

                    st.markdown(
                        f"""
                        <div class="wind-card">
                        <h2>🌬️ Wind Energy</h2>
                        <h3>{wind_output:.2f} units</h3>
                        <p>Estimated wind energy output</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

    else:

        st.info(
            "Enter a city above to view weather conditions "
            "and renewable energy predictions."
        )

    st.markdown("---")

    st.subheader("⚡ Application Features")

    feature_col1, feature_col2, feature_col3 = st.columns(3)

    with feature_col1:

        st.markdown(
            """
            <div class="card">
            <h3>☀️ Solar Prediction</h3>
            Estimate solar energy output using sunlight,
            temperature, humidity and cloud cover.
            </div>
            """,
            unsafe_allow_html=True
        )

    with feature_col2:

        st.markdown(
            """
            <div class="card">
            <h3>🌬️ Wind Prediction</h3>
            Predict wind energy production using wind speed,
            temperature and humidity.
            </div>
            """,
            unsafe_allow_html=True
        )

    with feature_col3:

        st.markdown(
            """
            <div class="card">
            <h3>🤖 AI Assistant</h3>
            Ask questions about renewable energy,
            weather and machine learning.
            </div>
            """,
            unsafe_allow_html=True
        )

# ==============================
# SOLAR PREDICTION PAGE
# ==============================

elif page == "☀️ Solar Prediction":

    st.markdown(
        "<h1>☀️ Solar Energy Prediction</h1>",
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="solar-card">
        <h3>🔆 Predict Solar Power Generation</h3>
        Enter your location and number of solar panels
        to estimate the total energy output.
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:

        solar_city = st.text_input(
            "📍 Enter City",
            placeholder="Example: Chennai",
            key="solar_city"
        )

    with col2:

        panel_count = st.number_input(
            "🔋 Number of Solar Panels",
            min_value=1,
            max_value=1000,
            value=5,
            step=1
        )

    st.markdown("---")

    if st.button("☀️ Predict Solar Output"):

        if solar_city.strip() == "":

            st.warning("Please enter a city name.")

        else:

            with st.spinner(
                "Collecting weather data and predicting..."
            ):

                weather = get_weather(solar_city)

            if weather is None:

                st.error(
                    "Weather data could not be retrieved. "
                    "Check your city name or internet connection."
                )

            else:

                st.success(
                    f"Weather data received for {weather['city']}"
                )

                st.subheader("🌦️ Current Weather Conditions")

                show_weather_metrics(weather)

                sunlight_hours = calculate_sunlight(
                    weather["cloud_cover"]
                )

                single_panel_output = get_solar_prediction(
                    weather
                )

                total_output = calculate_total(
                    single_panel_output,
                    panel_count
                )

                st.markdown("---")

                result_col1, result_col2 = st.columns(2)

                with result_col1:

                    st.metric(
                        "☀️ Sunlight Hours",
                        f"{sunlight_hours:.2f} hours"
                    )

                with result_col2:

                    st.metric(
                        "🔋 Energy per Panel",
                        f"{single_panel_output:.2f} units"
                    )

                st.markdown(
                    f"""
                    <div class="result-card">
                    <h2>⚡ Total Estimated Output</h2>
                    <h1>{total_output:.2f} units</h1>
                    <p>
                    Estimated output from {panel_count} solar panels
                    </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.markdown("---")

                st.subheader("📋 Prediction Summary")

                summary = pd.DataFrame({
                    "Parameter": [
                        "City",
                        "Temperature",
                        "Humidity",
                        "Cloud Cover",
                        "Sunlight Hours",
                        "Number of Panels",
                        "Output per Panel",
                        "Total Output"
                    ],
                    "Value": [
                        weather["city"],
                        f"{weather['temperature']} °C",
                        f"{weather['humidity']} %",
                        f"{weather['cloud_cover']} %",
                        f"{sunlight_hours:.2f} hours",
                        panel_count,
                        f"{single_panel_output:.2f} units",
                        f"{total_output:.2f} units"
                    ]
                })

                st.table(summary)

                st.info(
                    "💡 Solar output depends on sunlight, "
                    "cloud cover, temperature and humidity. "
                    "This is an AI-based estimated value, "
                    "not a guaranteed real-world measurement."
                )


# ==============================
# WIND PREDICTION PAGE
# ==============================

elif page == "🌬️ Wind Prediction":

    st.markdown(
        "<h1>🌬️ Wind Energy Prediction</h1>",
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="wind-card">
        <h3>🌪️ Predict Wind Power Generation</h3>
        Enter your city and estimate the energy generated
        using wind turbines.
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:

        wind_city = st.text_input(
            "📍 Enter City",
            placeholder="Example: Chennai",
            key="wind_city"
        )

    with col2:

        turbine_count = st.number_input(
            "⚙️ Number of Wind Turbines",
            min_value=1,
            max_value=1000,
            value=1,
            step=1
        )

    st.markdown("---")

    if st.button("🌬️ Predict Wind Output"):

        if wind_city.strip() == "":

            st.warning("Please enter a city name.")

        else:

            with st.spinner(
                "Collecting weather data and predicting..."
            ):

                weather = get_weather(wind_city)

            if weather is None:

                st.error(
                    "Weather data could not be retrieved. "
                    "Check your city name or internet connection."
                )

            else:

                st.success(
                    f"Weather data received for {weather['city']}"
                )

                st.subheader("🌦️ Current Weather Conditions")

                show_weather_metrics(weather)

                single_turbine_output = get_wind_prediction(
                    weather
                )

                total_wind_output = calculate_total(
                    single_turbine_output,
                    turbine_count
                )

                st.markdown("---")

                result_col1, result_col2 = st.columns(2)

                with result_col1:

                    st.metric(
                        "🌬️ Wind Speed",
                        f"{weather['wind_speed']} m/s"
                    )

                with result_col2:

                    st.metric(
                        "⚙️ Output per Turbine",
                        f"{single_turbine_output:.2f} units"
                    )

                st.markdown(
                    f"""
                    <div class="result-card">
                    <h2>⚡ Total Estimated Output</h2>
                    <h1>{total_wind_output:.2f} units</h1>
                    <p>
                    Estimated output from {turbine_count}
                    wind turbine(s)
                    </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.markdown("---")

                st.subheader("📋 Prediction Summary")

                summary = pd.DataFrame({
                    "Parameter": [
                        "City",
                        "Temperature",
                        "Humidity",
                        "Wind Speed",
                        "Number of Turbines",
                        "Output per Turbine",
                        "Total Output"
                    ],
                    "Value": [
                        weather["city"],
                        f"{weather['temperature']} °C",
                        f"{weather['humidity']} %",
                        f"{weather['wind_speed']} m/s",
                        turbine_count,
                        f"{single_turbine_output:.2f} units",
                        f"{total_wind_output:.2f} units"
                    ]
                })

                st.table(summary)

                st.info(
                    "💡 Wind energy production depends mainly "
                    "on wind speed and turbine characteristics. "
                    "This is an AI-based estimated value."
                )

# ==============================
# ANALYTICS PAGE
# ==============================

elif page == "📊 Analytics":

    st.markdown(
        "<h1>📊 Renewable Energy Analytics</h1>",
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="card">
        <h3>📈 Explore the Training Dataset</h3>
        This section displays the dataset used to train
        the machine learning models.
        </div>
        """,
        unsafe_allow_html=True
    )

    if os.path.exists("dataset.csv"):

        data = pd.read_csv("dataset.csv")

        st.subheader("📋 Dataset Preview")

        st.dataframe(
            data.head(20),
            use_container_width=True
        )

        st.markdown("---")

        st.subheader("📌 Dataset Information")

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "📄 Total Rows",
                len(data)
            )

        with col2:

            st.metric(
                "📊 Total Columns",
                len(data.columns)
            )

        with col3:

            st.metric(
                "🔢 Missing Values",
                int(data.isnull().sum().sum())
            )

        st.markdown("---")

        st.subheader("📈 Average Energy Output")

        average_solar = data["solar_output"].mean()
        average_wind = data["wind_output"].mean()

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "☀️ Average Solar Output",
                f"{average_solar:.2f} units"
            )

        with col2:

            st.metric(
                "🌬️ Average Wind Output",
                f"{average_wind:.2f} units"
            )

        st.markdown("---")

        st.subheader("☀️ Solar Output Analysis")

        solar_chart_data = data[
            [
                "temperature",
                "solar_output"
            ]
        ].head(100)

        st.line_chart(
            solar_chart_data.set_index("temperature")
        )

        st.markdown("---")

        st.subheader("🌬️ Wind Output Analysis")

        wind_chart_data = data[
            [
                "wind_speed",
                "wind_output"
            ]
        ].head(100)

        st.line_chart(
            wind_chart_data.set_index("wind_speed")
        )

        st.markdown("---")

        st.subheader("📊 Statistical Summary")

        st.dataframe(
            data.describe(),
            use_container_width=True
        )

    else:

        st.warning(
            "Dataset not found. Please refresh the application."
        )


# ==============================
# AI CHATBOT PAGE
# ==============================

elif page == "🤖 AI Chatbot":

    st.markdown(
        "<h1>🤖 Renewable Energy AI Assistant</h1>",
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="card">
        <h3>💬 Ask Your Renewable Energy Questions</h3>
        You can ask questions about solar energy, wind energy,
        weather and renewable energy.
        </div>
        """,
        unsafe_allow_html=True
    )

    user_question = st.text_input(
        "✍️ Enter your question",
        placeholder="Example: What is solar energy?"
    )

    if st.button("🤖 Ask AI Assistant"):

        if user_question.strip() == "":

            st.warning("Please enter a question.")

        else:

            response = get_bot_response(user_question)

            st.markdown(
                f"""
                <div class="result-card">
                <h3>🤖 AI Assistant Response</h3>
                <p>{response}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("---")

    st.subheader("💡 Try Asking")

    st.write("☀️ What is solar energy?")
    st.write("🌬️ How does wind energy work?")
    st.write("♻️ What is renewable energy?")
    st.write("☁️ How does cloud cover affect solar output?")


# ==============================
# ABOUT PROJECT PAGE
# ==============================

elif page == "ℹ️ About Project":

    st.markdown(
        "<h1>ℹ️ About This Project</h1>",
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="card">
        <h2>🌱 Renewable Energy Output Predictor</h2>

        This project uses Artificial Intelligence and
        Machine Learning to estimate renewable energy output.

        The application focuses on solar and wind energy.
        Weather conditions are used as inputs for prediction.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.subheader("🎯 Project Objectives")

    st.write(
        "• Predict solar energy output using weather conditions."
    )

    st.write(
        "• Predict wind energy output using wind speed and weather."
    )

    st.write(
        "• Demonstrate the use of Machine Learning."
    )

    st.write(
        "• Promote awareness of clean and renewable energy."
    )

    st.write(
        "• Support Sustainable Development Goal 7."
    )

    st.markdown("---")

    st.subheader("🛠️ Technologies Used")

    technologies = pd.DataFrame({
        "Technology": [
            "Python",
            "Streamlit",
            "Pandas",
            "NumPy",
            "Scikit-learn",
            "Random Forest",
            "OpenWeatherMap API"
        ],
        "Purpose": [
            "Programming language",
            "Web application",
            "Data processing",
            "Numerical calculations",
            "Machine Learning",
            "Prediction model",
            "Weather information"
        ]
    })

    st.table(technologies)

    st.markdown("---")

    st.subheader("🌍 Sustainable Development Goal")

    st.info(
        "This project supports SDG 7: Affordable and Clean Energy."
    )

    st.markdown("---")

    st.subheader("👨‍💻 Project Conclusion")

    st.write(
        "Artificial Intelligence can help estimate renewable "
        "energy production using environmental data. "
        "This project demonstrates how technology can support "
        "clean energy awareness and future energy planning."
    )


# ==============================
# FOOTER
# ==============================

st.markdown(
    """
    <div class="footer">
    ⚡ Renewable Energy AI Predictor |
    Built using Python and Streamlit |
    🌱 Supporting Clean Energy
    </div>
    """,
    unsafe_allow_html=True
)


