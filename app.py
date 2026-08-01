import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.ensemble import RandomForestRegressor

from weather import get_weather
from chatbot import get_bot_response


st.set_page_config(
    page_title="AI Renewable Energy Predictor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>

.main {
    background: linear-gradient(
        180deg,
        #e8f5ff,
        #ffffff
    );
}

h1, h2, h3 {
    color: #00695c;
}

.sidebar-title {
    font-size:28px;
    font-weight:bold;
    color:#00695c;
}

.card {
    background:white;
    padding:20px;
    border-radius:18px;
    box-shadow:0px 5px 15px rgba(0,0,0,0.12);
    margin-bottom:15px;
}

.result-card {
    background:linear-gradient(
        135deg,
        #b9f6ca,
        #80cbc4
    );
    padding:25px;
    border-radius:20px;
    text-align:center;
    font-size:24px;
    font-weight:bold;
}

.info-card {
    background:#ffffff;
    padding:18px;
    border-radius:15px;
    border-left:6px solid #00bcd4;
}

.stButton button {
    width:100%;
    border-radius:12px;
    height:3em;
    background:
    linear-gradient(
        90deg,
        #00695c,
        #00bcd4
    );
    color:white;
    font-weight:bold;
}

.footer {
    text-align:center;
    color:grey;
}

</style>
""", unsafe_allow_html=True)



def sunlight_calculation(cloud):

    sunlight = 12 - ((cloud / 100) * 12)

    if sunlight < 0:
        sunlight = 0

    return sunlight



def calculate_total(output, number):

    return output * number



def create_models():

    np.random.seed(42)

    rows = 4000

    temperature = np.random.randint(
        15,
        45,
        rows
    )

    humidity = np.random.randint(
        20,
        95,
        rows
    )

    wind_speed = np.random.uniform(
        0.5,
        30,
        rows
    )

    sunlight = np.random.uniform(
        0,
        12,
        rows
    )

    cloud = np.random.randint(
        0,
        100,
        rows
    )


    solar_output = (
        sunlight * 45
        + (100-cloud)*0.35
        - humidity*0.15
        + np.random.normal(0,8,rows)
    )


    wind_output = (
        (wind_speed**2)*2.3
        - humidity*0.12
        + np.random.normal(0,6,rows)
    )


    data = pd.DataFrame({

        "temperature":temperature,
        "humidity":humidity,
        "wind_speed":wind_speed,
        "sunlight_hours":sunlight,
        "cloud_cover":cloud,
        "solar_output":solar_output,
        "wind_output":wind_output

    })


    data.to_csv(
        "dataset.csv",
        index=False
    )


    solar_model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )


    solar_model.fit(
        data[
            [
                "temperature",
                "humidity",
                "sunlight_hours",
                "cloud_cover"
            ]
        ],
        data["solar_output"]
    )


    wind_model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )


    wind_model.fit(
        data[
            [
                "wind_speed",
                "humidity",
                "temperature"
            ]
        ],
        data["wind_output"]
    )
if not os.path.exists("solar_model.pkl") or not os.path.exists("wind_model.pkl"):

    with st.spinner("Training AI models for first use..."):
        create_models()



solar_model = joblib.load(
    "solar_model.pkl"
)

wind_model = joblib.load(
    "wind_model.pkl"
)



# SIDEBAR

with st.sidebar:

    st.markdown(
        "<div class='sidebar-title'>⚡ AI Energy Predictor</div>",
        unsafe_allow_html=True
    )

    st.write("")


    try:
        st.image(
            "images/logo.png",
            width=120
        )

    except:
        st.write("⚡ Renewable Energy AI")


    st.write("")


    page = st.selectbox(

        "Choose Section",

        [

            "🏠 Home",

            "☀ Solar Prediction",

            "🌬 Wind Prediction",

            "📊 Analytics",

            "🤖 AI Chatbot",

            "ℹ About Project"

        ]

    )


    st.divider()


    st.caption(
        "SDG 7 - Affordable and Clean Energy"
    )



# HOME PAGE


if page == "🏠 Home":


    try:

        st.image(
            "images/banner.jpg",
            use_container_width=True
        )

    except:

        pass



    st.title(
        "⚡ Renewable Energy Output Predictor"
    )


    st.markdown(
        """
        <div class="info-card">

        🌍 Predict renewable energy production
        using Artificial Intelligence and live weather data.

        <br><br>

        ☀ Solar Energy Prediction

        <br>

        🌬 Wind Energy Prediction

        <br>

        🌦 Live Weather Integration

        <br>

        🤖 Machine Learning Based Forecasting

        </div>

        """,
        unsafe_allow_html=True
    )


    st.write("")


    st.subheader(
        "🌱 How it works"
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        st.markdown(
            """
            <div class="card">

            🌦

            <h3>Weather Data</h3>

            Live temperature,
            humidity,
            wind speed,
            and cloud data.

            </div>

            """,
            unsafe_allow_html=True
        )


    with col2:

        st.markdown(
            """
            <div class="card">

            🤖

            <h3>AI Model</h3>

            Machine learning predicts
            energy output.

            </div>

            """,
            unsafe_allow_html=True
        )


    with col3:

        st.markdown(
            """
            <div class="card">

            ⚡

            <h3>Energy Output</h3>

            Shows estimated
            renewable production.

            </div>

            """,
            unsafe_allow_html=True
        )



    st.divider()



    st.subheader(
        "🌦 Check Live Weather Prediction"
    )


    city = st.text_input(
        "Enter City Name",
        placeholder="Example: Chennai"
    )


    if st.button(
        "Get Weather Prediction",
        key="home_weather"
    ):


        weather = get_weather(city)


        if weather is None:

            st.error(
                "Unable to fetch weather data."
            )


        else:


            st.success(
                f"Weather fetched for {weather['city']}"
            )


            c1, c2, c3, c4 = st.columns(4)


            with c1:

                st.metric(
                    "🌡 Temperature",
                    f"{weather['temperature']} °C"
                )


            with c2:

                st.metric(
                    "💧 Humidity",
                    f"{weather['humidity']} %"
                )


            with c3:

                st.metric(
                    "💨 Wind",
                    f"{weather['wind_speed']} m/s"
                )


            with c4:

                st.metric(
                    "☁ Cloud",
                    f"{weather['cloud_cover']} %"
                )


            sunlight = sunlight_calculation(
                weather["cloud_cover"]
            )


            solar_input = pd.DataFrame(
                [
                    {
                        "temperature":
                        weather["temperature"],

                        "humidity":
                        weather["humidity"],

                        "sunlight_hours":
                        sunlight,

                        "cloud_cover":
                        weather["cloud_cover"]
                    }
                ]
            )


            wind_input = pd.DataFrame(
                [
                    {
                        "wind_speed":
                        weather["wind_speed"],

                        "humidity":
                        weather["humidity"],

                        "temperature":
                        weather["temperature"]
                    }
                ]
            )


            solar_result = solar_model.predict(
                solar_input
            )[0]


            wind_result = wind_model.predict(
                wind_input
            )[0]


            r1, r2 = st.columns(2)


            with r1:

                st.markdown(
                    f"""
                    <div class="result-card">

                    ☀ Solar Output

                    <br><br>

                    {solar_result:.2f} kWh

                    </div>
                    """,
                    unsafe_allow_html=True
                )


            with r2:

                st.markdown(
                    f"""
                    <div class="result-card">

                    🌬 Wind Output

                    <br><br>

                    {wind_result:.2f} kWh

                    </div>
                    """,
                    unsafe_allow_html=True
                )
# SOLAR PREDICTION PAGE


elif page == "☀ Solar Prediction":


    st.title(
        "☀ Solar Energy Prediction"
    )


    st.markdown(
        """
        <div class="info-card">

        Predict solar energy output using
        live weather conditions and
        number of solar panels.

        </div>
        """,
        unsafe_allow_html=True
    )


    city = st.text_input(
        "🏙 Enter City Name",
        key="solar_city"
    )


    panels = st.number_input(

        "🔋 Number of Solar Panels",

        min_value=1,

        max_value=10000,

        value=1

    )


    if st.button(
        "Predict Solar Energy",
        key="solar_button"
    ):


        weather = get_weather(city)


        if weather is None:


            st.error(
                "❌ Unable to fetch weather data."
            )


        else:


            st.success(
                f"Weather data received for {weather['city']}"
            )


            sunlight = sunlight_calculation(
                weather["cloud_cover"]
            )


            solar_input = pd.DataFrame(
                [
                    {

                    "temperature":
                    weather["temperature"],


                    "humidity":
                    weather["humidity"],


                    "sunlight_hours":
                    sunlight,


                    "cloud_cover":
                    weather["cloud_cover"]

                    }
                ]
            )


            single_panel_output = solar_model.predict(
                solar_input
            )[0]


            total_output = calculate_total(
                single_panel_output,
                panels
            )


            st.divider()


            a,b = st.columns(2)


            with a:


                st.markdown(

                    f"""
                    <div class="result-card">

                    ☀ One Panel Output

                    <br><br>

                    {single_panel_output:.2f} kWh

                    </div>
                    """,

                    unsafe_allow_html=True

                )


            with b:


                st.markdown(

                    f"""
                    <div class="result-card">

                    ⚡ Total Output

                    <br><br>

                    {total_output:.2f} kWh

                    </div>
                    """,

                    unsafe_allow_html=True

                )



            st.write("")


            st.subheader(
                "🌦 Current Weather"
            )


            c1,c2 = st.columns(2)


            with c1:


                st.metric(

                    "🌡 Temperature",

                    f"{weather['temperature']} °C"

                )


                st.metric(

                    "💧 Humidity",

                    f"{weather['humidity']} %"

                )



            with c2:


                st.metric(

                    "☁ Cloud Cover",

                    f"{weather['cloud_cover']} %"

                )


                st.metric(

                    "☀ Sunlight",

                    f"{sunlight:.1f} hours"

                )



            st.info(

                f"Your


    joblib.dump(
        solar_model,
        "solar_model.pkl"
    )


    joblib.dump(
        wind_model,
        "wind_model.pkl"
    )
# WIND PREDICTION PAGE


elif page == "🌬 Wind Prediction":


    st.title(
        "🌬 Wind Energy Prediction"
    )


    st.markdown(
        """
        <div class="info-card">

        Predict wind energy output using
        live weather conditions and
        number of wind turbines.

        </div>
        """,
        unsafe_allow_html=True
    )


    city = st.text_input(
        "🏙 Enter City Name",
        key="wind_city"
    )


    windmills = st.number_input(

        "🌬 Number of Windmills",

        min_value=1,

        max_value=10000,

        value=1

    )


    if st.button(

        "Predict Wind Energy",

        key="wind_button"

    ):


        weather = get_weather(city)


        if weather is None:


            st.error(
                "❌ Unable to fetch weather data."
            )


        else:


            st.success(
                f"Weather data received for {weather['city']}"
            )


            wind_input = pd.DataFrame(

                [
                    {

                    "wind_speed":
                    weather["wind_speed"],


                    "humidity":
                    weather["humidity"],


                    "temperature":
                    weather["temperature"]

                    }

                ]

            )


            single_wind_output = wind_model.predict(
                wind_input
            )[0]


            total_wind_output = calculate_total(

                single_wind_output,

                windmills

            )


            st.divider()



            a,b = st.columns(2)



            with a:


                st.markdown(

                    f"""
                    <div class="result-card">

                    🌬 One Windmill Output

                    <br><br>

                    {single_wind_output:.2f} kWh

                    </div>

                    """,

                    unsafe_allow_html=True

                )



            with b:


                st.markdown(

                    f"""
                    <div class="result-card">

                    ⚡ Total Wind Output

                    <br><br>

                    {total_wind_output:.2f} kWh

                    </div>

                    """,

                    unsafe_allow_html=True

                )



            st.write("")


            st.subheader(
                "🌦 Current Weather"
            )



            c1,c2 = st.columns(2)



            with c1:


                st.metric(

                    "💨 Wind Speed",

                    f"{weather['wind_speed']} m/s"

                )


                st.metric(

                    "🌡 Temperature",

                    f"{weather['temperature']} °C"

                )



            with c2:


                st.metric(

                    "💧 Humidity",

                    f"{weather['humidity']} %"

                )


                st.metric(

                    "☁ Cloud Cover",

                    f"{weather['cloud_cover']} %"

                )



            st.info(

                f"Your {windmills} windmills can produce approximately {total_wind_output:.2f} kWh."

            )
# ANALYTICS PAGE


elif page == "📊 Analytics":


    st.title(
        "📊 Renewable Energy Analytics"
    )


    st.markdown(
        """
        <div class="info-card">

        Explore generated dataset,
        energy distribution and
        weather parameters.

        </div>
        """,
        unsafe_allow_html=True
    )


    if os.path.exists("dataset.csv"):


        df = pd.read_csv(
            "dataset.csv"
        )


        st.subheader(
            "📄 Dataset Preview"
        )


        st.dataframe(
            df.head(10),
            use_container_width=True
        )



        st.divider()



        col1,col2,col3 = st.columns(3)



        with col1:


            st.metric(

                "🌡 Average Temperature",

                f"{df['temperature'].mean():.1f} °C"

            )



        with col2:


            st.metric(

                "💧 Average Humidity",

                f"{df['humidity'].mean():.1f} %"

            )



        with col3:


            st.metric(

                "💨 Average Wind Speed",

                f"{df['wind_speed'].mean():.1f} m/s"

            )



        st.subheader(
            "☀ Solar Output Distribution"
        )


        st.line_chart(

            df["solar_output"]

        )



        st.subheader(
            "🌬 Wind Output Distribution"
        )


        st.line_chart(

            df["wind_output"]

        )



        st.subheader(
           
# ABOUT PROJECT PAGE


elif page == "ℹ About Project":


    st.title(
        "ℹ About The Project"
    )


    st.markdown(

        """
        <div class="info-card">

        <h3>⚡ AI Renewable Energy Output Predictor</h3>


        This project uses Artificial Intelligence
        and Machine Learning to predict renewable
        energy generation using weather conditions.


        <br><br>


        <b>Technologies Used:</b>

        <br>

        🐍 Python

        <br>

        🎨 Streamlit

        <br>

        🤖 Machine Learning

        <br>

        🌦 OpenWeatherMap API

        <br>

        📊 Pandas & NumPy


        </div>

        """,

        unsafe_allow_html=True

    )


    st.write("")


    c1,c2,c3 = st.columns(3)


    with c1:


        st.markdown(

            """
            <div class="card">

            ☀

            <h3>Solar Energy</h3>

            Predicts solar output based
            on sunlight, humidity and
            cloud conditions.

            </div>

            """,

            unsafe_allow_html=True

        )



    with c2:


        st.markdown(

            """
            <div class="card">

            🌬

            <h3>Wind Energy</h3>

            Predicts wind power output
            using wind speed and weather.

            </div>

            """,

            unsafe_allow_html=True

        )



    with c3:


        st.markdown(

            """
            <div class="card">

            🌍

            <h3>SDG 7</h3>

            Supports affordable,
            clean and sustainable energy.

            </div>

            """,

            unsafe_allow_html=True

        )



    st.divider()



    st.subheader(

        "🌱 Future Improvements"

    )


    st.write(

        """
        • Real-time energy monitoring

        • Satellite weather data integration

        • Better deep learning models

        • Battery storage prediction

        • Carbon emission tracking

        """

    )





# FOOTER


st.markdown(

    "---"

)



st.markdown(

    """
    <div class="footer">

    ⚡ AI Renewable Energy Output Predictor

    <br>

    Built using Python • Streamlit • Machine Learning

    <br>

    🌍 SDG 7: Affordable and Clean Energy

    </div>

    """,

    unsafe_allow_html=True

)
