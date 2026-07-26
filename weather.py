import requests

# Replace this with your own OpenWeatherMap API key
API_KEY = "6da8cbebad8d9c8a5e6b382d85ecd6b8"

def get_weather(city):
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"

    try:
        response = requests.get(url, timeout=10)

        if response.status_code != 200:
            return None

        data = response.json()

        return {
            "temperature": data["main"]["temp"],
            "humidity": data["main"]["humidity"],
            "wind_speed": data["wind"]["speed"],
            "cloud_cover": data["clouds"]["all"],
            "city": data["name"]
        }

    except Exception:
        return None