import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib

print("⚡ Starting full AI pipeline...")

# ---------------- DATASET ----------------
np.random.seed(42)
n = 2000

temperature = np.random.randint(15, 45, n)
humidity = np.random.randint(20, 95, n)
wind_speed = np.random.uniform(0.5, 25, n)
sunlight_hours = np.random.uniform(0, 12, n)
cloud_cover = np.random.randint(0, 100, n)

solar_output = (
    sunlight_hours * 40
    + (100 - cloud_cover) * 0.3
    - humidity * 0.2
    + np.random.normal(0, 8, n)
)

wind_output = (
    (wind_speed ** 2) * 2.5
    - humidity * 0.1
    + np.random.normal(0, 6, n)
)

df = pd.DataFrame({
    "temperature": temperature,
    "humidity": humidity,
    "wind_speed": wind_speed,
    "sunlight_hours": sunlight_hours,
    "cloud_cover": cloud_cover,
    "solar_output": solar_output,
    "wind_output": wind_output
})

df.to_csv("dataset.csv", index=False)
print("✅ Dataset created")

# ---------------- SOLAR MODEL ----------------
X_solar = df[["temperature", "humidity", "sunlight_hours", "cloud_cover"]]
y_solar = df["solar_output"]

solar_model = RandomForestRegressor(n_estimators=100, random_state=42)
solar_model.fit(X_solar, y_solar)

joblib.dump(solar_model, "solar_model.pkl")
print("☀️ Solar model trained")

# ---------------- WIND MODEL ----------------
X_wind = df[["wind_speed", "humidity", "temperature"]]
y_wind = df["wind_output"]

wind_model = RandomForestRegressor(n_estimators=100, random_state=42)
wind_model.fit(X_wind, y_wind)

joblib.dump(wind_model, "wind_model.pkl")
print("🌬️ Wind model trained")

print("🎉 ALL TASKS COMPLETED SUCCESSFULLY!")