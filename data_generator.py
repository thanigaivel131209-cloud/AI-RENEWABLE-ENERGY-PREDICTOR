import pandas as pd
import numpy as np

np.random.seed(42)

n = 2000

# ---------------- WEATHER DATA ----------------
temperature = np.random.randint(15, 45, n)
humidity = np.random.randint(20, 95, n)
wind_speed = np.random.uniform(0.5, 25, n)
sunlight_hours = np.random.uniform(0, 12, n)
cloud_cover = np.random.randint(0, 100, n)

# ---------------- SOLAR OUTPUT ----------------
solar_output = (
    sunlight_hours * 40
    + (100 - cloud_cover) * 0.3
    - humidity * 0.2
    + np.random.normal(0, 8, n)
)

# ---------------- WIND OUTPUT ----------------
wind_output = (
    (wind_speed ** 2) * 2.5
    - humidity * 0.1
    + np.random.normal(0, 6, n)
)

# ---------------- DATAFRAME ----------------
df = pd.DataFrame({
    "temperature": temperature,
    "humidity": humidity,
    "wind_speed": wind_speed,
    "sunlight_hours": sunlight_hours,
    "cloud_cover": cloud_cover,
    "solar_output": solar_output,
    "wind_output": wind_output
})

# ---------------- SAVE ----------------
df.to_csv("dataset.csv", index=False)

print("Dataset created successfully!")