import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load dataset
df = pd.read_csv("dataset.csv")

# Features (inputs)
X = df[["temperature", "humidity", "sunlight_hours", "cloud_cover"]]

# Target (output we want AI to learn)
y = df["solar_output"]

# Create model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train model
model.fit(X, y)

# Save model
joblib.dump(model, "solar_model.pkl")

print("Solar model trained and saved!")