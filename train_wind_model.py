import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load dataset
df = pd.read_csv("dataset.csv")

# Features (inputs)
X = df[["wind_speed", "humidity", "temperature"]]

# Target (output we want AI to learn)
y = df["wind_output"]

# Create model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train model
model.fit(X, y)

# Save model
joblib.dump(model, "wind_model.pkl")

print("Wind model trained and saved!")