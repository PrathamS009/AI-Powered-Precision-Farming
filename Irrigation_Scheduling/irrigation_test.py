import joblib
import numpy as np

# === Load trained model and label encoder ===
model = joblib.load("irrigation_xgb_model.pkl")
le = joblib.load("class_label_encoder.pkl")

# === Ask user for input ===
print("\n🌾 Irrigation Class Prediction\n")

temperature = float(input("Enter Temperature (°C): "))
pressure = float(input("Enter Pressure (hPa): "))
altitude = float(input("Enter Altitude (m): "))
soil_moisture = float(input("Enter Soil Moisture (% or sensor value): "))

# === Prepare input ===
features = np.array([[temperature, pressure, altitude, soil_moisture]])

# === Predict class ===
pred_class_encoded = model.predict(features)[0]
pred_class = le.inverse_transform([pred_class_encoded])[0]

print(f"\n💧 Predicted Irrigation Class: {pred_class}")
