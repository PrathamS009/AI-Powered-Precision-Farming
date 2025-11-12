import joblib
import numpy as np

# Load model, encoder, and scaler
model = joblib.load("irrigation_xgb_model.pkl")
le = joblib.load("class_label_encoder.pkl")
scaler = joblib.load("feature_scaler.pkl")

print("\n🌾 Irrigation Class Prediction\n")
temperature = float(input("Temperature (°C): "))
pressure = float(input("Pressure (hPa): "))
altitude = float(input("Altitude (m): "))
soil_moisture = float(input("Soil Moisture (sensor value): "))

# Prepare and scale input
X_input = np.array([[temperature, pressure, altitude, soil_moisture]])
X_scaled = scaler.transform(X_input)

# Predict
pred_encoded = model.predict(X_scaled)[0]
pred_label = le.inverse_transform([pred_encoded])[0]

print(f"\n💧 Predicted Irrigation Class: {pred_label}")
