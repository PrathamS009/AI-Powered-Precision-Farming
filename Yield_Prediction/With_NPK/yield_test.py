# test_yield_model.py
import joblib
import numpy as np

# Load model
model = joblib.load("yield_model.pkl")

print("\n🌾 Yield Prediction System 🌾\n")

# Get inputs
fertilizer = float(input("Enter Fertilizer amount: "))
temp = float(input("Enter Temperature (°C): "))
N = float(input("Enter Nitrogen (N) value: "))
P = float(input("Enter Phosphorus (P) value: "))
K = float(input("Enter Potassium (K) value: "))

# Prepare data
features = np.array([[fertilizer, temp, N, P, K]])

# Predict yield
pred_yield = model.predict(features)[0]

print(f"\nPredicted Yield: {pred_yield:.2f} tons/hectare\n")
