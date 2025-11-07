import joblib
import pandas as pd

# === Load model and encoders ===
model = joblib.load("yield_xgb_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")

# === Define input fields ===
fields = [
    "Soil_Type",
    "Crop",
    "Rainfall_mm",
    "Temperature_Celsius",
    "Fertilizer_Used",
    "Irrigation_Used",
    "Weather_Condition",
    "Days_to_Harvest"
]

# === Collect user inputs ===
input_data = {}
for field in fields:
    value = input(f"Enter {field}: ").strip()

    # Convert numeric fields
    if field in ["Rainfall_mm", "Temperature_Celsius", "Days_to_Harvest"]:
        value = float(value)

    # Normalize boolean fields
    elif field in ["Fertilizer_Used", "Irrigation_Used"]:
        value = str(value).lower() in ["true", "yes", "1"]

    # Normalize text fields (capitalize like training data)
    elif field in ["Soil_Type", "Crop", "Weather_Condition"]:
        value = value.capitalize()

    input_data[field] = value

# === Create DataFrame ===
X_new = pd.DataFrame([input_data])

# === Encode categorical features ===
for col, le in label_encoders.items():
    if col in X_new.columns:
        try:
            X_new[col] = le.transform(X_new[col])
        except ValueError:
            print(f"⚠️ Unseen category in '{col}'. Using default encoding.")
            X_new[col] = le.transform([le.classes_[0]])

# === Convert boolean fields to int ===
for col in ["Fertilizer_Used", "Irrigation_Used"]:
    if col in X_new.columns:
        X_new[col] = X_new[col].astype(int)

# === Predict ===
predicted_yield = model.predict(X_new)[0]
print(f"\n🌾 Predicted Yield: {predicted_yield:.2f} tons per hectare")
