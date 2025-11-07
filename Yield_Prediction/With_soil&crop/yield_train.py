import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import joblib

# === Load cleaned dataset ===
df = pd.read_csv("Yield_Data/cleaned_yield_data.csv")

# === Separate features and target ===
X = df.drop("Yield_tons_per_hectare", axis=1)
y = df["Yield_tons_per_hectare"]

# === Encode categorical features ===
label_encoders = {}
for col in X.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# === Train-test split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Initialize and train model ===
model = xgb.XGBRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
model.fit(X_train, y_train)

# === Evaluate model ===
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"R² Score: {r2:.3f}")
print(f"MAE: {mae:.3f}")
print(f"RMSE: {rmse:.3f}")

# === Save model and encoders ===
joblib.dump(model, "yield_xgb_model.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")
print("✅ Model and encoders saved successfully.")

# === Plot: Actual vs Predicted ===
plt.figure(figsize=(7, 5))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel("Actual Yield (tons/ha)")
plt.ylabel("Predicted Yield (tons/ha)")
plt.title("Actual vs Predicted Yield")
plt.grid(True)
plt.savefig("yield_prediction_accuracy.png", dpi=300, bbox_inches="tight")
plt.close()

# === Plot: Residual Distribution ===
residuals = y_test - y_pred
plt.figure(figsize=(7, 5))
plt.hist(residuals, bins=20, edgecolor="black", alpha=0.7)
plt.xlabel("Residual (Actual - Predicted)")
plt.ylabel("Frequency")
plt.title("Residuals Distribution")
plt.grid(True)
plt.savefig("yield_residuals_distribution.png", dpi=300, bbox_inches="tight")
plt.close()

print("📊 Charts saved: 'yield_prediction_accuracy.png', 'yield_residuals_distribution.png'")
