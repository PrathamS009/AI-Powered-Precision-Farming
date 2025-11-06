import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# === Load cleaned dataset ===
df = pd.read_csv("Irrigation_Data/cleaned_irrigation_data.csv")

# === Separate features and target ===
X = df.drop("class", axis=1)
y = df["class"]

# === Encode categorical target ===
le = LabelEncoder()
y = le.fit_transform(y)

# === Encode any categorical features (if present) ===
for col in X.select_dtypes(include=["object"]).columns:
    X[col] = LabelEncoder().fit_transform(X[col])

# === Train-test split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# === Initialize and train model ===
model = xgb.XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="mlogloss"
)
model.fit(X_train, y_train)

# === Evaluate model ===
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"Accuracy: {acc:.3f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

# === Confusion Matrix Plot ===
plt.figure(figsize=(7, 5))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues",
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("irrigation_confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.close()

# === Save model and encoder ===
joblib.dump(model, "irrigation_xgb_model.pkl")
joblib.dump(le, "class_label_encoder.pkl")
print("✅ Model and encoder saved successfully.")
print("📊 Chart saved: 'irrigation_confusion_matrix.png'")
