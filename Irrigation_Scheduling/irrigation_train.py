import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# === Load and clean ===
df = pd.read_csv("Irrigation_Data/cleaned_irrigation_data.csv").dropna().drop_duplicates()

X = df.drop("class", axis=1)
y = df["class"]

# Encode target
le = LabelEncoder()
y = le.fit_transform(y)

# Scale numeric features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Balance dataset
X_bal, y_bal = SMOTE(random_state=42).fit_resample(X_scaled, y)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_bal, y_bal, test_size=0.2, random_state=42, stratify=y_bal
)

# === Train Random Forest ===
model = RandomForestClassifier(
    n_estimators=400,
    max_depth=10,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# === Evaluate ===
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print("\n", classification_report(y_test, y_pred, target_names=le.classes_))

sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d",
            cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("rf_confusion_matrix.png", dpi=300)
plt.close()

# Save
joblib.dump(model, "irrigation_rf_model.pkl")
joblib.dump(le, "class_label_encoder.pkl")
joblib.dump(scaler, "feature_scaler.pkl")
print("✅ RF model, encoder, and scaler saved.")
