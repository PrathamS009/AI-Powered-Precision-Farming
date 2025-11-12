import pandas as pd

# Load CSV
df = pd.read_csv("Irrigation_Data/Irrigation_Scheduling.csv")

# Drop unwanted columns
df = df.drop(columns=["id", "note", "status", "date", "time"])

# Handle duplicates and missing values
df = df.drop_duplicates()
df = df.dropna()

# Optional: reset index
df = df.reset_index(drop=True)

# Save cleaned dataset
df.to_csv("Irrigation_Data/cleaned_irrigation_data.csv", index=False)

print("✅ Cleaned dataset saved as 'cleaned_irrigation_data.csv'")
print(df.head())
