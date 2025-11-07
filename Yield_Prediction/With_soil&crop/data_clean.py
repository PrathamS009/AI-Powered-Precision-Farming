import pandas as pd

# === Step 1: Load your dataset ===
# Change the filename to match your file
df = pd.read_csv("Yield_Data/crop_yield.csv")

# === Step 2: Define columns to keep ===
required_columns = [
    'Soil_Type',          # e.g. sandy, loamy, clay
    'Crop',               # crop name
    'Rainfall_mm',        # rainfall in mm
    'Temperature_Celsius',      # temperature in °C
    'Fertilizer_Used',    # yes/no or 1/0
    'Irrigation_Used',    # yes/no or 1/0
    'Weather_Condition',  # e.g. sunny, cloudy
    'Days_to_Harvest',
    'Yield_tons_per_hectare'              # target variable
]

# Keep only these columns if present
df = df[[col for col in required_columns if col in df.columns]]

# === Step 3: Filter crops ===
valid_crops = ['bajra', 'bottlegourd', 'brinjal', 'cauliflower', 'maize', 'rice', 'wheat']
df = df[df['Crop'].str.lower().isin(valid_crops)]

# === Step 4: Drop missing or invalid entries ===
df = df.dropna(subset=required_columns)  # drop rows with missing key values
df = df.reset_index(drop=True)

# === Step 5: Save cleaned dataset ===
df.to_csv("Yield_Data/cleaned_yield_data.csv", index=False)

print("✅ Cleaned dataset saved as 'cleaned_yield_data.csv'")
print("Rows:", len(df))
print("Columns:", list(df.columns))
