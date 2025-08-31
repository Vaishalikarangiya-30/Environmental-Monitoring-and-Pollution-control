
# preprocess_data.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

# Load dataset (Excel file)
df = pd.read_excel('AirQualityUCI.xlsx')

# Remove empty columns (if any)
df = df.iloc[:, :-2]

# Replace -200 with NaN and fill missing values
df.replace(-200, np.nan, inplace=True)
df.fillna(df.mean(numeric_only=True), inplace=True)

# Features (exclude target C6H6(GT))
features = ['CO(GT)', 'PT08.S1(CO)', 'NMHC(GT)', 'NOx(GT)', 'NO2(GT)']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])

# Save scaler
joblib.dump(scaler, 'scaler.pkl')
print("✅ Scaler saved successfully!")
