# train_and_save_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

# ---------------------------
# 1. Load dataset
# ---------------------------
df = pd.read_excel('AirQualityUCI.xlsx')  # Excel file
df = df.iloc[:, :-2]  # Remove empty columns at the end

# ---------------------------
# 2. Clean dataset
# ---------------------------
# Replace -200 with NaN
df.replace(-200, np.nan, inplace=True)

# Fill missing numeric values with column mean
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

# ---------------------------
# 3. Feature selection
# ---------------------------
features = ['CO(GT)', 'PT08.S1(CO)', 'NMHC(GT)', 'NOx(GT)', 'NO2(GT)']
target = 'C6H6(GT)'

X = df[features]
y = df[target]

# ---------------------------
# 4. Train-test split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------------
# 5. Scale features
# ---------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler
joblib.dump(scaler, 'scaler.pkl')
print("✅ Scaler saved as scaler.pkl")

# ---------------------------
# 6. Train model
# ---------------------------
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# ---------------------------
# 7. Evaluate model
# ---------------------------
y_pred = model.predict(X_test_scaled)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"✅ Model trained! RMSE: {rmse:.2f}")

# ---------------------------
# 8. Save trained model
# ---------------------------
joblib.dump(model, 'pollution_model.pkl')
print("✅ Model saved as pollution_model.pkl")
