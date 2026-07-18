import json
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load and preprocess dataset
columns = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", 
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
]
csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "heart.csv")
df = pd.read_csv(csv_path, header=None, names=columns)

df.replace("?", np.nan, inplace=True)
df["ca"] = pd.to_numeric(df["ca"])
df["thal"] = pd.to_numeric(df["thal"])
df.fillna(df.median(numeric_only=True), inplace=True)
df["target"] = df["target"].apply(lambda x: 0 if x == 0 else 1)

# Extract raw stats for the charts
raw_stats = {
    "total_patients": int(df.shape[0]),
    "healthy_count": int((df["target"] == 0).sum()),
    "disease_count": int((df["target"] == 1).sum()),
    # Age groups
    "age_groups": {
        "under_45": int((df["age"] < 45).sum()),
        "45_54": int(((df["age"] >= 45) & (df["age"] < 55)).sum()),
        "55_64": int(((df["age"] >= 55) & (df["age"] < 65)).sum()),
        "65_plus": int((df["age"] >= 65).sum())
    },
    # Gender disease breakdown
    "gender_stats": {
        "female_healthy": int(((df["sex"] == 0) & (df["target"] == 0)).sum()),
        "female_disease": int(((df["sex"] == 0) & (df["target"] == 1)).sum()),
        "male_healthy": int(((df["sex"] == 1) & (df["target"] == 0)).sum()),
        "male_disease": int(((df["sex"] == 1) & (df["target"] == 1)).sum())
    },
    # Chest pain stats
    "chest_pain_stats": {
        "typical_angina": int((df["cp"] == 0).sum()),
        "atypical_angina": int((df["cp"] == 1).sum()),
        "non_anginal": int((df["cp"] == 2).sum()),
        "asymptomatic": int((df["cp"] == 3).sum())
    }
}

# Perform train/test split to get exact training points
X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Package all data for dashboard_data.js
dashboard_data = {
    "scaler_mean": scaler.mean_.tolist(),
    "scaler_scale": scaler.scale_.tolist(),
    "X_train": X_train_scaled.tolist(),
    "y_train": y_train.tolist(),
    "raw_stats": raw_stats
}

# Write dashboard_data.js
output_path = os.path.join(os.path.dirname(__file__), "..", "dashboard_data.js")
with open(output_path, "w", encoding="utf-8") as f:
    f.write("window.DASHBOARD_DATA = ")
    json.dump(dashboard_data, f, indent=2)
    f.write(";\n")

print(f"Successfully exported dashboard data to: {output_path}")
