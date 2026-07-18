import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import joblib
import os

# Load dataset
columns = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", 
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
]
csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "heart.csv")
df = pd.read_csv(csv_path, header=None, names=columns)

# Preprocessing
df.replace("?", np.nan, inplace=True)
df["ca"] = pd.to_numeric(df["ca"])
df["thal"] = pd.to_numeric(df["thal"])
df.fillna(df.median(numeric_only=True), inplace=True)
df["target"] = df["target"].apply(lambda x: 0 if x == 0 else 1)

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Fit best KNN model (K=9)
final_model = KNeighborsClassifier(n_neighbors=9)
final_model.fit(X_train_scaled, y_train)

# Create models directory
models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(models_dir, exist_ok=True)

# Save best model & scaler
joblib.dump(final_model, os.path.join(models_dir, "knn_best.pkl"))
joblib.dump(scaler, os.path.join(models_dir, "scaler.pkl"))

print("Model and scaler saved successfully!")

# Test loading the model
loaded_model = joblib.load(os.path.join(models_dir, "knn_best.pkl"))
print(type(loaded_model))
