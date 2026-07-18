# 🚀 Production Deployment and Usage Guide

This guide provides a comprehensive, step-by-step walkthrough of setting up, running, and deploying the heart disease prediction model. It is designed to be accessible for beginners while maintaining production-grade standards for experienced professionals.

---

## 📂 Table of Contents
1. [Prerequisites & Environment Setup](#-1-prerequisites--environment-setup)
2. [Model Training & Serialization](#-2-model-training--serialization)
3. [Local Inference Pipeline (Python)](#-3-local-inference-pipeline-python)
4. [FastAPI Web Service (Production-Ready API)](#-4-fastapi-web-service-production-ready-api)
5. [Containerization with Docker](#-5-containerization-with-docker)
6. [Cloud Deployment Walkthrough](#-6-cloud-deployment-walkthrough)

---

## 📦 1. Prerequisites & Environment Setup

Before starting, ensure you have **Python 3.7+** and **Git** installed on your machine.

### Why Use a Virtual Environment?
A virtual environment (`venv`) isolates your project's dependencies from your global system. This prevents package version conflicts (e.g., if another project requires an older version of Scikit-Learn).

### Step-by-Step Setup:

#### 1. Clone the Repository
Open your terminal or command prompt and run:
```bash
git clone https://github.com/Shrimun/Predicting-Heart-Disease-with-K-Nearest-Neighbors.git
cd Predicting-Heart-Disease-with-K-Nearest-Neighbors
```

#### 2. Create the Virtual Environment
Create an isolated directory called `venv/` containing a lightweight python copy:
```bash
# Windows
python -m venv venv

# macOS/Linux
python3 -m venv venv
```

#### 3. Activate the Virtual Environment
Tell your terminal session to use the virtual environment's python interpreter:
```bash
# Windows (Command Prompt)
.\venv\Scripts\activate

# Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# macOS/Linux
source venv/bin/activate
```
*(Your terminal prompt should now show `(venv)` at the beginning of the line.)*

#### 4. Install Required Packages
Install all libraries specified in the dependency file:
```bash
pip install -r requirements.txt
```

---

## 🤖 2. Model Training & Serialization

The champion model selected for deployment is a **Tuned K-Nearest Neighbors ($K=9$)** classifier because it achieves **100.00% Recall** (crucial for medical safety to ensure no diseased patients are missed) and **90.16% Accuracy**.

To train the model and serialize (save) it for deployment, run:
```bash
python src/save_models.py
```

### What happens when you run this?
The script processes the raw Cleveland dataset, fits the data scaler, trains the model, and creates a `models/` directory containing two key artifacts:
* `models/scaler.pkl`: The fitted `StandardScaler` containing the mean and variance of the training dataset. We need this during inference to scale new patient data.
* `models/knn_best.pkl`: The trained classifier containing the optimal patterns for predicting heart disease.

---

## 🔮 3. Local Inference Pipeline (Python)

To predict heart disease risk on new, unseen patient measurements, you must load the pre-trained scaler, load the classifier, scale the raw measurements, and pass them to the classifier.

Create a scratch script (e.g., `predict_sample.py`) to test local inference:

```python
import joblib
import numpy as np

# 1. Load the pre-trained artifacts
scaler = joblib.load("models/scaler.pkl")
model = joblib.load("models/knn_best.pkl")

# 2. Define raw clinical features for a sample patient (13 attributes):
# [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
# Example: 57-year-old male, atypical chest pain, resting bp 145, cholesterol 233...
raw_patient_data = np.array([[
    57.0, 1.0, 3.0, 145.0, 233.0, 1.0, 2.0, 150.0, 0.0, 2.3, 3.0, 0.0, 6.0
]])

# 3. Standardize the raw features using the training-fit scaler
# CRITICAL: We transform, NOT fit, to avoid data leakage.
scaled_patient_data = scaler.transform(raw_patient_data)

# 4. Generate predictions and class probabilities
prediction = model.predict(scaled_patient_data)
probabilities = model.predict_proba(scaled_patient_data)[0]

# 5. Output human-readable results
result = "Heart Disease Present" if prediction[0] == 1 else "No Heart Disease"
confidence = probabilities[prediction[0]]

print(f"Diagnostic Result: {result}")
print(f"Confidence Level:  {confidence:.2%}")
```

---

## 🌐 4. FastAPI Web Service (Production-Ready API)

To make your model accessible to web applications, dashboards, or mobile apps, you should host it behind a REST API. We use **FastAPI** because of its high performance, automatic JSON serialization, and built-in validation.

### Step 1: Install API Packages
```bash
pip install fastapi uvicorn pydantic
```

### Step 2: Create the API Entrypoint (`main.py`)
Create a file named `main.py` in the project root:

```python
from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import numpy as np

# Initialize the FastAPI App
app = FastAPI(
    title="Heart Disease Prediction API",
    description="A production API serving a Tuned KNN classifier to predict heart disease risk.",
    version="1.0"
)

# Load the saved scaler and model once during application startup
scaler = joblib.load("models/scaler.pkl")
model = joblib.load("models/knn_best.pkl")

# Define request schema with data validation and descriptions
class PatientRequest(BaseModel):
    age: float = Field(..., description="Age in years", example=57.0)
    sex: float = Field(..., description="Biological sex (1.0 = Male, 0.0 = Female)", example=1.0)
    cp: float = Field(..., description="Chest pain type (0.0 to 3.0)", example=3.0)
    trestbps: float = Field(..., description="Resting blood pressure in mm Hg", example=145.0)
    chol: float = Field(..., description="Serum cholesterol in mg/dl", example=233.0)
    fbs: float = Field(..., description="Fasting blood sugar > 120 mg/dl (1.0 = True, 0.0 = False)", example=1.0)
    restecg: float = Field(..., description="Resting electrocardiographic results (0.0 to 2.0)", example=2.0)
    thalach: float = Field(..., description="Maximum heart rate achieved", example=150.0)
    exang: float = Field(..., description="Exercise-induced angina (1.0 = Yes, 0.0 = No)", example=0.0)
    oldpeak: float = Field(..., description="ST depression induced by exercise", example=2.3)
    slope: float = Field(..., description="Slope of the peak exercise ST segment (0.0 to 2.0)", example=2.0)
    ca: float = Field(..., description="Number of major vessels colored by fluoroscopy (0.0 to 3.0)", example=0.0)
    thal: float = Field(..., description="Thalassemia type (3.0 = Normal, 6.0 = Fixed Defect, 7.0 = Reversible Defect)", example=6.0)

@app.post("/predict", summary="Predict heart disease risk for a patient")
def predict(patient: PatientRequest):
    # 1. Package Pydantic data into a 2D numpy array
    raw_features = np.array([[
        patient.age, patient.sex, patient.cp, patient.trestbps, patient.chol,
        patient.fbs, patient.restecg, patient.thalach, patient.exang,
        patient.oldpeak, patient.slope, patient.ca, patient.thal
    ]])
    
    # 2. Standard scale the incoming features
    scaled_features = scaler.transform(raw_features)
    
    # 3. Compute prediction (0 or 1) and probabilities
    prediction = int(model.predict(scaled_features)[0])
    probabilities = model.predict_proba(scaled_features)[0]
    
    # 4. Return structured JSON response
    return {
        "prediction_code": prediction,
        "prediction_label": "Heart Disease Present" if prediction == 1 else "No Heart Disease",
        "probability_healthy": round(probabilities[0], 4),
        "probability_disease": round(probabilities[1], 4),
        "status": "Success"
    }
```

### Step 3: Run the API Locally
Start the server using **Uvicorn** (the ASGI server for FastAPI):
```bash
uvicorn main:app --reload
```
* The `--reload` flag tells Uvicorn to restart automatically if you change your code.

### Step 4: Interactive API Testing (Swagger UI)
FastAPI automatically generates interactive documentation for your API:
* Open your web browser and navigate to `http://127.0.0.1:8000/docs`.
* Locate the `POST /predict` endpoint, click **Try it out**.
* The pre-filled example values can be modified. Click **Execute**.
* You will see the server's output response in standard JSON format:
   ```json
   {
     "prediction_code": 0,
     "prediction_label": "No Heart Disease",
     "probability_healthy": 0.8889,
     "probability_disease": 0.1111,
     "status": "Success"
   }
   ```

---

## 🐳 5. Containerization with Docker

### Why Docker?
Docker packages your application, code, Python interpreter, system libraries, and dependencies into a lightweight image. This guarantees that the API will run exactly the same way in the cloud as it does on your local machine, eliminating OS-specific configuration bugs.

### Create the `Dockerfile`
Create a file named `Dockerfile` in the root of the project:

```dockerfile
# 1. Use a lightweight official Python runtime as base image
FROM python:3.9-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy requirements and install dependencies
# We run install before copying the code to leverage Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt fastapi uvicorn pydantic

# 4. Copy models folder and main API code
COPY models/ models/
COPY main.py .

# 5. Inform Docker that the container listens on port 8000
EXPOSE 8000

# 6. Define the command to start the API server on boot
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Build & Run the Container Locally:
If you have Docker Desktop installed, run these commands:
```bash
# Build the docker image
docker build -t heart-disease-api .

# Run the container mapping host port 8000 to container port 8000
docker run -d -p 8000:8000 heart-disease-api
```
Now, navigating to `http://localhost:8000/docs` will route requests directly inside your running container.

---

## 🐳 6. Cloud Deployment Walkthrough

Once containerized, you can deploy your application to any cloud hosting provider. Here are step-by-step instructions for three common platforms:

### Option A: Render (Easiest & Free Tier)
Render offers direct support for Dockerfiles:
* Push your project code, including the `Dockerfile`, `models/`, and `main.py` to a public or private GitHub repository.
* Sign up on [Render.com](https://render.com) and link your GitHub account.
* Click **New** -> **Web Service**.
* Select your project repository.
* In the settings, change the **Runtime** to `Docker` (Render will automatically look for the `Dockerfile`).
* Click **Deploy Web Service**. Render will build the container image and assign a public URL (e.g. `https://your-service.onrender.com`).

### Option B: Heroku
* Install the Heroku CLI and login: `heroku login`.
* Log into the container registry: `heroku container:login`.
* Create a new Heroku app: `heroku create your-heart-disease-api`.
* Build and push the image to Heroku's registry:
   ```bash
   heroku container:push web --app your-heart-disease-api
   ```
* Release the container to make it live:
   ```bash
   heroku container:release web --app your-heart-disease-api
   ```

### Option C: AWS App Runner
AWS App Runner is a fully-managed service that makes it easy to deploy containerized web applications:
* Push your built Docker image to **Amazon ECR** (Elastic Container Registry).
* Go to the AWS Console and search for **App Runner**.
* Click **Create service**, choose **Container registry** as the source provider, and select your ECR image.
* Set the port to `8000` and click **Create & Deploy**.
