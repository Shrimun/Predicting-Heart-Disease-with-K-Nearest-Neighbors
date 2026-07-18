# ❤️ Predicting Heart Disease with K-Nearest Neighbors

This repository contains a professional, production-grade machine learning pipeline to predict the presence of heart disease using patient clinical measurements from the famous **Cleveland Heart Disease Dataset** (obtained from the UCI Machine Learning Repository).

We build, optimize, and compare multiple supervised classifiers, culminating in a champion model that is serialized and ready for deployment. This guide is structured to be immediately understandable for **beginners** looking to learn machine learning workflows, and **professionals** evaluating model metrics and architecture.

> [!TIP]
> **🚀 Live Interactive Dashboard:** We developed a fully interactive, Power BI-styled storytelling dashboard in this repository! Simply open **[dashboard.html](file:///d:/Predicting-Heart-Disease-with-K-Nearest-Neighbour/Predicting-Heart-Disease-with-K-Nearest-Neighbors/dashboard.html)** in any web browser to explore dataset distributions, interact with model comparison graphs, and run patient diagnostic predictions using a **real-time client-side KNN machine learning engine**.

---

## 🗺️ 1. Project Workflow Architecture

Below is the workflow diagram showing how raw patient data is processed, trained, compared, and exported:

```mermaid
graph TD
    A["Raw Cleveland Data (heart.csv)"] --> B["01. Exploratory Data Analysis (EDA)"]
    B --> C["02. Preprocessing & Pre-Cleaning"]
    C -->|Replace '?' with NaN | D["Median Imputation"]
    D -->|Map classes 1-4 to 1| E["Target Binarization"]
    E -->|80/20 Stratified Split| F["Train/Test Split"]
    F -->|Fit on Train, Transform Test| G["StandardScaler Normalization"]
    G --> H["Model Development"]
    H -->|K=5 Baseline| I["03. KNN Baseline"]
    H -->|Range K=1 to K=31| J["04. KNN Hyperparameter Tuning"]
    H -->|Linear baseline| K["05. Logistic Regression"]
    H -->|Ensemble decision trees| L["06. Random Forest"]
    I & J & K & L --> M["07. Model Comparison & Charts"]
    M -->|Tuned KNN (K=9) Selected| N["Champion Model Export"]
    N -->|joblib.dump| O["models/knn_best.pkl & models/scaler.pkl"]
```

---

## 🏥 2. The Clinical Context: Why Recall is the Priority

In medical diagnostics, machine learning metrics must align with clinical risks. Prediction errors have asymmetrical consequences:
* **False Positive (FP)**: A healthy patient is flagged as having heart disease. This leads to additional, non-invasive diagnostic tests (such as a follow-up ECG or angiogram). While it causes stress, the patient is medically safe.
* **False Negative (FN)**: A patient with heart disease is flagged as completely healthy and sent home. This is **fatal**, as they miss life-saving treatment opportunities.

### Metric Definitions for Healthcare:
* **Recall (Sensitivity)**: $\frac{TP}{TP + FN}$ — Measures what percentage of *actual* heart disease cases were caught by the model. **Minimizing False Negatives means maximizing Recall.**
* **Specificity (True Negative Rate)**: $\frac{TN}{TN + FP}$ — Measures how well the model identifies healthy individuals, reducing false alarms.
* **Precision**: $\frac{TP}{TP + FP}$ — Out of all patients predicted to have disease, how many actually had it.
* **F1-Score**: The harmonic mean of Precision and Recall, showing the overall balance of the classifier.

---

## 📊 3. Dataset Overview

The dataset contains **303 patient records** characterized by **13 clinical attributes** and **1 target column**:

| Feature | Description | DataType | Values |
| :--- | :--- | :---: | :--- |
| `age` | Age of the patient | Numerical | Years |
| `sex` | Biological sex | Categorical | 1 = Male, 0 = Female |
| `cp` | Chest pain type | Categorical | 0 = Typical Angina, 1 = Atypical Angina, 2 = Non-anginal Pain, 3 = Asymptomatic |
| `trestbps` | Resting blood pressure | Numerical | mm Hg (measured on admission) |
| `chol` | Serum cholesterol | Numerical | mg/dl |
| `fbs` | Fasting blood sugar > 120 mg/dl | Categorical | 1 = True, 0 = False |
| `restecg` | Resting electrocardiographic results | Categorical | 0 = Normal, 1 = ST-T Wave Abnormality, 2 = Left Ventricular Hypertrophy |
| `thalach` | Maximum heart rate achieved | Numerical | bpm |
| `exang` | Exercise-induced angina | Categorical | 1 = Yes, 0 = No |
| `oldpeak` | ST depression induced by exercise | Numerical | ST segment depression value |
| `slope` | Slope of peak exercise ST segment | Categorical | 0 = Upsloping, 1 = Flat, 2 = Downsloping |
| `ca` | Major vessels colored by fluoroscopy | Numerical | 0–3 (imputed for missing values) |
| `thal` | Thalassemia type | Categorical | 3.0 = Normal, 6.0 = Fixed Defect, 7.0 = Reversible Defect |
| `target` | Heart disease diagnosis | Categorical | 0 = No Disease (Healthy), 1–4 = Levels of Disease Present |

*Note: Missing values in `ca` (4 records) and `thal` (2 records) are designated by `?` in raw data and imputed with column medians.*

---

## 🔧 4. Deep-Dive Preprocessing Rationale

Understanding *why* we modify data is as important as the code itself:

* **Target Binarization**:
   The raw target contains values `0, 1, 2, 3, 4`. Values `1` through `4` indicate different degrees of heart disease. To build a classifier that identifies the general presence of heart disease, we map all non-zero values (`1, 2, 3, 4`) to `1` (Disease Present) and `0` to `0` (No Disease).
* **Robust Median Imputation**:
   Rather than dropping records with missing values (which reduces our small dataset of 303 samples), we impute missing values. We use the **median** instead of the mean because the median is robust to extreme outliers and preserves integer bounds for discrete variables like `ca`.
* **Feature Scaling (StandardScaler)**:
   KNN determines classes based on Euclidean distance:
   $$d(p, q) = \sqrt{\sum (p_i - q_i)^2}$$
   Without scaling, features with large values like cholesterol (`chol` $\approx 240$) would overwhelm features with small values like ST depression (`oldpeak` $\approx 1.5$). `StandardScaler` standardizes each feature to have a mean of $0$ and a standard deviation of $1$, ensuring a fair distance calculation.

---

## 📓 5. Project Directory Structure & Notebook Execution

```text
Predicting-Heart-Disease-with-K-Nearest-Neighbors/
│
├── data/
│   ├── heart.csv                  # Cleveland dataset
│   └── data_dictionary.md         # Reference clinical definitions
│
├── models/
│   ├── knn_best.pkl               # Serialized champion model
│   └── scaler.pkl                 # Serialized fitted scaler
│
├── notebooks/                     # Step-by-Step Pipeline (Run in numerical order)
│   ├── 01_Data_Exploration.ipynb
│   ├── 02_Data_Preprocessing.ipynb
│   ├── 03_KNN_Model.ipynb
│   ├── 04_KNN_Hyperparameter_Tuning.ipynb
│   ├── 05_Logistic_Regression.ipynb
│   ├── 06_Random_Forest.ipynb
│   └── 07_Model_Comparison.ipynb
│
├── reports/
│   └── figures/                   # Exported pipeline charts
│
├── src/
│   └── save_models.py             # Script to export models for deployment
│
├── requirements.txt               # Required packages
├── deployment.md                  # Deployment & FastAPI hosting guide
└── README.md                      # Main documentation page
```

### Self-Contained Execution:
* The Jupyter Notebooks are designed to be run **independently**.
* Each notebook re-loads and cleans the raw dataset from `../data/heart.csv`. You do **not** need variables stored in memory from previous notebooks.
* Simply run them in numeric order (`01_` to `07_`) to follow the development pipeline.

---

## 📈 6. Experimental Results and Model Comparison

The classifiers were evaluated using a stratified 80/20 test split (242 training samples, 61 test samples):

| Model | Accuracy | Precision | Recall (Sensitivity) | Specificity | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **KNN (K=5 Baseline)** | 88.52% | 80.00% | **100.00%** | 78.79% | 88.89% |
| **Tuned KNN (Optimal K=9)** | **90.16%** | 82.35% | **100.00%** | **81.82%** | **90.32%** |
| **Logistic Regression** | 86.89% | 81.25% | 92.86% | **81.82%** | 86.67% |
| **Random Forest** | 88.52% | 81.82% | 96.43% | **81.82%** | 88.52% |

### Champion Model Selection:
* **Tuned KNN ($K=9$)** was selected as our deployable champion.
* It achieves a perfect **100.00% Recall** (0 False Negatives), meaning no patient with heart disease was missed.
* It yields the highest overall **Accuracy** (90.16%) and **F1-Score** (90.32%), outperforming the linear and ensemble baselines.

---

## 🚀 7. Quick Start & Production Deployment

* **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
* **Train and serialize the model:**
   ```bash
   python src/save_models.py
   ```
* **Deploy as an API / Containerize:**
   To deploy the model as a production-grade Web Service with FastAPI, Docker, and host it on cloud platforms, see the **[Production Deployment Guide (deployment.md)](file:///d:/Predicting-Heart-Disease-with-K-Nearest-Neighbour/Predicting-Heart-Disease-with-K-Nearest-Neighbors/deployment.md)**.
