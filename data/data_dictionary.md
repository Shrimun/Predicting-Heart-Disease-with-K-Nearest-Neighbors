# Heart Disease Dataset - Data Dictionary

## Dataset Information

This project uses the **Processed Cleveland Heart Disease Dataset** from the UCI Machine Learning Repository.

Each row represents one patient, while each column represents a clinical measurement or medical observation.

| Feature | Description | Values |
|----------|-------------|--------|
| age | Age of the patient | Years |
| sex | Biological sex | 1 = Male, 0 = Female |
| cp | Chest pain type | 0–3 |
| trestbps | Resting blood pressure | mm Hg |
| chol | Serum cholesterol | mg/dl |
| fbs | Fasting blood sugar > 120 mg/dl | 1 = True, 0 = False |
| restecg | Resting electrocardiographic results | 0–2 |
| thalach | Maximum heart rate achieved | bpm |
| exang | Exercise-induced angina | 1 = Yes, 0 = No |
| oldpeak | ST depression induced by exercise | Numeric |
| slope | Slope of peak exercise ST segment | 0–2 |
| ca | Number of major vessels colored by fluoroscopy | 0–3 |
| thal | Thalassemia | 3 = Normal, 6 = Fixed Defect, 7 = Reversible Defect |
| target | Heart disease diagnosis | 0 = No Disease, 1–4 = Disease Present |

## Notes

- Missing values are represented using `?`.
- During preprocessing, missing values will be handled before model training.
- The target variable will later be converted into a binary classification:
  - **0 → No Heart Disease**
  - **1 → Heart Disease**
