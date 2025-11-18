# Customer Churn Prediction

This repository contains an end-to-end machine learning project for predicting customer churn using the Telco Customer Churn dataset. It covers data preprocessing, model training, threshold tuning, explainability with SHAP, and a Streamlit web application for interactive predictions.

---

## 1. Project Overview

Customer churn is a key business problem: retaining existing customers is usually cheaper than acquiring new ones. The goal of this project is to:

- Predict whether a customer is likely to churn.
- Prioritize catching as many churners as possible (high recall).
- Explain which factors drive churn, using model interpretability tools.
- Provide an easy-to-use web interface for scoring new customers.

The project is implemented in Python using:

- pandas, numpy
- scikit-learn
- xgboost
- shap
- streamlit
- joblib

---

## 2. Dataset

- Source: Telco Customer Churn (Kaggle)
- Rows: 7,043
- Columns: 21
- Target column: `Churn` (Yes/No), converted to numeric `churn` (1 = churn, 0 = no churn)

Examples of features used:

- Numerical:
  - `tenure`
  - `MonthlyCharges`
  - `TotalCharges`
  - `SeniorCitizen`
- Categorical:
  - `gender`
  - `Partner`
  - `Dependents`
  - `PhoneService`
  - `MultipleLines`
  - `InternetService`
  - `OnlineSecurity`
  - `OnlineBackup`
  - `DeviceProtection`
  - `TechSupport`
  - `StreamingTV`
  - `StreamingMovies`
  - `Contract`
  - `PaperlessBilling`
  - `PaymentMethod`

The `customerID` column is dropped, as it is an identifier and not useful for prediction.

---

## 3. Methodology

### 3.1 Problem Type

- Binary classification:
  - 1 = customer churned
  - 0 = customer did not churn
- The dataset is imbalanced: churners are a minority.

### 3.2 Train/Test Split

- The data is split into:
  - Training set
  - Test set
- `train_test_split` with `stratify=y` is used to preserve class balance across splits.

---

## 4. Preprocessing and Pipeline

A scikit-learn `Pipeline` is used, with a `ColumnTransformer` to handle different data types.

### 4.1 Preprocessing

- Numeric columns:
  - Imputation: `SimpleImputer(strategy="median")`
  - Scaling: `StandardScaler(with_mean=False)`
- Categorical columns:
  - Imputation: `SimpleImputer(strategy="most_frequent")`
  - Encoding: `OneHotEncoder(handle_unknown="ignore")`

This logic is encapsulated in a `ColumnTransformer`, which ensures:

- Consistent preprocessing for both training and inference.
- Robust handling of missing values and unseen categories.

### 4.2 Models

1. **Baseline Model: Logistic Regression**
   - `LogisticRegression(max_iter=2000, class_weight="balanced")`
   - Used to establish a strong, interpretable baseline.

2. **Final Model: XGBoost Classifier**
   - `XGBClassifier` with typical settings such as:
     - `n_estimators = 400`
     - `max_depth = 5`
     - `learning_rate = 0.05`
     - `subsample = 0.9`
     - `colsample_bytree = 0.9`
     - `eval_metric = "logloss"`
     - `tree_method = "hist"`
   - Wrapped inside the same preprocessing pipeline:
     - `Pipeline([("prep", preprocess), ("clf", xgb_classifier)])`

---

## 5. Evaluation and Threshold Tuning

### 5.1 Evaluation Metrics

Because the dataset is imbalanced, evaluation emphasizes:

- ROC-AUC (Area Under ROC Curve)
- PR-AUC (Area Under Precision-Recall Curve)
- Precision, Recall, F1-score (especially for the positive class = churners)

Example test-set performance (approximate):

- ROC-AUC: 0.836
- PR-AUC: 0.651

### 5.2 Threshold Tuning (F2-Score)

The default classification threshold of 0.5 is not ideal for churn prediction. The goal is to catch as many churners as possible, even at the cost of more false positives.

To achieve this:

- A range of thresholds between 0.05 and 0.95 is evaluated.
- For each threshold, predictions are computed and the F2-score is calculated using `fbeta_score(beta=2.0)`.
- The best threshold is chosen based on the highest F2-score.

Example result:

- Chosen threshold: approximately 0.15
- At this threshold (on the test set), the model achieves roughly:
  - High recall for churners (around 0.87)
  - Moderate precision for churners
  - Very few false negatives (missed churners)

This configuration is suitable when the business cost of missing a churner is high.

---

## 6. Model Explainability with SHAP

SHAP (SHapley Additive exPlanations) is used to interpret the XGBoost model.

Steps:

1. The fitted preprocessing transformer transforms the data.
2. A `TreeExplainer` is created for the trained XGBoost model.
3. SHAP values are computed for the training data.
4. Global explanations are visualized via:
   - `shap.summary_plot(...)` (beeswarm)
   - `shap.summary_plot(..., plot_type="bar")` (bar chart)

### 6.1 Key Findings (Typical for This Dataset)

Top features contributing to churn:

- `Contract_Month-to-month`:
  - Month-to-month contracts strongly increase churn risk.
- `tenure`:
  - Short tenure (new customers) is associated with higher churn.
- `MonthlyCharges`:
  - Higher monthly charges are associated with higher churn.
- `TotalCharges`:
  - Low total charges correlate with new customers and thus higher churn.
- `OnlineSecurity_No`, `TechSupport_No`:
  - Customers without these services are more likely to churn.
- `PaymentMethod_Electronic check`:
  - Often correlated with higher churn.
- `InternetService_Fiber optic`:
  - Certain fiber customers exhibit higher churn compared to DSL or no internet.

These insights can be used for targeted retention strategies (e.g., offering long-term contracts, discounts, or additional support services to high-risk segments).

---

## 7. Streamlit Application

A Streamlit app is provided in `churn_app/app.py` to interactively score customers.

### 7.1 Features

- Form for entering customer attributes (matching original dataset columns except `customerID` and `Churn`).
- Model predictions:
  - Churn probability (between 0 and 1).
  - Churn classification using the tuned threshold (e.g., 0.15).
- Clear indication of whether a customer is likely to churn.

### 7.2 Running the App

From the project root:

```bash
cd churn_app
streamlit run app.py
```
Make sure the virtual environment is activated and dependencies are installed.

## 8. Saved Artifacts

The trained model and associated metadata are stored in the models/ directory:

xgb_churn_pipeline.pkl — the complete pipeline (preprocessing + XGBoost model).

metadata.json — contains:

Timestamp of model creation

Tuned threshold used for classification

Evaluation metrics (ROC-AUC, PR-AUC)

Feature names after encoding

These artifacts allow for efficient deployment and reuse without retraining the model.

## 9. Project Structure
```
Customer_Churn_Prediction/
│
├── data/
│   └── telco_churn.csv
│
├── notebooks/
│   └── churn_prediction.ipynb
│
├── models/
│   ├── xgb_churn_pipeline.pkl
│   └── metadata.json
│
├── churn_app/
│   └── app.py
│
└── README.md
```

## 10. Environment and Installation

To install the required packages, use one of the following methods:

Using a requirements file:

```pip install -r requirements.txt
```
Or install manually:

```pip install pandas numpy scikit-learn xgboost shap streamlit joblib
```

## 11. Using the Model Programmatically

Example Python code for loading and using the model:

```import joblib
import json
import pandas as pd
from pathlib import Path

# Load model and metadata
model = joblib.load("models/xgb_churn_pipeline.pkl")
meta = json.loads(Path("models/metadata.json").read_text())
threshold = meta["threshold"]

# Example input data
data = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 5,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 85.0,
    "TotalCharges": 400.0
}

df = pd.DataFrame([data])

# Predict churn probability
proba = model.predict_proba(df)[0, 1]

# Classification using tuned threshold
will_churn = proba >= threshold

print("Churn probability:", proba)
print("Will churn:", bool(will_churn))
```
## 12. Possible Extensions

-> Potential enhancements to this project include:

-> Hyperparameter tuning using cross-validation

-> Benchmarking additional models (Random Forest, LightGBM, CatBoost)

-> Cost-sensitive evaluation for business scenarios

-> Deployment to the cloud (Streamlit Cloud, Render, AWS)

-> Monitoring real-time predictions

-> Creating automated retraining pipelines