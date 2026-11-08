# Customer Salary Intelligence Pro
## Advanced Artificial Neural Network Regression Project

### 1. Project Overview

This project converts a customer churn classification workflow
into a regression application.

The objective is to predict EstimatedSalary as a continuous
numerical value using Artificial Neural Networks.

This is an academic learning exercise, not a production
salary-estimation system.

### 2. Dataset

Dataset: Churn_Modelling.csv

Target:
- EstimatedSalary

Input features:
- CreditScore
- Age
- Tenure
- Balance
- NumOfProducts
- HasCrCard
- IsActiveMember
- Geography
- Gender

Excluded:
- RowNumber
- CustomerId
- Surname
- Exited

### 3. Preprocessing

The dataset is split into training and testing sets.

Numeric features:
- Median imputation
- StandardScaler

Categorical features:
- Most-frequent imputation
- One-hot encoding

Preprocessing is fitted only on training data.

The fitted preprocessing pipeline and feature order
are saved and reused by the Streamlit application.

### 4. ANN Architectures

Three ANN architectures are compared:

Balanced ANN:
128 → 64 → 32

Deep ANN:
256 → 128 → 64 → 32 → 16

Wide ANN:
256 → 256 → 128

All models use:
- ReLU hidden-layer activation
- Dropout regularization
- Adam optimizer
- Mean Squared Error loss
- One linear output neuron

### 5. Evaluation

The models are evaluated using:

MAE:
Average absolute prediction error.

RMSE:
Penalizes larger prediction errors more heavily.

R²:
Measures performance relative to a mean-prediction baseline.

A median DummyRegressor baseline is also evaluated.

### 6. Application Features

- Executive Dashboard
- Individual Salary Prediction
- Customer Scenario Comparison
- Bulk CSV Prediction
- Dataset Explorer
- ANN Model Laboratory
- Research Notes

### 7. Installation

Install Python 3.11.

Create and activate a virtual environment:

    py -3.11 -m venv .venv
    .\.venv\Scripts\Activate.ps1

Install dependencies:

    python -m pip install --upgrade pip
    pip install -r requirements.txt

### 8. Dataset Placement

Place the dataset here:

    data/Churn_Modelling.csv

### 9. Train Models

Run:

    python train_model.py

The script generates the model and preprocessing
artifacts inside the artifacts folder.

### 10. Run Streamlit

Run:

    python -m streamlit run app.py

### 11. Classification vs Regression

Classification predicts categories such as churn or
non-churn. It commonly uses a sigmoid output,
binary cross-entropy and classification metrics.

Regression predicts continuous numerical values.
This project predicts EstimatedSalary, so it uses
one linear output neuron, MSE loss and MAE, RMSE
and R² evaluation metrics.

### 12. Limitations

EstimatedSalary in this classroom dataset is not
verified income. The dataset may have weak predictive
relationships and may not represent real customers.

The model must not be used for banking, lending,
hiring or other consequential decisions.

### 13. Submission Checklist

- GitHub repository URL
- Live Streamlit application URL
- Notebook runs end-to-end
- Complete README
- Application screenshot
- Model artifacts
- Preprocessing artifacts
- Classification vs regression explanation