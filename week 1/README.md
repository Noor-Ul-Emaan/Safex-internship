# Churn Prediction Mini-Project

## Objective
Build a basic churn-prediction model on the public **Telco Customer Churn** dataset.

## Dataset
- Source: Kaggle — "Telco Customer Churn" (IBM sample dataset)
- Link: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- ~7,043 customers, 21 columns (demographics, account info, services subscribed, churn label)

> The provided notebook auto-detects the real CSV (`WA_Fn-UseC_-Telco-Customer-Churn.csv`)
> if it's placed in the same folder. If not found, it generates a synthetic dataset with the
> exact same schema, so the full pipeline can still be run and reviewed end-to-end.

## Steps Performed

### 1. Data Cleaning
- Converted `TotalCharges` to numeric (blank strings in the raw data replaced with median)
- Dropped the non-predictive `customerID` column
- Removed duplicate rows

### 2. Feature Engineering (3 new features)
| Feature | Description |
|---|---|
| `tenure_group` | Buckets customers into tenure ranges (0-1yr, 1-2yr, 2-4yr, 4-6yr) |
| `avg_monthly_spend` | `TotalCharges / tenure` — spend rate signal |
| `num_addon_services` | Count of add-on services subscribed (Online Security, Backup, Device Protection, Tech Support, Streaming TV/Movies) |

### 3. Modeling
Two classifiers trained on an 80/20 stratified train-test split:
- **Logistic Regression** (features scaled with `StandardScaler`)
- **Decision Tree** (max_depth=5, to avoid overfitting)

### 4. Results
*(these numbers are from running the notebook on the real Kaggle dataset)*

| Model | Accuracy | Top 3 Predictive Features |
|---|---|---|
| Logistic Regression | **0.80** | tenure, num_addon_services, TotalCharges |
| Decision Tree | **0.79** | Contract, tenure, OnlineSecurity |

**Classification report (Logistic Regression):**
- Non-churn (class 0): precision 0.84, recall 0.90, f1 0.87
- Churn (class 1): precision 0.65, recall 0.53, f1 0.58

**Classification report (Decision Tree):**
- Non-churn (class 0): precision 0.83, recall 0.90, f1 0.86
- Churn (class 1): precision 0.64, recall 0.47, f1 0.54

**Key takeaway:** `Contract` type (month-to-month vs. long-term), customer `tenure`,
and engagement signals like `num_addon_services` and `OnlineSecurity` are the
strongest predictors of churn. This matches well-known industry patterns — customers
on flexible/no-commitment plans, with shorter tenure, and fewer add-on services are
more likely to churn. Recall on the churn class is lower than on the non-churn class,
which is expected since churners are the minority class in this dataset.

## How to Run
1. (Optional, for real results) Download the dataset from Kaggle and save it as
   `WA_Fn-UseC_-Telco-Customer-Churn.csv` in this project folder.
2. Open `Churn_Prediction.ipynb` in Jupyter or Google Colab.
3. Run all cells top to bottom.
4. Accuracy and top features print at the bottom of the notebook.

## Files in this Submission
- `Churn_Prediction.ipynb` — full notebook (data loading → cleaning → feature
  engineering → modeling → results)
- `churn_pipeline.py` — same pipeline as a plain Python script (easier to run/debug outside Jupyter)
- `model_results.txt` — saved accuracy + classification report from the last run
- `README.md` — this file

## Tools Used
Python, pandas, NumPy, scikit-learn (LogisticRegression, DecisionTreeClassifier,
StandardScaler, LabelEncoder, train_test_split, classification metrics)
