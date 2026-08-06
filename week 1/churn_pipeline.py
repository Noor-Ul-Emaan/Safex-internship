"""
Churn Prediction Mini-Project
=============================
Dataset: Telco Customer Churn (public dataset, same schema as Kaggle's
"Telco Customer Churn" dataset by IBM/Kaggle).

NOTE: This script auto-detects a real downloaded CSV first. If none is
found, it builds a realistic synthetic dataset with the EXACT same
columns/structure as the real Kaggle dataset, so you can develop and
test everything now, then just drop in the real CSV later (same file
name) to get results on real data -- no code changes needed.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

RANDOM_STATE = 42
DATA_PATH = "WA_Fn-UseC_-Telco-Customer-Churn.csv"  # standard Kaggle file name

# ---------------------------------------------------------------------
# STEP 1: LOAD DATA
# ---------------------------------------------------------------------
def load_data(path=DATA_PATH, n_synthetic=2000):
    if os.path.exists(path):
        print(f"Loading REAL dataset from {path}")
        df = pd.read_csv(path)
        return df, False

    print("Real Kaggle CSV not found -> generating synthetic dataset "
          "with identical schema for development/testing.")
    rng = np.random.default_rng(RANDOM_STATE)

    genders = rng.choice(["Male", "Female"], n_synthetic)
    senior = rng.choice([0, 1], n_synthetic, p=[0.84, 0.16])
    partner = rng.choice(["Yes", "No"], n_synthetic)
    dependents = rng.choice(["Yes", "No"], n_synthetic, p=[0.3, 0.7])
    tenure = rng.integers(0, 73, n_synthetic)
    phone_service = rng.choice(["Yes", "No"], n_synthetic, p=[0.9, 0.1])
    multiple_lines = rng.choice(["Yes", "No", "No phone service"], n_synthetic)
    internet_service = rng.choice(["DSL", "Fiber optic", "No"], n_synthetic, p=[0.35, 0.44, 0.21])
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n_synthetic, p=[0.55, 0.21, 0.24])
    paperless = rng.choice(["Yes", "No"], n_synthetic, p=[0.59, 0.41])
    payment_method = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        n_synthetic
    )
    monthly_charges = np.round(rng.uniform(18, 120, n_synthetic), 2)
    total_charges = np.round(monthly_charges * tenure + rng.normal(0, 50, n_synthetic), 2)
    total_charges = np.clip(total_charges, 0, None)

    online_security = rng.choice(["Yes", "No", "No internet service"], n_synthetic)
    tech_support = rng.choice(["Yes", "No", "No internet service"], n_synthetic)
    streaming_tv = rng.choice(["Yes", "No", "No internet service"], n_synthetic)
    streaming_movies = rng.choice(["Yes", "No", "No internet service"], n_synthetic)
    online_backup = rng.choice(["Yes", "No", "No internet service"], n_synthetic)
    device_protection = rng.choice(["Yes", "No", "No internet service"], n_synthetic)

    # churn probability driven mainly by contract type + tenure + monthly charges
    # (mirrors real-world patterns: month-to-month & short tenure & high charges -> higher churn)
    churn_score = (
        (contract == "Month-to-month") * 0.35
        + (tenure < 12) * 0.25
        + (monthly_charges > 80) * 0.15
        + rng.normal(0, 0.15, n_synthetic)
    )
    churn_prob = 1 / (1 + np.exp(-(churn_score - 0.3) * 4))
    churn = (rng.uniform(0, 1, n_synthetic) < churn_prob)
    churn = np.where(churn, "Yes", "No")

    df = pd.DataFrame({
        "customerID": [f"CUST-{i:05d}" for i in range(n_synthetic)],
        "gender": genders,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "Churn": churn,
    })
    return df, True


# ---------------------------------------------------------------------
# STEP 2: CLEAN DATA
# ---------------------------------------------------------------------
def clean_data(df):
    df = df.copy()

    # TotalCharges sometimes has blank strings in the real dataset -> convert & fill
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    # Drop ID column (not predictive)
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # Drop exact duplicate rows
    df = df.drop_duplicates()

    return df


# ---------------------------------------------------------------------
# STEP 3: FEATURE ENGINEERING (2-3 new features)
# ---------------------------------------------------------------------
def engineer_features(df):
    df = df.copy()

    # Feature 1: tenure buckets (new customers churn more)
    df["tenure_group"] = pd.cut(
        df["tenure"], bins=[-1, 12, 24, 48, 72],
        labels=["0-1yr", "1-2yr", "2-4yr", "4-6yr"]
    )

    # Feature 2: average monthly spend so far (total / tenure), avoids div-by-zero
    df["avg_monthly_spend"] = df["TotalCharges"] / df["tenure"].replace(0, 1)

    # Feature 3: count of add-on services subscribed (engagement signal)
    addon_cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                  "TechSupport", "StreamingTV", "StreamingMovies"]
    df["num_addon_services"] = (df[addon_cols] == "Yes").sum(axis=1)

    return df


# ---------------------------------------------------------------------
# STEP 4: ENCODE + TRAIN
# ---------------------------------------------------------------------
def prepare_model_data(df):
    df = df.copy()
    target = df["Churn"].map({"Yes": 1, "No": 0})
    features = df.drop(columns=["Churn"])

    # Label-encode all categorical columns
    encoders = {}
    for col in features.columns:
        if str(features[col].dtype) in ("object", "category", "str", "string"):
            le = LabelEncoder()
            features[col] = le.fit_transform(features[col].astype(str))
            encoders[col] = le

    return features, target, encoders


def train_and_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    results = {}

    # Logistic Regression (scaled features help convergence + fair coefficient comparison)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    log_reg = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
    log_reg.fit(X_train_scaled, y_train)
    log_preds = log_reg.predict(X_test_scaled)
    log_acc = accuracy_score(y_test, log_preds)
    results["Logistic Regression"] = {
        "model": log_reg,
        "accuracy": log_acc,
        "report": classification_report(y_test, log_preds),
    }

    # Decision Tree
    tree = DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE)
    tree.fit(X_train, y_train)
    tree_preds = tree.predict(X_test)
    tree_acc = accuracy_score(y_test, tree_preds)
    results["Decision Tree"] = {
        "model": tree,
        "accuracy": tree_acc,
        "report": classification_report(y_test, tree_preds),
    }

    return results, X_train.columns


def top_features(model, feature_names, model_type, n=3):
    if model_type == "Logistic Regression":
        importance = np.abs(model.coef_[0])
    else:
        importance = model.feature_importances_

    idx = np.argsort(importance)[::-1][:n]
    return [(feature_names[i], round(importance[i], 4)) for i in idx]


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
if __name__ == "__main__":
    df_raw, is_synthetic = load_data()
    print(f"\nDataset shape: {df_raw.shape}")
    print(f"Data source: {'SYNTHETIC (replace CSV for real results)' if is_synthetic else 'REAL Kaggle CSV'}")

    df_clean = clean_data(df_raw)
    df_features = engineer_features(df_clean)
    X, y, encoders = prepare_model_data(df_features)

    results, feature_names = train_and_evaluate(X, y)

    print("\n" + "=" * 50)
    print("MODEL RESULTS")
    print("=" * 50)
    for name, res in results.items():
        print(f"\n--- {name} ---")
        print(f"Accuracy: {res['accuracy']:.4f}")
        top3 = top_features(res["model"], list(feature_names), name)
        print(f"Top 3 predictive features: {top3}")

    # Save a small results summary to file
    with open("model_results.txt", "w") as f:
        f.write("CHURN PREDICTION - MODEL RESULTS\n")
        f.write("=" * 50 + "\n")
        f.write(f"Data source: {'Synthetic (dev/test)' if is_synthetic else 'Real Kaggle dataset'}\n")
        f.write(f"Dataset shape: {df_raw.shape}\n\n")
        for name, res in results.items():
            f.write(f"\n--- {name} ---\n")
            f.write(f"Accuracy: {res['accuracy']:.4f}\n")
            top3 = top_features(res["model"], list(feature_names), name)
            f.write(f"Top 3 predictive features: {top3}\n")
            f.write(res["report"] + "\n")

    print("\nSaved detailed results to model_results.txt")
