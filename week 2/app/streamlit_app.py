"""
streamlit_app.py
-----------------
A small internal tool for SafeX BD reps: enter a lead's details and get an
instant conversion-likelihood score + priority tier.

Run locally with:
    pip install -r ../requirements.txt
    streamlit run streamlit_app.py

On first run this trains the Logistic Regression model on data/leads.csv
(same pipeline as the notebook) and caches it.
"""

import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

DATA_PATH = "../data/leads.csv"
RANDOM_SEED = 42

FEATURE_COLS = [
    "company_size", "source", "industry", "engagement_score",
    "response_time_hours", "prior_interactions", "budget_confirmed",
    "attended_demo", "lead_age_months",
]
CATEGORICAL = ["company_size", "source", "industry"]
NUMERIC = [
    "engagement_score", "response_time_hours", "prior_interactions",
    "budget_confirmed", "attended_demo", "lead_age_months",
]


@st.cache_resource
def train_model():
    df = pd.read_csv(DATA_PATH)
    X, y = df[FEATURE_COLS], df["converted"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_SEED, stratify=y
    )

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, NUMERIC),
        ("cat", categorical_transformer, CATEGORICAL),
    ])
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)),
    ])
    pipe.fit(X_train, y_train)
    return pipe, df


def score_lead(lead: dict, pipe) -> dict:
    lead_df = pd.DataFrame([lead])[FEATURE_COLS]
    proba = pipe.predict_proba(lead_df)[0, 1]
    score = round(proba * 100, 1)
    if proba >= 0.65:
        priority = "High"
    elif proba >= 0.35:
        priority = "Medium"
    else:
        priority = "Low"
    return {"score": score, "priority": priority, "probability": round(float(proba), 3)}


st.set_page_config(page_title="SafeX Lead Scoring", page_icon="🎯", layout="centered")
st.title("🎯 SafeX Lead Scoring Tool")
st.caption(
    "Enter a lead's details to get an instant conversion-likelihood score. "
    "Model is trained on simulated data — retrain on real historical leads before "
    "using in production (see README)."
)

pipe, df = train_model()

with st.form("lead_form"):
    col1, col2 = st.columns(2)
    with col1:
        company_size = st.selectbox("Company / Org Size", ["1-50", "51-200", "201-1000", "1000+"])
        source = st.selectbox(
            "Lead Source",
            ["Referral", "University Partnership", "Cold Outreach", "Webinar", "Inbound Website", "Conference"],
        )
        industry = st.selectbox(
            "Industry",
            ["Higher Education", "Government", "Corporate Security", "Healthcare", "NGO/Nonprofit", "Other"],
        )
        engagement_score = st.slider("Engagement Score (0-100)", 0, 100, 50)
    with col2:
        response_time_hours = st.number_input("Response Time (hours)", min_value=0.0, value=12.0, step=1.0)
        prior_interactions = st.number_input("Prior Interactions", min_value=0, value=2, step=1)
        budget_confirmed = st.checkbox("Budget / Authority Confirmed?")
        attended_demo = st.checkbox("Attended Demo?")
        lead_age_months = st.number_input("Lead Age (months)", min_value=0.0, value=1.0, step=0.5)

    submitted = st.form_submit_button("Score Lead")

if submitted:
    lead = {
        "company_size": company_size,
        "source": source,
        "industry": industry,
        "engagement_score": engagement_score,
        "response_time_hours": response_time_hours,
        "prior_interactions": prior_interactions,
        "budget_confirmed": int(budget_confirmed),
        "attended_demo": int(attended_demo),
        "lead_age_months": lead_age_months,
    }
    result = score_lead(lead, pipe)

    st.subheader("Result")
    priority_color = {"High": "green", "Medium": "orange", "Low": "red"}[result["priority"]]
    st.metric("Lead Score", f"{result['score']} / 100")
    st.markdown(f"**Priority:** :{priority_color}[{result['priority']}]")
    st.progress(result["probability"])
    st.caption(f"Estimated conversion probability: {result['probability']:.1%}")
