"""
generate_data.py
-----------------
Creates a SIMULATED dataset of marketing/sales leads for SafeX.

Why simulated data?
SafeX does not yet have a labeled historical dataset of past leads with
known outcomes (converted / not converted). To build and validate a
scoring approach now, we simulate a dataset whose features have a
plausible, documented relationship to conversion, based on common
B2B sales-development assumptions (faster response time, higher
engagement, and warmer lead sources convert better). This lets the
team validate the *modeling pipeline* end-to-end today, and swap in
real historical data later with zero code changes (just replace this
CSV / point the notebook at the real export).

Run:
    python generate_data.py
Produces:
    leads.csv
"""

import numpy as np
import pandas as pd

RANDOM_SEED = 42
N_LEADS = 1200

rng = np.random.default_rng(RANDOM_SEED)

# ---------------------------------------------------------------------
# Feature definitions
# ---------------------------------------------------------------------

sources = ["Referral", "University Partnership", "Cold Outreach", "Webinar", "Inbound Website", "Conference"]
source_weights = [0.15, 0.15, 0.20, 0.15, 0.20, 0.15]

industries = ["Higher Education", "Government", "Corporate Security", "Healthcare", "NGO/Nonprofit", "Other"]
industry_weights = [0.25, 0.15, 0.20, 0.15, 0.15, 0.10]

df = pd.DataFrame({
    "lead_id": range(1, N_LEADS + 1),
    "company_size": rng.choice(
        ["1-50", "51-200", "201-1000", "1000+"], size=N_LEADS, p=[0.30, 0.30, 0.25, 0.15]
    ),
    "source": rng.choice(sources, size=N_LEADS, p=source_weights),
    "industry": rng.choice(industries, size=N_LEADS, p=industry_weights),
    # engagement_score: 0-100, composite of email opens, site visits, content downloads
    "engagement_score": np.clip(rng.normal(50, 20, N_LEADS), 0, 100).round(1),
    # response_time_hours: how long it took the lead to reply to first outreach
    "response_time_hours": np.round(np.clip(rng.exponential(24, N_LEADS), 0.5, 240), 1),
    # number of prior touchpoints/interactions (calls, emails, meetings) before scoring
    "prior_interactions": rng.poisson(3, N_LEADS),
    # whether a budget/authority signal was captured during discovery (0/1)
    "budget_confirmed": rng.choice([0, 1], size=N_LEADS, p=[0.65, 0.35]),
    # whether the lead attended a demo or product walkthrough
    "attended_demo": rng.choice([0, 1], size=N_LEADS, p=[0.6, 0.4]),
    # months since first contact
    "lead_age_months": np.round(np.clip(rng.exponential(2, N_LEADS), 0.1, 18), 1),
})

# ---------------------------------------------------------------------
# Build a latent "propensity to convert" score from a realistic,
# documented weighting of the features above, then sample a binary
# outcome from it. This creates real (not random) signal for the
# model to discover -- mirroring patterns commonly reported in B2B
# sales-development benchmarks (fast response + high engagement +
# budget/authority confirmed + warm source => higher conversion).
# ---------------------------------------------------------------------

size_map = {"1-50": 0.0, "51-200": 0.3, "201-1000": 0.6, "1000+": 0.5}
source_map = {
    "Referral": 1.1,
    "University Partnership": 1.0,
    "Webinar": 0.5,
    "Inbound Website": 0.4,
    "Conference": 0.3,
    "Cold Outreach": -0.6,
}
industry_map = {
    "Higher Education": 0.4,
    "Government": 0.2,
    "Corporate Security": 0.5,
    "Healthcare": 0.1,
    "NGO/Nonprofit": 0.0,
    "Other": -0.1,
}

logit = (
    -3.7
    + df["company_size"].map(size_map)
    + df["source"].map(source_map)
    + df["industry"].map(industry_map)
    + 0.035 * df["engagement_score"]
    - 0.02 * df["response_time_hours"]
    + 0.18 * df["prior_interactions"]
    + 0.9 * df["budget_confirmed"]
    + 0.7 * df["attended_demo"]
    - 0.05 * df["lead_age_months"]
    + rng.normal(0, 0.9, N_LEADS)  # noise: real-world unpredictability
)

prob = 1 / (1 + np.exp(-logit))
df["converted"] = (rng.uniform(0, 1, N_LEADS) < prob).astype(int)

# Introduce a small amount of realistic missingness in engagement_score
missing_idx = rng.choice(df.index, size=int(0.03 * N_LEADS), replace=False)
df.loc[missing_idx, "engagement_score"] = np.nan

out_path = "leads.csv"
df.to_csv(out_path, index=False)
print(f"Saved {len(df)} leads to {out_path}")
print(f"Conversion rate: {df['converted'].mean():.1%}")
