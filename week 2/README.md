# SafeX Predictive Lead Scoring Model

Scores marketing/sales leads (university contacts, prospective clients) by
likelihood to convert, so Business Development can prioritize follow-up.

## Project structure

```
lead-scoring/
├── data/
│   ├── generate_data.py     # Generates the simulated lead dataset
│   └── leads.csv            # Simulated dataset (1,200 leads)
├── notebook/
│   ├── lead_scoring.py      # Notebook source (jupytext percent format)
│   └── lead_scoring.ipynb   # Executed notebook: EDA, modeling, evaluation, demo
├── app/
│   └── streamlit_app.py     # Optional: interactive scoring tool for BD reps
├── requirements.txt
└── README.md
```

## Data assumptions

SafeX does not yet have a labeled historical dataset of leads with known
outcomes (converted / not converted). To validate the modeling *pipeline*
now — rather than wait for a real data export — `data/generate_data.py`
creates a **simulated** dataset of 1,200 leads with:

- **Categorical features:** `company_size`, `source` (Referral, University
  Partnership, Cold Outreach, Webinar, Inbound Website, Conference),
  `industry` (Higher Education, Government, Corporate Security, Healthcare,
  NGO/Nonprofit, Other)
- **Numeric features:** `engagement_score` (0-100 composite of opens/visits/
  downloads), `response_time_hours`, `prior_interactions`, `lead_age_months`
- **Binary features:** `budget_confirmed`, `attended_demo`
- **Target:** `converted` (0/1)

The outcome isn't random — it's generated from a documented, plausible
weighting of the features (warmer source + higher engagement + faster
response + confirmed budget/authority → higher conversion probability),
based on common B2B sales-development benchmarks, plus random noise to
mimic real-world unpredictability. Overall simulated conversion rate is
~40%, and ~3% of `engagement_score` values are missing to mimic real
tracking gaps.

**This is simulated data, not SafeX's actual pipeline.** The direction and
relative importance of features (e.g. "response time matters," "referrals
convert better than cold outreach") are reasonable industry priors, but the
exact coefficients should **not** be treated as ground truth about SafeX's
real leads. Once real historical lead + outcome data is available, swap
`data/leads.csv` for the real export (same column names) and re-run the
notebook — no code changes needed.

## Model choice

Two models were trained and compared (see `notebook/lead_scoring.ipynb`,
Section 4-5):

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | ~0.72 | ~0.70 | ~0.54 | ~0.61 | see notebook |
| Decision Tree | ~0.64 | ~0.56 | ~0.52 | ~0.54 | see notebook |

*(exact numbers are in the executed notebook output; they'll shift slightly if
re-run with different data)*

**Logistic Regression was selected as the primary scoring model** because:

1. It outputs a smooth, well-calibrated **probability**, which is what a lead
   score should be. A decision tree only outputs as many distinct
   probabilities as it has leaves — much coarser.
2. Coefficients are directly interpretable ("all else equal, a confirmed
   budget increases the odds of conversion") — important for BD to trust and
   for the team to update over time.
3. It's less prone to overfitting on a dataset this size.

The Decision Tree is kept in the notebook as a comparison model and as a
plain-English sanity check (its top splits read like a checklist: e.g. "if
response time is under ~12 hours and budget is confirmed → high
likelihood").

## How to interpret a lead score

`score_lead()` (in the notebook) and the Streamlit app both return:

- **`probability`** — the model's raw estimated likelihood of conversion (0-1)
- **`score`** — the same thing as a 0-100 number, for a friendlier display
- **`priority`** — a bucketed tier for quick triage:
  - **High** — probability ≥ 0.65 → call first
  - **Medium** — 0.35 ≤ probability < 0.65 → follow up this week
  - **Low** — probability < 0.35 → low-touch / nurture campaign

These thresholds are a starting point, set in `score_lead()` — they should be
tuned once real conversion rates and BD team capacity are known (e.g. if the
team can only call the top 20% of leads each week, set the "High" threshold
at the 80th percentile of scores instead of a fixed probability).

## Top conversion drivers (from this simulated run)

See Section 6 of the notebook for the full ranked list and chart. In this
run, the strongest drivers were lead **source** (Referral/University
Partnership vs. Cold Outreach), **engagement score**, and **budget
confirmation** — all consistent with the assumptions baked into the
simulation and with common B2B sales-development findings. These should be
re-validated against SafeX's real conversion data once available.

## Running it yourself

```bash
pip install -r requirements.txt

# Regenerate the simulated dataset (optional, already included in data/leads.csv)
python data/generate_data.py

# Open and run the notebook
jupyter notebook notebook/lead_scoring.ipynb

# Or launch the interactive scoring tool
cd app
streamlit run streamlit_app.py
```

## Next steps for a real pilot

1. Export SafeX's actual historical lead data (same or mapped column names)
   with a true `converted` outcome column.
2. Replace `data/leads.csv` with that export and re-run the notebook.
3. Re-validate feature importance and re-tune priority thresholds against
   real BD team capacity.
4. If accuracy on real data is unsatisfactory, consider adding more
   features (e.g. CRM activity logs, email content signals) or trying
   additional models (Random Forest, Gradient Boosting).
