"""
kpi_analysis.py
----------------
This is the "pandas" layer of the project: it takes the raw actuals/budget/
order tables and turns them into the exact shapes the dashboard needs.

Run this after generate_data.py (or after swapping in real data with the
same column structure). It writes dashboard_data.js, which the dashboard
reads directly -- no server or database required, so the whole thing runs
by opening dashboard.html in a browser.
"""

import json
import pandas as pd

actuals = pd.read_csv("../data/monthly_actuals.csv")
budget = pd.read_csv("../data/monthly_budget.csv")
orders = pd.read_csv("../data/order_line_items.csv")

MONTH_ORDER = sorted(actuals["month"].unique())


def month_label(m):
    return pd.to_datetime(m + "-01").strftime("%b %Y")


# ---------------------------------------------------------------------------
# KPI summary cards (whole-year + latest-month figures)
# ---------------------------------------------------------------------------
rev = actuals[actuals["type"] == "revenue"]
exp = actuals[actuals["type"] == "expense"]

monthly_rev = rev.groupby("month")["amount"].sum().reindex(MONTH_ORDER)
monthly_exp = exp.groupby("month")["amount"].sum().reindex(MONTH_ORDER)
monthly_profit = monthly_rev - monthly_exp
monthly_margin = (monthly_profit / monthly_rev * 100).round(1)

latest = MONTH_ORDER[-1]
prior = MONTH_ORDER[-2]

def pct_change(curr, prev):
    if prev == 0:
        return 0.0
    return round((curr - prev) / prev * 100, 1)

kpi_cards = [
    {
        "label": "Total Revenue (YTD)",
        "value": round(monthly_rev.sum(), 2),
        "format": "currency",
        "delta_label": f"{month_label(latest)} vs {month_label(prior)}",
        "delta_pct": pct_change(monthly_rev[latest], monthly_rev[prior]),
    },
    {
        "label": "Total Expenses (YTD)",
        "value": round(monthly_exp.sum(), 2),
        "format": "currency",
        "delta_label": f"{month_label(latest)} vs {month_label(prior)}",
        "delta_pct": pct_change(monthly_exp[latest], monthly_exp[prior]),
    },
    {
        "label": "Net Profit (YTD)",
        "value": round(monthly_profit.sum(), 2),
        "format": "currency",
        "delta_label": f"{month_label(latest)} vs {month_label(prior)}",
        "delta_pct": pct_change(monthly_profit[latest], monthly_profit[prior]),
    },
    {
        "label": "Net Margin (latest month)",
        "value": float(monthly_margin[latest]),
        "format": "percent",
        "delta_label": f"vs {month_label(prior)}",
        "delta_pct": round(float(monthly_margin[latest] - monthly_margin[prior]), 1),
    },
]

# ---------------------------------------------------------------------------
# Monthly trend chart data
# ---------------------------------------------------------------------------
trend = {
    "labels": [month_label(m) for m in MONTH_ORDER],
    "revenue": [round(v, 2) for v in monthly_rev.tolist()],
    "expense": [round(v, 2) for v in monthly_exp.tolist()],
    "profit": [round(v, 2) for v in monthly_profit.tolist()],
}

# ---------------------------------------------------------------------------
# Revenue by product line (for the revenue breakdown visual + drill-down)
# ---------------------------------------------------------------------------
rev_by_line = (
    rev.groupby(["month", "category"])["amount"].sum().unstack(fill_value=0).reindex(MONTH_ORDER)
)
product_lines = list(rev_by_line.columns)
revenue_breakdown = {
    "labels": [month_label(m) for m in MONTH_ORDER],
    "series": [
        {"name": line, "data": [round(v, 2) for v in rev_by_line[line].tolist()]}
        for line in product_lines
    ],
}

# ---------------------------------------------------------------------------
# Expense breakdown by category (latest month, for a composition view)
# ---------------------------------------------------------------------------
latest_exp = exp[exp["month"] == latest].groupby("category")["amount"].sum().sort_values(ascending=False)
expense_breakdown = {
    "month": month_label(latest),
    "labels": latest_exp.index.tolist(),
    "values": [round(v, 2) for v in latest_exp.tolist()],
}

# ---------------------------------------------------------------------------
# Variance to budget (advanced feature #1)
# ---------------------------------------------------------------------------
merged = actuals.merge(
    budget, on=["month", "type", "category"], suffixes=("_actual", "_budget")
)
merged["variance"] = merged["amount_actual"] - merged["amount_budget"]
merged["variance_pct"] = (merged["variance"] / merged["amount_budget"] * 100).round(1)

variance_rows = []
for month in MONTH_ORDER:
    sub = merged[merged["month"] == month]
    rev_actual = sub[sub["type"] == "revenue"]["amount_actual"].sum()
    rev_budget = sub[sub["type"] == "revenue"]["amount_budget"].sum()
    exp_actual = sub[sub["type"] == "expense"]["amount_actual"].sum()
    exp_budget = sub[sub["type"] == "expense"]["amount_budget"].sum()
    variance_rows.append(
        {
            "month": month_label(month),
            "revenue_actual": round(rev_actual, 2),
            "revenue_budget": round(rev_budget, 2),
            "revenue_variance_pct": round((rev_actual - rev_budget) / rev_budget * 100, 1),
            "expense_actual": round(exp_actual, 2),
            "expense_budget": round(exp_budget, 2),
            "expense_variance_pct": round((exp_actual - exp_budget) / exp_budget * 100, 1),
        }
    )

# category-level variance for the latest month (drives a detail table)
latest_variance = (
    merged[merged["month"] == latest][
        ["type", "category", "amount_actual", "amount_budget", "variance", "variance_pct"]
    ]
    .sort_values("variance_pct", key=lambda s: s.abs(), ascending=False)
    .to_dict(orient="records")
)

# ---------------------------------------------------------------------------
# Drill-down: summary -> line items (advanced feature #2)
# category summary per product line, plus the underlying orders
# ---------------------------------------------------------------------------
orders["month"] = orders["month"].astype(str)
drilldown = {}
for line in product_lines:
    line_orders = orders[orders["product_line"] == line].copy()
    by_month = line_orders.groupby("month")["revenue"].agg(["sum", "count"]).reindex(MONTH_ORDER, fill_value=0)
    drilldown[line] = {
        "monthly_revenue": [round(v, 2) for v in by_month["sum"].tolist()],
        "monthly_orders": [int(v) for v in by_month["count"].tolist()],
        "sample_orders": (
            line_orders.sort_values("revenue", ascending=False)
            .head(25)[["order_id", "date", "channel", "units", "revenue", "returned"]]
            .to_dict(orient="records")
        ),
    }

# ---------------------------------------------------------------------------
# Write everything the dashboard needs as a single JS file
# ---------------------------------------------------------------------------
payload = {
    "generated_from": "monthly_actuals.csv / monthly_budget.csv / order_line_items.csv",
    "months": [month_label(m) for m in MONTH_ORDER],
    "kpi_cards": kpi_cards,
    "trend": trend,
    "revenue_breakdown": revenue_breakdown,
    "expense_breakdown": expense_breakdown,
    "variance_by_month": variance_rows,
    "latest_month_variance_detail": latest_variance,
    "drilldown": drilldown,
    "product_lines": product_lines,
}

with open("../dashboard_data.js", "w") as f:
    f.write("// Auto-generated by scripts/kpi_analysis.py -- do not edit by hand.\n")
    f.write("const DASHBOARD_DATA = ")
    json.dump(payload, f, indent=2)
    f.write(";\n")

print("Wrote dashboard_data.js")
print(f"YTD Revenue: ${monthly_rev.sum():,.0f}")
print(f"YTD Expenses: ${monthly_exp.sum():,.0f}")
print(f"YTD Net Profit: ${monthly_profit.sum():,.0f}")
