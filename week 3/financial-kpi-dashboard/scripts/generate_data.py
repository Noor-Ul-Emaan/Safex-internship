"""
generate_data.py
-----------------
Creates a realistic synthetic financial dataset for an e-commerce retailer.

Why synthetic data: real finance data is confidential, but the shape of this
data (monthly revenue/expense by category, a budget plan, and order-level
line items) mirrors what you'd actually pull from a retailer's order
management system + accounting ledger.

Outputs (all written to ../data/):
  - monthly_actuals.csv     : month x category actual $ (revenue & expense)
  - monthly_budget.csv      : month x category budgeted $ (for variance view)
  - order_line_items.csv    : order-level detail for the drill-down feature
  - ecommerce_financials.xlsx : all three tables as separate sheets, the
                                 "single source file" a finance team would
                                 actually hand you
"""

import numpy as np
import pandas as pd
from datetime import date

rng = np.random.default_rng(42)

MONTHS = pd.date_range("2025-01-01", periods=12, freq="MS")
PRODUCT_LINES = ["Apparel", "Electronics", "Home & Garden", "Beauty", "Sports & Outdoors"]
CHANNELS = ["Website", "Mobile App", "Marketplace"]
EXPENSE_CATEGORIES = [
    "COGS",
    "Marketing & Ads",
    "Shipping & Fulfillment",
    "Payment Processing",
    "Platform & Tech",
    "Payroll & Ops",
    "Returns & Refunds",
]

# ---------------------------------------------------------------------------
# 1. Order-level line items (this is the "ground truth" everything rolls up from)
# ---------------------------------------------------------------------------
rows = []
order_id = 100000
for m_idx, month_start in enumerate(MONTHS):
    days_in_month = (month_start + pd.offsets.MonthEnd(0)).day
    # gentle seasonal growth + a Nov/Dec holiday bump
    seasonal = 1.0 + 0.03 * m_idx
    if month_start.month in (11, 12):
        seasonal *= 1.45
    n_orders = int(rng.normal(loc=520 * seasonal, scale=25))

    for _ in range(max(n_orders, 1)):
        order_id += 1
        day = rng.integers(1, days_in_month + 1)
        order_date = date(month_start.year, month_start.month, day)
        product_line = rng.choice(PRODUCT_LINES, p=[0.27, 0.24, 0.18, 0.16, 0.15])
        channel = rng.choice(CHANNELS, p=[0.55, 0.33, 0.12])

        base_price = {
            "Apparel": 48,
            "Electronics": 165,
            "Home & Garden": 72,
            "Beauty": 34,
            "Sports & Outdoors": 89,
        }[product_line]
        order_value = max(8, rng.normal(base_price, base_price * 0.35))
        units = rng.integers(1, 4)
        revenue = round(order_value * units, 2)
        was_returned = rng.random() < 0.06

        rows.append(
            {
                "order_id": order_id,
                "date": order_date,
                "month": month_start.strftime("%Y-%m"),
                "product_line": product_line,
                "channel": channel,
                "units": int(units),
                "revenue": revenue,
                "returned": was_returned,
            }
        )

orders = pd.DataFrame(rows)
orders.to_csv("../data/order_line_items.csv", index=False)

# ---------------------------------------------------------------------------
# 2. Monthly actuals: revenue by product line + expenses by category
# ---------------------------------------------------------------------------
rev_by_month_line = (
    orders.groupby(["month", "product_line"])["revenue"].sum().reset_index()
)
rev_by_month_line["category"] = rev_by_month_line["product_line"]
rev_by_month_line["type"] = "revenue"
rev_by_month_line = rev_by_month_line[["month", "type", "category", "revenue"]].rename(
    columns={"revenue": "amount"}
)

monthly_revenue_totals = orders.groupby("month")["revenue"].sum()

expense_rows = []
for month_str, total_rev in monthly_revenue_totals.items():
    ratios = {
        "COGS": rng.normal(0.38, 0.015),
        "Marketing & Ads": rng.normal(0.14, 0.02),
        "Shipping & Fulfillment": rng.normal(0.09, 0.01),
        "Payment Processing": rng.normal(0.029, 0.002),
        "Platform & Tech": rng.normal(0.035, 0.004),
        "Payroll & Ops": rng.normal(0.11, 0.008),
        "Returns & Refunds": rng.normal(0.045, 0.006),
    }
    for cat, ratio in ratios.items():
        expense_rows.append(
            {
                "month": month_str,
                "type": "expense",
                "category": cat,
                "amount": round(max(ratio, 0.005) * total_rev, 2),
            }
        )

expenses_df = pd.DataFrame(expense_rows)
actuals = pd.concat([rev_by_month_line, expenses_df], ignore_index=True)
actuals["amount"] = actuals["amount"].round(2)
actuals.to_csv("../data/monthly_actuals.csv", index=False)

# ---------------------------------------------------------------------------
# 3. Monthly budget: what finance planned for, for the variance-to-budget view
# ---------------------------------------------------------------------------
budget_rows = []
for month_str in monthly_revenue_totals.index:
    sub = actuals[actuals["month"] == month_str]
    for _, r in sub.iterrows():
        # budget is set at the start of the year with modest, deliberate error
        # so the variance view has something real to show
        noise = rng.normal(1.0, 0.09)
        budget_rows.append(
            {
                "month": month_str,
                "type": r["type"],
                "category": r["category"],
                "amount": round(r["amount"] * noise, 2),
            }
        )

budget = pd.DataFrame(budget_rows)
budget.to_csv("../data/monthly_budget.csv", index=False)

# ---------------------------------------------------------------------------
# 4. One workbook a finance stakeholder could open directly in Excel
# ---------------------------------------------------------------------------
with pd.ExcelWriter("../data/ecommerce_financials.xlsx", engine="openpyxl") as writer:
    actuals.to_excel(writer, sheet_name="Monthly Actuals", index=False)
    budget.to_excel(writer, sheet_name="Monthly Budget", index=False)
    orders.to_excel(writer, sheet_name="Order Line Items", index=False)

print(f"Generated {len(orders):,} orders across {len(MONTHS)} months.")
print("Wrote: monthly_actuals.csv, monthly_budget.csv, order_line_items.csv, ecommerce_financials.xlsx")
