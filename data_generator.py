import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta


N = 100000

industries = ["Retail", "Construction", "Healthcare", "Technology", "Hospitality", "Manufacturing"]
regions = ["Texas", "California", "Florida", "New York", "Illinois", "Arizona"]

industry_weights = [0.25, 0.20, 0.18, 0.15, 0.12, 0.10]
region_weights = [0.22, 0.20, 0.18, 0.16, 0.14, 0.10]

data = []

for _ in range(N):

    credit_score = int(np.clip(np.random.normal(680, 60), 500, 820))
    monthly_income = np.random.lognormal(mean=8.5, sigma=0.5)
    loan_amount = np.random.lognormal(mean=9, sigma=0.8)
    loan_term = random.choice([24, 36, 48, 60])
    interest_rate = round(np.random.uniform(6, 22), 2)
    loan_id = f"L{_+1:06d}"
    customer_id = f"CUST{_+1:06d}"

    dti = min(loan_amount / (monthly_income * 12), 1.5)

    industry = random.choices(industries, weights=industry_weights, k=1)[0]
    region = random.choices(regions, weights=region_weights, k=1)[0]

    # Default probability logic
    base_pd = 0.02

    # if credit_score < 620:
    #     base_pd += 0.10
    # if dti > 0.45:
    #     base_pd += 0.08
    # if interest_rate > 18:
    #     base_pd += 0.05
    # if industry in ["Retail", "Construction"]:
    #     base_pd += 0.04

    logit = (
        -4
        - 0.008 * credit_score
        + 3.5 * dti
        + 0.07 * interest_rate
    )

    base_pd = 1 / (1 + np.exp(-logit)) # base_pd = 1 / (1 + np.exp(-logit))

    macro_stress_factor = np.random.normal(1, 0.05)
    base_pd *= macro_stress_factor

    base_pd = min(max(base_pd, 0), 0.6) # cap at 60% for realism
    default = np.random.rand() < base_pd
    
    late = False

    if not default:
        late = np.random.rand() < 0.10

    if default:
        status = "defaulted"
        days_late = random.randint(90, 180)
    elif late:
        status = "late"
        days_late = random.randint(15, 60)
    else:
        status = "active"
        days_late = 0

    default_flag = 1 if status == "defaulted" else 0

    origination_date = datetime.now() - timedelta(days=random.randint(0, 1460))

    data.append([
        loan_id,
        customer_id,
        round(float(loan_amount), 2),
        interest_rate,
        industry,
        region,
        credit_score,
        round(float(monthly_income), 2),
        loan_term,
        round(float(dti), 2),
        status,
        round(base_pd, 4),
        days_late,
        default_flag,
        origination_date.date()
    ])

df = pd.DataFrame(data, columns=[
    "loan_id",
    "customer_id",
    "loan_amount",
    "interest_rate",
    "industry",
    "region",
    "credit_score",
    "monthly_income",
    "loan_term_months",
    "debt_to_income_ratio",
    "status",
    "default_probability_true",
    "days_late",
    "default_flag",
    "origination_date"
])

# Save to CSV in same folder
df.to_csv("synthetic_loans_100k.csv", index=False)

print("CSV and Parquet files saved successfully in current folder.")
print(df["status"].value_counts(normalize=True))
print("Default Rate:", (df["status"]=="defaulted").mean())
print("Avg Credit Score:", df["credit_score"].mean())
print("Avg DTI:", df["debt_to_income_ratio"].mean())
print("Avg Loan Amount:", df["loan_amount"].mean())
print(df["default_flag"].mean())
print(df["default_probability_true"].describe())
