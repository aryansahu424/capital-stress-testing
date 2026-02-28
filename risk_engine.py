import pandas as pd
import risk_engine as re


LGD = 0.60  # Assume 60% loss given default


# -------------------------------
# Portfolio Level Metrics
# -------------------------------

def total_exposure(df):
    return df["loan_amount"].sum()


def exposure_by_industry(df):
    return df.groupby("industry")["loan_amount"].sum().sort_values(ascending=False)


def exposure_by_region(df):
    return df.groupby("region")["loan_amount"].sum().sort_values(ascending=False)


def delinquency_rate(df):
    return (df["days_late"] > 30).mean()


def default_rate(df):
    return df["default_flag"].mean()


# -------------------------------
# Expected Loss
# -------------------------------

def calculate_expected_loss(df):
    df = df.copy()
    df["expected_loss"] = (
        df["default_probability_true"] * LGD * df["loan_amount"]
    )
    return df["expected_loss"].sum()


# -------------------------------
# Concentration Risk
# -------------------------------

def concentration_by_industry(df, threshold=0.30):
    exposure = exposure_by_industry(df)
    total = total_exposure(df)
    concentration_ratio = exposure / total
    return concentration_ratio[concentration_ratio > threshold]


# -------------------------------
# Risk Bucketing
# -------------------------------

def risk_bucket(pd_value):
    if pd_value < 0.05:
        return "Low Risk"
    elif pd_value < 0.12:
        return "Medium Risk"
    else:
        return "High Risk"


def apply_risk_buckets(df):
    df = df.copy()
    df["risk_bucket"] = df["default_probability_true"].apply(risk_bucket)
    return df

def portfolio_loss_ratio(df):
    total_exp = total_exposure(df)
    el = calculate_expected_loss(df)
    return el / total_exp if total_exp > 0 else 0

df = pd.read_csv("synthetic_loans_100k.csv")

print("Total Exposure:", re.total_exposure(df))
print("Default Rate:", re.default_rate(df))
print("Delinquency Rate:", re.delinquency_rate(df))
print("Expected Loss:", re.calculate_expected_loss(df))

print("\nTop Industry Exposure:")
print(re.exposure_by_industry(df))

print("\nConcentration Flags (>30%):")
print(re.concentration_by_industry(df))
print("\nPortfolio Loss Ratio:", re.portfolio_loss_ratio(df))