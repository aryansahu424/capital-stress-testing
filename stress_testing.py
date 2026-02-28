import pandas as pd
import stress_testing as st


LGD = 0.60


# ---------------------------------
# Helper: Calculate Expected Loss
# ---------------------------------

def calculate_expected_loss(df, pd_column="default_probability_true"):
    return (df[pd_column] * LGD * df["loan_amount"]).sum()


# ---------------------------------
# Scenario 1: Recession
# ---------------------------------

def recession_scenario(df, stress_multiplier=1.25):
    stressed_df = df.copy()
    stressed_df["stressed_pd"] = (
        stressed_df["default_probability_true"] * stress_multiplier
    ).clip(upper=0.99)

    return stressed_df


# ---------------------------------
# Scenario 2: Interest Rate Shock
# High interest loans deteriorate
# ---------------------------------

def interest_rate_shock(df, threshold=18, multiplier=1.40):
    stressed_df = df.copy()

    stressed_df["stressed_pd"] = stressed_df["default_probability_true"]

    mask = stressed_df["interest_rate"] > threshold
    stressed_df.loc[mask, "stressed_pd"] *= multiplier

    stressed_df["stressed_pd"] = stressed_df["stressed_pd"].clip(upper=0.99)

    return stressed_df


# ---------------------------------
# Scenario 3: Industry Crash
# ---------------------------------

def industry_crash(df, industry_name="Retail", multiplier=1.50):
    stressed_df = df.copy()

    stressed_df["stressed_pd"] = stressed_df["default_probability_true"]

    mask = stressed_df["industry"] == industry_name
    stressed_df.loc[mask, "stressed_pd"] *= multiplier

    stressed_df["stressed_pd"] = stressed_df["stressed_pd"].clip(upper=0.99)

    return stressed_df


# ---------------------------------
# Compare Base vs Stressed
# ---------------------------------

def stress_summary(df_stressed):
    base_el = calculate_expected_loss(df_stressed, "default_probability_true")
    stressed_el = calculate_expected_loss(df_stressed, "stressed_pd")

    return {
        "Base Expected Loss": base_el,
        "Stressed Expected Loss": stressed_el,
        "Increase in Loss": stressed_el - base_el,
        "Percentage Increase": (stressed_el - base_el) / base_el
    }

def stress_by_industry(df_stressed):
    base = (
        df_stressed["default_probability_true"] * 0.60 * df_stressed["loan_amount"]
    )
    stressed = (
        df_stressed["stressed_pd"] * 0.60 * df_stressed["loan_amount"]
    )

    df_temp = df_stressed.copy()
    df_temp["base_el"] = base
    df_temp["stressed_el"] = stressed

    summary = df_temp.groupby("industry")[["base_el", "stressed_el"]].sum()
    summary["increase"] = summary["stressed_el"] - summary["base_el"]
    summary["pct_increase"] = summary["increase"] / summary["base_el"]

    return summary.sort_values("increase", ascending=False)

df = pd.read_csv("synthetic_loans_100k.csv")

# Recession scenario
recession_df = st.recession_scenario(df)
summary = st.stress_summary(recession_df)

print("\nRecession Scenario")
for k, v in summary.items():
    print(k, ":", v)

print(st.stress_by_industry(recession_df))

# Interest Rate Shock