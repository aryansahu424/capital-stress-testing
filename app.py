import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import requests
import plotly.express as px
import random
from datetime import datetime, timedelta

st.set_page_config(layout="wide")

st.markdown("""
<style>
.title-container {
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.tooltip {
    position: relative;
    display: inline-block;
    cursor: help;
    color: transparent;
    font-size: 0.8em;
    margin-left: 4px;
    width: 12px;
    height: 12px;
    border: 1px solid #999;
    border-radius: 50%;
    text-align: center;
    line-height: 12px;
}
.tooltip:before {
    content: '?';
    color: #999;
    font-size: 10px;
    font-weight: bold;
}
.tooltip .tooltiptext {
    visibility: hidden;
    width: 250px;
    background-color: #f9f9f9;
    color: #333;
    text-align: left;
    border-radius: 6px;
    border: 1px solid #ddd;
    padding: 8px;
    position: absolute;
    z-index: 1;
    bottom: 125%;
    left: 50%;
    margin-left: -125px;
    opacity: 0;
    transition: opacity 0.3s;
    font-size: 12px;
    line-height: 1.4;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.tooltip:hover .tooltiptext {
    visibility: visible;
    opacity: 1;
}
</style>
""", unsafe_allow_html=True)

col_title, col_btn1, col_btn2 = st.columns([3, 1, 1])
with col_title:
    st.title("Portfolio Monitoring Agent")
with col_btn1:
    if st.button("Generate New Data", key="gen_data_btn"):
        with st.spinner("Generating new data..."):
            try:
                exec(open("data_generator.py").read())
                st.cache_data.clear()
                st.success("New data generated!")
                st.rerun()
            except Exception as e:
                st.error(f"Error: {str(e)}")
with col_btn2:
    if "show_ai" not in st.session_state:
        st.session_state.show_ai = False
    if st.button("AI Assistant", key="ai_assistant_btn"):
        st.session_state.show_ai = not st.session_state.show_ai

# Project Details Expander
with st.expander("About This Project"):
    st.markdown("""
    ### Portfolio Monitoring Agent
    
    An AI-powered credit risk portfolio monitoring dashboard designed for investment banking professionals.
    
    **Key Features:**
    - **Real-Time Analytics**: Monitor 100K+ loan portfolio with live metrics
    - **AI Risk Commentary**: Executive summaries powered by Llama 3.2 via Hugging Face
    - **Concentration Analysis**: Interactive industry and regional exposure visualizations
    - **Predictive Modeling**: HistGradientBoosting classifier with AUC scoring
    - **Stress Testing**: Dynamic scenario analysis with adjustable PD multipliers (1.0x-3.0x)
    - **Interactive Q&A**: Context-aware AI assistant for portfolio insights
    
    **Tech Stack:**
    - Frontend: Streamlit
    - ML/Analytics: scikit-learn, pandas, numpy
    - Visualization: Plotly Express
    - AI: Hugging Face Inference API
    
    **Data:**
    - Synthetic loan portfolio with realistic distributions
    - Weighted industry/region concentrations
    - Logistic regression-based default probability modeling
    
    **Use Cases:**
    - Portfolio risk assessment
    - Concentration risk monitoring
    - Stress testing and scenario analysis
    - Executive reporting and decision support
    """)

HF_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"

from huggingface_hub import InferenceClient

# --------------------------------------------------
# Load Data
# --------------------------------------------------
@st.cache_data
def load_data():
    import os
    if not os.path.exists("synthetic_loans_100k.csv"):
        exec(open("data_generator.py").read())
    return pd.read_csv("synthetic_loans_100k.csv")

df = load_data()

# --------------------------------------------------
# Train Model
# --------------------------------------------------
features = [
    "credit_score",
    "loan_amount",
    "monthly_income",
    "interest_rate",
    "debt_to_income_ratio",
    "loan_term_months",
]

X = df[features]
y = df["default_flag"]

model = HistGradientBoostingClassifier()
model.fit(X, y)

df["predicted_pd"] = model.predict_proba(X)[:, 1]
model_auc = roc_auc_score(y, df["predicted_pd"])

# --------------------------------------------------
# Portfolio Metrics
# --------------------------------------------------
total_portfolio_value = df["loan_amount"].sum()
avg_interest = df["interest_rate"].mean()
default_rate = df["default_flag"].mean()
delinquency_rate = (df["status"] != "active").mean()
total_loans = len(df)
avg_loan_size = df["loan_amount"].mean()
avg_credit_score = df["credit_score"].mean()
avg_dti = df["debt_to_income_ratio"].mean()

industry_exposure = df.groupby("industry")["loan_amount"].sum()
industry_pct = industry_exposure / industry_exposure.sum()
largest_industry = industry_pct.idxmax()
largest_industry_pct = industry_pct.max()

# --------------------------------------------------
# AI Summary
# --------------------------------------------------
if "summary" not in st.session_state:
    st.session_state.summary = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def generate_ai_summary():
    token = st.secrets.get("HF_TOKEN")

    if not token:
        return "❌ HF_TOKEN not found."

    try:
        client = InferenceClient(token=token)
        
        context = f"""
Total Exposure: {total_portfolio_value:,.0f}
Default Rate: {default_rate:.2%}
Delinquency Rate: {delinquency_rate:.2%}
Largest Industry: {largest_industry} ({largest_industry_pct:.2%})
Model ROC AUC: {model_auc:.3f}
"""

        messages = [
            {"role": "user", "content": f"""You are a Managing Director in Credit Risk at a top-tier investment bank preparing an executive summary for the Risk Committee.

Provide a concise assessment covering:
1. Portfolio Quality: Overall credit health and key risk indicators
2. Concentration Risk: Industry/geographic exposures requiring attention
3. Model Performance: Predictive accuracy and reliability
4. Strategic Recommendations: Risk mitigation actions and capital allocation guidance

Portfolio Metrics:
{context}

Deliver in executive summary format with actionable insights."""}
        ]

        response = client.chat_completion(
            messages=messages,
            model="meta-llama/Llama-3.2-3B-Instruct",
            max_tokens=800,
            temperature=0.4
        )
        
        return response.choices[0].message.content

    except Exception as e:
        return f"❌ Exception: {str(e)}"

def chat_with_ai(user_message):
    token = st.secrets.get("HF_TOKEN")
    if not token:
        return "❌ HF_TOKEN not found."
    
    try:
        client = InferenceClient(token=token)
        
        context = f"""
Total Exposure: {total_portfolio_value:,.0f}
Default Rate: {default_rate:.2%}
Delinquency Rate: {delinquency_rate:.2%}
Largest Industry: {largest_industry} ({largest_industry_pct:.2%})
Model ROC AUC: {model_auc:.3f}
"""
        
        messages = [{"role": "system", "content": f"""You are a Managing Director in Credit Risk at a leading investment bank. Provide strategic insights on portfolio risk management, capital allocation, and regulatory considerations.

Current Portfolio Context:
{context}

Respond with senior-level analysis appropriate for executive decision-making."""}]
        messages.extend(st.session_state.chat_history)
        messages.append({"role": "user", "content": user_message})
        
        response = client.chat_completion(
            messages=messages,
            model="meta-llama/Llama-3.2-3B-Instruct",
            max_tokens=800,
            temperature=0.4
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ Exception: {str(e)}"

# --------------------------------------------------
# PAGE LAYOUT
# --------------------------------------------------

st.markdown("## Portfolio Overview")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown("**Total Portfolio Value** <span class='tooltip'><span class='tooltiptext'>Total outstanding loan amount across all borrowers</span></span>", unsafe_allow_html=True)
    st.markdown(f"<h2 style='margin:0'>${total_portfolio_value/1e6:.1f}M</h2>", unsafe_allow_html=True)
    st.caption(f"Total Loans: {total_loans:,}")

with col2:
    st.markdown("**Avg Loan Size** <span class='tooltip'><span class='tooltiptext'>Average loan amount per borrower</span></span>", unsafe_allow_html=True)
    st.markdown(f"<h2 style='margin:0'>${avg_loan_size:,.0f}</h2>", unsafe_allow_html=True)
    st.caption(f"Interest Rate: {avg_interest:.2f}%")

with col3:
    st.markdown("**Default Rate** <span class='tooltip'><span class='tooltiptext'>Percentage of loans that have defaulted (failed to repay)</span></span>", unsafe_allow_html=True)
    st.markdown(f"<h2 style='margin:0'>{default_rate:.2%}</h2>", unsafe_allow_html=True)
    st.caption(f"Delinquency: {delinquency_rate:.2%}")

with col4:
    st.markdown("**Avg Credit Score** <span class='tooltip'><span class='tooltiptext'>Average FICO score indicating borrower creditworthiness (300-850)</span></span>", unsafe_allow_html=True)
    st.markdown(f"<h2 style='margin:0'>{avg_credit_score:.0f}</h2>", unsafe_allow_html=True)
    st.caption(f"Avg DTI: {avg_dti:.2f}")

with col5:
    st.markdown("**Model AUC** <span class='tooltip'><span class='tooltiptext'>Area Under Curve - Model's ability to distinguish between defaulters and non-defaulters (0.5-1.0, higher is better)</span></span>", unsafe_allow_html=True)
    st.markdown(f"<h2 style='margin:0'>{model_auc:.3f}</h2>", unsafe_allow_html=True)
    st.caption(f"Top Industry: {largest_industry}")

st.divider()

# AI Commentary Modal
if st.session_state.show_ai:
    st.header("AI Risk Commentary")
    
    if st.button("Generate AI Summary"):
        with st.spinner("Generating summary..."):
            st.session_state.summary = generate_ai_summary()
    
    if st.session_state.summary:
        st.write(st.session_state.summary)
        
        st.divider()
        st.subheader("Ask Follow-up Questions")
        
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
        
        if user_input := st.chat_input("Ask about the portfolio..."):
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.write(user_input)
            
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = chat_with_ai(user_input)
                    st.write(response)
                    st.session_state.chat_history.append({"role": "assistant", "content": response})

st.divider()

# Concentration
st.header("Concentration Analysis")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Industry Exposure")
    fig = px.pie(industry_exposure, values=industry_exposure.values, names=industry_exposure.index, 
                 hole=0.4, color_discrete_sequence=px.colors.qualitative.Set3)
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

with col2:
    st.subheader("Region Exposure")
    region_exposure = df.groupby("region")["loan_amount"].sum()
    fig = px.pie(region_exposure, values=region_exposure.values, names=region_exposure.index,
                 hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel)
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

st.divider()

# Risk Map
st.header("Top 10 Riskiest Loans")

top_risk = df.sort_values("predicted_pd", ascending=False)[
    ["loan_id", "industry", "region", "credit_score", "loan_amount", "predicted_pd"]
].head(10).reset_index(drop=True)

top_risk["predicted_pd"] = (top_risk["predicted_pd"] * 100).round(2)
top_risk.columns = ["Loan ID", "Industry", "Region", "Credit Score", "Loan Amount", "Default Risk (%)"]

st.dataframe(
    top_risk,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Loan Amount": st.column_config.NumberColumn(format="$%d"),
        "Default Risk (%)": st.column_config.ProgressColumn(
            min_value=0,
            max_value=100,
            format="%.2f%%"
        ),
        "Credit Score": st.column_config.NumberColumn(format="%d")
    }
)
st.metric("Model ROC AUC", f"{model_auc:.3f}")

st.divider()

# Stress Test
st.header("Stress Testing")

col1, col2 = st.columns([1, 2])

with col1:
    multiplier = st.slider(
        "PD Stress Multiplier",
        min_value=1.0,
        max_value=3.0,
        value=1.5,
        step=0.1,
        help="Multiply baseline default probability by this factor"
    )
    st.caption(f"Selected: {multiplier}x PD")

stressed_pd = np.minimum(df["predicted_pd"] * multiplier, 0.99)
lgd = 0.6

base_loss = (df["predicted_pd"] * lgd * df["loan_amount"]).sum()
projected_loss = (stressed_pd * lgd * df["loan_amount"]).sum()
loss_increase = projected_loss - base_loss
pct_increase = (loss_increase / base_loss) * 100

with col2:
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Base Loss", f"${base_loss:,.0f}")
    metric_col2.metric("Stressed Loss", f"${projected_loss:,.0f}", delta=f"{pct_increase:.1f}%", delta_color="inverse")
    metric_col3.metric("Loss Increase", f"${loss_increase:,.0f}")

df_temp = df.copy()
df_temp["base_el"] = df_temp["predicted_pd"] * lgd * df_temp["loan_amount"]
df_temp["stressed_pd"] = stressed_pd
df_temp["stressed_el"] = stressed_pd * lgd * df_temp["loan_amount"]

industry_stress = df_temp.groupby("industry")[["base_el", "stressed_el"]].sum()
industry_stress["increase"] = industry_stress["stressed_el"] - industry_stress["base_el"]
industry_stress = industry_stress.sort_values("increase", ascending=False).reset_index()

fig = px.bar(industry_stress, x="industry", y=["base_el", "stressed_el"],
             barmode="group", 
             labels={"value": "Expected Loss ($)", "industry": "Industry", "base_el": "Base", "stressed_el": "Stressed"},
             color_discrete_sequence=["#636EFA", "#EF553B"])
fig.update_layout(legend_title_text="Scenario", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
fig.for_each_trace(lambda t: t.update(name=t.name.replace("base_el", "Base").replace("stressed_el", "Stressed")))
st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

st.divider()

# Risk Engine Analytics
st.markdown("## Risk Engine Analytics")

LGD = 0.60
expected_loss = (df["default_probability_true"] * LGD * df["loan_amount"]).sum()
loss_ratio = expected_loss / total_portfolio_value

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("**Expected Loss** <span class='tooltip'><span class='tooltiptext'>Probability of Default × Loss Given Default × Exposure - Expected monetary loss from defaults</span></span>", unsafe_allow_html=True)
    st.markdown(f"<h2 style='margin:0'>${expected_loss/1e6:.2f}M</h2>", unsafe_allow_html=True)
with col2:
    st.markdown("**Loss Ratio** <span class='tooltip'><span class='tooltiptext'>Expected Loss / Total Portfolio Value - Percentage of portfolio expected to be lost</span></span>", unsafe_allow_html=True)
    st.markdown(f"<h2 style='margin:0'>{loss_ratio:.2%}</h2>", unsafe_allow_html=True)
with col3:
    delinq_30 = (df["days_late"] > 30).mean()
    st.markdown("**Delinquency Rate (30+ days)** <span class='tooltip'><span class='tooltiptext'>Percentage of loans with payments overdue by more than 30 days</span></span>", unsafe_allow_html=True)
    st.markdown(f"<h2 style='margin:0'>{delinq_30:.2%}</h2>", unsafe_allow_html=True)
with col4:
    st.markdown("**LGD Assumption** <span class='tooltip'><span class='tooltiptext'>Loss Given Default - Percentage of exposure lost when a borrower defaults (industry standard: 40-60%)</span></span>", unsafe_allow_html=True)
    st.markdown(f"<h2 style='margin:0'>{LGD:.0%}</h2>", unsafe_allow_html=True)

# Risk Bucketing
df_risk = df.copy()
def risk_bucket(pd_value):
    if pd_value < 0.05:
        return "Low Risk"
    elif pd_value < 0.12:
        return "Medium Risk"
    else:
        return "High Risk"

df_risk["risk_bucket"] = df_risk["default_probability_true"].apply(risk_bucket)
risk_dist = df_risk.groupby("risk_bucket").agg({
    "loan_amount": "sum",
    "loan_id": "count"
}).reset_index()
risk_dist.columns = ["Risk Bucket", "Total Exposure", "Loan Count"]
risk_dist["Exposure %"] = (risk_dist["Total Exposure"] / total_portfolio_value * 100).round(2)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Risk Distribution")
    fig = px.bar(risk_dist, x="Risk Bucket", y="Total Exposure",
                 color="Risk Bucket",
                 color_discrete_map={"Low Risk": "#2ecc71", "Medium Risk": "#f39c12", "High Risk": "#e74c3c"})
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

with col2:
    st.subheader("Risk Bucket Summary")
    st.dataframe(
        risk_dist,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Total Exposure": st.column_config.NumberColumn(format="$%.0f"),
            "Exposure %": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.2f%%")
        }
    )

# Concentration Risk
concentration_threshold = 0.30
industry_conc = industry_pct[industry_pct > concentration_threshold]
if len(industry_conc) > 0:
    st.warning(f"⚠️ Concentration Alert: {len(industry_conc)} industry(ies) exceed {concentration_threshold:.0%} threshold")
    conc_df = pd.DataFrame({
        "Industry": industry_conc.index,
        "Concentration %": (industry_conc.values * 100).round(2)
    })
    st.dataframe(conc_df, use_container_width=True, hide_index=True)

st.divider()
