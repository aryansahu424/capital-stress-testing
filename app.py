import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import plotly.express as px
import plotly.graph_objects as go
from huggingface_hub import InferenceClient

st.set_page_config(layout="wide")

# Sidebar with Info
st.sidebar.title("Portfolio Monitor")
st.sidebar.markdown("### Quick Navigation")
st.sidebar.markdown("""
- [Portfolio Overview](#portfolio-overview)
- [Concentration Analysis](#concentration-analysis)
- [Top 10 Riskiest Loans](#top-10-riskiest-loans)
- [Stress Testing](#stress-testing)
- [Risk Engine Analytics](#risk-engine-analytics)
- [Scenario Analysis](#scenario-analysis)
- [Capital Stress Engine](#capital-stress-engine-9-quarter-projection)
""")

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
@st.cache_resource
def train_model(X, y):
    model = HistGradientBoostingClassifier()
    model.fit(X, y)
    return model

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

model = train_model(X, y)

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
        return "ERROR: HF_TOKEN not found."

    try:
        client = InferenceClient(token=token)
        
        # Get top 10 risky loans
        top_10 = df.nlargest(10, 'predicted_pd')[['industry', 'region', 'loan_amount', 'credit_score', 'predicted_pd']]
        
        # Get industry breakdown
        industry_stats = df.groupby('industry').agg({
            'loan_amount': 'sum',
            'predicted_pd': 'mean'
        }).sort_values('loan_amount', ascending=False).head(5)
        
        context = f"""
Total Exposure: ${total_portfolio_value:,.0f}
Total Loans: {total_loans:,}
Default Rate: {default_rate:.2%}
Delinquency Rate: {delinquency_rate:.2%}
Avg Credit Score: {avg_credit_score:.0f}
Avg DTI: {avg_dti:.2f}
Largest Industry: {largest_industry} ({largest_industry_pct:.2%})
Model ROC AUC: {model_auc:.3f}

Top 5 Industries by Exposure:
{industry_stats.to_string()}

Top 10 Riskiest Loans Summary:
{top_10.to_string()}
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
        return f"ERROR: Exception: {str(e)}"

def chat_with_ai(user_message):
    token = st.secrets.get("HF_TOKEN")
    if not token:
        return "ERROR: HF_TOKEN not found."
    
    try:
        client = InferenceClient(token=token)
        
        # Calculate additional metrics with default LGD
        lgd_default = 0.60
        expected_loss = (df["predicted_pd"] * lgd_default * df["loan_amount"]).sum()
        loss_ratio = expected_loss / total_portfolio_value
        delinq_30 = (df["days_late"] > 30).mean()
        
        # Region breakdown
        region_stats = df.groupby('region').agg({
            'loan_amount': 'sum',
            'predicted_pd': 'mean'
        }).sort_values('loan_amount', ascending=False).head(3)
        
        context = f"""
Total Exposure: ${total_portfolio_value:,.0f}
Total Loans: {total_loans:,}
Avg Loan Size: ${avg_loan_size:,.0f}
Avg Interest Rate: {avg_interest:.2f}%
Default Rate: {default_rate:.2%}
Delinquency Rate: {delinquency_rate:.2%}
Delinquency 30+ days: {delinq_30:.2%}
Avg Credit Score: {avg_credit_score:.0f}
Avg DTI: {avg_dti:.2f}
Expected Loss: ${expected_loss:,.0f}
Loss Ratio: {loss_ratio:.2%}
Largest Industry: {largest_industry} ({largest_industry_pct:.2%})
Model ROC AUC: {model_auc:.3f}

Top 3 Regions by Exposure:
{region_stats.to_string()}
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
        return f"ERROR: Exception: {str(e)}"

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
    industry_exposure_display = industry_exposure.copy()
    industry_exposure_display.index = industry_exposure_display.index.str.replace('_', ' ').str.title()
    fig = px.pie(industry_exposure_display, values=industry_exposure_display.values, names=industry_exposure_display.index, 
                 hole=0.4, color_discrete_sequence=px.colors.qualitative.Set3)
    fig.update_traces(textposition='inside', textinfo='percent+label', hovertemplate='<b>%{label}</b><br>$%{value:,.0f}<br>%{percent}<extra></extra>')
    fig.update_layout(showlegend=False, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

with col2:
    st.subheader("Region Exposure")
    region_exposure = df.groupby("region")["loan_amount"].sum()
    region_exposure.index = region_exposure.index.str.replace('_', ' ').str.title()
    fig = px.pie(region_exposure, values=region_exposure.values, names=region_exposure.index,
                 hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel)
    fig.update_traces(textposition='inside', textinfo='percent+label', hovertemplate='<b>%{label}</b><br>$%{value:,.0f}<br>%{percent}<extra></extra>')
    fig.update_layout(showlegend=False, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
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
    LGD = st.slider("LGD Assumption", 0.3, 0.8, 0.6, 0.05, help="Loss Given Default percentage")
    multiplier = st.slider(
        "PD Stress Multiplier",
        min_value=1.0,
        max_value=3.0,
        value=1.5,
        step=0.1,
        help="Multiply baseline default probability by this factor"
    )
    st.caption(f"Selected: {multiplier}x PD")

base_loss = (df["predicted_pd"] * LGD * df["loan_amount"]).sum()
stressed_pd = np.minimum(df["predicted_pd"] * multiplier, 0.99)
projected_loss = (stressed_pd * LGD * df["loan_amount"]).sum()
loss_increase = projected_loss - base_loss
pct_increase = (loss_increase / base_loss) * 100

with col2:
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Base Loss", f"${base_loss:,.0f}")
    metric_col2.metric("Stressed Loss", f"${projected_loss:,.0f}", delta=f"{pct_increase:.1f}%", delta_color="inverse")
    metric_col3.metric("Loss Increase", f"${loss_increase:,.0f}")

df_temp = df.copy()
df_temp["base_el"] = df_temp["predicted_pd"] * LGD * df_temp["loan_amount"]
df_temp["stressed_pd"] = stressed_pd
df_temp["stressed_el"] = stressed_pd * LGD * df_temp["loan_amount"]

industry_stress = df_temp.groupby("industry")[["base_el", "stressed_el"]].sum()
industry_stress["increase"] = industry_stress["stressed_el"] - industry_stress["base_el"]
industry_stress = industry_stress.sort_values("increase", ascending=False).reset_index()
industry_stress["industry"] = industry_stress["industry"].str.replace('_', ' ').str.title()

fig = px.bar(industry_stress, x="industry", y=["base_el", "stressed_el"],
             barmode="group", 
             labels={"value": "Expected Loss ($)", "industry": "Industry"},
             color_discrete_sequence=["#636EFA", "#EF553B"])
fig.update_layout(legend_title_text="Scenario", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                  plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(size=12))
fig.for_each_trace(lambda t: t.update(
    name="Base" if "base" in t.name else "Stressed",
    hovertemplate='<b>%{x}</b><br>%{fullData.name}: $%{y:,.0f}<extra></extra>'
))
fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

st.divider()

# Risk Engine Analytics
st.markdown("## Risk Engine Analytics")

col_thresh = st.columns(1)[0]
with col_thresh:
    concentration_threshold = st.slider("Concentration Threshold", 0.15, 0.50, 0.30, 0.05, help="Alert threshold for industry concentration")

expected_loss = (df["predicted_pd"] * LGD * df["loan_amount"]).sum()
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

col_low, col_med = st.columns(2)
with col_low:
    low_threshold = st.slider("Low Risk Threshold", 0.01, 0.10, 0.05, 0.01, help="PD threshold for low risk classification")
with col_med:
    med_threshold = st.slider("Medium Risk Threshold", 0.06, 0.20, 0.12, 0.01, help="PD threshold for medium risk classification")

def risk_bucket(pd_value):
    if pd_value < low_threshold:
        return "Low Risk"
    elif pd_value < med_threshold:
        return "Medium Risk"
    else:
        return "High Risk"

df_risk["risk_bucket"] = df_risk["predicted_pd"].apply(risk_bucket)
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
    fig.update_layout(showlegend=False, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(size=12))
    fig.update_traces(hovertemplate='<b>%{x}</b><br>Total Exposure: $%{y:,.0f}<extra></extra>')
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
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
industry_conc = industry_pct[industry_pct > concentration_threshold]
if len(industry_conc) > 0:
    st.warning(f"WARNING: Concentration Alert: {len(industry_conc)} industry(ies) exceed {concentration_threshold:.0%} threshold")
    conc_df = pd.DataFrame({
        "Industry": industry_conc.index,
        "Concentration %": (industry_conc.values * 100).round(2)
    })
    st.dataframe(conc_df, use_container_width=True, hide_index=True)

st.divider()

# Scenario Analysis
st.header("Scenario Analysis")

tabs = st.tabs(["Recession", "Interest Rate Shock", "Industry Crash"])

with tabs[0]:
    st.subheader("Recession Scenario")
    recession_mult = st.slider("Recession PD Multiplier", 1.0, 4.0, 2.5, 0.1, key="recession")
    recession_pd = np.minimum(df["predicted_pd"] * recession_mult, 0.99)
    recession_loss = (recession_pd * LGD * df["loan_amount"]).sum()
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Base Loss", f"${base_loss:,.0f}")
        st.metric("Recession Loss", f"${recession_loss:,.0f}", delta=f"{((recession_loss-base_loss)/base_loss*100):.1f}%", delta_color="inverse")
    
    with col2:
        scenario_data = pd.DataFrame({
            "Scenario": ["Base", "Recession"],
            "Expected Loss": [base_loss, recession_loss]
        })
        fig = px.bar(scenario_data, x="Scenario", y="Expected Loss", color="Scenario",
                     color_discrete_map={"Base": "#636EFA", "Recession": "#EF553B"})
        fig.update_layout(showlegend=False, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(size=12))
        fig.update_traces(hovertemplate='<b>%{x}</b><br>$%{y:,.0f}<extra></extra>')
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

with tabs[1]:
    st.subheader("Interest Rate Shock")
    rate_mult = st.slider("Rate Shock PD Multiplier", 1.0, 3.0, 1.8, 0.1, key="rate")
    rate_shock_pd = np.minimum(df["predicted_pd"] * rate_mult, 0.99)
    rate_shock_loss = (rate_shock_pd * LGD * df["loan_amount"]).sum()
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Base Loss", f"${base_loss:,.0f}")
        st.metric("Rate Shock Loss", f"${rate_shock_loss:,.0f}", delta=f"{((rate_shock_loss-base_loss)/base_loss*100):.1f}%", delta_color="inverse")
    
    with col2:
        scenario_data = pd.DataFrame({
            "Scenario": ["Base", "Rate Shock"],
            "Expected Loss": [base_loss, rate_shock_loss]
        })
        fig = px.bar(scenario_data, x="Scenario", y="Expected Loss", color="Scenario",
                     color_discrete_map={"Base": "#636EFA", "Rate Shock": "#FF6692"})
        fig.update_layout(showlegend=False, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(size=12))
        fig.update_traces(hovertemplate='<b>%{x}</b><br>$%{y:,.0f}<extra></extra>')
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

with tabs[2]:
    st.subheader("Industry-Specific Crash")
    col_a, col_b = st.columns(2)
    with col_a:
        crash_industry = st.selectbox("Select Industry", df["industry"].unique())
    with col_b:
        crash_mult = st.slider("Industry Crash PD Multiplier", 1.0, 5.0, 3.0, 0.1, key="crash")
    
    df_crash = df.copy()
    df_crash.loc[df_crash["industry"] == crash_industry, "crash_pd"] = np.minimum(df_crash.loc[df_crash["industry"] == crash_industry, "predicted_pd"] * crash_mult, 0.99)
    df_crash["crash_pd"].fillna(df_crash["predicted_pd"], inplace=True)
    crash_loss = (df_crash["crash_pd"] * LGD * df_crash["loan_amount"]).sum()
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Base Loss", f"${base_loss:,.0f}")
        st.metric(f"{crash_industry} Crash Loss", f"${crash_loss:,.0f}", delta=f"{((crash_loss-base_loss)/base_loss*100):.1f}%", delta_color="inverse")
    
    with col2:
        industry_impact = df_crash.groupby("industry").apply(
            lambda x: (x["crash_pd"] * LGD * x["loan_amount"]).sum()
        ).reset_index(name="Loss")
        industry_impact = industry_impact.sort_values("Loss", ascending=False).head(5)
        industry_impact["industry"] = industry_impact["industry"].str.replace('_', ' ').str.title()
        
        fig = px.bar(industry_impact, x="industry", y="Loss", color="industry")
        fig.update_layout(showlegend=False, xaxis_title="Industry", yaxis_title="Expected Loss ($)",
                          plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(size=12))
        fig.update_traces(hovertemplate='<b>%{x}</b><br>Expected Loss: $%{y:,.0f}<extra></extra>')
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

st.divider()

st.header("Capital Stress Engine (9-Quarter Projection)")

# -----------------------------
# Starting Capital Inputs
# -----------------------------
col1, col2, col3 = st.columns(3)

with col1:
    starting_cet1 = st.number_input("Starting CET1 ($B)", value=100.0)

with col2:
    starting_rwa = st.number_input("Starting RWA ($B)", value=800.0)

with col3:
    regulatory_min = st.number_input("Regulatory Minimum CET1 (%)", value=10.5)

starting_ratio = (starting_cet1 / starting_rwa) * 100

st.metric("Starting CET1 Ratio", f"{starting_ratio:.2f}%")

# -----------------------------
# Stress Inputs
# -----------------------------
st.subheader("Stress Assumptions")

col1, col2, col3, col4 = st.columns(4)

with col1:
    total_ppnr = st.slider("9Q PPNR ($B)", 0.0, 80.0, 40.0)

with col2:
    total_credit_losses = st.slider("9Q Credit Losses ($B)", 0.0, 80.0, 35.0)

with col3:
    total_trading_losses = st.slider("Trading Losses ($B)", 0.0, 30.0, 8.0)

with col4:
    total_op_losses = st.slider("Operational Losses ($B)", 0.0, 20.0, 5.0)

rwa_inflation_pct = st.slider("RWA Inflation (%)", 0.0, 30.0, 12.5)
ppnr_decline_rate = st.slider("PPNR Decline Rate", 0.0, 0.5, 0.4, 0.05, help="Rate at which PPNR declines over 9 quarters")

# -----------------------------
# 9-Quarter Projection
# -----------------------------
quarters = np.arange(1, 10)

# Spread PPNR with decline
# Convert % decline to decimal
decline = ppnr_decline_rate / 100  

# Build stress path dynamically
ppnr_shape = np.array([
    1.0,
    1.0 - decline * 0.5,
    1.0 - decline,
    1.0 - decline,
    1.0 - decline * 0.8,
    1.0 - decline * 0.6,
    1.0 - decline * 0.4,
    1.0 - decline * 0.2,
    1.0
])
ppnr_path = (total_ppnr / ppnr_shape.sum()) * ppnr_shape

# Front-load credit losses (more realistic recession shape)
credit_loss_path = np.linspace(0.05, 0.15, 9)
credit_loss_path = credit_loss_path / credit_loss_path.sum() * total_credit_losses

# Assume trading & op losses occur early
trading_path = np.zeros(9)
trading_path[1] = total_trading_losses

op_path = np.zeros(9)
op_path[2] = total_op_losses

# Capital path
capital_path_adj = []
capital = starting_cet1

for q in range(9):
    capital += ppnr_path[q]
    capital -= credit_loss_path[q]
    capital -= trading_path[q]
    capital -= op_path[q]
    capital_path_adj.append(capital)

capital_path_adj = np.array(capital_path_adj)

# RWA path (inflate gradually)
rwa_path = np.linspace(starting_rwa, starting_rwa * (1 + rwa_inflation_pct/100), 9)

cet1_ratio_path = (capital_path_adj / rwa_path) * 100

ending_ratio = cet1_ratio_path[-1]

# -----------------------------
# Capital Ratio Chart
# -----------------------------

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=quarters,
    y=cet1_ratio_path,
    mode="lines+markers",
    name="CET1 Ratio",
    line=dict(color="#636EFA", width=3),
    marker=dict(size=8)
))

fig.add_trace(go.Scatter(
    x=quarters,
    y=[regulatory_min]*9,
    mode="lines",
    name="Regulatory Minimum",
    line=dict(dash="dash", color="#EF553B", width=2)
))

fig.update_layout(
    title="9-Quarter CET1 Ratio Projection",
    xaxis_title="Quarter",
    yaxis_title="CET1 Ratio (%)",
    showlegend=True,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(size=12),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)')

st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

# -----------------------------
# Final Metrics
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    st.metric("Ending CET1 ($B)", f"{capital_path_adj[-1]:.2f}")

with col2:
    delta = ending_ratio - starting_ratio
    st.metric("Ending CET1 Ratio", f"{ending_ratio:.2f}%", delta=f"{delta:.2f}%")

# Regulatory Breach Alert
if ending_ratio < regulatory_min:
    st.error("WARNING: Capital Breach: CET1 ratio falls below regulatory minimum.")
else:
    st.success("OK: Capital remains above regulatory minimum.")

st.divider()

# -----------------------------
# Additional Institutional Controls
# -----------------------------
st.subheader("Advanced Capital Controls")

col1, col2, col3 = st.columns(3)
with col1:
    tax_rate = st.slider("Tax Rate (%)", 0, 40, 21) / 100
with col2:
    dividend_payout = st.slider("Dividend Payout Ratio (%)", 0, 100, 40) / 100
with col3:
    management_overlay = st.slider("Management Overlay Loss ($B)", 0.0, 20.0, 0.0)

suspend_dividends = st.checkbox("Suspend Dividends If Capital Falls")

col_buf1, col_buf2 = st.columns(2)
with col_buf1:
    ccb = st.slider("Capital Conservation Buffer (%)", 0.0, 5.0, 2.5, 0.5, help="CCB requirement")
with col_buf2:
    gsib = st.slider("G-SIB Buffer (%)", 0.0, 3.5, 1.5, 0.5, help="Global Systemically Important Bank buffer")

# Recalculate capital with advanced controls
capital = starting_cet1
capital_path_adj = []
dividends_paid = []

for q in range(9):
    pre_tax_income = ppnr_path[q] - credit_loss_path[q] - trading_path[q] - op_path[q]
    tax = max(pre_tax_income, 0) * tax_rate
    net_income = pre_tax_income - tax
    
    if suspend_dividends and capital < starting_cet1:
        dividend = 0
    else:
        dividend = max(net_income, 0) * dividend_payout
    
    capital += net_income - dividend
    capital_path_adj.append(capital)

capital_path_adj = np.array(capital_path_adj)
capital_path_adj[-1] -= management_overlay

credit_stress_factor = total_credit_losses / 35
cumulative_loss_pct = np.cumsum(credit_loss_path) / total_credit_losses
rwa_path_adj = starting_rwa * (1 + cumulative_loss_pct * rwa_inflation_pct/100)
cet1_ratio_path_adj = (capital_path_adj / rwa_path_adj) * 100
min_ratio = cet1_ratio_path_adj.min()
scb = starting_ratio - min_ratio
ratio_drop = starting_ratio - min_ratio

# Enhanced Chart with Buffer Requirements
fig2 = go.Figure()

fig2.add_trace(go.Scatter(
    x=quarters, y=cet1_ratio_path_adj, mode="lines+markers", name="Adjusted CET1 Ratio",
    line=dict(color="#636EFA", width=3), marker=dict(size=8)
))

fig2.add_trace(go.Scatter(
    x=quarters, y=[regulatory_min]*9, mode="lines", name="Regulatory Minimum",
    line=dict(dash="dash", color="#EF553B", width=2)
))

total_required = regulatory_min + ccb + gsib

fig2.add_trace(go.Scatter(
    x=quarters, y=[total_required]*9, mode="lines", name="Total Buffer Requirement",
    line=dict(dash="dot", color="orange", width=2)
))

fig2.update_layout(
    title="Adjusted 9-Quarter CET1 Ratio with Buffers",
    xaxis_title="Quarter", yaxis_title="CET1 Ratio (%)", showlegend=True,
    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(size=12),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

min_q = quarters[np.argmin(cet1_ratio_path_adj)]

fig2.add_trace(go.Scatter(
    x=[min_q],
    y=[min_ratio],
    mode="markers",
    marker=dict(size=12, color="red"),
    name="Stress Low Point"
))
fig2.add_hrect(
    y0=0,
    y1=regulatory_min,
    fillcolor="red",
    opacity=0.1,
    layer="below",
    line_width=0,
)
fig2.update_xaxes(showgrid=False)
fig2.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)')

st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': False})

# Additional Metrics


col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Minimum CET1 Ratio (9Q Low)", f"{min_ratio:.2f}%")
with col2:
    st.metric("Stress Capital Buffer (SCB)", f"{scb:.2f}%")
with col3:
    st.metric("Adjusted Ending CET1", f"${capital_path_adj[-1]:.2f}B")
with col4:
    st.metric("Peak-to-Trough CET1 Decline", f"{ratio_drop:.2f}%")
with col5:
    st.metric("Lowest CET1 Quarter", f"Q{min_q}")

shortfall = regulatory_min - min_ratio
if shortfall > 0:
    st.metric("Capital Shortfall", f"{shortfall:.2f}%")

breach_quarters = quarters[cet1_ratio_path_adj < regulatory_min]
if len(breach_quarters) > 0:
    st.error(f"WARNING: Capital breach occurs in Quarter(s): {breach_quarters.tolist()}")
else:
    st.success("OK: No capital breaches under adjusted scenario")

st.divider()

# -----------------------------
# Capital Waterfall Chart
# -----------------------------
st.subheader("Capital Waterfall (9-Quarter Cumulative)")

# Calculate cumulative values
total_ppnr_cum = ppnr_path.sum()
total_credit_loss_cum = credit_loss_path.sum()
total_trading_cum = trading_path.sum()
total_op_cum = op_path.sum()
total_tax_cum = sum([max(ppnr_path[q] - credit_loss_path[q] - trading_path[q] - op_path[q], 0) * tax_rate for q in range(9)])
total_dividend_cum = sum([max(ppnr_path[q] - credit_loss_path[q] - trading_path[q] - op_path[q] - max(ppnr_path[q] - credit_loss_path[q] - trading_path[q] - op_path[q], 0) * tax_rate, 0) * dividend_payout for q in range(9) if not (suspend_dividends and capital_path_adj[q] < starting_cet1)])

# Waterfall data
waterfall_labels = ["Starting CET1", "PPNR", "Credit Losses", "Trading Losses", "Op Losses", "Taxes", "Dividends", "Mgmt Overlay", "Ending CET1"]
waterfall_values = [starting_cet1, total_ppnr_cum, -total_credit_loss_cum, -total_trading_cum, -total_op_cum, -total_tax_cum, -total_dividend_cum, -management_overlay, 0]
waterfall_measure = ["absolute", "relative", "relative", "relative", "relative", "relative", "relative", "relative", "total"]

fig_waterfall = go.Figure(go.Waterfall(
    x=waterfall_labels,
    y=waterfall_values,
    measure=waterfall_measure,
    text=[f"${v:.1f}B" for v in waterfall_values],
    textposition="outside",
    connector={"line": {"color": "rgb(150,150,150)"}},
    increasing={"marker": {"color": "#2ecc71"}},
    decreasing={"marker": {"color": "#e74c3c"}},
    totals={"marker": {"color": "#636EFA"}}
))

fig_waterfall.update_layout(
    title="9-Quarter Capital Movement",
    yaxis_title="Capital ($B)",
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(size=12),
    showlegend=False
)
fig_waterfall.update_xaxes(showgrid=False)
fig_waterfall.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)')

st.plotly_chart(fig_waterfall, use_container_width=True, config={'displayModeBar': False})
