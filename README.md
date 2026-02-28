# Capital Stress Testing Engine

An AI-powered credit risk portfolio monitoring dashboard built with Streamlit, featuring real-time analytics, stress testing, scenario analysis, and executive-level risk commentary.

**Live Demo:** [https://portfolio-analytics-ai.streamlit.app/](https://portfolio-analytics-ai.streamlit.app/)

## Features

### Portfolio Overview
- **Real-time Metrics**: Interactive tooltips for total exposure, default rates, credit scores, and model performance
- **Concentration Analysis**: Interactive pie charts for industry and regional exposures with subtle soft backgrounds
- **Top Loans**: Top 10 riskiest loans with predictive default probabilities and progress indicators

### Risk Analysis
- **Stress Testing**: Dynamic PD multiplier slider (1.0x - 3.0x) with base vs stressed loss comparison and industry-level impact visualization
- **Risk Engine Analytics**: Expected loss calculations, loss ratios, risk bucketing (Low/Medium/High), and concentration alerts with adjustable thresholds

### Stress Testing (Detailed)
- Advanced stress testing scenarios with comprehensive impact analysis
- Industry-level stress propagation
- Visual comparison of base vs stressed portfolios with soft gray chart backgrounds

### Scenario Analysis
- **Recession**: Adjustable multiplier (1.0x-4.0x PD multiplier) with comparison charts
- **Interest Rate Shock**: Adjustable multiplier (1.0x-3.0x PD multiplier) with impact visualization
- **Industry-Specific Crash**: Select industry and multiplier (1.0x-5.0x) with top 5 impact chart

### Capital Stress Engine
- **9-Quarter Projection**: Forward-looking capital adequacy analysis
- **Advanced Controls**: Tax rates, dividend policies, management overlays
- **Regulatory Buffers**: CCB and G-SIB buffer monitoring with visual indicators
- **Capital Waterfall**: Cumulative capital movement breakdown
- **Breach Detection**: Automatic alerts for regulatory minimum violations

### AI Risk Commentary
- **Executive Summaries**: AI-powered risk analysis powered by Llama 3.2 via Hugging Face
- **Interactive Q&A**: Ask follow-up questions with comprehensive portfolio context (activated via AI Assistant button)
- **Context-Aware**: Includes Expected Loss, Loss Ratio, delinquency rates, top industries/regions, and riskiest loans

### Additional Features
- **Interactive Tooltips**: Hover tooltips on all key financial terminologies explaining concepts
- **Info Expanders**: Expandable sections explaining calculation logic for each module
- **Data Generation**: Built-in synthetic loan data generator with realistic distributions
- **Sidebar Navigation**: Organized section mapping for easy navigation between analysis modules
- **Space-Efficient Design**: Clean, professional interface optimized for readability with subtle chart backgrounds

## Tech Stack

- **Frontend**: Streamlit
- **ML/Analytics**: scikit-learn, pandas, numpy
- **Visualization**: Plotly Express (interactive charts with custom hover templates)
- **AI**: Hugging Face Inference API (Llama 3.2-3B-Instruct)
- **Caching**: Streamlit cache_data and cache_resource for optimized performance

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd "Capital Stress Testing Engine"
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up Hugging Face API token:
   - Create `.streamlit/secrets.toml` file
   - Add your token:
```toml
HF_TOKEN = "your_huggingface_token_here"
```

## Usage

1. Generate synthetic data (first time):
```bash
python data_generator.py
```

2. Run the Streamlit app:
```bash
streamlit run app.py
```

3. Access the dashboard at `http://localhost:8501`

## Key Components

### Data Generator (`data_generator.py`)
- Generates 100k synthetic loan records
- Weighted industry/region distributions for realistic concentration
- HistGradientBoosting classifier for default probability prediction

### Main Dashboard (`app.py`)
- Portfolio metrics calculation with caching
- ML model training (HistGradientBoostingClassifier) with cache_resource
- Interactive visualizations with Plotly
- AI-powered risk analysis with comprehensive context
- Custom CSS for tooltips and styling
- Info expanders for calculation methodology
- Modular section-based navigation

### Section Mapping (`app_sections.py`)
- Organized navigation structure
- Section groupings: Portfolio Overview, Risk Analysis, Stress Testing, Scenario Analysis, Capital Stress Engine

### AI Functions
- Executive summary generation with portfolio context
- Interactive Q&A with chat history
- Context-aware responses including risk metrics, concentration alerts, stress scenarios, and capital stress engine projections

## Features in Detail

### Portfolio Metrics
- Total Portfolio Value with loan count
- Average Loan Size with interest rate
- Default Rate with delinquency percentage
- Average Credit Score with DTI ratio
- Model AUC Score with top industry
- All metrics include hover tooltips explaining concepts

### AI Assistant
- Generate executive risk summaries with portfolio data analysis
- Ask follow-up questions with comprehensive context including:
  - Expected Loss & Loss Ratio
  - Delinquency rates (30+ days)
  - Top 5 industries and top 3 regions by exposure
  - Top 10 riskiest loans
  - Capital Stress Engine metrics (CET1 ratios, PPNR, credit losses, regulatory buffers)
- Context-aware responses with portfolio data
- Chat history maintained in session state

### Stress Testing
- Adjustable PD multiplier slider
- Base vs. stressed loss comparison with delta indicators
- Industry-level impact analysis with grouped bar charts
- Visual comparison charts with custom hover templates

### Risk Engine Analytics
- Adjustable LGD assumption slider (30%-80%)
- Adjustable concentration threshold slider (15%-50%)
- Adjustable risk bucket thresholds:
  - Low Risk: 1%-10% (default 5%)
  - Medium Risk: 6%-20% (default 12%)
- Risk distribution visualization
- Concentration alerts for industries exceeding threshold

### Scenario Analysis
- **Recession**: Adjustable multiplier with comparison charts
- **Interest Rate Shock**: Adjustable multiplier with impact visualization
- **Industry Crash**: Select industry and multiplier with top 5 impact chart

### Capital Stress Engine
- **Capital Ratios**: CET1 (11.5%), Tier 1 (13.0%), Total Capital (15.5%)
- **Regulatory Thresholds**: Minimum requirements with buffer zones
- **RWA Distribution**: Risk-weighted assets by Low/Medium/High risk buckets
- **Stress Scenarios**: Capital impact under recession, rate shock, and industry crash
- **Buffer Analysis**: CCB (2.5%), CCyB (0-2.5%), G-SIB (0-3.5%)

## Configuration

Edit `data_generator.py` to customize:
- Number of loans (`N = 100000`)
- Industries and regions
- Concentration weights
- Default probability logic

## Requirements

See `requirements.txt` for full dependencies:
- streamlit>=1.28.0
- pandas
- numpy
- scikit-learn
- plotly
- huggingface_hub
- requests

## Performance Optimizations

- `@st.cache_data` for data loading
- `@st.cache_resource` for model training
- Efficient data aggregations
- Optimized chart rendering

## Calculation Logic

### Expected Loss Calculation
```
Expected Loss (EL) = PD × LGD × EAD
```
- **PD (Probability of Default)**: Predicted by HistGradientBoostingClassifier
- **LGD (Loss Given Default)**: User-adjustable slider (30%-80%)
- **EAD (Exposure at Default)**: Loan amount
- **Loss Ratio**: EL / Total Portfolio Value

### Risk Bucketing
- **Low Risk**: PD < Low Risk Threshold (adjustable slider 1%-10%)
- **Medium Risk**: Low Risk Threshold ≤ PD < Medium Risk Threshold (adjustable slider 6%-20%)
- **High Risk**: PD ≥ Medium Risk Threshold

### Stress Testing
```
Stressed PD = min(Base PD × Multiplier, 0.99)
Stressed Loss = Stressed PD × LGD × Loan Amount
```
- **Multipliers**: User-adjustable sliders (1.0x-3.0x general, 1.0x-4.0x recession, 1.0x-5.0x industry crash)

### Capital Stress Engine (9-Quarter Projection)

**Starting Position:**
```
CET1 Ratio = CET1 Capital / RWA × 100
```

**Quarterly Capital Movement:**
```
For each quarter q:
  Capital += PPNR[q] - Credit Losses[q] - Trading Losses[q] - Op Losses[q]
  Tax = Capital × Tax Rate (slider)
  Capital -= Tax
  Dividend = Dividend Payout (slider, suspended if checkbox enabled and capital < starting level)
  Capital -= Dividend
```

**PPNR Distribution:**
- Declines over stress period based on PPNR Decline Rate slider
- Shape: [1.0, 1.0-0.5d, 1.0-d, 1.0-d, 1.0-0.8d, 1.0-0.6d, 1.0-0.4d, 1.0-0.2d, 1.0]
- Where d = PPNR Decline Rate slider value

**Credit Loss Distribution:**
- Front-loaded using linear interpolation (0.05 to 0.15 weights)
- Normalized to sum to total 9Q credit losses from slider

**RWA Inflation:**
```
RWA[q] = Starting RWA × (1 + Cumulative Loss % × RWA Inflation % slider)
```

**Regulatory Buffers:**
- **Minimum CET1**: User input (Regulatory Minimum CET1 %)
- **CCB (Capital Conservation Buffer)**: Adjustable slider (0-5%)
- **G-SIB Buffer**: Adjustable slider (0-3.5%)
- **Total Requirement**: Minimum + CCB + G-SIB

**Stress Capital Buffer (SCB):**
```
SCB = Starting CET1 Ratio - Minimum CET1 Ratio (9Q Low)
```
