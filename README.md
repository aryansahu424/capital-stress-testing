# Portfolio Monitoring Agent

An AI-powered credit risk portfolio monitoring dashboard built with Streamlit, featuring real-time analytics, stress testing, scenario analysis, and executive-level risk commentary.

## Features

- **Portfolio Overview**: Real-time metrics with interactive tooltips including total exposure, default rates, credit scores, and model performance
- **AI Risk Commentary**: Executive summaries and interactive Q&A powered by Llama 3.2 via Hugging Face with comprehensive portfolio context
- **Concentration Analysis**: Interactive pie charts for industry and regional exposures with clean hover tooltips
- **Risk Assessment**: Top 10 riskiest loans with predictive default probabilities and progress indicators
- **Stress Testing**: Dynamic PD multiplier slider (1.0x - 3.0x) with base vs stressed loss comparison and industry-level impact visualization
- **Risk Engine Analytics**: Expected loss calculations, loss ratios, risk bucketing (Low/Medium/High), and concentration alerts with adjustable thresholds
- **Scenario Analysis**: Three scenario tabs with adjustable sliders:
  - Recession (1.0x-4.0x PD multiplier)
  - Interest Rate Shock (1.0x-3.0x PD multiplier)
  - Industry-Specific Crash (1.0x-5.0x PD multiplier with industry selector)
- **Interactive Tooltips**: Hover tooltips on all key financial terminologies explaining concepts
- **Data Generation**: Built-in synthetic loan data generator with realistic distributions

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
cd "Portfolio Monitoring Agent"
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
- Logistic regression-based default probability modeling

### Main Dashboard (`app.py`)
- Portfolio metrics calculation with caching
- ML model training (HistGradientBoostingClassifier) with cache_resource
- Interactive visualizations with Plotly
- AI-powered risk analysis with comprehensive context
- Custom CSS for tooltips and styling

### Risk Engine (`risk_engine.py`)
- Expected loss calculations
- Risk bucketing logic
- Concentration analysis

### Stress Testing (`stress_testing.py`)
- Recession scenarios
- Interest rate shock analysis
- Industry-specific stress tests

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
