# Portfolio Monitoring Agent

An AI-powered credit risk portfolio monitoring dashboard built with Streamlit, featuring real-time analytics, stress testing, and executive-level risk commentary.

## Features

- **Portfolio Overview**: Real-time metrics including total exposure, default rates, credit scores, and model performance
- **AI Risk Commentary**: Executive summaries and interactive Q&A powered by Llama 3.2 via Hugging Face
- **Concentration Analysis**: Interactive visualizations of industry and regional exposures
- **Risk Assessment**: Top 10 riskiest loans with predictive default probabilities
- **Stress Testing**: Dynamic scenario analysis with adjustable PD multipliers (1.0x - 3.0x)
- **Data Generation**: Built-in synthetic loan data generator with realistic distributions

## Tech Stack

- **Frontend**: Streamlit
- **ML/Analytics**: scikit-learn, pandas, numpy
- **Visualization**: Plotly Express
- **AI**: Hugging Face Inference API (Llama 3.2-3B-Instruct)

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
- Portfolio metrics calculation
- ML model training (HistGradientBoostingClassifier)
- Interactive visualizations
- AI-powered risk analysis

### Stress Testing (`stress_testing.py`)
- Recession scenarios
- Interest rate shock analysis
- Industry-specific stress tests

## Features in Detail

### Portfolio Metrics
- Total Portfolio Value
- Average Loan Size & Interest Rate
- Default & Delinquency Rates
- Average Credit Score & DTI
- Model AUC Score

### AI Assistant
- Generate executive risk summaries
- Ask follow-up questions about portfolio
- Context-aware responses with portfolio data

### Stress Testing
- Adjustable PD multiplier (slider)
- Base vs. stressed loss comparison
- Industry-level impact analysis
- Visual comparison charts

## Configuration

Edit `data_generator.py` to customize:
- Number of loans (`N = 100000`)
- Industries and regions
- Concentration weights
- Default probability logic

## Requirements

See `requirements.txt` for full dependencies:
- streamlit
- pandas
- numpy
- scikit-learn
- plotly
- huggingface_hub
- requests
