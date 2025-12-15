# Stock Performance Analyzer

A financial analysis web application that provides comprehensive stock performance metrics, interactive visualizations, and multi-stock comparisons. Track returns, volatility, drawdowns, and technical indicators with real-time data from Yahoo Finance.

## Features

**Data Management:** Integrates with Yahoo Finance API for real-time stock data (no API key required). Implements intelligent caching with 24-hour TTL to reduce API calls by 90%. Automatic retry logic with exponential backoff ensures reliable data fetching.

**Financial Analysis:** Calculate 30+ financial metrics including daily/cumulative/annualized returns, standard and rolling volatility, maximum drawdown analysis, and performance ratios (Sharpe, Sortino, Calmar). Advanced technical indicators with moving averages and golden/death cross detection.

**Interactive Visualizations:** View your stock analysis with 6 interactive chart types - candlestick price charts with volume, daily and cumulative returns, rolling volatility, drawdown tracking, moving average overlays, and multi-stock comparison.

**Export Capabilities:** Download your analysis in multiple formats including CSV for raw data, JSON for metrics, HTML for interactive charts, and formatted text reports.

---

## 🔧 Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)

---

## 🚀 Run Locally

Clone the project

```bash
git clone https://github.com/Hansss-Dengg/Stock-Metrics-Analyzer.git
```

Navigate to the project directory

```bash
cd Stock-Metrics-Analyzer
```

Create and activate virtual environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

Start the application

```bash
python run_app.py
```

The app will open in your browser at `http://localhost:8501`

---

## 📊 Utilize the Backend API Separately

You can create your own front-end to interface with user analysis and data as they're implemented through a backend RESTful API.

### Core Modules:

#### `data_fetcher`
- Fetches stock data from Yahoo Finance (no API key required)
- Automatic caching with 24-hour TTL
- Retry logic with exponential backoff

#### `data_processor`
- Comprehensive financial analysis (30+ metrics)
- Returns, volatility, drawdown calculations
- Risk ratios (Sharpe, Sortino, Calmar)

#### `visualizer`
- Creates 6+ interactive Plotly charts
- Candlestick, returns, volatility, drawdown charts
- Multi-stock comparison visualizations

#### `data_cleaner`
- Input validation and data normalization
- Handles missing data and outliers
- Ensures data quality for analysis

---

## 💻 Python API Usage

```python
from spa.data_fetcher import fetch_stock_data
from spa.data_processor import get_comprehensive_analysis
from spa.visualizer import create_price_chart

# Fetch stock data
df = fetch_stock_data('AAPL', start_date='2024-01-01', end_date='2024-12-31')

# Analyze performance
analysis = get_comprehensive_analysis(df)
print(f"Total Return: {analysis['returns']['total_return']*100:.2f}%")
print(f"Sharpe Ratio: {analysis['ratios']['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {analysis['drawdown']['max_drawdown']*100:.2f}%")

# Create interactive chart
fig = create_price_chart(df, ticker='AAPL')
fig.show()
```

---

## 🧪 Running Tests

```bash
pytest tests/
```

Run with coverage

```bash
pytest --cov=src/spa tests/
```

---

## 📁 Project Structure

```
Stock-Metrics-Analyzer/
├── src/
│   └── spa/
│       ├── __init__.py
│       ├── app.py              # Main Streamlit application
│       ├── data_fetcher.py     # Yahoo Finance API integration
│       ├── data_processor.py   # Financial metrics calculations (30+ metrics)
│       ├── data_cleaner.py     # Data validation and normalization
│       ├── visualizer.py       # Interactive Plotly charts
│       ├── cache.py           # File-based caching system
│       ├── retry.py           # Retry logic with exponential backoff
│       └── exceptions.py      # Custom exception hierarchy
├── tests/
│   └── test_data_processor.py # Unit tests (33 tests, 100% pass rate)
├── requirements.txt          # Python dependencies
├── run_app.py               # Local development launcher
├── .gitignore
└── README.md
```

---

## ✨ Key Features

### 💰 Data Management
- Fetch historical stock data from Yahoo Finance API
- Intelligent file-based caching with 24-hour TTL (reduces API calls by 90%)
- Automatic retry logic with exponential backoff for network resilience
- Comprehensive input validation and error handling

### 📈 Financial Metrics (30+ Calculations)
- **Returns:** Daily, cumulative, and annualized returns
- **Volatility:** Standard deviation, rolling volatility, downside volatility
- **Drawdown:** Current drawdown, maximum drawdown, recovery analysis
- **Risk Ratios:** Sharpe ratio, Sortino ratio, Calmar ratio
- **Technical Indicators:** SMA, EMA with golden/death cross detection

### 📊 Interactive Visualizations (6+ Chart Types)
- **Candlestick Charts:** Price action with volume overlay
- **Returns Analysis:** Daily and cumulative return visualization
- **Volatility Charts:** Rolling volatility with statistical bands
- **Drawdown Tracking:** Underwater plots with maximum drawdown highlighting
- **Moving Average Overlays:** Technical analysis with crossover markers
- **Multi-Stock Comparison:** Normalized price, returns, and volatility comparison

### 🌐 Web Dashboard
- Intuitive Streamlit interface with 7 analysis pages
- Real-time data fetching with progress indicators
- Customizable date ranges and analysis parameters
- Multi-stock comparison tool for portfolio analysis
- Export functionality (CSV, JSON, HTML, TXT)
- Session state tracking and performance optimizations
- Responsive design with custom CSS styling

---


