# Stock Performance Analyzer

Python app to fetch historical stock prices, compute returns, volatility, drawdown, and visualize results.

## Tech Stack
- Python, pandas, yfinance, plotly
- Streamlit (optional)
- Yahoo Finance API

## Installation

```bash
# Clone repository
git clone https://github.com/Hansss-Dengg/stock-performance-analyzer.git
cd stock-performance-analyzer

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Web Application (Streamlit)

Launch the interactive web dashboard:

```bash
python run_app.py
```

The app will open in your browser at http://localhost:8501

### Python API

```python
from spa.data_fetcher import fetch_stock_data
from spa.data_processor import calculate_comprehensive_analysis
from spa.visualizer import create_price_chart

# Fetch stock data
df = fetch_stock_data('AAPL', period='1y')

# Analyze performance
analysis = calculate_comprehensive_analysis(df)
print(f"Total Return: {analysis['returns']['total_return']*100:.2f}%")
print(f"Sharpe Ratio: {analysis['ratios']['sharpe_ratio']:.2f}")

# Create interactive chart
fig = create_price_chart(df, ticker='AAPL')
fig.show()
```

## Running Tests

```bash
pytest
```

## Features

### Data Management
- Fetch historical stock data from Yahoo Finance
- Automatic caching with 24-hour TTL (90% API call reduction)
- Retry logic with exponential backoff for network errors
- Comprehensive error handling and validation

### Financial Metrics (30+ calculations)
- Returns: daily, cumulative, annualized
- Volatility: standard, rolling, downside
- Drawdown: current, maximum, recovery analysis
- Risk ratios: Sharpe, Sortino, Calmar
- Moving averages: SMA, EMA with golden/death cross detection

### Visualizations (6+ interactive charts)
- Candlestick price charts with volume
- Daily and cumulative returns
- Rolling volatility analysis
- Drawdown tracking with max DD highlighting
- Moving average overlays with crossover markers
- Multi-stock comparison (normalized price, returns, volatility)

### Web Dashboard
- Interactive Streamlit interface
- Real-time stock data fetching
- Multiple analysis pages
- Customizable date ranges and parameters
- Multi-stock comparison tool
