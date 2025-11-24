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

```python
from spa.data_fetcher import fetch_stock_data

# Fetch stock data
df = fetch_stock_data('AAPL', period='1y')
print(df.head())
```

## Running Tests

```bash
pytest
```

## Features

- Fetch historical stock data from Yahoo Finance
- Automatic caching (24-hour TTL)
- Retry logic for network errors
- Error handling and validation
