"""
Data fetcher module for retrieving historical stock data.
"""
import yfinance as yf
import pandas as pd
from typing import Optional
from datetime import datetime, timedelta
import re

from .exceptions import InvalidTickerError, DataFetchError, NoDataError
from .cache import DataCache

# Global cache instance
_cache = DataCache()


def _validate_ticker(ticker: str) -> str:
    """
    Validate and clean ticker symbol.
    
    Args:
        ticker: Stock ticker symbol
    
    Returns:
        Cleaned ticker symbol in uppercase
    
    Raises:
        InvalidTickerError: If ticker is invalid
    """
    if not ticker or not isinstance(ticker, str):
        raise InvalidTickerError("Ticker symbol cannot be empty")
    
    ticker = ticker.upper().strip()
    
    # Basic validation: alphanumeric, dots, and hyphens only
    if not re.match(r'^[A-Z0-9.\-]+$', ticker):
        raise InvalidTickerError(
            f"Invalid ticker format: '{ticker}'. "
            "Ticker must contain only letters, numbers, dots, and hyphens."
        )
    
    # Check reasonable length
    if len(ticker) > 10:
        raise InvalidTickerError(f"Ticker symbol too long: '{ticker}'")
    
    return ticker


def fetch_stock_data(
    ticker: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    period: str = "1y",
    use_cache: bool = True
) -> pd.DataFrame:
    """
    Fetch historical stock data from Yahoo Finance with caching.
    
    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'MSFT')
        start_date: Start date in 'YYYY-MM-DD' format (optional)
        end_date: End date in 'YYYY-MM-DD' format (optional)
        period: Time period if dates not specified (e.g., '1y', '6mo', '1mo')
                Valid periods: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
        use_cache: Whether to use cached data (default: True)
    
    Returns:
        DataFrame with columns: Open, High, Low, Close, Volume, Adj Close
        Indexed by date
    
    Raises:
        InvalidTickerError: If ticker format is invalid
        NoDataError: If no data found for the ticker
        DataFetchError: If there's an error fetching data
    """
    # Validate ticker format
    ticker = _validate_ticker(ticker)
    
    # Create cache key parameters
    cache_params = {
        'start_date': start_date,
        'end_date': end_date,
        'period': period
    }
    
    # Try to get from cache first
    if use_cache:
        cached_data = _cache.get(ticker, **cache_params)
        if cached_data is not None:
            return cached_data
    
    try:
        stock = yf.Ticker(ticker)
        
        # Fetch data based on parameters
        if start_date and end_date:
            df = stock.history(start=start_date, end=end_date)
        else:
            df = stock.history(period=period)
        
        if df.empty:
            raise NoDataError(
                f"No data found for ticker '{ticker}'. "
                "The ticker may not exist or may be delisted."
            )
        
        # Additional validation: check if we got real data
        if len(df) < 2:
            raise NoDataError(
                f"Insufficient data for ticker '{ticker}'. "
                "At least 2 data points are required."
            )
        
        # Cache the data
        if use_cache:
            _cache.set(ticker, df, **cache_params)
        
        return df
    
    except (InvalidTickerError, NoDataError):
        # Re-raise our custom exceptions
        raise
    
    except Exception as e:
        raise DataFetchError(f"Error fetching data for '{ticker}': {str(e)}")


def get_stock_info(ticker: str) -> dict:
    """
    Get basic information about a stock.
    
    Args:
        ticker: Stock ticker symbol
    
    Returns:
        Dictionary containing stock information (name, sector, industry, etc.)
    
    Raises:
        InvalidTickerError: If ticker format is invalid
        DataFetchError: If there's an error fetching info
    """
    ticker = _validate_ticker(ticker)
    
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Check if we got valid info
        if not info or 'symbol' not in info:
            raise NoDataError(f"No information found for ticker '{ticker}'")
        
        return {
            'symbol': ticker,
            'name': info.get('longName', 'N/A'),
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),
            'market_cap': info.get('marketCap', 0),
            'currency': info.get('currency', 'USD')
        }
    
    except (InvalidTickerError, NoDataError):
        raise
    
    except Exception as e:
        raise DataFetchError(f"Error fetching info for '{ticker}': {str(e)}")


def clear_cache(ticker: Optional[str] = None):
    """
    Clear the data cache.
    
    Args:
        ticker: If provided, clear cache for specific ticker (clears all if None)
    """
    _cache.clear(ticker)


def get_cache_info() -> dict:
    """
    Get information about the cache.
    
    Returns:
        Dictionary with cache statistics
    """
    return _cache.get_cache_info()


def fetch_multiple_stocks(
    tickers: list[str],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    period: str = "1y",
    skip_errors: bool = True,
    use_cache: bool = True
) -> dict[str, pd.DataFrame]:
    """
    Fetch historical data for multiple stocks.
    
    Args:
        tickers: List of stock ticker symbols
        start_date: Start date in 'YYYY-MM-DD' format (optional)
        end_date: End date in 'YYYY-MM-DD' format (optional)
        period: Time period if dates not specified
        skip_errors: If True, skip tickers with errors; if False, raise on first error
        use_cache: Whether to use cached data (default: True)
    
    Returns:
        Dictionary mapping ticker symbols to their DataFrames
    
    Raises:
        InvalidTickerError, NoDataError, DataFetchError: If skip_errors is False
    """
    if not tickers:
        raise InvalidTickerError("Ticker list cannot be empty")
    
    results = {}
    errors = {}
    
    for ticker in tickers:
        try:
            results[ticker] = fetch_stock_data(
                ticker, start_date, end_date, period, use_cache
            )
        except (InvalidTickerError, NoDataError, DataFetchError) as e:
            if skip_errors:
                errors[ticker] = str(e)
                print(f"Warning: Skipping {ticker} - {str(e)}")
            else:
                raise
    
    if errors and not results:
        raise DataFetchError(f"Failed to fetch data for all tickers: {errors}")
    
    return results
