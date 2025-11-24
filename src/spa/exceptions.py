"""
Custom exceptions for the stock performance analyzer.
"""


class StockDataError(Exception):
    """Base exception for stock data related errors."""
    pass


class InvalidTickerError(StockDataError):
    """Raised when an invalid ticker symbol is provided."""
    pass


class DataFetchError(StockDataError):
    """Raised when there's an error fetching data from the API."""
    pass


class NoDataError(StockDataError):
    """Raised when no data is available for the requested period."""
    pass


class CacheError(Exception):
    """Raised when there's an error with the caching mechanism."""
    pass
