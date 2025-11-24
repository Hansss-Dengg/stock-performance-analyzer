"""
Unit tests for the data_fetcher module.
"""
import pytest
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from spa.data_fetcher import (
    fetch_stock_data,
    get_stock_info,
    fetch_multiple_stocks,
    clear_cache,
    get_cache_info,
    _validate_ticker
)
from spa.exceptions import InvalidTickerError, NoDataError, DataFetchError


class TestValidateTicker:
    """Tests for ticker validation."""
    
    def test_valid_ticker(self):
        """Test that valid tickers are accepted."""
        assert _validate_ticker("AAPL") == "AAPL"
        assert _validate_ticker("msft") == "MSFT"
        assert _validate_ticker("brk.b") == "BRK.B"
        assert _validate_ticker("brk-b") == "BRK-B"
    
    def test_empty_ticker(self):
        """Test that empty tickers raise InvalidTickerError."""
        with pytest.raises(InvalidTickerError, match="cannot be empty"):
            _validate_ticker("")
        
        with pytest.raises(InvalidTickerError, match="cannot be empty"):
            _validate_ticker(None)
    
    def test_invalid_characters(self):
        """Test that tickers with invalid characters are rejected."""
        with pytest.raises(InvalidTickerError, match="Invalid ticker format"):
            _validate_ticker("AAPL@")
        
        with pytest.raises(InvalidTickerError, match="Invalid ticker format"):
            _validate_ticker("AAPL#123")
        
        with pytest.raises(InvalidTickerError, match="Invalid ticker format"):
            _validate_ticker("AAPL$")
    
    def test_ticker_too_long(self):
        """Test that overly long tickers are rejected."""
        with pytest.raises(InvalidTickerError, match="too long"):
            _validate_ticker("VERYLONGTICKER123")
    
    def test_whitespace_handling(self):
        """Test that whitespace is stripped."""
        assert _validate_ticker("  AAPL  ") == "AAPL"
        assert _validate_ticker("\tMSFT\n") == "MSFT"


class TestFetchStockData:
    """Tests for fetch_stock_data function."""
    
    @patch('spa.data_fetcher._fetch_from_api')
    def test_fetch_valid_ticker(self, mock_fetch):
        """Test fetching data for a valid ticker."""
        # Create mock DataFrame
        mock_df = pd.DataFrame({
            'Open': [100, 101],
            'High': [102, 103],
            'Low': [99, 100],
            'Close': [101, 102],
            'Volume': [1000000, 1100000]
        })
        mock_fetch.return_value = mock_df
        
        # Clear cache to ensure fresh fetch
        clear_cache()
        
        # Fetch data
        result = fetch_stock_data('AAPL', period='1mo', use_cache=False)
        
        # Verify
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        mock_fetch.assert_called_once()
    
    def test_fetch_invalid_ticker(self):
        """Test that invalid tickers raise InvalidTickerError."""
        with pytest.raises(InvalidTickerError):
            fetch_stock_data('INVALID@TICKER')
    
    @patch('spa.data_fetcher._fetch_from_api')
    def test_fetch_no_data(self, mock_fetch):
        """Test handling of tickers with no data."""
        # Mock empty DataFrame
        mock_fetch.return_value = pd.DataFrame()
        
        with pytest.raises(NoDataError, match="No data found"):
            fetch_stock_data('NOTREAL', use_cache=False)
    
    @patch('spa.data_fetcher._fetch_from_api')
    def test_fetch_insufficient_data(self, mock_fetch):
        """Test handling of tickers with insufficient data."""
        # Mock DataFrame with only 1 row
        mock_df = pd.DataFrame({'Close': [100]})
        mock_fetch.return_value = mock_df
        
        with pytest.raises(NoDataError, match="Insufficient data"):
            fetch_stock_data('AAPL', use_cache=False)
    
    @patch('spa.data_fetcher._fetch_from_api')
    def test_caching_enabled(self, mock_fetch):
        """Test that caching works when enabled."""
        mock_df = pd.DataFrame({
            'Close': [100, 101, 102]
        })
        mock_fetch.return_value = mock_df
        
        # Clear cache
        clear_cache()
        
        # First call - should fetch from API
        result1 = fetch_stock_data('AAPL', period='1mo', use_cache=True)
        assert mock_fetch.call_count == 1
        
        # Second call - should use cache
        result2 = fetch_stock_data('AAPL', period='1mo', use_cache=True)
        assert mock_fetch.call_count == 1  # Still 1, not called again
        
        # Verify results are the same
        pd.testing.assert_frame_equal(result1, result2)
    
    @patch('spa.data_fetcher._fetch_from_api')
    def test_caching_disabled(self, mock_fetch):
        """Test that caching can be disabled."""
        mock_df = pd.DataFrame({
            'Close': [100, 101, 102]
        })
        mock_fetch.return_value = mock_df
        
        # Both calls should fetch from API
        fetch_stock_data('AAPL', period='1mo', use_cache=False)
        fetch_stock_data('AAPL', period='1mo', use_cache=False)
        
        assert mock_fetch.call_count == 2
    
    @patch('spa.data_fetcher._fetch_from_api')
    def test_different_periods_different_cache(self, mock_fetch):
        """Test that different periods use different cache entries."""
        mock_df = pd.DataFrame({
            'Close': [100, 101, 102]
        })
        mock_fetch.return_value = mock_df
        
        clear_cache()
        
        # Fetch with different periods
        fetch_stock_data('AAPL', period='1mo', use_cache=True)
        fetch_stock_data('AAPL', period='1y', use_cache=True)
        
        # Should call API twice (different cache keys)
        assert mock_fetch.call_count == 2


class TestGetStockInfo:
    """Tests for get_stock_info function."""
    
    @patch('spa.data_fetcher.yf.Ticker')
    def test_get_valid_stock_info(self, mock_ticker):
        """Test getting info for a valid stock."""
        mock_info = {
            'symbol': 'AAPL',
            'longName': 'Apple Inc.',
            'sector': 'Technology',
            'industry': 'Consumer Electronics',
            'marketCap': 3000000000000,
            'currency': 'USD'
        }
        mock_ticker.return_value.info = mock_info
        
        result = get_stock_info('AAPL')
        
        assert result['symbol'] == 'AAPL'
        assert result['name'] == 'Apple Inc.'
        assert result['sector'] == 'Technology'
        assert result['industry'] == 'Consumer Electronics'
    
    @patch('spa.data_fetcher.yf.Ticker')
    def test_get_info_missing_fields(self, mock_ticker):
        """Test handling of missing info fields."""
        mock_info = {'symbol': 'AAPL'}
        mock_ticker.return_value.info = mock_info
        
        result = get_stock_info('AAPL')
        
        assert result['name'] == 'N/A'
        assert result['sector'] == 'N/A'
        assert result['market_cap'] == 0
    
    def test_get_info_invalid_ticker(self):
        """Test that invalid tickers raise InvalidTickerError."""
        with pytest.raises(InvalidTickerError):
            get_stock_info('INVALID@')


class TestFetchMultipleStocks:
    """Tests for fetch_multiple_stocks function."""
    
    @patch('spa.data_fetcher._fetch_from_api')
    def test_fetch_multiple_valid_tickers(self, mock_fetch):
        """Test fetching data for multiple valid tickers."""
        mock_df = pd.DataFrame({
            'Close': [100, 101, 102]
        })
        mock_fetch.return_value = mock_df
        
        clear_cache()
        
        tickers = ['AAPL', 'MSFT', 'GOOGL']
        result = fetch_multiple_stocks(tickers, period='1mo', use_cache=False)
        
        assert len(result) == 3
        assert 'AAPL' in result
        assert 'MSFT' in result
        assert 'GOOGL' in result
    
    def test_fetch_empty_ticker_list(self):
        """Test that empty ticker list raises error."""
        with pytest.raises(InvalidTickerError, match="cannot be empty"):
            fetch_multiple_stocks([])
    
    @patch('spa.data_fetcher._fetch_from_api')
    def test_fetch_multiple_with_errors_skip(self, mock_fetch):
        """Test that errors can be skipped when fetching multiple stocks."""
        def side_effect(ticker, *args):
            if ticker == 'INVALID':
                return pd.DataFrame()  # Empty = NoDataError
            return pd.DataFrame({'Close': [100, 101, 102]})
        
        mock_fetch.side_effect = side_effect
        
        clear_cache()
        
        tickers = ['AAPL', 'INVALID', 'MSFT']
        result = fetch_multiple_stocks(
            tickers, 
            period='1mo', 
            skip_errors=True, 
            use_cache=False
        )
        
        # Should have 2 successful results, skipping INVALID
        assert len(result) == 2
        assert 'AAPL' in result
        assert 'MSFT' in result
        assert 'INVALID' not in result


class TestCacheFunctions:
    """Tests for cache management functions."""
    
    def test_clear_cache(self):
        """Test clearing the cache."""
        # This should not raise any errors
        clear_cache()
        clear_cache('AAPL')
    
    def test_get_cache_info(self):
        """Test getting cache information."""
        info = get_cache_info()
        
        assert 'num_files' in info
        assert 'total_size_mb' in info
        assert 'cache_dir' in info
        assert 'ttl_hours' in info
        assert info['ttl_hours'] == 24


class TestRetryLogic:
    """Tests for retry functionality."""
    
    @patch('spa.data_fetcher.yf.Ticker')
    def test_retry_on_transient_error(self, mock_ticker):
        """Test that transient errors are retried."""
        # First call fails, second succeeds
        mock_history = Mock()
        mock_history.side_effect = [
            ConnectionError("Network error"),
            pd.DataFrame({'Close': [100, 101, 102]})
        ]
        mock_ticker.return_value.history = mock_history
        
        clear_cache()
        
        # Should succeed after retry
        result = fetch_stock_data('AAPL', period='1mo', use_cache=False)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        # Verify it was called twice (initial + 1 retry)
        assert mock_history.call_count == 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
