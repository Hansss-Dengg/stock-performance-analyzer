"""
Caching mechanism for stock data to reduce API calls.
"""
import pandas as pd
import pickle
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Any

from .exceptions import CacheError


class DataCache:
    """Simple file-based cache for stock data."""
    
    def __init__(self, cache_dir: str = ".cache", ttl_hours: int = 24):
        """
        Initialize the cache.
        
        Args:
            cache_dir: Directory to store cache files
            ttl_hours: Time-to-live for cached data in hours (default: 24)
        """
        self.cache_dir = Path(cache_dir)
        self.ttl_hours = ttl_hours
        self._ensure_cache_dir()
    
    def _ensure_cache_dir(self):
        """Create cache directory if it doesn't exist."""
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise CacheError(f"Failed to create cache directory: {str(e)}")
    
    def _generate_key(self, ticker: str, **kwargs) -> str:
        """
        Generate a unique cache key for the given parameters.
        
        Args:
            ticker: Stock ticker symbol
            **kwargs: Additional parameters (start_date, end_date, period, etc.)
        
        Returns:
            MD5 hash of the parameters
        """
        # Sort kwargs to ensure consistent hashing
        params = f"{ticker}_{sorted(kwargs.items())}"
        return hashlib.md5(params.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get the full path for a cache file."""
        return self.cache_dir / f"{cache_key}.pkl"
    
    def _is_expired(self, cache_path: Path) -> bool:
        """Check if cached data has expired."""
        if not cache_path.exists():
            return True
        
        try:
            modified_time = datetime.fromtimestamp(cache_path.stat().st_mtime)
            age = datetime.now() - modified_time
            return age > timedelta(hours=self.ttl_hours)
        except Exception:
            return True
    
    def get(self, ticker: str, **kwargs) -> Optional[pd.DataFrame]:
        """
        Retrieve cached data if available and not expired.
        
        Args:
            ticker: Stock ticker symbol
            **kwargs: Additional parameters used for cache key
        
        Returns:
            Cached DataFrame or None if not found/expired
        """
        cache_key = self._generate_key(ticker, **kwargs)
        cache_path = self._get_cache_path(cache_key)
        
        if self._is_expired(cache_path):
            return None
        
        try:
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            return data
        except Exception as e:
            print(f"Warning: Failed to load cache for {ticker}: {str(e)}")
            return None
    
    def set(self, ticker: str, data: pd.DataFrame, **kwargs):
        """
        Store data in cache.
        
        Args:
            ticker: Stock ticker symbol
            data: DataFrame to cache
            **kwargs: Additional parameters used for cache key
        """
        cache_key = self._generate_key(ticker, **kwargs)
        cache_path = self._get_cache_path(cache_key)
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            raise CacheError(f"Failed to cache data for {ticker}: {str(e)}")
    
    def clear(self, ticker: Optional[str] = None):
        """
        Clear cache files.
        
        Args:
            ticker: If provided, only clear cache for this ticker.
                   If None, clear all cache files.
        """
        try:
            if ticker:
                # Clear all cache files that might be related to this ticker
                # (we can't easily reverse the hash, so we clear all)
                pattern = "*.pkl"
            else:
                pattern = "*.pkl"
            
            for cache_file in self.cache_dir.glob(pattern):
                cache_file.unlink()
        except Exception as e:
            raise CacheError(f"Failed to clear cache: {str(e)}")
    
    def get_cache_info(self) -> dict:
        """
        Get information about the cache.
        
        Returns:
            Dictionary with cache statistics
        """
        try:
            cache_files = list(self.cache_dir.glob("*.pkl"))
            total_size = sum(f.stat().st_size for f in cache_files)
            
            return {
                'num_files': len(cache_files),
                'total_size_mb': round(total_size / (1024 * 1024), 2),
                'cache_dir': str(self.cache_dir),
                'ttl_hours': self.ttl_hours
            }
        except Exception as e:
            raise CacheError(f"Failed to get cache info: {str(e)}")
