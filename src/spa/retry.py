"""
Retry decorator with exponential backoff for handling transient network errors.
"""
import time
import functools
from typing import Callable, Type, Tuple
import logging

logger = logging.getLogger(__name__)


def retry_on_failure(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,)
) -> Callable:
    """
    Decorator that retries a function with exponential backoff on failure.
    
    Args:
        max_retries: Maximum number of retry attempts (default: 3)
        initial_delay: Initial delay between retries in seconds (default: 1.0)
        backoff_factor: Multiplier for delay after each retry (default: 2.0)
        exceptions: Tuple of exception types to catch and retry (default: all Exception)
    
    Returns:
        Decorated function that retries on failure
    
    Example:
        @retry_on_failure(max_retries=3, initial_delay=1.0, backoff_factor=2.0)
        def fetch_data():
            # This will retry up to 3 times with delays: 1s, 2s, 4s
            return api_call()
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                
                except exceptions as e:
                    last_exception = e
                    
                    # Don't retry on the last attempt
                    if attempt == max_retries:
                        logger.error(
                            f"{func.__name__} failed after {max_retries} retries: {str(e)}"
                        )
                        raise
                    
                    # Log the retry attempt
                    logger.warning(
                        f"{func.__name__} failed (attempt {attempt + 1}/{max_retries}), "
                        f"retrying in {delay:.1f}s: {str(e)}"
                    )
                    
                    # Wait before retrying
                    time.sleep(delay)
                    delay *= backoff_factor
            
            # This should never be reached, but just in case
            if last_exception:
                raise last_exception
        
        return wrapper
    return decorator


def retry_with_timeout(
    max_retries: int = 3,
    timeout_seconds: float = 30.0,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,)
) -> Callable:
    """
    Decorator that retries with both exponential backoff and a total timeout.
    
    Args:
        max_retries: Maximum number of retry attempts
        timeout_seconds: Maximum total time to spend retrying
        initial_delay: Initial delay between retries in seconds
        backoff_factor: Multiplier for delay after each retry
        exceptions: Tuple of exception types to catch and retry
    
    Returns:
        Decorated function that retries with timeout
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            delay = initial_delay
            last_exception = None
            
            for attempt in range(max_retries + 1):
                # Check if we've exceeded the timeout
                elapsed = time.time() - start_time
                if elapsed >= timeout_seconds:
                    logger.error(
                        f"{func.__name__} timed out after {elapsed:.1f}s"
                    )
                    raise TimeoutError(
                        f"Operation timed out after {elapsed:.1f}s"
                    )
                
                try:
                    return func(*args, **kwargs)
                
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        logger.error(
                            f"{func.__name__} failed after {max_retries} retries"
                        )
                        raise
                    
                    logger.warning(
                        f"{func.__name__} failed (attempt {attempt + 1}/{max_retries}), "
                        f"retrying in {delay:.1f}s"
                    )
                    
                    time.sleep(delay)
                    delay *= backoff_factor
            
            if last_exception:
                raise last_exception
        
        return wrapper
    return decorator
