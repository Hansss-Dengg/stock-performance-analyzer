"""
Data processing module for calculating financial metrics and returns.
"""
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


def calculate_daily_returns(
    df: pd.DataFrame,
    price_column: str = 'Close',
    method: str = 'simple'
) -> pd.Series:
    """
    Calculate daily returns from price data.
    
    Args:
        df: Input DataFrame with price data
        price_column: Column to use for return calculation (default: 'Close')
        method: 'simple' for arithmetic returns or 'log' for logarithmic returns
    
    Returns:
        Series with daily returns
    
    Raises:
        ValueError: If price column not found or invalid method
    """
    if df.empty:
        return pd.Series(dtype=float)
    
    if price_column not in df.columns:
        raise ValueError(f"Column '{price_column}' not found in DataFrame")
    
    prices = df[price_column]
    
    if method == 'simple':
        # Simple arithmetic returns: (P_t - P_t-1) / P_t-1
        returns = prices.pct_change()
    
    elif method == 'log':
        # Logarithmic returns: ln(P_t / P_t-1)
        returns = np.log(prices / prices.shift(1))
    
    else:
        raise ValueError(f"Invalid method: {method}. Use 'simple' or 'log'")
    
    # Drop the first NaN value (no previous price for first day)
    returns = returns.dropna()
    
    logger.info(f"Calculated {len(returns)} daily returns using {method} method")
    
    return returns


def calculate_return_statistics(returns: pd.Series) -> Dict[str, float]:
    """
    Calculate statistical metrics for returns.
    
    Args:
        returns: Series of returns
    
    Returns:
        Dictionary with return statistics
    """
    if returns.empty:
        return {}
    
    stats = {
        'mean': float(returns.mean()),
        'std': float(returns.std()),
        'min': float(returns.min()),
        'max': float(returns.max()),
        'median': float(returns.median()),
        'skewness': float(returns.skew()),
        'kurtosis': float(returns.kurtosis()),
        'count': len(returns)
    }
    
    return stats


def calculate_percentage_returns(
    df: pd.DataFrame,
    price_column: str = 'Close'
) -> pd.Series:
    """
    Calculate daily returns as percentages.
    
    Args:
        df: Input DataFrame with price data
        price_column: Column to use for calculation
    
    Returns:
        Series with percentage returns (e.g., 0.05 = 5%)
    """
    returns = calculate_daily_returns(df, price_column, method='simple')
    return returns * 100  # Convert to percentage


def calculate_period_return(
    df: pd.DataFrame,
    price_column: str = 'Close',
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> float:
    """
    Calculate return over a specific period.
    
    Args:
        df: Input DataFrame with price data
        price_column: Column to use for calculation
        start_date: Start date (if None, uses first date)
        end_date: End date (if None, uses last date)
    
    Returns:
        Period return as decimal (e.g., 0.45 = 45%)
    """
    if df.empty:
        return 0.0
    
    df_period = df.copy()
    
    # Filter by date range if specified
    if start_date:
        df_period = df_period[df_period.index >= start_date]
    if end_date:
        df_period = df_period[df_period.index <= end_date]
    
    if len(df_period) < 2:
        return 0.0
    
    start_price = df_period[price_column].iloc[0]
    end_price = df_period[price_column].iloc[-1]
    
    return (end_price - start_price) / start_price


def calculate_cumulative_returns(
    df: pd.DataFrame,
    price_column: str = 'Close',
    method: str = 'simple'
) -> pd.Series:
    """
    Calculate cumulative returns over time.
    
    Shows the total return if you bought at the start and held until each point.
    
    Args:
        df: Input DataFrame with price data
        price_column: Column to use for calculation
        method: 'simple' or 'compound' for calculation method
    
    Returns:
        Series with cumulative returns
    """
    if df.empty:
        return pd.Series(dtype=float)
    
    if price_column not in df.columns:
        raise ValueError(f"Column '{price_column}' not found in DataFrame")
    
    prices = df[price_column]
    
    if method == 'simple':
        # Simple cumulative return from start
        start_price = prices.iloc[0]
        cumulative = (prices - start_price) / start_price
    
    elif method == 'compound':
        # Compound returns: (1 + r1) * (1 + r2) * ... - 1
        daily_returns = calculate_daily_returns(df, price_column, method='simple')
        cumulative = (1 + daily_returns).cumprod() - 1
        
        # Add initial 0 for the first day
        first_date = df.index[0]
        cumulative = pd.concat([pd.Series([0.0], index=[first_date]), cumulative])
    
    else:
        raise ValueError(f"Invalid method: {method}. Use 'simple' or 'compound'")
    
    logger.info(f"Calculated cumulative returns using {method} method")
    
    return cumulative


def calculate_total_return(
    df: pd.DataFrame,
    price_column: str = 'Close'
) -> float:
    """
    Calculate total return from start to end.
    
    Args:
        df: Input DataFrame with price data
        price_column: Column to use for calculation
    
    Returns:
        Total return as decimal (e.g., 0.45 = 45% gain)
    """
    if df.empty or len(df) < 2:
        return 0.0
    
    start_price = df[price_column].iloc[0]
    end_price = df[price_column].iloc[-1]
    
    return (end_price - start_price) / start_price


def calculate_annualized_return(
    df: pd.DataFrame,
    price_column: str = 'Close'
) -> float:
    """
    Calculate annualized return (CAGR - Compound Annual Growth Rate).
    
    Args:
        df: Input DataFrame with price data
        price_column: Column to use for calculation
    
    Returns:
        Annualized return as decimal
    """
    if df.empty or len(df) < 2:
        return 0.0
    
    start_price = df[price_column].iloc[0]
    end_price = df[price_column].iloc[-1]
    
    # Calculate number of years
    start_date = df.index[0]
    end_date = df.index[-1]
    years = (end_date - start_date).days / 365.25
    
    if years <= 0:
        return 0.0
    
    # CAGR formula: (End/Start)^(1/years) - 1
    annualized = (end_price / start_price) ** (1 / years) - 1
    
    return annualized


def calculate_rolling_returns(
    df: pd.DataFrame,
    window: int = 30,
    price_column: str = 'Close'
) -> pd.Series:
    """
    Calculate rolling returns over a moving window.
    
    Args:
        df: Input DataFrame with price data
        window: Number of periods for rolling window
        price_column: Column to use for calculation
    
    Returns:
        Series with rolling returns
    """
    if df.empty or len(df) < window:
        return pd.Series(dtype=float)
    
    prices = df[price_column]
    
    # Calculate rolling return
    rolling_returns = prices.pct_change(periods=window)
    
    return rolling_returns


def calculate_volatility(
    df: pd.DataFrame,
    price_column: str = 'Close',
    window: Optional[int] = None,
    annualize: bool = True
) -> float:
    """
    Calculate historical volatility (standard deviation of returns).
    
    Args:
        df: Input DataFrame with price data
        price_column: Column to use for calculation
        window: Number of periods for calculation (None = use all data)
        annualize: If True, annualize the volatility
    
    Returns:
        Volatility as decimal (e.g., 0.18 = 18% volatility)
    """
    if df.empty:
        return 0.0
    
    # Calculate daily returns
    returns = calculate_daily_returns(df, price_column, method='simple')
    
    if returns.empty:
        return 0.0
    
    # Use specific window if provided
    if window and len(returns) >= window:
        returns = returns.tail(window)
    
    # Calculate standard deviation
    volatility = returns.std()
    
    # Annualize if requested (assuming 252 trading days per year)
    if annualize:
        volatility = volatility * np.sqrt(252)
    
    logger.info(f"Calculated volatility: {volatility:.4f}")
    
    return float(volatility)


def calculate_rolling_volatility(
    df: pd.DataFrame,
    window: int = 30,
    price_column: str = 'Close',
    annualize: bool = True
) -> pd.Series:
    """
    Calculate rolling volatility over time.
    
    Args:
        df: Input DataFrame with price data
        window: Number of periods for rolling window
        price_column: Column to use for calculation
        annualize: If True, annualize the volatility
    
    Returns:
        Series with rolling volatility
    """
    if df.empty or len(df) < window:
        return pd.Series(dtype=float)
    
    # Calculate daily returns
    returns = calculate_daily_returns(df, price_column, method='simple')
    
    # Calculate rolling standard deviation
    rolling_vol = returns.rolling(window=window).std()
    
    # Annualize if requested
    if annualize:
        rolling_vol = rolling_vol * np.sqrt(252)
    
    return rolling_vol


def calculate_downside_volatility(
    df: pd.DataFrame,
    price_column: str = 'Close',
    target_return: float = 0.0,
    annualize: bool = True
) -> float:
    """
    Calculate downside volatility (semi-deviation).
    
    Only considers returns below the target (typically 0).
    
    Args:
        df: Input DataFrame with price data
        price_column: Column to use for calculation
        target_return: Threshold return (default: 0)
        annualize: If True, annualize the volatility
    
    Returns:
        Downside volatility as decimal
    """
    if df.empty:
        return 0.0
    
    # Calculate daily returns
    returns = calculate_daily_returns(df, price_column, method='simple')
    
    if returns.empty:
        return 0.0
    
    # Only consider returns below target
    downside_returns = returns[returns < target_return]
    
    if downside_returns.empty:
        return 0.0
    
    # Calculate standard deviation of downside returns
    downside_vol = downside_returns.std()
    
    # Annualize if requested
    if annualize:
        downside_vol = downside_vol * np.sqrt(252)
    
    return float(downside_vol)


def get_return_volatility_summary(
    df: pd.DataFrame,
    price_column: str = 'Close'
) -> Dict[str, Any]:
    """
    Generate a comprehensive summary of returns and volatility.
    
    Args:
        df: Input DataFrame with price data
        price_column: Column to use for calculation
    
    Returns:
        Dictionary with return and volatility metrics
    """
    if df.empty:
        return {}
    
    # Calculate returns
    daily_returns = calculate_daily_returns(df, price_column)
    total_return = calculate_total_return(df, price_column)
    annualized_return = calculate_annualized_return(df, price_column)
    
    # Calculate volatility
    volatility = calculate_volatility(df, price_column, annualize=True)
    downside_vol = calculate_downside_volatility(df, price_column, annualize=True)
    
    # Return statistics
    return_stats = calculate_return_statistics(daily_returns)
    
    summary = {
        'total_return': float(total_return),
        'total_return_pct': round(total_return * 100, 2),
        'annualized_return': float(annualized_return),
        'annualized_return_pct': round(annualized_return * 100, 2),
        'volatility': float(volatility),
        'volatility_pct': round(volatility * 100, 2),
        'downside_volatility': float(downside_vol),
        'downside_volatility_pct': round(downside_vol * 100, 2),
        'daily_return_mean': return_stats.get('mean', 0),
        'daily_return_std': return_stats.get('std', 0),
        'sharpe_ratio': float(annualized_return / volatility) if volatility > 0 else 0,
        'trading_days': len(daily_returns),
        'period_days': (df.index[-1] - df.index[0]).days if len(df) > 0 else 0
    }
    
    return summary
