"""
Data processing module for calculating financial metrics and returns.

This module provides comprehensive financial analysis tools including:
- Return calculations (daily, cumulative, annualized)
- Risk metrics (volatility, drawdown, downside deviation)
- Technical indicators (moving averages, crossovers)
- Performance summaries and ratios (Sharpe, Calmar)
"""
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List, Union
import logging

logger = logging.getLogger(__name__)

# Constants
TRADING_DAYS_PER_YEAR = 252
BUSINESS_DAYS_PER_YEAR = 252


def _validate_price_data(
    df: pd.DataFrame,
    price_column: str
) -> None:
    """
    Validate that DataFrame has required price data.
    
    Args:
        df: Input DataFrame
        price_column: Column name to validate
    
    Raises:
        ValueError: If validation fails
    """
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    if price_column not in df.columns:
        raise ValueError(f"Column '{price_column}' not found in DataFrame")
    
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have DatetimeIndex")


def _ensure_sufficient_data(
    df: pd.DataFrame,
    min_periods: int,
    operation: str = "calculation"
) -> None:
    """
    Ensure DataFrame has sufficient data points.
    
    Args:
        df: Input DataFrame
        min_periods: Minimum required periods
        operation: Name of operation for error message
    
    Raises:
        ValueError: If insufficient data
    """
    if len(df) < min_periods:
        raise ValueError(
            f"Insufficient data for {operation}. "
            f"Required: {min_periods}, Available: {len(df)}"
        )


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
    
    # Annualize if requested
    if annualize:
        volatility = volatility * np.sqrt(TRADING_DAYS_PER_YEAR)
    
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
        rolling_vol = rolling_vol * np.sqrt(TRADING_DAYS_PER_YEAR)
    
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
        downside_vol = downside_vol * np.sqrt(TRADING_DAYS_PER_YEAR)
    
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


def calculate_drawdown(
    df: pd.DataFrame,
    price_column: str = 'Close'
) -> pd.Series:
    """
    Calculate drawdown series (percentage decline from peak).
    
    Args:
        df: Input DataFrame with price data
        price_column: Column to use for calculation
    
    Returns:
        Series with drawdown values (negative percentages)
    """
    if df.empty:
        return pd.Series(dtype=float)
    
    prices = df[price_column]
    
    # Calculate running maximum (peak)
    running_max = prices.expanding().max()
    
    # Calculate drawdown
    drawdown = (prices - running_max) / running_max
    
    return drawdown


def calculate_max_drawdown(
    df: pd.DataFrame,
    price_column: str = 'Close'
) -> float:
    """
    Calculate maximum drawdown (worst peak-to-trough decline).
    
    Args:
        df: Input DataFrame with price data
        price_column: Column to use for calculation
    
    Returns:
        Maximum drawdown as decimal (e.g., -0.124 = -12.4% decline)
    """
    if df.empty:
        return 0.0
    
    drawdown = calculate_drawdown(df, price_column)
    max_dd = drawdown.min()
    
    logger.info(f"Maximum drawdown: {max_dd:.4f}")
    
    return float(max_dd)


def calculate_drawdown_details(
    df: pd.DataFrame,
    price_column: str = 'Close'
) -> Dict[str, Any]:
    """
    Calculate detailed drawdown information including dates and recovery.
    
    Args:
        df: Input DataFrame with price data
        price_column: Column to use for calculation
    
    Returns:
        Dictionary with drawdown details
    """
    if df.empty:
        return {}
    
    prices = df[price_column]
    drawdown = calculate_drawdown(df, price_column)
    
    # Find max drawdown
    max_dd = drawdown.min()
    max_dd_date = drawdown.idxmin()
    
    # Find peak before max drawdown
    prices_before_dd = prices[:max_dd_date]
    if not prices_before_dd.empty:
        peak_value = prices_before_dd.max()
        peak_date = prices_before_dd.idxmax()
    else:
        peak_value = prices.iloc[0]
        peak_date = prices.index[0]
    
    # Find recovery (if any)
    prices_after_dd = prices[max_dd_date:]
    recovered = (prices_after_dd >= peak_value).any()
    
    if recovered:
        recovery_date = prices_after_dd[prices_after_dd >= peak_value].index[0]
        recovery_days = (recovery_date - max_dd_date).days
    else:
        recovery_date = None
        recovery_days = None
    
    trough_value = prices[max_dd_date]
    drawdown_days = (max_dd_date - peak_date).days
    
    details = {
        'max_drawdown': float(max_dd),
        'max_drawdown_pct': round(max_dd * 100, 2),
        'peak_date': peak_date,
        'peak_value': float(peak_value),
        'trough_date': max_dd_date,
        'trough_value': float(trough_value),
        'drawdown_days': drawdown_days,
        'recovered': recovered,
        'recovery_date': recovery_date,
        'recovery_days': recovery_days,
        'current_drawdown': float(drawdown.iloc[-1]),
        'current_drawdown_pct': round(drawdown.iloc[-1] * 100, 2)
    }
    
    return details


def calculate_calmar_ratio(
    df: pd.DataFrame,
    price_column: str = 'Close'
) -> float:
    """
    Calculate Calmar ratio (annualized return / max drawdown).
    
    Measures return per unit of downside risk.
    
    Args:
        df: Input DataFrame with price data
        price_column: Column to use for calculation
    
    Returns:
        Calmar ratio
    """
    if df.empty:
        return 0.0
    
    annualized_return = calculate_annualized_return(df, price_column)
    max_dd = abs(calculate_max_drawdown(df, price_column))
    
    if max_dd == 0:
        return 0.0
    
    calmar = annualized_return / max_dd
    
    return float(calmar)


def calculate_moving_average(
    df: pd.DataFrame,
    window: int = 20,
    price_column: str = 'Close',
    ma_type: str = 'simple'
) -> pd.Series:
    """
    Calculate moving average.
    
    Args:
        df: Input DataFrame with price data
        window: Number of periods for moving average
        price_column: Column to use for calculation
        ma_type: Type of MA - 'simple' (SMA) or 'exponential' (EMA)
    
    Returns:
        Series with moving average values
    """
    if df.empty or len(df) < window:
        return pd.Series(dtype=float)
    
    prices = df[price_column]
    
    if ma_type == 'simple':
        ma = prices.rolling(window=window).mean()
    
    elif ma_type == 'exponential':
        ma = prices.ewm(span=window, adjust=False).mean()
    
    else:
        raise ValueError(f"Invalid ma_type: {ma_type}. Use 'simple' or 'exponential'")
    
    return ma


def calculate_sma(
    df: pd.DataFrame,
    window: int = 20,
    price_column: str = 'Close'
) -> pd.Series:
    """
    Calculate Simple Moving Average (SMA).
    
    Args:
        df: Input DataFrame with price data
        window: Number of periods (default: 20)
        price_column: Column to use for calculation
    
    Returns:
        Series with SMA values
    """
    return calculate_moving_average(df, window, price_column, ma_type='simple')


def calculate_ema(
    df: pd.DataFrame,
    window: int = 20,
    price_column: str = 'Close'
) -> pd.Series:
    """
    Calculate Exponential Moving Average (EMA).
    
    Args:
        df: Input DataFrame with price data
        window: Number of periods (default: 20)
        price_column: Column to use for calculation
    
    Returns:
        Series with EMA values
    """
    return calculate_moving_average(df, window, price_column, ma_type='exponential')


def calculate_multiple_mas(
    df: pd.DataFrame,
    windows: list = [20, 50, 200],
    price_column: str = 'Close',
    ma_type: str = 'simple'
) -> pd.DataFrame:
    """
    Calculate multiple moving averages at once.
    
    Args:
        df: Input DataFrame with price data
        windows: List of window periods (default: [20, 50, 200])
        price_column: Column to use for calculation
        ma_type: Type of MA - 'simple' or 'exponential'
    
    Returns:
        DataFrame with moving averages for each window
    """
    if df.empty:
        return pd.DataFrame()
    
    ma_df = pd.DataFrame(index=df.index)
    
    for window in windows:
        col_name = f'MA_{window}'
        ma_df[col_name] = calculate_moving_average(df, window, price_column, ma_type)
    
    return ma_df


def detect_golden_cross(
    df: pd.DataFrame,
    short_window: int = 50,
    long_window: int = 200,
    price_column: str = 'Close'
) -> Optional[pd.Timestamp]:
    """
    Detect golden cross (bullish signal when short MA crosses above long MA).
    
    Args:
        df: Input DataFrame with price data
        short_window: Short-term MA window (default: 50)
        long_window: Long-term MA window (default: 200)
        price_column: Column to use for calculation
    
    Returns:
        Date of most recent golden cross, or None if not found
    """
    if df.empty or len(df) < long_window:
        return None
    
    short_ma = calculate_sma(df, short_window, price_column)
    long_ma = calculate_sma(df, long_window, price_column)
    
    # Find crossover points (short MA crosses above long MA)
    crosses = (short_ma > long_ma) & (short_ma.shift(1) <= long_ma.shift(1))
    
    if crosses.any():
        return crosses[crosses].index[-1]
    
    return None


def detect_death_cross(
    df: pd.DataFrame,
    short_window: int = 50,
    long_window: int = 200,
    price_column: str = 'Close'
) -> Optional[pd.Timestamp]:
    """
    Detect death cross (bearish signal when short MA crosses below long MA).
    
    Args:
        df: Input DataFrame with price data
        short_window: Short-term MA window (default: 50)
        long_window: Long-term MA window (default: 200)
        price_column: Column to use for calculation
    
    Returns:
        Date of most recent death cross, or None if not found
    """
    if df.empty or len(df) < long_window:
        return None
    
    short_ma = calculate_sma(df, short_window, price_column)
    long_ma = calculate_sma(df, long_window, price_column)
    
    # Find crossover points (short MA crosses below long MA)
    crosses = (short_ma < long_ma) & (short_ma.shift(1) >= long_ma.shift(1))
    
    if crosses.any():
        return crosses[crosses].index[-1]
    
    return None


def get_moving_average_signals(
    df: pd.DataFrame,
    price_column: str = 'Close'
) -> Dict[str, Any]:
    """
    Get current moving average positions and recent crossover signals.
    
    Args:
        df: Input DataFrame with price data
        price_column: Column to use for calculation
    
    Returns:
        Dictionary with MA signals and positions
    """
    if df.empty:
        return {}
    
    # Calculate MAs
    ma_20 = calculate_sma(df, 20, price_column)
    ma_50 = calculate_sma(df, 50, price_column)
    ma_200 = calculate_sma(df, 200, price_column)
    
    current_price = df[price_column].iloc[-1]
    
    signals = {
        'current_price': float(current_price),
        'ma_20': float(ma_20.iloc[-1]) if not ma_20.empty else None,
        'ma_50': float(ma_50.iloc[-1]) if not ma_50.empty else None,
        'ma_200': float(ma_200.iloc[-1]) if not ma_200.empty else None,
        'price_above_ma_20': bool(current_price > ma_20.iloc[-1]) if not ma_20.empty else None,
        'price_above_ma_50': bool(current_price > ma_50.iloc[-1]) if not ma_50.empty else None,
        'price_above_ma_200': bool(current_price > ma_200.iloc[-1]) if not ma_200.empty else None,
    }
    
    # Detect crosses
    golden_cross = detect_golden_cross(df, 50, 200, price_column)
    death_cross = detect_death_cross(df, 50, 200, price_column)
    
    signals['golden_cross_date'] = golden_cross
    signals['death_cross_date'] = death_cross
    
    # Determine trend
    if len(df) >= 200:
        if ma_20.iloc[-1] > ma_50.iloc[-1] > ma_200.iloc[-1]:
            signals['trend'] = 'STRONG_BULLISH'
        elif ma_20.iloc[-1] > ma_50.iloc[-1]:
            signals['trend'] = 'BULLISH'
        elif ma_20.iloc[-1] < ma_50.iloc[-1] < ma_200.iloc[-1]:
            signals['trend'] = 'STRONG_BEARISH'
        elif ma_20.iloc[-1] < ma_50.iloc[-1]:
            signals['trend'] = 'BEARISH'
        else:
            signals['trend'] = 'NEUTRAL'
    else:
        signals['trend'] = 'INSUFFICIENT_DATA'
    
    return signals


def get_comprehensive_analysis(
    df: pd.DataFrame,
    price_column: str = 'Close'
) -> Dict[str, Any]:
    """
    Get comprehensive financial analysis with all metrics.
    
    Combines returns, volatility, drawdown, and moving average analysis
    into a single comprehensive report.
    
    Args:
        df: Input DataFrame with price data
        price_column: Column to use for calculations
    
    Returns:
        Dictionary with comprehensive analysis
    """
    if df.empty:
        return {'error': 'Empty DataFrame'}
    
    try:
        analysis = {}
        
        # Return metrics
        analysis['returns'] = {
            'total': calculate_total_return(df, price_column),
            'annualized': calculate_annualized_return(df, price_column),
        }
        
        # Risk metrics
        analysis['risk'] = {
            'volatility': calculate_volatility(df, price_column),
            'downside_volatility': calculate_downside_volatility(df, price_column),
            'max_drawdown': calculate_max_drawdown(df, price_column),
        }
        
        # Drawdown details
        try:
            analysis['drawdown_details'] = calculate_drawdown_details(df, price_column)
        except Exception as e:
            logger.warning(f"Could not calculate drawdown details: {e}")
            analysis['drawdown_details'] = {}
        
        # Performance ratios
        analysis['ratios'] = {}
        if analysis['risk']['volatility'] > 0:
            analysis['ratios']['sharpe'] = (
                analysis['returns']['annualized'] / analysis['risk']['volatility']
            )
        
        if abs(analysis['risk']['max_drawdown']) > 0:
            analysis['ratios']['calmar'] = calculate_calmar_ratio(df, price_column)
        
        # Moving averages and signals
        try:
            analysis['technical'] = get_moving_average_signals(df, price_column)
        except Exception as e:
            logger.warning(f"Could not calculate MA signals: {e}")
            analysis['technical'] = {}
        
        # Summary statistics
        analysis['summary'] = {
            'start_date': df.index[0],
            'end_date': df.index[-1],
            'trading_days': len(df),
            'calendar_days': (df.index[-1] - df.index[0]).days,
            'start_price': float(df[price_column].iloc[0]),
            'end_price': float(df[price_column].iloc[-1]),
            'high': float(df[price_column].max()),
            'low': float(df[price_column].min()),
        }
        
        return analysis
    
    except Exception as e:
        logger.error(f"Error in comprehensive analysis: {e}")
        return {'error': str(e)}
