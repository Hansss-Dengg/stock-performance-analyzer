"""
Visualization module for creating interactive charts.

This module provides functions to create interactive Plotly visualizations
for stock performance analysis including:
- Price history charts with volume
- Returns visualizations (daily and cumulative)
- Volatility analysis charts
- Drawdown visualizations
- Moving average overlays
- Multi-stock comparison charts
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Optional, List, Dict, Any, Union
import logging

from .data_processor import (
    calculate_daily_returns,
    calculate_cumulative_returns,
    calculate_rolling_volatility,
    calculate_drawdown,
    calculate_sma,
    calculate_ema
)

logger = logging.getLogger(__name__)


# Color scheme for consistent visualizations
COLORS = {
    'primary': '#1f77b4',      # Blue
    'secondary': '#ff7f0e',    # Orange
    'positive': '#2ca02c',     # Green
    'negative': '#d62728',     # Red
    'neutral': '#7f7f7f',      # Gray
    'volume': 'rgba(128, 128, 128, 0.3)',
    'ma_short': '#e377c2',     # Pink
    'ma_long': '#9467bd'       # Purple
}


def _validate_dataframe(df: pd.DataFrame, required_columns: Optional[List[str]] = None) -> None:
    """
    Validate DataFrame for visualization.
    
    Args:
        df: Input DataFrame
        required_columns: List of required column names
    
    Raises:
        ValueError: If validation fails
    """
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have DatetimeIndex")
    
    if required_columns:
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")


def _format_large_number(value: float) -> str:
    """
    Format large numbers with appropriate suffixes (K, M, B).
    
    Args:
        value: Number to format
    
    Returns:
        Formatted string
    """
    if abs(value) >= 1e9:
        return f"{value / 1e9:.2f}B"
    elif abs(value) >= 1e6:
        return f"{value / 1e6:.2f}M"
    elif abs(value) >= 1e3:
        return f"{value / 1e3:.2f}K"
    else:
        return f"{value:.2f}"


def create_base_layout(
    title: str,
    xaxis_title: str = "Date",
    yaxis_title: str = "Value",
    height: int = 600
) -> Dict[str, Any]:
    """
    Create base layout configuration for charts.
    
    Args:
        title: Chart title
        xaxis_title: X-axis label
        yaxis_title: Y-axis label
        height: Chart height in pixels
    
    Returns:
        Dictionary with layout configuration
    """
    return {
        'title': {
            'text': title,
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        'xaxis': {
            'title': xaxis_title,
            'showgrid': True,
            'gridcolor': 'rgba(128, 128, 128, 0.2)'
        },
        'yaxis': {
            'title': yaxis_title,
            'showgrid': True,
            'gridcolor': 'rgba(128, 128, 128, 0.2)'
        },
        'hovermode': 'x unified',
        'height': height,
        'template': 'plotly_white',
        'showlegend': True,
        'legend': {
            'orientation': 'h',
            'yanchor': 'bottom',
            'y': 1.02,
            'xanchor': 'right',
            'x': 1
        }
    }


def _add_range_selector() -> Dict[str, Any]:
    """
    Create range selector buttons for time-based filtering.
    
    Returns:
        Dictionary with range selector configuration
    """
    return {
        'buttons': [
            {'count': 1, 'label': '1M', 'step': 'month', 'stepmode': 'backward'},
            {'count': 3, 'label': '3M', 'step': 'month', 'stepmode': 'backward'},
            {'count': 6, 'label': '6M', 'step': 'month', 'stepmode': 'backward'},
            {'count': 1, 'label': 'YTD', 'step': 'year', 'stepmode': 'todate'},
            {'count': 1, 'label': '1Y', 'step': 'year', 'stepmode': 'backward'},
            {'step': 'all', 'label': 'All'}
        ],
        'bgcolor': 'rgba(150, 150, 150, 0.1)',
        'x': 0,
        'xanchor': 'left',
        'y': 1.15,
        'yanchor': 'top'
    }


def save_chart(fig: go.Figure, filepath: str, **kwargs) -> None:
    """
    Save chart to file.
    
    Args:
        fig: Plotly figure object
        filepath: Output file path (supports .html, .png, .jpg, .svg, .pdf)
        **kwargs: Additional arguments passed to fig.write_*
    
    Example:
        >>> fig = create_price_chart(stock_df, ticker="AAPL")
        >>> save_chart(fig, "price_chart.html")
    """
    try:
        if filepath.endswith('.html'):
            fig.write_html(filepath, **kwargs)
        elif filepath.endswith('.png'):
            fig.write_image(filepath, **kwargs)
        elif filepath.endswith('.jpg') or filepath.endswith('.jpeg'):
            fig.write_image(filepath, **kwargs)
        elif filepath.endswith('.svg'):
            fig.write_image(filepath, **kwargs)
        elif filepath.endswith('.pdf'):
            fig.write_image(filepath, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {filepath}")
        
        logger.info(f"Chart saved to {filepath}")
    
    except Exception as e:
        logger.error(f"Error saving chart: {e}")
        raise


# Placeholder functions for future implementation
def create_price_chart(
    df: pd.DataFrame,
    ticker: str = "Stock",
    show_volume: bool = True
) -> go.Figure:
    """
    Create interactive price history chart.
    
    Args:
        df: DataFrame with OHLCV data
        ticker: Stock ticker symbol for title
        show_volume: Whether to show volume subplot
    
    Returns:
        Plotly Figure object
    
    Example:
        >>> fig = create_price_chart(stock_df, ticker="AAPL")
        >>> fig.show()
    """
    raise NotImplementedError("Price chart will be implemented in commit 26")


def create_returns_chart(
    df: pd.DataFrame,
    ticker: str = "Stock"
) -> go.Figure:
    """
    Create returns visualization chart.
    
    Args:
        df: DataFrame with price data
        ticker: Stock ticker symbol for title
    
    Returns:
        Plotly Figure object
    
    Example:
        >>> fig = create_returns_chart(stock_df, ticker="AAPL")
        >>> fig.show()
    """
    raise NotImplementedError("Returns chart will be implemented in commit 27")


def create_volatility_chart(
    df: pd.DataFrame,
    ticker: str = "Stock"
) -> go.Figure:
    """
    Create volatility analysis chart.
    
    Args:
        df: DataFrame with price data
        ticker: Stock ticker symbol for title
    
    Returns:
        Plotly Figure object
    
    Example:
        >>> fig = create_volatility_chart(stock_df, ticker="AAPL")
        >>> fig.show()
    """
    raise NotImplementedError("Volatility chart will be implemented in commit 28")


def create_drawdown_chart(
    df: pd.DataFrame,
    ticker: str = "Stock"
) -> go.Figure:
    """
    Create drawdown visualization chart.
    
    Args:
        df: DataFrame with price data
        ticker: Stock ticker symbol for title
    
    Returns:
        Plotly Figure object
    
    Example:
        >>> fig = create_drawdown_chart(stock_df, ticker="AAPL")
        >>> fig.show()
    """
    raise NotImplementedError("Drawdown chart will be implemented in commit 29")


def create_ma_overlay_chart(
    df: pd.DataFrame,
    ticker: str = "Stock",
    windows: List[int] = [20, 50, 200]
) -> go.Figure:
    """
    Create price chart with moving average overlays.
    
    Args:
        df: DataFrame with price data
        ticker: Stock ticker symbol for title
        windows: List of MA window periods
    
    Returns:
        Plotly Figure object
    
    Example:
        >>> fig = create_ma_overlay_chart(stock_df, ticker="AAPL", windows=[50, 200])
        >>> fig.show()
    """
    raise NotImplementedError("MA overlay chart will be implemented in commit 30")


def create_comparison_chart(
    data_dict: Dict[str, pd.DataFrame],
    metric: str = "price"
) -> go.Figure:
    """
    Create comparison chart for multiple stocks.
    
    Args:
        data_dict: Dictionary mapping ticker symbols to DataFrames
        metric: Metric to compare ('price', 'returns', 'volatility')
    
    Returns:
        Plotly Figure object
    
    Example:
        >>> data = {"AAPL": aapl_df, "MSFT": msft_df, "GOOGL": googl_df}
        >>> fig = create_comparison_chart(data, metric="returns")
        >>> fig.show()
    """
    raise NotImplementedError("Comparison chart will be implemented in commit 31")
