"""
Stock Performance Analyzer - Streamlit Web Application

Main application entry point for the interactive web dashboard.

Performance optimizations:
- Streamlit caching with 1-hour TTL
- Progress indicators for long operations
- Efficient data serialization for caching
- Chart rendering optimizations
"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import logging

# Configure pandas display options for better performance
pd.options.plotting.backend = "plotly"

from spa.data_fetcher import fetch_stock_data, get_stock_info
from spa.data_processor import get_comprehensive_analysis
from spa.visualizer import (
    create_price_chart,
    create_returns_chart,
    create_volatility_chart,
    create_drawdown_chart,
    create_ma_overlay_chart,
    create_comparison_chart
)
from spa.exceptions import StockDataError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Stock Performance Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stAlert {
        margin-top: 1rem;
    }
    .export-section {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    /* Improve button styling */
    .stDownloadButton button {
        width: 100%;
    }
    /* Improve metric display */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem;
    }
    /* Improve sidebar */
    [data-testid="stSidebar"] {
        background-color: #f8f9fa;
    }
    </style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables."""
    if 'stock_data' not in st.session_state:
        st.session_state.stock_data = None
    if 'stock_info' not in st.session_state:
        st.session_state.stock_info = None
    if 'ticker' not in st.session_state:
        st.session_state.ticker = None
    if 'analysis' not in st.session_state:
        st.session_state.analysis = None
    if 'last_fetch_time' not in st.session_state:
        st.session_state.last_fetch_time = None
    if 'fetch_count' not in st.session_state:
        st.session_state.fetch_count = 0


def sidebar_navigation():
    """Create sidebar with navigation and input controls."""
    st.sidebar.title("📊 Navigation")
    
    # Page selection
    page = st.sidebar.radio(
        "Select Page",
        ["Overview", "Price Analysis", "Returns Analysis", "Volatility Analysis", 
         "Drawdown Analysis", "Technical Analysis", "Multi-Stock Comparison"]
    )
    
    st.sidebar.markdown("---")
    
    # Stock input section
    st.sidebar.subheader("Stock Selection")
    
    ticker = st.sidebar.text_input(
        "Enter Stock Ticker",
        value="AAPL",
        help="Enter a valid stock ticker symbol (e.g., AAPL, MSFT, GOOGL)",
        max_chars=10
    ).upper().strip()
    
    # Validate ticker format
    ticker_valid = True
    if ticker:
        if not ticker.replace('.', '').replace('-', '').isalnum():
            st.sidebar.error("⚠️ Ticker should only contain letters, numbers, dots, or hyphens")
            ticker_valid = False
    
    # Date range selection
    st.sidebar.subheader("Date Range")
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime.now() - timedelta(days=365),
            max_value=datetime.now()
        )
    
    with col2:
        end_date = st.date_input(
            "End Date",
            value=datetime.now(),
            max_value=datetime.now()
        )
    
    # Validate date range
    date_valid = True
    if start_date >= end_date:
        st.sidebar.error("⚠️ Start date must be before end date")
        date_valid = False
    elif (end_date - start_date).days < 7:
        st.sidebar.warning("⚠️ Date range is very short. Consider at least 1 month for meaningful analysis.")
    elif (end_date - start_date).days > 365 * 20:
        st.sidebar.warning("⚠️ Very long date range may slow down loading.")
    
    # Fetch data button (disabled if validation fails)
    fetch_button = st.sidebar.button(
        "Fetch Data", 
        type="primary", 
        use_container_width=True,
        disabled=not (ticker_valid and date_valid and ticker)
    )
    
    if not ticker:
        st.sidebar.info("💡 Enter a ticker symbol to begin")
    
    # Additional options
    st.sidebar.markdown("---")
    st.sidebar.subheader("Options")
    
    show_volume = st.sidebar.checkbox("Show Volume", value=True)
    ma_windows = st.sidebar.multiselect(
        "Moving Average Windows",
        options=[20, 50, 100, 200],
        default=[50, 200]
    )
    
    # Cache management
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Settings")
    
    if st.sidebar.button("🗑️ Clear Cache", help="Clear all cached data and charts"):
        st.cache_data.clear()
        st.sidebar.success("Cache cleared!")
    
    # Session info
    if st.session_state.stock_data is not None:
        st.sidebar.markdown("---")
        st.sidebar.caption("📊 **Current Session**")
        st.sidebar.caption(f"Ticker: **{st.session_state.ticker}**")
        st.sidebar.caption(f"Data points: **{len(st.session_state.stock_data)}**")
        if st.session_state.last_fetch_time:
            time_ago = datetime.now() - st.session_state.last_fetch_time
            minutes_ago = int(time_ago.total_seconds() / 60)
            st.sidebar.caption(f"Fetched: **{minutes_ago}m ago**")
    
    # App info
    st.sidebar.markdown("---")
    st.sidebar.caption("💡 **Tips:**")
    st.sidebar.caption("• Data cached for 1 hour")
    st.sidebar.caption("• Charts are interactive")
    st.sidebar.caption("• Download any chart or data")
    
    return {
        'page': page,
        'ticker': ticker,
        'start_date': start_date,
        'end_date': end_date,
        'fetch_button': fetch_button,
        'show_volume': show_volume,
        'ma_windows': ma_windows
    }


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_stock_data_cached(ticker: str, start_date: str, end_date: str):
    """
    Fetch stock data with Streamlit caching.
    
    Args:
        ticker: Stock ticker symbol
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD)
    
    Returns:
        Tuple of (stock_data, stock_info)
    """
    # Fetch stock data
    stock_data = fetch_stock_data(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date
    )
    
    # Fetch stock info
    try:
        stock_info = get_stock_info(ticker)
    except Exception as e:
        logger.warning(f"Could not fetch stock info: {e}")
        stock_info = {}
    
    return stock_data, stock_info


@st.cache_data(ttl=3600, show_spinner=False)
def compute_analysis_cached(stock_data_dict: dict):
    """
    Compute comprehensive analysis with Streamlit caching.
    
    Args:
        stock_data_dict: Dictionary representation of stock data
    
    Returns:
        Analysis dictionary
    """
    # Convert dict back to DataFrame
    stock_data = pd.DataFrame(stock_data_dict)
    stock_data.index = pd.to_datetime(stock_data.index)
    
    # Calculate comprehensive analysis
    return get_comprehensive_analysis(stock_data)


def fetch_and_store_data(ticker: str, start_date, end_date):
    """Fetch stock data and store in session state."""
    try:
        # Progress bar
        progress_bar = st.progress(0, text=f"Fetching data for {ticker}...")
        
        # Fetch stock data
        progress_bar.progress(25, text="Fetching stock data...")
        stock_data, stock_info = fetch_stock_data_cached(
            ticker=ticker,
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d')
        )
        progress_bar.progress(50, text="Data fetched successfully...")
            
            if stock_data.empty:
                st.error(
                    f"❌ No data found for **{ticker}**\n\n"
                    "**Possible reasons:**\n"
                    "- Invalid ticker symbol\n"
                    "- Stock not traded during this period\n"
                    "- Data not available from Yahoo Finance\n\n"
                    "**Suggestions:**\n"
                    "- Verify ticker symbol is correct\n"
                    "- Try a different date range\n"
                    "- Check if the company is publicly traded"
                )
                return False
            
            # Check if we have sufficient data
            data_points = len(stock_data)
            if data_points < 30:
                st.warning(
                    f"⚠️ Only {data_points} data points available. "
                    "Some metrics may be less accurate with limited data."
                )
            
            # Calculate analysis
            progress_bar.progress(75, text="Calculating metrics...")
            # Use cached analysis function
            # Convert DataFrame to dict for caching (DataFrames aren't hashable)
            stock_data_dict = stock_data.to_dict()
            stock_data_dict['index'] = stock_data.index.astype(str).tolist()
            analysis = compute_analysis_cached(stock_data_dict)
            
            # Store in session state
            progress_bar.progress(100, text="Complete!")
            st.session_state.stock_data = stock_data
            st.session_state.stock_info = stock_info
            st.session_state.ticker = ticker
            st.session_state.analysis = analysis
            st.session_state.last_fetch_time = datetime.now()
            st.session_state.fetch_count += 1
            
            # Clear progress bar
            progress_bar.empty()
            
            # Show success message with data summary
            st.success(
                f"✅ Successfully fetched **{ticker}** data!\n\n"
                f"📊 **{data_points}** trading days from "
                f"**{stock_data.index[0].strftime('%Y-%m-%d')}** to "
                f"**{stock_data.index[-1].strftime('%Y-%m-%d')}**"
            )
            return True
    
    except StockDataError as e:
        st.error(
            f"❌ **Data Fetch Error**\n\n"
            f"{str(e)}\n\n"
            "Please try again or contact support if the issue persists."
        )
        return False
    except ConnectionError:
        st.error(
            "❌ **Connection Error**\n\n"
            "Unable to connect to Yahoo Finance. Please check:\n"
            "- Your internet connection\n"
            "- Yahoo Finance service status\n\n"
            "Try again in a few moments."
        )
        return False
    except Exception as e:
        st.error(
            f"❌ **Unexpected Error**\n\n"
            f"```\n{str(e)}\n```\n\n"
            "Please try again. If the error persists, try:\n"
            "- Using the 'Clear Cache' button\n"
            "- Refreshing the page\n"
            "- Selecting a different date range"
        )
        logger.error(f"Error in fetch_and_store_data: {e}", exc_info=True)
        return False


@st.cache_data(show_spinner="Creating chart...")
def create_cached_price_chart(ticker: str, data_dict: dict, show_volume: bool):
    """Create price chart with caching."""
    df = pd.DataFrame(data_dict)
    df.index = pd.to_datetime(df.index)
    return create_price_chart(df, ticker=ticker, show_volume=show_volume)


@st.cache_data(show_spinner="Creating chart...")
def create_cached_returns_chart(ticker: str, data_dict: dict):
    """Create returns chart with caching."""
    df = pd.DataFrame(data_dict)
    df.index = pd.to_datetime(df.index)
    return create_returns_chart(df, ticker=ticker)


@st.cache_data(show_spinner="Creating chart...")
def create_cached_volatility_chart(ticker: str, data_dict: dict, window: int = 30):
    """Create volatility chart with caching."""
    df = pd.DataFrame(data_dict)
    df.index = pd.to_datetime(df.index)
    return create_volatility_chart(df, ticker=ticker, window=window)


@st.cache_data(show_spinner="Creating chart...")
def create_cached_drawdown_chart(ticker: str, data_dict: dict):
    """Create drawdown chart with caching."""
    df = pd.DataFrame(data_dict)
    df.index = pd.to_datetime(df.index)
    return create_drawdown_chart(df, ticker=ticker)


@st.cache_data(show_spinner="Creating chart...")
def create_cached_ma_chart(ticker: str, data_dict: dict, windows: list):
    """Create MA overlay chart with caching."""
    df = pd.DataFrame(data_dict)
    df.index = pd.to_datetime(df.index)
    return create_ma_overlay_chart(df, ticker=ticker, windows=windows)


@st.cache_data(show_spinner="Creating chart...")
def create_cached_comparison_chart(data_dicts: dict, metric: str):
    """Create comparison chart with caching."""
    data_dict_converted = {}
    for ticker, data_dict in data_dicts.items():
        df = pd.DataFrame(data_dict)
        df.index = pd.to_datetime(df.index)
        data_dict_converted[ticker] = df
    return create_comparison_chart(data_dict_converted, metric=metric)


def display_overview_page():
    """Display overview page with key metrics and summary."""
    st.markdown('<div class="main-header">📈 Stock Overview</div>', unsafe_allow_html=True)
    
    if st.session_state.stock_data is None:
        st.info("👈 **Get Started:** Enter a stock ticker in the sidebar and click 'Fetch Data'")
        
        # Quick start guide
        st.markdown("---")
        st.subheader("📚 Quick Start Guide")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Popular Stock Tickers:**
            - **AAPL** - Apple Inc.
            - **MSFT** - Microsoft
            - **GOOGL** - Alphabet (Google)
            - **TSLA** - Tesla
            - **AMZN** - Amazon
            - **NVDA** - NVIDIA
            - **META** - Meta (Facebook)
            """)
        
        with col2:
            st.markdown("""
            **Features:**
            - 📊 Price analysis with candlestick charts
            - 💰 Returns and performance metrics
            - 📉 Volatility and risk analysis
            - 📈 Drawdown tracking
            - 🎯 Technical indicators (Moving Averages)
            - 🔄 Multi-stock comparison
            """)
        
        st.markdown("---")
        st.markdown("💡 **Tip:** Data is cached for 1 hour to speed up your analysis!")
        return
    
    ticker = st.session_state.ticker
    stock_data = st.session_state.stock_data
    stock_info = st.session_state.stock_info
    analysis = st.session_state.analysis
    
    # Stock information section
    st.subheader(f"{ticker} - {stock_info.get('longName', ticker)}")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Current Price",
            f"${stock_data['Close'].iloc[-1]:.2f}",
            f"{analysis['returns']['total_return']*100:.2f}%"
        )
    
    with col2:
        st.metric(
            "Market Cap",
            f"${stock_info.get('marketCap', 0)/1e9:.2f}B" if stock_info.get('marketCap') else "N/A"
        )
    
    with col3:
        st.metric(
            "Volatility (Annual)",
            f"{analysis['volatility']['annualized_volatility']*100:.2f}%"
        )
    
    # Key metrics
    st.markdown("---")
    st.subheader("Key Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Return", f"{analysis['returns']['total_return']*100:.2f}%")
        st.metric("Daily Avg Return", f"{analysis['returns']['average_daily_return']*100:.2f}%")
    
    with col2:
        st.metric("Max Drawdown", f"{analysis['drawdown']['max_drawdown']*100:.2f}%")
        st.metric("Current Drawdown", f"{analysis['drawdown']['current_drawdown']*100:.2f}%")
    
    with col3:
        st.metric("Sharpe Ratio", f"{analysis['ratios']['sharpe_ratio']:.2f}")
        st.metric("Sortino Ratio", f"{analysis['ratios']['sortino_ratio']:.2f}")
    
    with col4:
        st.metric("Calmar Ratio", f"{analysis['ratios']['calmar_ratio']:.2f}")
        st.metric("Downside Vol", f"{analysis['volatility']['downside_volatility']*100:.2f}%")
    
    # Export options
    st.markdown("---")
    st.subheader("📥 Export Data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Export stock data to CSV
        csv_data = stock_data.to_csv()
        st.download_button(
            label="📊 Download Price Data (CSV)",
            data=csv_data,
            file_name=f"{ticker}_price_data.csv",
            mime="text/csv",
            help="Download raw OHLCV data"
        )
    
    with col2:
        # Export analysis results to JSON
        import json
        analysis_json = json.dumps(analysis, indent=2, default=str)
        st.download_button(
            label="📈 Download Analysis (JSON)",
            data=analysis_json,
            file_name=f"{ticker}_analysis.json",
            mime="application/json",
            help="Download calculated metrics and statistics"
        )
    
    with col3:
        # Export summary report
        summary_report = f"""Stock Performance Report: {ticker}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Company: {stock_info.get('longName', ticker)}
Period: {stock_data.index[0].strftime('%Y-%m-%d')} to {stock_data.index[-1].strftime('%Y-%m-%d')}
Trading Days: {len(stock_data)}

PRICE METRICS
Current Price: ${stock_data['Close'].iloc[-1]:.2f}
Period High: ${stock_data['High'].max():.2f}
Period Low: ${stock_data['Low'].min():.2f}
Average Price: ${stock_data['Close'].mean():.2f}

RETURNS
Total Return: {analysis['returns']['total_return']*100:.2f}%
Annualized Return: {analysis['returns']['annualized_return']*100:.2f}%
Best Day: {analysis['returns']['best_day']*100:.2f}%
Worst Day: {analysis['returns']['worst_day']*100:.2f}%
Positive Days: {analysis['returns']['positive_days']}

RISK METRICS
Annualized Volatility: {analysis['volatility']['annualized_volatility']*100:.2f}%
Downside Volatility: {analysis['volatility']['downside_volatility']*100:.2f}%
Maximum Drawdown: {analysis['drawdown']['max_drawdown']*100:.2f}%
Current Drawdown: {analysis['drawdown']['current_drawdown']*100:.2f}%

PERFORMANCE RATIOS
Sharpe Ratio: {analysis['ratios']['sharpe_ratio']:.2f}
Sortino Ratio: {analysis['ratios']['sortino_ratio']:.2f}
Calmar Ratio: {analysis['ratios']['calmar_ratio']:.2f}
"""
        st.download_button(
            label="📄 Download Report (TXT)",
            data=summary_report,
            file_name=f"{ticker}_report.txt",
            mime="text/plain",
            help="Download formatted text report"
        )
    
    # Quick price chart
    st.markdown("---")
    st.subheader("Price History")
    data_dict = stock_data.to_dict()
    data_dict['index'] = stock_data.index.astype(str).tolist()
    fig = create_cached_price_chart(ticker, data_dict, show_volume=True)
    st.plotly_chart(fig, use_container_width=True)
    
    # Export chart
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        chart_html = fig.to_html()
        st.download_button(
            label="💾 Download Chart (HTML)",
            data=chart_html,
            file_name=f"{ticker}_price_chart.html",
            mime="text/html",
            help="Interactive HTML chart"
        )


def display_price_analysis_page(show_volume: bool):
    """Display detailed price analysis page."""
    st.markdown('<div class="main-header">📊 Price Analysis</div>', unsafe_allow_html=True)
    
    if st.session_state.stock_data is None:
        st.info("👈 Enter a stock ticker and fetch data to get started!")
        return
    
    ticker = st.session_state.ticker
    stock_data = st.session_state.stock_data
    
    st.subheader(f"{ticker} Price History")
    
    data_dict = stock_data.to_dict()
    data_dict['index'] = stock_data.index.astype(str).tolist()
    fig = create_cached_price_chart(ticker, data_dict, show_volume=show_volume)
    st.plotly_chart(fig, use_container_width=True)
    
    # Export chart
    chart_html = fig.to_html()
    st.download_button(
        label="💾 Download Chart (HTML)",
        data=chart_html,
        file_name=f"{ticker}_price_chart.html",
        mime="text/html"
    )
    
    # Price statistics
    st.markdown("---")
    st.subheader("Price Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Current", f"${stock_data['Close'].iloc[-1]:.2f}")
        st.metric("Open", f"${stock_data['Open'].iloc[-1]:.2f}")
    
    with col2:
        st.metric("High", f"${stock_data['High'].iloc[-1]:.2f}")
        st.metric("Low", f"${stock_data['Low'].iloc[-1]:.2f}")
    
    with col3:
        st.metric("Period High", f"${stock_data['High'].max():.2f}")
        st.metric("Period Low", f"${stock_data['Low'].min():.2f}")
    
    with col4:
        st.metric("Average", f"${stock_data['Close'].mean():.2f}")
        st.metric("Median", f"${stock_data['Close'].median():.2f}")


def display_returns_analysis_page():
    """Display returns analysis page."""
    st.markdown('<div class="main-header">💰 Returns Analysis</div>', unsafe_allow_html=True)
    
    if st.session_state.stock_data is None:
        st.info("👈 Enter a stock ticker and fetch data to get started!")
        return
    
    ticker = st.session_state.ticker
    stock_data = st.session_state.stock_data
    analysis = st.session_state.analysis
    
    st.subheader(f"{ticker} Returns")
    
    data_dict = stock_data.to_dict()
    data_dict['index'] = stock_data.index.astype(str).tolist()
    fig = create_cached_returns_chart(ticker, data_dict)
    st.plotly_chart(fig, use_container_width=True)
    
    # Export chart
    chart_html = fig.to_html()
    st.download_button(
        label="💾 Download Chart (HTML)",
        data=chart_html,
        file_name=f"{ticker}_returns_chart.html",
        mime="text/html"
    )
    
    # Returns statistics3
    st.markdown("---")
    st.subheader("Returns Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Return", f"{analysis['returns']['total_return']*100:.2f}%")
        st.metric("Annualized Return", f"{analysis['returns']['annualized_return']*100:.2f}%")
    
    with col2:
        st.metric("Average Daily", f"{analysis['returns']['average_daily_return']*100:.2f}%")
        st.metric("Best Day", f"{analysis['returns']['best_day']*100:.2f}%")
    
    with col3:
        st.metric("Worst Day", f"{analysis['returns']['worst_day']*100:.2f}%")
        st.metric("Positive Days", f"{analysis['returns']['positive_days']}")


def display_volatility_analysis_page():
    """Display volatility analysis page."""
    st.markdown('<div class="main-header">📉 Volatility Analysis</div>', unsafe_allow_html=True)
    
    if st.session_state.stock_data is None:
        st.info("👈 Enter a stock ticker and fetch data to get started!")
        return
    
    ticker = st.session_state.ticker
    stock_data = st.session_state.stock_data
    analysis = st.session_state.analysis
    
    st.subheader(f"{ticker} Volatility")
    
    data_dict = stock_data.to_dict()
    data_dict['index'] = stock_data.index.astype(str).tolist()
    fig = create_cached_volatility_chart(ticker, data_dict, window=30)
    st.plotly_chart(fig, use_container_width=True)
    
    # Export chart
    chart_html = fig.to_html()
    st.download_button(
        label="💾 Download Chart (HTML)",
        data=chart_html,
        file_name=f"{ticker}_volatility_chart.html",
        mime="text/html"
    )
    
    # Volatility statistics
    st.markdown("---")
    st.subheader("Volatility Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Annual Volatility", f"{analysis['volatility']['annualized_volatility']*100:.2f}%")
    
    with col2:
        st.metric("Downside Volatility", f"{analysis['volatility']['downside_volatility']*100:.2f}%")
    
    with col3:
        st.metric("Daily Std Dev", f"{analysis['volatility']['daily_volatility']*100:.2f}%")


def display_drawdown_analysis_page():
    """Display drawdown analysis page."""
    st.markdown('<div class="main-header">📉 Drawdown Analysis</div>', unsafe_allow_html=True)
    
    if st.session_state.stock_data is None:
        st.info("👈 Enter a stock ticker and fetch data to get started!")
        return
    
    ticker = st.session_state.ticker
    stock_data = st.session_state.stock_data
    analysis = st.session_state.analysis
    
    st.subheader(f"{ticker} Drawdown")
    
    data_dict = stock_data.to_dict()
    data_dict['index'] = stock_data.index.astype(str).tolist()
    fig = create_cached_drawdown_chart(ticker, data_dict)
    st.plotly_chart(fig, use_container_width=True)
    
    # Export chart
    chart_html = fig.to_html()
    st.download_button(
        label="💾 Download Chart (HTML)",
        data=chart_html,
        file_name=f"{ticker}_drawdown_chart.html",
        mime="text/html"
    )
    
    # Drawdown statistics
    st.markdown("---")
    st.subheader("Drawdown Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Max Drawdown", f"{analysis['drawdown']['max_drawdown']*100:.2f}%")
    
    with col2:
        st.metric("Current Drawdown", f"{analysis['drawdown']['current_drawdown']*100:.2f}%")
    
    with col3:
        st.metric("Recovery Status", "Recovered" if analysis['drawdown']['current_drawdown'] == 0 else "In Drawdown")


def display_technical_analysis_page(ma_windows: list):
    """Display technical analysis page."""
    st.markdown('<div class="main-header">📊 Technical Analysis</div>', unsafe_allow_html=True)
    
    if st.session_state.stock_data is None:
        st.info("👈 Enter a stock ticker and fetch data to get started!")
        return
    
    ticker = st.session_state.ticker
    stock_data = st.session_state.stock_data
    
    st.subheader(f"{ticker} with Moving Averages")
    
    if not ma_windows:
        st.warning("Please select at least one moving average window from the sidebar.")
        return
    
    data_dict = stock_data.to_dict()
    data_dict['index'] = stock_data.index.astype(str).tolist()
    fig = create_cached_ma_chart(ticker, data_dict, windows=ma_windows)
    st.plotly_chart(fig, use_container_width=True)
    
    # Export chart
    chart_html = fig.to_html()
    st.download_button(
        label="💾 Download Chart (HTML)",
        data=chart_html,
        file_name=f"{ticker}_technical_analysis.html",
        mime="text/html"
    )


def display_comparison_page():
    """Display multi-stock comparison page."""
    st.markdown('<div class="main-header">📊 Stock Comparison</div>', unsafe_allow_html=True)
    
    st.subheader("Compare Multiple Stocks")
    
    # Stock ticker inputs
    col1, col2 = st.columns([3, 1])
    
    with col1:
        tickers_input = st.text_input(
            "Enter stock tickers (comma-separated)",
            value="AAPL,MSFT,GOOGL",
            help="Enter multiple ticker symbols separated by commas"
        )
    
    with col2:
        metric = st.selectbox("Comparison Metric", ["price", "returns", "volatility"])
    
    # Date range
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime.now() - timedelta(days=365)
        )
    with col2:
        end_date = st.date_input("End Date", value=datetime.now())
    
    compare_button = st.button("Compare Stocks", type="primary")
    
    if compare_button:
        tickers = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]
        
        # Validation
        if len(tickers) < 2:
            st.error("❌ Please enter at least **2 stock tickers** for comparison.")
            return
        
        if len(tickers) > 10:
            st.error("❌ Too many tickers! Please compare **10 or fewer** stocks at once.")
            return
        
        # Check for duplicates
        if len(tickers) != len(set(tickers)):
            st.warning("⚠️ Duplicate tickers removed")
            tickers = list(set(tickers))
        
        try:
            with st.spinner(f"Fetching data for {len(tickers)} stocks..."):
                from spa.data_fetcher import fetch_multiple_stocks
                
                data_dict = fetch_multiple_stocks(
                    tickers=tickers,
                    start_date=start_date.strftime('%Y-%m-%d'),
                    end_date=end_date.strftime('%Y-%m-%d')
                )
                
                if not data_dict:
                    st.error(
                        "❌ **No Data Found**\n\n"
                        "Could not fetch data for any of the specified tickers.\n\n"
                        "**Please verify:**\n"
                        "- Ticker symbols are correct\n"
                        "- Stocks were trading during the selected period\n"
                        "- Date range is valid"
                    )
                    return
                
                # Report which tickers failed
                failed_tickers = set(tickers) - set(data_dict.keys())
                if failed_tickers:
                    st.warning(
                        f"⚠️ Could not fetch data for: **{', '.join(sorted(failed_tickers))}**\n\n"
                        f"Comparing {len(data_dict)} stock(s): **{', '.join(sorted(data_dict.keys()))}**"
                    )
                
                # Convert DataFrames to dicts for caching
                data_dicts = {}
                for ticker, df in data_dict.items():
                    df_dict = df.to_dict()
                    df_dict['index'] = df.index.astype(str).tolist()
                    data_dicts[ticker] = df_dict
                
                fig = create_cached_comparison_chart(data_dicts, metric=metric)
                st.plotly_chart(fig, use_container_width=True)
                
                # Export options
                col1, col2 = st.columns(2)
                
                with col1:
                    # Export chart
                    chart_html = fig.to_html()
                    st.download_button(
                        label="💾 Download Chart (HTML)",
                        data=chart_html,
                        file_name=f"comparison_{metric}_{'_'.join(sorted(data_dict.keys()))}.html",
                        mime="text/html"
                    )
                
                with col2:
                    # Export combined data
                    combined_data = pd.concat(
                        {ticker: df['Close'] for ticker, df in data_dict.items()},
                        axis=1
                    )
                    csv_data = combined_data.to_csv()
                    st.download_button(
                        label="📊 Download Data (CSV)",
                        data=csv_data,
                        file_name=f"comparison_{'_'.join(sorted(data_dict.keys()))}.csv",
                        mime="text/csv"
                    )
                
                st.success(
                    f"✅ Successfully compared **{len(data_dict)}** stocks!\n\n"
                    f"Showing {metric} comparison"
                )
        
        except ConnectionError:
            st.error(
                "❌ **Connection Error**\n\n"
                "Unable to fetch comparison data. Please check your internet connection."
            )
        except Exception as e:
            st.error(
                f"❌ **Comparison Error**\n\n"
                f"```\n{str(e)}\n```\n\n"
                "**Suggestions:**\n"
                "- Verify all ticker symbols are valid\n"
                "- Try with fewer stocks\n"
                "- Clear cache and try again"
            )
            logger.error(f"Error in comparison: {e}", exc_info=True)


def main():
    """Main application entry point."""
    # Initialize session state
    initialize_session_state()
    
    # Sidebar navigation and controls
    nav_state = sidebar_navigation()
    
    # Fetch data if button clicked
    if nav_state['fetch_button']:
        fetch_and_store_data(
            nav_state['ticker'],
            nav_state['start_date'],
            nav_state['end_date']
        )
    
    # Display selected page
    if nav_state['page'] == "Overview":
        display_overview_page()
    elif nav_state['page'] == "Price Analysis":
        display_price_analysis_page(nav_state['show_volume'])
    elif nav_state['page'] == "Returns Analysis":
        display_returns_analysis_page()
    elif nav_state['page'] == "Volatility Analysis":
        display_volatility_analysis_page()
    elif nav_state['page'] == "Drawdown Analysis":
        display_drawdown_analysis_page()
    elif nav_state['page'] == "Technical Analysis":
        display_technical_analysis_page(nav_state['ma_windows'])
    elif nav_state['page'] == "Multi-Stock Comparison":
        display_comparison_page()
    
    # Footer
    st.markdown("---")
    
    footer_col1, footer_col2, footer_col3 = st.columns([1, 2, 1])
    
    with footer_col1:
        if st.session_state.fetch_count > 0:
            st.caption(f"📊 Fetches: {st.session_state.fetch_count}")
    
    with footer_col2:
        st.markdown(
            "<div style='text-align: center; color: #666;'>"
            "Stock Performance Analyzer | Built with Streamlit & Python<br>"
            "<small>Data from Yahoo Finance • Cached for optimal performance</small>"
            "</div>",
            unsafe_allow_html=True
        )
    
    with footer_col3:
        st.caption(
            "[GitHub](https://github.com/Hansss-Dengg/stock-performance-analyzer)",
            unsafe_allow_html=True
        )


if __name__ == "__main__":
    main()
