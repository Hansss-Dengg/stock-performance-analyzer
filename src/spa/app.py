"""
Stock Performance Analyzer - Streamlit Web Application

Main application entry point for the interactive web dashboard.
"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import logging

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
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stAlert {
        margin-top: 1rem;
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
        help="Enter a valid stock ticker symbol (e.g., AAPL, MSFT, GOOGL)"
    ).upper()
    
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
    
    # Fetch data button
    fetch_button = st.sidebar.button("Fetch Data", type="primary", use_container_width=True)
    
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
    if st.sidebar.button("🗑️ Clear Cache", help="Clear all cached data and charts"):
        st.cache_data.clear()
        st.sidebar.success("Cache cleared!")
    
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
        with st.spinner(f"Fetching data for {ticker}..."):
            # Use cached fetch function
            stock_data, stock_info = fetch_stock_data_cached(
                ticker=ticker,
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d')
            )
            
            if stock_data.empty:
                st.error(f"No data found for {ticker}. Please check the ticker symbol and date range.")
                return False
            
            # Use cached analysis function
            # Convert DataFrame to dict for caching (DataFrames aren't hashable)
            stock_data_dict = stock_data.to_dict()
            stock_data_dict['index'] = stock_data.index.astype(str).tolist()
            analysis = compute_analysis_cached(stock_data_dict)
            
            # Store in session state
            st.session_state.stock_data = stock_data
            st.session_state.stock_info = stock_info
            st.session_state.ticker = ticker
            st.session_state.analysis = analysis
            
            st.success(f"✅ Successfully fetched data for {ticker}!")
            return True
    
    except StockDataError as e:
        st.error(f"❌ Error fetching data: {str(e)}")
        return False
    except Exception as e:
        st.error(f"❌ Unexpected error: {str(e)}")
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
        st.info("👈 Enter a stock ticker and fetch data to get started!")
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
    
    # Quick price chart
    st.markdown("---")
    st.subheader("Price History")
    data_dict = stock_data.to_dict()
    data_dict['index'] = stock_data.index.astype(str).tolist()
    fig = create_cached_price_chart(ticker, data_dict, show_volume=True)
    st.plotly_chart(fig, use_container_width=True)


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
        tickers = [t.strip().upper() for t in tickers_input.split(',')]
        
        if len(tickers) < 2:
            st.error("Please enter at least 2 stock tickers for comparison.")
            return
        
        try:
            with st.spinner("Fetching comparison data..."):
                from spa.data_fetcher import fetch_multiple_stocks
                
                data_dict = fetch_multiple_stocks(
                    tickers=tickers,
                    start_date=start_date.strftime('%Y-%m-%d'),
                    end_date=end_date.strftime('%Y-%m-%d')
                )
                
                if not data_dict:
                    st.error("Could not fetch data for any of the specified tickers.")
                    return
                
                # Convert DataFrames to dicts for caching
                data_dicts = {}
                for ticker, df in data_dict.items():
                    df_dict = df.to_dict()
                    df_dict['index'] = df.index.astype(str).tolist()
                    data_dicts[ticker] = df_dict
                
                fig = create_cached_comparison_chart(data_dicts, metric=metric)
                st.plotly_chart(fig, use_container_width=True)
                
                st.success(f"✅ Successfully compared {len(data_dict)} stocks!")
        
        except Exception as e:
            st.error(f"Error comparing stocks: {str(e)}")
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
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "Stock Performance Analyzer | Built with Streamlit & Python"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
