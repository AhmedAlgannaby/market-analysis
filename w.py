import os
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Import local modules
from utils.data_fetcher import DataFetcher
from analysis.technical_analysis import TechnicalAnalyzer
from indicators.custom_indicators import (
    calculate_support_resistance,
    calculate_momentum,
    calculate_volume_profile,
    calculate_pivot_points
)
from config.settings import TRADING_CONFIG, TA_PARAMS, BACKTEST_PARAMS

# Load environment variables
load_dotenv("api.env")

# Initialize DataFetcher
data_fetcher = DataFetcher()

def initialize_session_state():
    """Initialize session state variables"""
    if 'selected_symbol' not in st.session_state:
        st.session_state.selected_symbol = None
    if 'current_data' not in st.session_state:
        st.session_state.current_data = None

def display_header():
    """Display application header and description"""
    st.title("🤖 Crypto Trading Bot")
    st.markdown("""
    Advanced cryptocurrency trading analysis and recommendation system.
    This bot provides technical analysis, trading signals, and market insights.
    """)

@st.cache_data(ttl=3600)
def get_available_symbols():
    """Fetch and cache available trading symbols"""
    return data_fetcher.get_available_pairs()

@st.cache_data(ttl=300)
def get_hot_cryptos(limit=5):
    """Get top cryptocurrencies by volume"""
    symbols = data_fetcher.get_available_pairs()
    # This is a simplified version. In reality, you'd want to sort by actual volume
    return symbols[:limit]

def fetch_and_analyze_data(symbol):
    """Fetch and analyze cryptocurrency data"""
    # Fetch historical data
    data = data_fetcher.fetch_ohlcv(
        symbol,
        timeframe=TRADING_CONFIG['default_timeframe'],
        limit=TRADING_CONFIG['default_limit']
    )
    
    if data.empty:
        st.error(f"Failed to fetch data for {symbol}")
        return None

    # Perform technical analysis
    analyzer = TechnicalAnalyzer(data)
    analyzed_data = analyzer.calculate_all_indicators()
    
    # Add additional indicators
    analyzed_data = calculate_support_resistance(analyzed_data)
    analyzed_data = calculate_momentum(analyzed_data)
    analyzed_data = calculate_pivot_points(analyzed_data)
    
    return analyzed_data

def generate_trading_signals(data):
    """Generate trading signals based on technical indicators"""
    signals = {
        'action': None,
        'confidence': 0,
        'reasons': []
    }

    # RSI Analysis
    latest_rsi = data['RSI'].iloc[-1]
    if latest_rsi < TA_PARAMS['rsi_oversold']:
        signals['reasons'].append(f"RSI oversold ({latest_rsi:.2f})")
        signals['confidence'] += 0.3
        signals['action'] = 'BUY'
    elif latest_rsi > TA_PARAMS['rsi_overbought']:
        signals['reasons'].append(f"RSI overbought ({latest_rsi:.2f})")
        signals['confidence'] += 0.3
        signals['action'] = 'SELL'

    # MACD Analysis
    if data['MACD'].iloc[-1] > data['Signal_Line'].iloc[-1]:
        signals['reasons'].append("MACD crossed above signal line")
        signals['confidence'] += 0.3
        signals['action'] = 'BUY'
    elif data['MACD'].iloc[-1] < data['Signal_Line'].iloc[-1]:
        signals['reasons'].append("MACD crossed below signal line")
        signals['confidence'] += 0.3
        signals['action'] = 'SELL'

    # Bollinger Bands Analysis
    latest_close = data['close'].iloc[-1]
    if latest_close < data['BB_lower'].iloc[-1]:
        signals['reasons'].append("Price below lower Bollinger Band")
        signals['confidence'] += 0.2
        signals['action'] = 'BUY'
    elif latest_close > data['BB_upper'].iloc[-1]:
        signals['reasons'].append("Price above upper Bollinger Band")
        signals['confidence'] += 0.2
        signals['action'] = 'SELL'

    return signals

def plot_technical_analysis(data, symbol):
    """Plot technical analysis charts"""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # Create figure with secondary y-axis
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.03, row_heights=[0.7, 0.3])

    # Add candlestick chart
    fig.add_trace(go.Candlestick(x=data.index,
                                open=data['open'],
                                high=data['high'],
                                low=data['low'],
                                close=data['close'],
                                name='OHLC'),
                  row=1, col=1)

    # Add Bollinger Bands
    fig.add_trace(go.Scatter(x=data.index, y=data['BB_upper'],
                            name='Upper BB', line=dict(dash='dash')),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['BB_lower'],
                            name='Lower BB', line=dict(dash='dash')),
                  row=1, col=1)

    # Add RSI
    fig.add_trace(go.Scatter(x=data.index, y=data['RSI'],
                            name='RSI'),
                  row=2, col=1)

    # Update layout
    fig.update_layout(
        title=f'{symbol} Technical Analysis',
        yaxis_title='Price',
        yaxis2_title='RSI',
        xaxis_rangeslider_visible=False
    )

    st.plotly_chart(fig)

def display_trading_dashboard(symbol):
    """Display the main trading dashboard"""
    data = fetch_and_analyze_data(symbol)
    if data is not None:
        # Plot technical analysis
        plot_technical_analysis(data, symbol)

        # Generate and display trading signals
        signals = generate_trading_signals(data)
        
        # Display signals in a nice format
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Trading Signals")
            if signals['action']:
                st.write(f"**Recommended Action:** {signals['action']}")
                st.write(f"**Confidence:** {signals['confidence']*100:.1f}%")
                st.write("**Reasons:**")
                for reason in signals['reasons']:
                    st.write(f"- {reason}")
        
        with col2:
            st.subheader("Market Statistics")
            latest_data = data.iloc[-1]
            st.write(f"**Current Price:** ${latest_data['close']:.2f}")
            st.write(f"**24h High:** ${latest_data['high']:.2f}")
            st.write(f"**24h Low:** ${latest_data['low']:.2f}")

def main():
    """Main application function"""
    initialize_session_state()
    display_header()

    # Sidebar
    st.sidebar.header("Settings")
    
    # Symbol selection
    symbols = get_available_symbols()
    selected_symbol = st.sidebar.selectbox(
        "Select Cryptocurrency",
        options=symbols,
        index=symbols.index('BTC/USDT') if 'BTC/USDT' in symbols else 0
    )

    # Hot cryptocurrencies
    st.sidebar.subheader("Hot Cryptocurrencies")
    hot_cryptos = get_hot_cryptos()
    for crypto in hot_cryptos:  # تم تصحيح الخطأ هنا
        st.sidebar.write(crypto)

    # Display trading dashboard
    display_trading_dashboard(selected_symbol)

if __name__ == "__main__":
    main()
