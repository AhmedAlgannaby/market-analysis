import numpy as np
import os
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
from statsmodels.tsa.statespace.sarimax import SARIMAX
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# Import local modules
from utils.data_fetcher import DataFetcher
from analysis.technical_analysis import TechnicalAnalyzer
from predictors.forecast_model import AdvancedForecaster
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
    return symbols[:limit]

def fetch_and_analyze_data(symbol, timeframe):
    """Fetch and analyze cryptocurrency data"""
    # Fetch historical data with the selected timeframe
    data = data_fetcher.fetch_ohlcv(
        symbol,
        timeframe=timeframe,  # استخدام الفريم المحدد
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
        'reasons': [],
        'stop_loss': None,
        'take_profit': None
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

    # Calculate Stop Loss and Take Profit
    if signals['action'] == 'BUY':
        signals['stop_loss'] = latest_close * (1 - TRADING_CONFIG['stop_loss_percentage'])
        signals['take_profit'] = latest_close * (1 + TRADING_CONFIG['take_profit_percentage'])
    elif signals['action'] == 'SELL':
        signals['stop_loss'] = latest_close * (1 + TRADING_CONFIG['stop_loss_percentage'])
        signals['take_profit'] = latest_close * (1 - TRADING_CONFIG['take_profit_percentage'])

    return signals


def plot_technical_analysis(data, symbol):
    """Plot main technical analysis chart"""
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

def prepare_forecast_data(data):
    """Prepare data for forecasting by adding technical indicators and handling NaN values"""
    # Create a copy of the data
    df = data.copy()
    
    # Add Moving Averages
    df['MA5'] = df['close'].rolling(window=5).mean()
    df['MA20'] = df['close'].rolling(window=20).mean()
    df['MA50'] = df['close'].rolling(window=50).mean()
    
    # Handle NaN values
    df = df.dropna()
    
    return df

def plot_forecast(data, symbol, forecast_days=30):
    """Plot forecast chart with stop loss and take profit levels"""
    try:
        # Prepare data for forecasting
        forecast_data = prepare_forecast_data(data)
        
        # Get the last known price
        current_price = forecast_data['close'].iloc[-1]
        
        # Simple forecast calculation (you can replace this with more sophisticated methods)
        last_ma5 = forecast_data['MA5'].iloc[-1]
        last_ma20 = forecast_data['MA20'].iloc[-1]
        
        # Create forecast dates
        last_date = forecast_data.index[-1]
        forecast_dates = pd.date_range(start=last_date, periods=forecast_days + 1)[1:]
        
        # Simple forecast calculation
        if last_ma5 > last_ma20:
            # Bullish trend
            forecast_values = np.linspace(current_price, current_price * 1.1, forecast_days)
            upper_bound = forecast_values * 1.05
            lower_bound = forecast_values * 0.95
        else:
            # Bearish trend
            forecast_values = np.linspace(current_price, current_price * 0.9, forecast_days)
            upper_bound = forecast_values * 1.05
            lower_bound = forecast_values * 0.95
        
        # Create forecast DataFrame
        forecast_df = pd.DataFrame({
            'forecast': forecast_values,
            'upper_bound': upper_bound,
            'lower_bound': lower_bound
        }, index=forecast_dates)
        
        # Calculate Stop Loss and Take Profit levels
        stop_loss_pct = 0.02  # 2% stop loss
        risk_ratio = 2  # Risk:Reward ratio
        
        if forecast_values[-1] > current_price:  # Bullish prediction
            stop_loss = current_price * (1 - stop_loss_pct)
            take_profit = current_price + (risk_ratio * (current_price - stop_loss))
        else:  # Bearish prediction
            stop_loss = current_price * (1 + stop_loss_pct)
            take_profit = current_price - (risk_ratio * (stop_loss - current_price))

        # Create figure
        fig = go.Figure()

        # Add historical price
        fig.add_trace(go.Scatter(
            x=forecast_data.index,
            y=forecast_data['close'],
            name='Historical Price',
            line=dict(color='blue')
        ))


         # Plotting with candlestick chart for historical data
        fig = make_subplots(rows=1, cols=1, shared_xaxes=True)

        # Add historical candlestick chart
        fig.add_trace(go.Candlestick(
            x=forecast_data.index,
            open=forecast_data['open'],
            high=forecast_data['high'],
            low=forecast_data['low'],
            close=forecast_data['close'],
            name='Historical Price'
        ))


         # Update layout
        fig.update_layout(
            title=f'{symbol} Price Forecast with Candlestick Data',
            yaxis_title='Price',
            xaxis_title='Date'
        )

        st.plotly_chart(fig)

        # Add forecast
        fig.add_trace(go.Scatter(
            x=forecast_df.index,
            y=forecast_df['forecast'],
            name='Forecast',
            line=dict(color='red', dash='dash')
        ))

        # Add confidence intervals
        fig.add_trace(go.Scatter(
            x=forecast_df.index,
            y=forecast_df['upper_bound'],
            name='Upper Bound',
            line=dict(color='rgba(0,100,80,0.2)'),
            showlegend=False
        ))

        fig.add_trace(go.Scatter(
            x=forecast_df.index,
            y=forecast_df['lower_bound'],
            name='Lower Bound',
            line=dict(color='rgba(0,100,80,0.2)'),
            fill='tonexty',
            showlegend=False
        ))

        # Add Stop Loss and Take Profit lines
        fig.add_hline(
            y=stop_loss,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Stop Loss: {stop_loss:.2f}"
        )
        
        fig.add_hline(
            y=take_profit,
            line_dash="dash",
            line_color="green",
            annotation_text=f"Take Profit: {take_profit:.2f}"
        )

        # Update layout
        fig.update_layout(
            title=f'{symbol} Price Forecast with Stop Loss and Take Profit Levels',
            yaxis_title='Price',
            showlegend=True,
            xaxis_title='Date'
        )

        st.plotly_chart(fig)

        # Display trading information
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Price", f"${current_price:.2f}")
        with col2:
            st.metric("Stop Loss", f"${stop_loss:.2f}", 
                     f"{((stop_loss/current_price)-1)*100:.1f}%")
        with col3:
            st.metric("Take Profit", f"${take_profit:.2f}", 
                     f"{((take_profit/current_price)-1)*100:.1f}%")

        # Calculate risk metrics
        risk_amount = abs(current_price - stop_loss)
        reward_amount = abs(take_profit - current_price)
        risk_reward_ratio = reward_amount / risk_amount

        # Display additional trading metrics
        st.subheader("Trading Metrics")
        metrics_col1, metrics_col2 = st.columns(2)
        with metrics_col1:
            st.write(f"Risk Amount: ${risk_amount:.2f}")
            st.write(f"Reward Amount: ${reward_amount:.2f}")
        with metrics_col2:
            st.write(f"Risk:Reward Ratio: {risk_reward_ratio:.2f}")

    except Exception as e:
        st.error(f"Error generating forecast: {str(e)}")
def display_trading_dashboard(symbol, timeframe, forecast_days):
    """Display the main trading dashboard"""
    data = fetch_and_analyze_data(symbol, timeframe)
    if data is not None:
        # Create tabs for different charts
        tab1, tab2 = st.tabs(["Technical Analysis", "Price Forecast"])
        
        with tab1:
            plot_technical_analysis(data, symbol)
            
        with tab2:
            plot_forecast(data, symbol, forecast_days)

        # Generate and display trading signals
        signals = generate_trading_signals(data)
        
        # Display signals in a nice format
        st.subheader("Trading Signals")
        col1, col2 = st.columns(2)
        with col1:
            if signals['action']:
                st.write(f"**Recommended Action:** {signals['action']}")
                st.write(f"**Confidence:** {signals['confidence']*100:.1f}%")
                st.write("**Reasons:**")
                for reason in signals['reasons']:
                    st.write(f"- {reason}")
                if signals['stop_loss'] is not None:
                    st.write(f"**Stop Loss:** ${signals['stop_loss']:.2f}")
                if signals['take_profit'] is not None:
                    st.write(f"**Take Profit:** ${signals['take_profit']:.2f}")
        
        with col2:
            st.subheader("Market Statistics")
            latest_data = data.iloc[-1]
            st.write(f"**Current Price:** ${latest_data['close']:.2f}")
            st.write(f"**24h High:** ${latest_data['high']:.2f}")
            st.write(f"**24h Low:** ${latest_data['low']:.2f}")

def main():
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

    # Timeframe selection
    timeframe = st.sidebar.selectbox(
        "اختر الفريم الزمني",
        options=["1h", "30m", "15m", "5m", "1m"]  # قائمة بفريمات أصغر
    )


    # Customizable short forecast period in hours
    forecast_hours = st.sidebar.slider("Forecast Period (Hours)", min_value=1, max_value=48, value=12, step=1)

    # Hot cryptocurrencies
    st.sidebar.subheader("Hot Cryptocurrencies")
    hot_cryptos = get_hot_cryptos()
    for crypto in hot_cryptos:
        st.sidebar.write(crypto)

    # Display trading dashboard with selected timeframe
    display_trading_dashboard(selected_symbol, timeframe, forecast_hours)

if __name__ == "__main__":
    main()
