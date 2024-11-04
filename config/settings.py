# Trading parameters
TRADING_CONFIG = {
    'default_timeframe': '1d',
    'default_limit': 365,
    'risk_percentage': 2,
    'stop_loss_multiplier': 2,
    'take_profit_multiplier': 3
}

# Technical Analysis parameters
TA_PARAMS = {
    'rsi_period': 14,
    'rsi_overbought': 70,
    'rsi_oversold': 30,
    'macd_fast': 12,
    'macd_slow': 26,
    'macd_signal': 9,
    'bb_period': 20,
    'bb_std': 2
}

# Backtesting parameters
BACKTEST_PARAMS = {
    'initial_balance': 10000,
    'commission': 0.001,
    'slippage': 0.0005
}

# API configuration
API_CONFIG = {
    'exchange': 'binance',
    'rate_limit': True,
    'rate_limit_wait': 1
}