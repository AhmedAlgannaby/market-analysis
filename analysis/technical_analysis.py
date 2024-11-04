import pandas as pd
import numpy as np

class TechnicalAnalyzer:
    def __init__(self, data):
        self.data = data

    def calculate_all_indicators(self):
        self.calculate_rsi()
        self.calculate_macd()
        self.calculate_bollinger_bands()
        self.calculate_fibonacci_levels()
        return self.data

    def calculate_rsi(self, period=14):
        delta = self.data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        self.data['RSI'] = 100 - (100 / (1 + rs))
        return self.data

    def calculate_macd(self, short_period=12, long_period=26, signal_period=9):
        exp1 = self.data['close'].ewm(span=short_period, adjust=False).mean()
        exp2 = self.data['close'].ewm(span=long_period, adjust=False).mean()
        self.data['MACD'] = exp1 - exp2
        self.data['Signal_Line'] = self.data['MACD'].ewm(span=signal_period, adjust=False).mean()
        return self.data

    def calculate_bollinger_bands(self, period=20, std_dev=2):
        self.data['BB_middle'] = self.data['close'].rolling(window=period).mean()
        bb_std = self.data['close'].rolling(window=period).std()
        self.data['BB_upper'] = self.data['BB_middle'] + (bb_std * std_dev)
        self.data['BB_lower'] = self.data['BB_middle'] - (bb_std * std_dev)
        return self.data

    def calculate_fibonacci_levels(self, period=14):
        high_max = self.data['high'].rolling(window=period).max()
        low_min = self.data['low'].rolling(window=period).min()
        diff = high_max - low_min

        self.data['Fib_0'] = low_min
        self.data['Fib_236'] = low_min + diff * 0.236
        self.data['Fib_382'] = low_min + diff * 0.382
        self.data['Fib_500'] = low_min + diff * 0.500
        self.data['Fib_618'] = low_min + diff * 0.618
        self.data['Fib_100'] = high_max
        return self.data