import ccxt
import pandas as pd
from datetime import datetime, timedelta

class DataFetcher:
    def __init__(self, exchange_id='binance'):
        self.exchange = getattr(ccxt, exchange_id)()

    def fetch_ohlcv(self, symbol, timeframe='1d', limit=365):
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            print(f"Error fetching data: {e}")
            return pd.DataFrame()

    def get_available_pairs(self):
        try:
            markets = self.exchange.load_markets()
            return list(markets.keys())
        except Exception as e:
            print(f"Error loading markets: {e}")
            return []