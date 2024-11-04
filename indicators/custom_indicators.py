import pandas as pd
import numpy as np

def calculate_support_resistance(data, window=20):
    """Calculate support and resistance levels"""
    data['Support'] = data['low'].rolling(window=window).min()
    data['Resistance'] = data['high'].rolling(window=window).max()
    return data

def calculate_momentum(data, period=14):
    """Calculate momentum indicator"""
    data['Momentum'] = data['close'].diff(period)
    return data

def calculate_volume_profile(data, bins=10):
    """Calculate volume profile"""
    price_range = np.linspace(data['low'].min(), data['high'].max(), bins)
    volume_profile = []
    
    for i in range(len(price_range)-1):
        mask = (data['close'] >= price_range[i]) & (data['close'] < price_range[i+1])
        volume_profile.append(data.loc[mask, 'volume'].sum())
    
    return price_range, volume_profile

def calculate_pivot_points(data):
    """Calculate pivot points"""
    data['PP'] = (data['high'] + data['low'] + data['close']) / 3
    data['R1'] = 2 * data['PP'] - data['low']
    data['S1'] = 2 * data['PP'] - data['high']
    data['R2'] = data['PP'] + (data['high'] - data['low'])
    data['S2'] = data['PP'] - (data['high'] - data['low'])
    return data
