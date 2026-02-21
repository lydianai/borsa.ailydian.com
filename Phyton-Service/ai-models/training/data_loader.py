"""
Crypto Data Loader for AI Models
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
import requests
import json

class CryptoDataLoader:
    """
    Data loader for cryptocurrency data
    """
    
    def __init__(self, api_base_url: str = "https://api.binance.com/api/v3"):
        self.api_base_url = api_base_url
        
    def fetch_binance_data(self, symbol: str = "BTCUSDT", interval: str = "1h", limit: int = 1000) -> pd.DataFrame:
        """
        Fetch cryptocurrency data from Binance API
        
        Args:
            symbol: Trading pair symbol
            interval: Time interval (1m, 5m, 15m, 1h, 4h, 1d, etc.)
            limit: Number of data points to fetch
            
        Returns:
            DataFrame with OHLCV data
        """
        endpoint = f"{self.api_base_url}/klines"
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }
        
        try:
            response = requests.get(endpoint, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Convert to DataFrame
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Convert data types
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
            
            return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            
        except Exception as e:
            print(f"Error fetching data from Binance: {e}")
            # Return empty DataFrame with correct structure
            return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    
    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Prepare features for AI models
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Feature array for model input
        """
        if df.empty or len(df) < 2:
            # Return zeros for empty data
            return np.zeros((1, 60, 150))
            
        # Calculate technical indicators
        features = []
        
        # Price-based features
        df = df.copy()
        df['returns'] = df['close'].pct_change()
        df['high_low_pct'] = (df['high'] - df['low']) / df['close']
        df['open_close_pct'] = (df['close'] - df['open']) / df['open']
        
        # Moving averages
        df['ma_5'] = df['close'].rolling(window=5).mean()
        df['ma_10'] = df['close'].rolling(window=10).mean()
        df['ma_20'] = df['close'].rolling(window=20).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Select features
        feature_columns = [
            'open', 'high', 'low', 'close', 'volume',
            'returns', 'high_low_pct', 'open_close_pct',
            'ma_5', 'ma_10', 'ma_20',
            'rsi', 'bb_position', 'volume_ratio'
        ]
        
        # Fill NaN values
        df[feature_columns] = df[feature_columns].fillna(method='bfill').fillna(method='ffill')
        
        # Create sequences
        sequence_length = 60
        if len(df) >= sequence_length:
            # Use last 60 data points
            sequence_data = df[feature_columns].tail(sequence_length).values
        else:
            # Pad with zeros if not enough data
            padding = np.zeros((sequence_length - len(df), len(feature_columns)))
            sequence_data = df[feature_columns].values
            sequence_data = np.vstack([padding, sequence_data])
            
        # Reshape to (1, sequence_length, num_features)
        features = sequence_data.reshape(1, sequence_length, -1)
        
        # If we don't have enough features, pad with zeros
        if features.shape[2] < 150:
            padding = np.zeros((1, sequence_length, 150 - features.shape[2]))
            features = np.concatenate([features, padding], axis=2)
            
        return features
    
    def get_sample_data(self, symbol: str = "BTCUSDT") -> Tuple[np.ndarray, np.ndarray]:
        """
        Get sample data for testing
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Tuple of (features, labels) for training/testing
        """
        # Generate sample data
        np.random.seed(42)
        features = np.random.rand(1, 60, 150).astype(np.float32)
        labels = np.random.randint(0, 3, (1,)).astype(np.int64)
        
        return features, labels

# Example usage
if __name__ == "__main__":
    loader = CryptoDataLoader()
    
    # Fetch real data (commented out to avoid API calls during testing)
    # df = loader.fetch_binance_data("BTCUSDT", "1h", 100)
    # features = loader.prepare_features(df)
    # print(f"Features shape: {features.shape}")
    
    # Get sample data
    features, labels = loader.get_sample_data()
    print(f"Sample features shape: {features.shape}")
    print(f"Sample labels shape: {labels.shape}")