"""
Data Loader for LSTM-GRU Hybrid Model
======================================

Loads 150+ features from Feature Engineering service and prepares
training/validation/test datasets for PyTorch model.

Features:
- Sequence generation (60 timesteps sliding window)
- Feature scaling (StandardScaler)
- Train/Val/Test split (70/15/15)
- Label generation (BUY/SELL/HOLD based on future returns)
- PyTorch Dataset and DataLoader
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, List, Optional
import requests
import json


class TradingDataset(Dataset):
    """
    PyTorch Dataset for trading data with sequences

    Each sample:
        X: (seq_length, num_features) - Historical features
        y: scalar - Label (0=BUY, 1=SELL, 2=HOLD)
    """

    def __init__(
        self,
        sequences: np.ndarray,
        labels: np.ndarray,
        device: str = 'cpu'
    ):
        """
        Args:
            sequences: (num_samples, seq_length, num_features)
            labels: (num_samples,)
            device: 'cpu' or 'cuda'
        """
        self.sequences = torch.FloatTensor(sequences).to(device)
        self.labels = torch.LongTensor(labels).to(device)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.labels[idx]


class FeatureDataLoader:
    """
    Loads features from Feature Engineering service and prepares datasets
    """

    def __init__(
        self,
        feature_service_url: str = "http://localhost:5006",
        seq_length: int = 60,
        look_ahead: int = 5,
        buy_threshold: float = 0.002,  # +0.2%
        sell_threshold: float = -0.002,  # -0.2%
        device: str = 'cpu'
    ):
        self.feature_service_url = feature_service_url
        self.seq_length = seq_length
        self.look_ahead = look_ahead
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.device = device

        self.scaler = StandardScaler()
        self.feature_names = None
        self.num_features = None

    def fetch_features(
        self,
        symbol: str,
        interval: str = '1h',
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Fetch features from Feature Engineering service

        Returns:
            DataFrame with 150+ features per candle
        """
        url = f"{self.feature_service_url}/features/generate"
        payload = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }

        try:
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()

            if not result.get('success'):
                raise ValueError(f"Feature generation failed: {result.get('error')}")

            features_data = result.get('data', [])
            if not features_data:
                raise ValueError("No features returned from service")

            df = pd.DataFrame(features_data)
            self.feature_names = list(df.columns)
            self.num_features = len(self.feature_names)

            print(f"✅ Fetched {len(df)} candles with {self.num_features} features")
            return df

        except Exception as e:
            print(f"❌ Error fetching features: {e}")
            raise

    def generate_labels(self, df: pd.DataFrame) -> np.ndarray:
        """
        Generate labels based on future returns

        Label Logic:
            future_return = (close[t+look_ahead] - close[t]) / close[t]

            if future_return > buy_threshold:   label = 0 (BUY)
            elif future_return < sell_threshold: label = 1 (SELL)
            else:                                label = 2 (HOLD)

        Args:
            df: DataFrame with 'close' column

        Returns:
            labels: (num_samples,) numpy array
        """
        if 'close' not in df.columns:
            raise ValueError("DataFrame must contain 'close' column")

        close_prices = df['close'].values
        labels = []

        for i in range(len(close_prices) - self.look_ahead):
            current_price = close_prices[i]
            future_price = close_prices[i + self.look_ahead]
            future_return = (future_price - current_price) / current_price

            if future_return > self.buy_threshold:
                label = 0  # BUY
            elif future_return < self.sell_threshold:
                label = 1  # SELL
            else:
                label = 2  # HOLD

            labels.append(label)

        # Last look_ahead samples have no future data
        # Assign HOLD (2) as default
        labels.extend([2] * self.look_ahead)

        return np.array(labels)

    def create_sequences(
        self,
        features: np.ndarray,
        labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences with sliding window

        Args:
            features: (num_candles, num_features)
            labels: (num_candles,)

        Returns:
            X: (num_samples, seq_length, num_features)
            y: (num_samples,)
        """
        X, y = [], []

        for i in range(len(features) - self.seq_length):
            # Sequence: features[i : i+seq_length]
            sequence = features[i : i + self.seq_length]
            # Label: labels[i + seq_length] (predict next candle)
            label = labels[i + self.seq_length]

            X.append(sequence)
            y.append(label)

        X = np.array(X)
        y = np.array(y)

        print(f"✅ Created {len(X)} sequences")
        print(f"   Sequence shape: {X.shape}")
        print(f"   Label distribution: BUY={np.sum(y==0)}, SELL={np.sum(y==1)}, HOLD={np.sum(y==2)}")

        return X, y

    def prepare_data(
        self,
        symbol: str,
        interval: str = '1h',
        limit: int = 1000,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15
    ) -> Tuple[TradingDataset, TradingDataset, TradingDataset]:
        """
        Full pipeline: Fetch → Scale → Sequence → Split

        Returns:
            train_dataset, val_dataset, test_dataset
        """
        print(f"\n📊 Preparing data for {symbol} ({interval})...")

        # 1. Fetch features
        df = self.fetch_features(symbol, interval, limit)

        # 2. Generate labels
        labels = self.generate_labels(df)

        # 3. Extract feature values (drop non-numeric columns if any)
        feature_columns = [col for col in df.columns if col != 'timestamp']
        features = df[feature_columns].values.astype(np.float32)

        # 4. Scale features
        features_scaled = self.scaler.fit_transform(features)
        print(f"✅ Features scaled (mean=0, std=1)")

        # 5. Create sequences
        X, y = self.create_sequences(features_scaled, labels)

        # 6. Train/Val/Test split
        # First split: train vs (val+test)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y,
            test_size=(val_ratio + test_ratio),
            shuffle=False,  # Time-series: no shuffle
            random_state=42
        )

        # Second split: val vs test
        val_size = val_ratio / (val_ratio + test_ratio)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=(1 - val_size),
            shuffle=False,
            random_state=42
        )

        print(f"\n✅ Data split:")
        print(f"   Train: {len(X_train)} samples ({train_ratio*100:.0f}%)")
        print(f"   Val:   {len(X_val)} samples ({val_ratio*100:.0f}%)")
        print(f"   Test:  {len(X_test)} samples ({test_ratio*100:.0f}%)")

        # 7. Create PyTorch datasets
        train_dataset = TradingDataset(X_train, y_train, device=self.device)
        val_dataset = TradingDataset(X_val, y_val, device=self.device)
        test_dataset = TradingDataset(X_test, y_test, device=self.device)

        return train_dataset, val_dataset, test_dataset

    def get_class_weights(self, labels: np.ndarray) -> Dict[int, float]:
        """
        Calculate class weights for weighted loss

        Handle class imbalance (HOLD is usually overrepresented)

        Returns:
            {0: weight_buy, 1: weight_sell, 2: weight_hold}
        """
        unique, counts = np.unique(labels, return_counts=True)
        total = len(labels)

        weights = {}
        for cls, count in zip(unique, counts):
            # Inverse frequency weighting
            weight = total / (len(unique) * count)
            weights[int(cls)] = weight

        print(f"\n⚖️  Class weights:")
        print(f"   BUY (0):  {weights.get(0, 1.0):.2f}")
        print(f"   SELL (1): {weights.get(1, 1.0):.2f}")
        print(f"   HOLD (2): {weights.get(2, 1.0):.2f}")

        return weights


def create_dataloaders(
    train_dataset: TradingDataset,
    val_dataset: TradingDataset,
    test_dataset: TradingDataset,
    batch_size: int = 32,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders

    Args:
        train_dataset, val_dataset, test_dataset: TradingDataset instances
        batch_size: Batch size for training
        num_workers: Number of workers for data loading

    Returns:
        train_loader, val_loader, test_loader
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle for training
        num_workers=num_workers,
        pin_memory=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )

    print(f"\n✅ DataLoaders created:")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches:   {len(val_loader)}")
    print(f"   Test batches:  {len(test_loader)}")

    return train_loader, val_loader, test_loader


# Example usage
if __name__ == "__main__":
    # Initialize data loader
    data_loader = FeatureDataLoader(
        feature_service_url="http://localhost:5006",
        seq_length=60,
        look_ahead=5,
        device='cpu'
    )

    # Prepare datasets
    train_dataset, val_dataset, test_dataset = data_loader.prepare_data(
        symbol="BTCUSDT",
        interval="1h",
        limit=500
    )

    # Calculate class weights
    all_labels = np.concatenate([
        train_dataset.labels.cpu().numpy(),
        val_dataset.labels.cpu().numpy(),
        test_dataset.labels.cpu().numpy()
    ])
    class_weights = data_loader.get_class_weights(all_labels)

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset, val_dataset, test_dataset,
        batch_size=32
    )

    print(f"\n🎉 Data preparation complete!")
    print(f"   Feature count: {data_loader.num_features}")
    print(f"   Sequence length: {data_loader.seq_length}")
