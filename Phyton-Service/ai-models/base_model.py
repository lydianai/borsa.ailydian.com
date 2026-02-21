"""
BASE MODEL CLASS
All AI models inherit from this professional base class
Ensures consistency, logging, and best practices
"""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import numpy as np
import joblib
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import json
import os


class BaseAIModel(ABC, nn.Module):
    """
    Professional base class for all AI trading models

    Features:
    - Standardized interface
    - Model versioning
    - Performance tracking
    - Save/load functionality
    - Logging and monitoring
    """

    def __init__(
        self,
        model_name: str,
        model_type: str,
        version: str = "1.0.0",
        device: str = "auto"
    ):
        super().__init__()

        self.model_name = model_name
        self.model_type = model_type  # 'lstm', 'gru', 'transformer', etc.
        self.version = version

        # Auto-detect device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Performance metrics
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'accuracy': [],
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'total_predictions': 0,
            'correct_predictions': 0
        }

        # Model metadata
        self.metadata = {
            'created_at': datetime.now().isoformat(),
            'last_trained': None,
            'training_samples': 0,
            'input_features': 0,
            'output_classes': 0
        }

        print(f"✅ {model_name} initialized on {self.device}")

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass - must be implemented by child classes
        """
        pass

    @abstractmethod
    def predict(self, x: np.ndarray) -> Dict[str, Any]:
        """
        Make prediction on input data
        Returns: {
            'prediction': float,
            'confidence': float,
            'action': 'BUY' | 'SELL' | 'HOLD'
        }
        """
        pass

    def preprocess(self, x: np.ndarray) -> torch.Tensor:
        """
        Preprocess input data (normalization, scaling)
        """
        # Convert to tensor
        x_tensor = torch.FloatTensor(x)

        # Move to device
        x_tensor = x_tensor.to(self.device)

        return x_tensor

    def postprocess(self, output: torch.Tensor) -> np.ndarray:
        """
        Postprocess model output
        """
        # Move to CPU and convert to numpy
        return output.detach().cpu().numpy()

    def save_model(self, path: str):
        """
        Save model weights and metadata
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Save state dict
        torch.save({
            'model_state_dict': self.state_dict(),
            'metrics': self.metrics,
            'metadata': self.metadata,
            'model_name': self.model_name,
            'model_type': self.model_type,
            'version': self.version
        }, path)

        print(f"✅ Model saved to {path}")

    def load_model(self, path: str):
        """
        Load model weights and metadata
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.load_state_dict(checkpoint['model_state_dict'])
        self.metrics = checkpoint['metrics']
        self.metadata = checkpoint['metadata']

        print(f"✅ Model loaded from {path}")

    def update_metrics(self, metric_name: str, value: float):
        """
        Update performance metrics
        """
        if metric_name in self.metrics:
            if isinstance(self.metrics[metric_name], list):
                self.metrics[metric_name].append(value)
            else:
                self.metrics[metric_name] = value

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current performance metrics
        """
        return {
            'model_name': self.model_name,
            'model_type': self.model_type,
            'version': self.version,
            'device': str(self.device),
            'metrics': self.metrics,
            'metadata': self.metadata
        }

    def reset_metrics(self):
        """
        Reset all metrics
        """
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'accuracy': [],
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'total_predictions': 0,
            'correct_predictions': 0
        }

    def count_parameters(self) -> int:
        """
        Count trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def summary(self):
        """
        Print model summary
        """
        print(f"\n{'='*60}")
        print(f"MODEL SUMMARY: {self.model_name}")
        print(f"{'='*60}")
        print(f"Type: {self.model_type}")
        print(f"Version: {self.version}")
        print(f"Device: {self.device}")
        print(f"Parameters: {self.count_parameters():,}")
        print(f"Created: {self.metadata['created_at']}")
        print(f"Last Trained: {self.metadata['last_trained']}")
        print(f"\nPerformance Metrics:")
        print(f"  Accuracy: {self.metrics.get('accuracy', 0):.2%}")
        print(f"  Sharpe Ratio: {self.metrics.get('sharpe_ratio', 0):.2f}")
        print(f"  Win Rate: {self.metrics.get('win_rate', 0):.2%}")
        print(f"  Total Predictions: {self.metrics.get('total_predictions', 0):,}")
        print(f"{'='*60}\n")


class BaseTimeSeriesModel(BaseAIModel):
    """
    Base class for time-series models (LSTM, GRU, Transformer)
    """

    def __init__(
        self,
        model_name: str,
        model_type: str,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        output_size: int,
        sequence_length: int,
        dropout: float = 0.2,
        **kwargs
    ):
        super().__init__(model_name, model_type, **kwargs)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.sequence_length = sequence_length
        self.dropout = dropout

        self.metadata['input_features'] = input_size
        self.metadata['output_classes'] = output_size

    def prepare_sequences(self, data: np.ndarray) -> List[np.ndarray]:
        """
        Create sequences from time-series data
        """
        sequences = []
        for i in range(len(data) - self.sequence_length):
            seq = data[i:i + self.sequence_length]
            sequences.append(seq)
        return sequences


class BaseEnsembleModel:
    """
    Base class for ensemble models (combining multiple models)
    """

    def __init__(self, models: List[BaseAIModel], weights: Optional[List[float]] = None):
        self.models = models

        if weights is None:
            # Equal weights
            self.weights = [1.0 / len(models)] * len(models)
        else:
            # Normalize weights
            total = sum(weights)
            self.weights = [w / total for w in weights]

        print(f"✅ Ensemble created with {len(models)} models")

    def predict(self, x: np.ndarray) -> Dict[str, Any]:
        """
        Ensemble prediction (weighted average)
        """
        predictions = []
        confidences = []

        for model in self.models:
            result = model.predict(x)
            predictions.append(result['prediction'])
            confidences.append(result['confidence'])

        # Weighted average
        ensemble_prediction = sum(p * w for p, w in zip(predictions, self.weights))
        ensemble_confidence = sum(c * w for c, w in zip(confidences, self.weights))

        # Determine action
        if ensemble_prediction > 0.6:
            action = 'BUY'
        elif ensemble_prediction < 0.4:
            action = 'SELL'
        else:
            action = 'HOLD'

        return {
            'prediction': ensemble_prediction,
            'confidence': ensemble_confidence,
            'action': action,
            'individual_predictions': predictions,
            'model_weights': self.weights
        }

    def update_weights(self, new_weights: List[float]):
        """
        Update model weights based on performance
        """
        total = sum(new_weights)
        self.weights = [w / total for w in new_weights]
        print(f"✅ Ensemble weights updated")
