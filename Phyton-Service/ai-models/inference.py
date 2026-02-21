"""
Inference Service for LSTM-GRU Hybrid Model
============================================

Production-ready inference with:
- Model loading from checkpoint
- Feature preprocessing
- Prediction with confidence scores
- Attention weight interpretation
- Performance monitoring
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time
import json

from lstm_gru_hybrid import LSTMGRUHybrid, create_model
from sklearn.preprocessing import StandardScaler
import joblib


class ModelInference:
    """
    Production inference for LSTM-GRU Hybrid model
    """

    def __init__(
        self,
        model_path: str = './checkpoints/lstm_gru_best.pth',
        scaler_path: str = './checkpoints/scaler.pkl',
        input_dim: int = 150,
        seq_length: int = 60,
        device: str = None
    ):
        """
        Args:
            model_path: Path to trained model checkpoint
            scaler_path: Path to fitted StandardScaler
            input_dim: Number of input features
            seq_length: Sequence length (timesteps)
            device: 'cpu' or 'cuda' (auto-detect if None)
        """
        # Device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        # Hyperparameters
        self.input_dim = input_dim
        self.seq_length = seq_length

        # Load model
        print(f"📥 Loading model from: {model_path}")
        self.model = create_model(input_dim=input_dim, device=self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        print(f"✅ Model loaded successfully")

        # Load scaler
        print(f"📥 Loading scaler from: {scaler_path}")
        self.scaler = joblib.load(scaler_path)
        print(f"✅ Scaler loaded successfully")

        # Class names
        self.class_names = ['BUY', 'SELL', 'HOLD']

        # Performance metrics
        self.inference_times = []

    def preprocess_features(self, features: np.ndarray) -> torch.Tensor:
        """
        Preprocess features for inference

        Args:
            features: (seq_length, num_features) numpy array

        Returns:
            tensor: (1, seq_length, num_features) torch tensor
        """
        # Validate shape
        if features.shape[0] != self.seq_length:
            raise ValueError(f"Expected {self.seq_length} timesteps, got {features.shape[0]}")

        if features.shape[1] != self.input_dim:
            raise ValueError(f"Expected {self.input_dim} features, got {features.shape[1]}")

        # Scale features
        features_scaled = self.scaler.transform(features)

        # Convert to tensor
        tensor = torch.FloatTensor(features_scaled).unsqueeze(0)  # Add batch dimension
        tensor = tensor.to(self.device)

        return tensor

    def predict(
        self,
        features: np.ndarray,
        return_attention: bool = False
    ) -> Dict:
        """
        Make prediction

        Args:
            features: (seq_length, num_features) numpy array
            return_attention: Whether to return attention weights

        Returns:
            prediction_result: {
                'signal': 'BUY' | 'SELL' | 'HOLD',
                'confidence': float (0.0-1.0),
                'probabilities': {'BUY': float, 'SELL': float, 'HOLD': float},
                'inference_time_ms': float,
                'attention_weights': Optional[array]
            }
        """
        start_time = time.time()

        # Preprocess
        X = self.preprocess_features(features)

        # Predict
        with torch.no_grad():
            logits, attention_weights = self.model(X)
            probabilities = torch.softmax(logits, dim=-1)
            prediction = torch.argmax(probabilities, dim=-1)

        # Convert to numpy
        prediction = prediction.cpu().item()
        probabilities = probabilities.cpu().numpy()[0]

        # Get signal name and confidence
        signal = self.class_names[prediction]
        confidence = float(probabilities[prediction])

        # Inference time
        inference_time = (time.time() - start_time) * 1000  # ms
        self.inference_times.append(inference_time)

        # Result
        result = {
            'signal': signal,
            'confidence': confidence,
            'strength': int(confidence * 10),
            'probabilities': {
                'BUY': float(probabilities[0]),
                'SELL': float(probabilities[1]),
                'HOLD': float(probabilities[2])
            },
            'inference_time_ms': round(inference_time, 2)
        }

        # Optionally include attention weights
        if return_attention:
            # Average attention weights across heads
            # Shape: (batch=1, num_heads, seq_len, seq_len)
            attention_avg = attention_weights.mean(dim=1)[0].cpu().numpy()
            # Get attention scores for last timestep (what it focuses on for prediction)
            last_timestep_attention = attention_avg[-1, :]
            result['attention_weights'] = last_timestep_attention.tolist()

        return result

    def predict_from_dict(
        self,
        feature_data: List[Dict[str, float]],
        return_attention: bool = False
    ) -> Dict:
        """
        Predict from feature dictionaries (from Feature Engineering service)

        Args:
            feature_data: List of feature dicts (length = seq_length)
            return_attention: Whether to return attention weights

        Returns:
            prediction_result
        """
        # Convert to numpy array
        # Assume all dicts have same keys
        if not feature_data:
            raise ValueError("feature_data is empty")

        if len(feature_data) < self.seq_length:
            raise ValueError(f"Need at least {self.seq_length} candles, got {len(feature_data)}")

        # Take last seq_length candles
        feature_data = feature_data[-self.seq_length:]

        # Extract feature values (exclude timestamp if present)
        feature_names = [k for k in feature_data[0].keys() if k != 'timestamp']
        features = np.array([[candle[feat] for feat in feature_names] for candle in feature_data])

        return self.predict(features, return_attention=return_attention)

    def get_performance_stats(self) -> Dict:
        """
        Get inference performance statistics

        Returns:
            stats: {
                'avg_inference_time_ms': float,
                'min_inference_time_ms': float,
                'max_inference_time_ms': float,
                'total_predictions': int
            }
        """
        if not self.inference_times:
            return {
                'avg_inference_time_ms': 0.0,
                'min_inference_time_ms': 0.0,
                'max_inference_time_ms': 0.0,
                'total_predictions': 0
            }

        return {
            'avg_inference_time_ms': round(np.mean(self.inference_times), 2),
            'min_inference_time_ms': round(np.min(self.inference_times), 2),
            'max_inference_time_ms': round(np.max(self.inference_times), 2),
            'total_predictions': len(self.inference_times)
        }

    def interpret_attention(self, attention_weights: List[float]) -> Dict:
        """
        Interpret attention weights

        Returns which timesteps are most important for the prediction

        Args:
            attention_weights: List of attention scores (length = seq_length)

        Returns:
            interpretation: {
                'most_important_timesteps': List[int],  # Top 5 timesteps
                'focus_recent': float,  # % attention on last 10 candles
                'focus_distribution': str  # 'recent' | 'distributed' | 'historical'
            }
        """
        attention_weights = np.array(attention_weights)

        # Top 5 most important timesteps
        top_indices = np.argsort(attention_weights)[-5:][::-1]

        # Focus on recent candles (last 10)
        recent_focus = np.sum(attention_weights[-10:]) / np.sum(attention_weights)

        # Determine focus distribution
        if recent_focus > 0.7:
            focus_type = 'recent'
        elif recent_focus > 0.4:
            focus_type = 'distributed'
        else:
            focus_type = 'historical'

        return {
            'most_important_timesteps': top_indices.tolist(),
            'focus_recent_pct': round(recent_focus * 100, 1),
            'focus_distribution': focus_type
        }


# Flask API Integration
def create_inference_api():
    """
    Create Flask API for inference service

    Endpoints:
        POST /predict - Make prediction
        GET /health - Health check
        GET /stats - Performance statistics
    """
    from flask import Flask, request, jsonify
    from flask_cors import CORS

    app = Flask(__name__)
    CORS(app)

    # Initialize inference
    try:
        inference = ModelInference(
            model_path='./checkpoints/lstm_gru_best.pth',
            scaler_path='./checkpoints/scaler.pkl',
            input_dim=150,
            seq_length=60
        )
        inference_ready = True
    except Exception as e:
        print(f"⚠️  Warning: Could not load model - {e}")
        inference_ready = False

    @app.route('/health', methods=['GET'])
    def health():
        return jsonify({
            'status': 'healthy' if inference_ready else 'model_not_loaded',
            'service': 'LSTM-GRU Inference Service',
            'model_loaded': inference_ready,
            'device': inference.device if inference_ready else None
        })

    @app.route('/predict', methods=['POST'])
    def predict():
        if not inference_ready:
            return jsonify({
                'success': False,
                'error': 'Model not loaded'
            }), 503

        try:
            data = request.json
            feature_data = data.get('features')
            return_attention = data.get('return_attention', False)

            if not feature_data:
                return jsonify({
                    'success': False,
                    'error': 'Missing features'
                }), 400

            # Predict
            result = inference.predict_from_dict(
                feature_data,
                return_attention=return_attention
            )

            # Add interpretation if attention weights returned
            if return_attention and 'attention_weights' in result:
                interpretation = inference.interpret_attention(result['attention_weights'])
                result['attention_interpretation'] = interpretation

            return jsonify({
                'success': True,
                'symbol': data.get('symbol', 'UNKNOWN'),
                'interval': data.get('interval', '1h'),
                'prediction': result,
                'model': 'lstm_gru_hybrid'
            })

        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

    @app.route('/stats', methods=['GET'])
    def stats():
        if not inference_ready:
            return jsonify({
                'success': False,
                'error': 'Model not loaded'
            }), 503

        return jsonify({
            'success': True,
            'performance': inference.get_performance_stats()
        })

    return app


# Example usage
if __name__ == "__main__":
    # Test inference
    print("Testing LSTM-GRU Inference...")

    try:
        # Initialize
        inference = ModelInference(
            model_path='./checkpoints/lstm_gru_best.pth',
            scaler_path='./checkpoints/scaler.pkl',
            input_dim=150,
            seq_length=60
        )

        # Dummy features for testing
        dummy_features = np.random.randn(60, 150).astype(np.float32)

        # Predict
        result = inference.predict(dummy_features, return_attention=True)

        print(f"\n✅ Prediction successful:")
        print(f"   Signal: {result['signal']}")
        print(f"   Confidence: {result['confidence']*100:.1f}%")
        print(f"   Inference Time: {result['inference_time_ms']:.2f}ms")

        if 'attention_weights' in result:
            interpretation = inference.interpret_attention(result['attention_weights'])
            print(f"\n   Attention Interpretation:")
            print(f"     Focus Recent: {interpretation['focus_recent_pct']}%")
            print(f"     Distribution: {interpretation['focus_distribution']}")

    except FileNotFoundError:
        print("⚠️  Model checkpoint not found. Train model first with train.py")
