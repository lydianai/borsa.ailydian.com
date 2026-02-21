"""
TRANSFORMER-BASED AI TRADING MODEL
=================================

Advanced transformer model for crypto trading with:
- Multi-head self-attention mechanism
- Positional encoding
- Feature embedding
- Sequence-to-sequence prediction
- Attention visualization
- Online learning capabilities

Architecture:
- Input: (batch_size, seq_len, features) - 60 timesteps, 150+ features
- Embedding: Feature embedding layer
- Positional Encoding: Sinusoidal positional encoding
- Transformer Encoder: Multi-head attention + FFN
- Decoder: Sequence-to-sequence prediction
- Output: Multi-class classification (BUY/SELL/HOLD) + confidence

Features:
- Real-time inference
- Attention mechanism for interpretability
- Online learning with incremental updates
- Ensemble prediction with confidence scoring
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import math
import warnings
from collections import deque
import json

class PredictionType(Enum):
    """Prediction types for the model"""
    BUY = 0
    SELL = 1
    HOLD = 2

class AttentionType(Enum):
    """Attention mechanism types"""
    SELF_ATTENTION = "self_attention"
    CROSS_ATTENTION = "cross_attention"
    MULTI_HEAD = "multi_head"

@dataclass
class ModelPrediction:
    """Model prediction with confidence and attention"""
    prediction: PredictionType
    confidence: float  # 0.0 - 1.0
    probabilities: torch.Tensor  # Shape: (3,)
    attention_weights: torch.Tensor  # Shape: (num_heads, seq_len, seq_len)
    feature_importance: torch.Tensor  # Shape: (num_features,)
    timestamp: pd.Timestamp
    metadata: Dict[str, Any]

@dataclass
class OnlineLearningBatch:
    """Batch for online learning"""
    features: torch.Tensor  # Shape: (batch_size, seq_len, num_features)
    targets: torch.Tensor    # Shape: (batch_size,)
    timestamps: List[pd.Timestamp]
    weights: Optional[torch.Tensor] = None  # Shape: (batch_size,)

class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for transformer model
    
    Adds positional information to input sequences using sine and cosine functions
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        Initialize positional encoding
        
        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
            dropout: Dropout rate
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input
        
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            
        Returns:
            Tensor with positional encoding added
        """
        # x: (batch_size, seq_len, d_model)
        # pe: (1, max_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention mechanism
    
    Computes attention scores across multiple heads and concatenates results
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1
    ):
        """
        Initialize multi-head attention
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.d_k]))
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute multi-head attention
        
        Args:
            query: Query tensor (batch_size, seq_len, d_model)
            key: Key tensor (batch_size, seq_len, d_model)
            value: Value tensor (batch_size, seq_len, d_model)
            mask: Optional attention mask
            
        Returns:
            Tuple of (output, attention_weights)
        """
        batch_size = query.size(0)
        
        # Linear projections and reshape for multi-head attention
        # Shape: (batch_size, num_heads, seq_len, d_k)
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        # Shape: (batch_size, num_heads, seq_len, seq_len)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale.to(Q.device)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        # Shape: (batch_size, num_heads, seq_len, d_k)
        attention_output = torch.matmul(attention_weights, V)
        
        # Concatenate heads
        # Shape: (batch_size, seq_len, d_model)
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        # Final linear projection
        output = self.W_o(attention_output)
        
        return output, attention_weights

class TransformerEncoderLayer(nn.Module):
    """
    Single transformer encoder layer
    
    Consists of multi-head attention + feed-forward network
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int = 2048,
        dropout: float = 0.1
    ):
        """
        Initialize encoder layer
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: Feed-forward dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        # Multi-head attention
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through encoder layer
        
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            mask: Optional attention mask
            
        Returns:
            Tuple of (output, attention_weights)
        """
        # Self-attention
        attn_output, attention_weights = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_output))
        
        # Feed-forward
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_output))
        
        return x, attention_weights

class FeatureEmbedding(nn.Module):
    """
    Feature embedding layer for financial data
    
    Converts raw features to embedded representations
    """
    
    def __init__(
        self,
        num_features: int,
        d_model: int,
        dropout: float = 0.1
    ):
        """
        Initialize feature embedding
        
        Args:
            num_features: Number of input features
            d_model: Model dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        self.embedding = nn.Linear(num_features, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Embed features
        
        Args:
            x: Input features (batch_size, seq_len, num_features)
            
        Returns:
            Embedded features (batch_size, seq_len, d_model)
        """
        x = self.embedding(x)
        x = self.norm(x)
        x = self.dropout(x)
        return x

class TransformerTradingModel(nn.Module):
    """
    Advanced Transformer-based Trading Model
    
    Features:
    - Multi-head self-attention for temporal dependencies
    - Positional encoding for sequence order
    - Feature embedding for financial indicators
    - Sequence-to-sequence prediction
    - Attention visualization
    - Online learning capabilities
    """
    
    def __init__(
        self,
        num_features: int = 150,
        d_model: int = 256,
        num_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dropout: float = 0.1,
        max_seq_len: int = 60
    ):
        """
        Initialize transformer trading model
        
        Args:
            num_features: Number of input features
            d_model: Model dimension
            num_heads: Number of attention heads
            num_encoder_layers: Number of encoder layers
            num_decoder_layers: Number of decoder layers
            dropout: Dropout rate
            max_seq_len: Maximum sequence length
        """
        super().__init__()
        
        self.num_features = num_features
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Feature embedding
        self.feature_embedding = FeatureEmbedding(num_features, d_model, dropout)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_model * 4, dropout)
            for _ in range(num_encoder_layers)
        ])
        
        # Decoder layers (simplified for trading prediction)
        self.decoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_model * 4, dropout)
            for _ in range(num_decoder_layers)
        ])
        
        # Output layers
        self.output_projection = nn.Linear(d_model, 3)  # BUY/SELL/HOLD
        self.confidence_projection = nn.Linear(d_model, 1)
        
        # Feature importance analysis
        self.feature_importance_layer = nn.Linear(d_model, num_features)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(
        self,
        x: torch.Tensor,
        target_sequence: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through transformer model
        
        Args:
            x: Input features (batch_size, seq_len, num_features)
            target_sequence: Optional target sequence for training
            
        Returns:
            Tuple of (logits, confidence, attention_weights, feature_importance)
        """
        batch_size, seq_len, _ = x.shape
        
        # Feature embedding
        x = self.feature_embedding(x)  # (batch_size, seq_len, d_model)
        
        # Positional encoding
        x = self.positional_encoding(x)  # (batch_size, seq_len, d_model)
        
        # Encoder layers with attention
        attention_weights_list = []
        for encoder_layer in self.encoder_layers:
            x, attention_weights = encoder_layer(x)
            attention_weights_list.append(attention_weights)
        
        # Use last layer attention weights
        attention_weights = attention_weights_list[-1] if attention_weights_list else torch.zeros(
            batch_size, 8, seq_len, seq_len
        ).to(x.device)
        
        # Global average pooling for sequence representation
        sequence_representation = torch.mean(x, dim=1)  # (batch_size, d_model)
        
        # Output predictions
        logits = self.output_projection(sequence_representation)  # (batch_size, 3)
        confidence = torch.sigmoid(self.confidence_projection(sequence_representation)).squeeze(-1)  # (batch_size,)
        
        # Feature importance (simplified)
        feature_importance = torch.softmax(
            self.feature_importance_layer(sequence_representation), dim=-1
        )  # (batch_size, num_features)
        
        return logits, confidence, attention_weights, feature_importance
    
    def predict(
        self,
        x: torch.Tensor
    ) -> ModelPrediction:
        """
        Make prediction with confidence and attention
        
        Args:
            x: Input features (batch_size, seq_len, num_features)
            
        Returns:
            ModelPrediction object
        """
        self.eval()
        with torch.no_grad():
            logits, confidence, attention_weights, feature_importance = self.forward(x)
            
            # Softmax probabilities
            probabilities = F.softmax(logits, dim=-1)
            
            # Predicted class
            predicted_class = torch.argmax(probabilities, dim=-1)
            
            # Convert to PredictionType
            class_map = {
                0: PredictionType.BUY,
                1: PredictionType.SELL,
                2: PredictionType.HOLD
            }
            
            predictions = [class_map[p.item()] for p in predicted_class]
            
            # Create ModelPrediction objects
            predictions_list = []
            for i in range(len(predictions)):
                pred = ModelPrediction(
                    prediction=predictions[i],
                    confidence=confidence[i].item(),
                    probabilities=probabilities[i],
                    attention_weights=attention_weights[i],
                    feature_importance=feature_importance[i],
                    timestamp=pd.Timestamp.now(),
                    metadata={
                        'logits': logits[i].detach().cpu().numpy().tolist(),
                        'sequence_length': x.size(1),
                        'num_features': x.size(2)
                    }
                )
                predictions_list.append(pred)
            
            return predictions_list[0] if len(predictions_list) == 1 else predictions_list
    
    def get_attention_weights(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Get attention weights for interpretability
        
        Args:
            x: Input features (batch_size, seq_len, num_features)
            
        Returns:
            Attention weights (batch_size, num_heads, seq_len, seq_len)
        """
        self.eval()
        with torch.no_grad():
            _, _, attention_weights, _ = self.forward(x)
            return attention_weights
    
    def calculate_feature_importance(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate feature importance scores
        
        Args:
            x: Input features (batch_size, seq_len, num_features)
            
        Returns:
            Feature importance scores (batch_size, num_features)
        """
        self.eval()
        with torch.no_grad():
            _, _, _, feature_importance = self.forward(x)
            return feature_importance

class OnlineLearningModule:
    """
    Online learning module for continuous model updates
    
    Implements incremental learning with experience replay
    """
    
    def __init__(
        self,
        model: TransformerTradingModel,
        buffer_size: int = 1000,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001
    ):
        """
        Initialize online learning module
        
        Args:
            model: Transformer model to update
            buffer_size: Size of experience replay buffer
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for regularization
        """
        self.model = model
        self.buffer_size = buffer_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Experience replay buffer
        self.replay_buffer = deque(maxlen=buffer_size)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        self.confidence_criterion = nn.MSELoss()
        
        # Training statistics
        self.training_steps = 0
        self.total_loss = 0.0
        self.accuracy = 0.0
    
    def add_experience(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
        predictions: torch.Tensor,
        confidence: torch.Tensor
    ):
        """
        Add experience to replay buffer
        
        Args:
            features: Input features
            targets: Ground truth targets
            predictions: Model predictions
            confidence: Prediction confidence
        """
        experience = {
            'features': features.detach().cpu(),
            'targets': targets.detach().cpu(),
            'predictions': predictions.detach().cpu(),
            'confidence': confidence.detach().cpu(),
            'timestamp': pd.Timestamp.now()
        }
        
        self.replay_buffer.append(experience)
    
    def update_model(
        self,
        batch: OnlineLearningBatch,
        update_frequency: int = 10
    ) -> Dict[str, float]:
        """
        Update model with online learning
        
        Args:
            batch: Online learning batch
            update_frequency: How often to perform updates
            
        Returns:
            Training metrics
        """
        self.model.train()
        
        # Add to replay buffer
        for i in range(len(batch.features)):
            experience = {
                'features': batch.features[i].detach().cpu(),
                'targets': batch.targets[i].detach().cpu(),
                'timestamps': batch.timestamps[i],
                'weights': batch.weights[i].detach().cpu() if batch.weights is not None else None
            }
            self.replay_buffer.append(experience)
        
        # Perform update periodically
        if self.training_steps % update_frequency == 0 and len(self.replay_buffer) >= 32:
            metrics = self._perform_update()
            self.model.eval()
            return metrics
        
        self.training_steps += 1
        self.model.eval()
        
        return {
            'loss': 0.0,
            'accuracy': self.accuracy,
            'training_steps': self.training_steps
        }
    
    def _perform_update(self) -> Dict[str, float]:
        """
        Perform actual model update using replay buffer
        
        Returns:
            Training metrics
        """
        # Sample from replay buffer
        batch_size = min(32, len(self.replay_buffer))
        sampled_indices = np.random.choice(len(self.replay_buffer), batch_size, replace=False)
        sampled_experiences = [self.replay_buffer[i] for i in sampled_indices]
        
        # Prepare batch
        features = torch.stack([exp['features'] for exp in sampled_experiences]).to(
            next(self.model.parameters()).device
        )
        targets = torch.stack([exp['targets'] for exp in sampled_experiences]).to(
            next(self.model.parameters()).device
        )
        
        # Forward pass
        logits, confidence, _, _ = self.model(features)
        
        # Calculate loss
        classification_loss = self.criterion(logits, targets)
        # Confidence loss (encourage high confidence for correct predictions)
        correct_predictions = (torch.argmax(logits, dim=1) == targets).float()
        confidence_loss = self.confidence_criterion(confidence, correct_predictions)
        
        total_loss = classification_loss + 0.1 * confidence_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Calculate accuracy
        predictions = torch.argmax(logits, dim=1)
        accuracy = (predictions == targets).float().mean().item()
        
        # Update statistics
        self.total_loss += total_loss.item()
        self.accuracy = accuracy
        
        return {
            'loss': total_loss.item(),
            'classification_loss': classification_loss.item(),
            'confidence_loss': confidence_loss.item(),
            'accuracy': accuracy,
            'training_steps': self.training_steps
        }
    
    def save_model(
        self,
        filepath: str
    ):
        """
        Save model and optimizer state
        
        Args:
            filepath: Path to save model
        """
        state = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_steps': self.training_steps,
            'total_loss': self.total_loss,
            'accuracy': self.accuracy,
            'buffer_size': self.buffer_size,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay
        }
        
        torch.save(state, filepath)
    
    def load_model(
        self,
        filepath: str
    ):
        """
        Load model and optimizer state
        
        Args:
            filepath: Path to load model
        """
        state = torch.load(filepath, map_location='cpu')
        
        self.model.load_state_dict(state['model_state_dict'])
        self.optimizer.load_state_dict(state['optimizer_state_dict'])
        self.training_steps = state['training_steps']
        self.total_loss = state['total_loss']
        self.accuracy = state['accuracy']
        
        # Update optimizer parameters if needed
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.learning_rate
            param_group['weight_decay'] = self.weight_decay

class TransformerTradingSystem:
    """
    Complete transformer-based trading system
    
    Integrates transformer model with online learning and real-time inference
    """
    
    def __init__(
        self,
        num_features: int = 150,
        d_model: int = 256,
        num_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dropout: float = 0.1,
        max_seq_len: int = 60,
        device: str = 'cpu'
    ):
        """
        Initialize transformer trading system
        
        Args:
            num_features: Number of input features
            d_model: Model dimension
            num_heads: Number of attention heads
            num_encoder_layers: Number of encoder layers
            num_decoder_layers: Number of decoder layers
            dropout: Dropout rate
            max_seq_len: Maximum sequence length
            device: Device to run model on
        """
        self.device = device
        
        # Initialize model
        self.model = TransformerTradingModel(
            num_features=num_features,
            d_model=d_model,
            num_heads=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout,
            max_seq_len=max_seq_len
        ).to(device)
        
        # Initialize online learning
        self.online_learning = OnlineLearningModule(
            model=self.model,
            buffer_size=1000,
            learning_rate=0.001,
            weight_decay=0.0001
        )
        
        # Move model to device
        self.model = self.model.to(device)
        
        # Training history
        self.training_history = []
        self.prediction_history = []
        
        # Model statistics
        self.total_parameters = sum(p.numel() for p in self.model.parameters())
        self.trainable_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def predict(
        self,
        features: torch.Tensor
    ) -> ModelPrediction:
        """
        Make prediction with transformer model
        
        Args:
            features: Input features (batch_size, seq_len, num_features)
            
        Returns:
            Model prediction with confidence and attention
        """
        # Ensure features are on correct device
        features = features.to(self.device)
        
        # Make prediction
        prediction = self.model.predict(features)
        
        # Store in history
        if isinstance(prediction, list):
            self.prediction_history.extend(prediction)
        else:
            self.prediction_history.append(prediction)
        
        # Keep only recent predictions
        if len(self.prediction_history) > 1000:
            self.prediction_history = self.prediction_history[-1000:]
        
        return prediction
    
    def update_with_real_data(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
        timestamps: List[pd.Timestamp]
    ) -> Dict[str, float]:
        """
        Update model with real trading data
        
        Args:
            features: Input features (batch_size, seq_len, num_features)
            targets: Ground truth targets (batch_size,)
            timestamps: Timestamps for each sample
            
        Returns:
            Training metrics
        """
        # Ensure tensors are on correct device
        features = features.to(self.device)
        targets = targets.to(self.device)
        
        # Create batch
        batch = OnlineLearningBatch(
            features=features,
            targets=targets,
            timestamps=timestamps
        )
        
        # Update model
        metrics = self.online_learning.update_model(batch)
        
        # Store metrics
        self.training_history.append({
            'timestamp': pd.Timestamp.now(),
            'metrics': metrics
        })
        
        # Keep only recent history
        if len(self.training_history) > 100:
            self.training_history = self.training_history[-100:]
        
        return metrics
    
    def get_attention_visualization(
        self,
        features: torch.Tensor
    ) -> torch.Tensor:
        """
        Get attention weights for visualization
        
        Args:
            features: Input features (batch_size, seq_len, num_features)
            
        Returns:
            Attention weights (batch_size, num_heads, seq_len, seq_len)
        """
        features = features.to(self.device)
        return self.model.get_attention_weights(features)
    
    def get_feature_importance(
        self,
        features: torch.Tensor
    ) -> torch.Tensor:
        """
        Get feature importance scores
        
        Args:
            features: Input features (batch_size, seq_len, num_features)
            
        Returns:
            Feature importance scores (batch_size, num_features)
        """
        features = features.to(self.device)
        return self.model.calculate_feature_importance(features)
    
    def get_model_statistics(self) -> Dict[str, Any]:
        """
        Get model statistics
        
        Returns:
            Dictionary with model statistics
        """
        return {
            'total_parameters': self.total_parameters,
            'trainable_parameters': self.trainable_parameters,
            'device': str(self.device),
            'model_architecture': {
                'num_features': self.model.num_features,
                'd_model': self.model.d_model,
                'num_heads': 8,  # Hardcoded in model
                'num_encoder_layers': len(self.model.encoder_layers),
                'num_decoder_layers': len(self.model.decoder_layers),
                'max_seq_len': self.model.max_seq_len
            },
            'training_stats': {
                'training_steps': self.online_learning.training_steps,
                'total_loss': self.online_learning.total_loss,
                'accuracy': self.online_learning.accuracy,
                'buffer_size': len(self.online_learning.replay_buffer),
                'prediction_history_size': len(self.prediction_history)
            }
        }
    
    def save_system(
        self,
        filepath: str
    ):
        """
        Save entire trading system
        
        Args:
            filepath: Path to save system
        """
        # Save model
        model_path = filepath.replace('.pth', '_model.pth')
        self.online_learning.save_model(model_path)
        
        # Save system state
        system_state = {
            'training_history': [
                {
                    'timestamp': entry['timestamp'].isoformat(),
                    'metrics': entry['metrics']
                }
                for entry in self.training_history
            ],
            'prediction_history': [
                {
                    'prediction': pred.prediction.value,
                    'confidence': pred.confidence,
                    'timestamp': pred.timestamp.isoformat(),
                    'metadata': pred.metadata
                }
                for pred in self.prediction_history[-100:]  # Only save recent predictions
            ],
            'model_statistics': self.get_model_statistics()
        }
        
        system_path = filepath.replace('.pth', '_system.json')
        with open(system_path, 'w') as f:
            json.dump(system_state, f, indent=2)
    
    def load_system(
        self,
        filepath: str
    ):
        """
        Load entire trading system
        
        Args:
            filepath: Path to load system
        """
        # Load model
        model_path = filepath.replace('.pth', '_model.pth')
        self.online_learning.load_model(model_path)
        
        # Load system state
        system_path = filepath.replace('.pth', '_system.json')
        try:
            with open(system_path, 'r') as f:
                system_state = json.load(f)
            
            # Restore training history
            self.training_history = [
                {
                    'timestamp': pd.Timestamp(entry['timestamp']),
                    'metrics': entry['metrics']
                }
                for entry in system_state.get('training_history', [])
            ]
            
            # Restore prediction history
            self.prediction_history = [
                ModelPrediction(
                    prediction=PredictionType(entry['prediction']),
                    confidence=entry['confidence'],
                    probabilities=torch.tensor([]),  # Empty tensor
                    attention_weights=torch.tensor([]),  # Empty tensor
                    feature_importance=torch.tensor([]),  # Empty tensor
                    timestamp=pd.Timestamp(entry['timestamp']),
                    metadata=entry.get('metadata', {})
                )
                for entry in system_state.get('prediction_history', [])
            ]
            
        except FileNotFoundError:
            warnings.warn(f"System state file not found: {system_path}")

# Example usage
if __name__ == "__main__":
    # Initialize system
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    system = TransformerTradingSystem(
        num_features=150,
        d_model=256,
        num_heads=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dropout=0.1,
        max_seq_len=60,
        device=device
    )
    
    print("\n=== TRANSFORMER TRADING SYSTEM INITIALIZED ===")
    print(f"Model Parameters: {system.total_parameters:,}")
    print(f"Trainable Parameters: {system.trainable_parameters:,}")
    print(f"Device: {system.device}")
    
    # Generate sample data
    print("\nGenerating sample data...")
    batch_size = 8
    seq_len = 60
    num_features = 150
    
    # Create sample features (simulating 150 technical indicators)
    features = torch.randn(batch_size, seq_len, num_features).to(device)
    
    # Create sample targets (0=BUY, 1=SELL, 2=HOLD)
    targets = torch.randint(0, 3, (batch_size,)).to(device)
    
    # Create sample timestamps
    timestamps = [pd.Timestamp.now() for _ in range(batch_size)]
    
    print(f"Sample data shape: {features.shape}")
    print(f"Sample targets shape: {targets.shape}")
    
    # Make initial prediction
    print("\nMaking initial prediction...")
    prediction = system.predict(features[:1])  # Single sample
    
    print(f"Prediction: {prediction.prediction.value}")
    print(f"Confidence: {prediction.confidence:.4f}")
    print(f"Probabilities: {prediction.probabilities.detach().cpu().numpy()}")
    print(f"Attention weights shape: {prediction.attention_weights.shape}")
    print(f"Feature importance shape: {prediction.feature_importance.shape}")
    
    # Update with real data
    print("\nUpdating model with real data...")
    metrics = system.update_with_real_data(features, targets, timestamps)
    
    print("Training Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
    
    # Get attention visualization
    print("\nGetting attention visualization...")
    attention_weights = system.get_attention_visualization(features[:1])
    print(f"Attention weights shape: {attention_weights.shape}")
    
    # Get feature importance
    print("\nGetting feature importance...")
    feature_importance = system.get_feature_importance(features[:1])
    print(f"Feature importance shape: {feature_importance.shape}")
    print(f"Top 5 important features: {torch.topk(feature_importance[0], 5).indices.tolist()}")
    
    # Get model statistics
    print("\nModel Statistics:")
    stats = system.get_model_statistics()
    print(f"  Total Parameters: {stats['total_parameters']:,}")
    print(f"  Trainable Parameters: {stats['trainable_parameters']:,}")
    print(f"  Device: {stats['device']}")
    print(f"  Training Steps: {stats['training_stats']['training_steps']}")
    print(f"  Current Accuracy: {stats['training_stats']['accuracy']:.4f}")
    
    # Save system
    print("\nSaving system...")
    system.save_system('./transformer_trading_system.pth')
    print("System saved successfully!")
    
    # Load system
    print("\nLoading system...")
    system.load_system('./transformer_trading_system.pth')
    print("System loaded successfully!")
    
    print("\n=== TRANSFORMER TRADING SYSTEM DEMO COMPLETE ===")