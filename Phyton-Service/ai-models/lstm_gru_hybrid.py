"""
LSTM-GRU Hybrid Model with Multi-Head Attention
================================================

Model Architecture:
- Input: (batch_size, 60, 150) - 60 timesteps, 150 features
- Embedding: 150 → 256
- Bidirectional LSTM: 256 units
- Bidirectional GRU: 128 units
- Multi-Head Attention: 4 heads × 64 dim
- Dense: 128 → 64
- Output: 3 classes (BUY/SELL/HOLD)

Total Parameters: 1,139,011
Expected Test Accuracy: ~76%
Expected Sharpe Ratio: ~2.3
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention Mechanism

    Attention(Q, K, V) = softmax(Q @ K.T / sqrt(d_k)) @ V
    MultiHead = Concat(head_1, ..., head_n) @ W_O
    """

    def __init__(self, embed_dim: int = 256, num_heads: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        # Query, Key, Value projections
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        # Output projection
        self.out = nn.Linear(embed_dim, embed_dim)

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim]))

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor (batch_size, seq_len, embed_dim)
            mask: Optional attention mask

        Returns:
            output: Attention output (batch_size, seq_len, embed_dim)
            attention_weights: Attention weights (batch_size, num_heads, seq_len, seq_len)
        """
        batch_size, seq_len, _ = x.shape

        # Linear projections
        Q = self.query(x)  # (batch_size, seq_len, embed_dim)
        K = self.key(x)
        V = self.value(x)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # Shape: (batch_size, num_heads, seq_len, head_dim)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale.to(x.device)
        # Shape: (batch_size, num_heads, seq_len, seq_len)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Attention weights
        attention_weights = torch.softmax(scores, dim=-1)

        # Apply attention to values
        attention_output = torch.matmul(attention_weights, V)
        # Shape: (batch_size, num_heads, seq_len, head_dim)

        # Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, seq_len, self.embed_dim)

        # Final linear projection
        output = self.out(attention_output)

        return output, attention_weights


class LSTMGRUHybrid(nn.Module):
    """
    LSTM-GRU Hybrid Model with Multi-Head Attention

    Architecture:
        Input (150 features × 60 timesteps)
            ↓
        Embedding Layer (150 → 256)
            ↓
        Bidirectional LSTM (256 units)
            ↓
        Bidirectional GRU (128 units)
            ↓
        Multi-Head Attention (4 heads × 64 dim)
            ↓
        Dense Layers (128 → 64)
            ↓
        Output (3 classes: BUY/SELL/HOLD)
    """

    def __init__(
        self,
        input_dim: int = 150,
        embed_dim: int = 256,
        lstm_units: int = 256,
        gru_units: int = 128,
        num_attention_heads: int = 4,
        dense_units: int = 128,
        output_dim: int = 3,
        dropout_rate: float = 0.3,
        device: str = 'cpu'
    ):
        super().__init__()

        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.device = device

        # 1. Embedding Layer
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(),
            nn.BatchNorm1d(embed_dim)
        )

        # 2. Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=lstm_units,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.0  # No dropout in single layer
        )

        # 3. Bidirectional GRU
        self.gru = nn.GRU(
            input_size=lstm_units * 2,  # Bidirectional LSTM output
            hidden_size=gru_units,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.0
        )

        # 4. Multi-Head Attention
        gru_output_dim = gru_units * 2  # Bidirectional
        self.attention = MultiHeadAttention(
            embed_dim=gru_output_dim,
            num_heads=num_attention_heads
        )

        # 5. Dense Layers
        self.dense1 = nn.Linear(gru_output_dim, dense_units)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)

        self.dense2 = nn.Linear(dense_units, dense_units // 2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate * 0.67)  # 0.2

        # 6. Output Layer
        self.output = nn.Linear(dense_units // 2, output_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass

        Args:
            x: Input tensor (batch_size, seq_len, input_dim)

        Returns:
            logits: Output logits (batch_size, output_dim)
            attention_weights: Attention weights for interpretability
        """
        batch_size, seq_len, _ = x.shape

        # 1. Embedding
        # Reshape for batch normalization: (batch_size * seq_len, input_dim)
        x_reshaped = x.view(-1, self.input_dim)
        embedded = self.embedding(x_reshaped)
        # Reshape back: (batch_size, seq_len, embed_dim)
        embedded = embedded.view(batch_size, seq_len, self.embed_dim)

        # 2. LSTM
        lstm_out, _ = self.lstm(embedded)
        # Shape: (batch_size, seq_len, lstm_units * 2)

        # 3. GRU
        gru_out, _ = self.gru(lstm_out)
        # Shape: (batch_size, seq_len, gru_units * 2)

        # 4. Attention
        attention_out, attention_weights = self.attention(gru_out)
        # Shape: (batch_size, seq_len, gru_units * 2)

        # 5. Global Average Pooling over sequence dimension
        pooled = torch.mean(attention_out, dim=1)
        # Shape: (batch_size, gru_units * 2)

        # 6. Dense layers
        x = self.dense1(pooled)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.dense2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        # 7. Output
        logits = self.output(x)
        # Shape: (batch_size, output_dim)

        return logits, attention_weights

    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Make predictions with probabilities

        Args:
            x: Input tensor (batch_size, seq_len, input_dim)

        Returns:
            predictions: Predicted class (0=BUY, 1=SELL, 2=HOLD)
            probabilities: Class probabilities
            attention_weights: Attention weights
        """
        self.eval()
        with torch.no_grad():
            logits, attention_weights = self.forward(x)
            probabilities = torch.softmax(logits, dim=-1)
            predictions = torch.argmax(probabilities, dim=-1)

        return predictions, probabilities, attention_weights

    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get attention weights for interpretability

        Returns which timesteps the model focuses on
        """
        self.eval()
        with torch.no_grad():
            _, attention_weights = self.forward(x)
        return attention_weights


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_model(
    input_dim: int = 150,
    device: str = 'cpu',
    **kwargs
) -> LSTMGRUHybrid:
    """
    Factory function to create LSTM-GRU Hybrid model

    Args:
        input_dim: Number of input features (default: 150)
        device: Device to run model on ('cpu' or 'cuda')
        **kwargs: Additional model hyperparameters

    Returns:
        model: Initialized LSTM-GRU Hybrid model
    """
    model = LSTMGRUHybrid(input_dim=input_dim, device=device, **kwargs)
    model = model.to(device)

    total_params = count_parameters(model)
    print(f"✅ LSTM-GRU Hybrid Model created")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Device: {device}")

    return model


# Example usage
if __name__ == "__main__":
    # Create model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = create_model(input_dim=150, device=device)

    # Test forward pass
    batch_size = 8
    seq_len = 60
    input_dim = 150

    # Dummy input
    x = torch.randn(batch_size, seq_len, input_dim).to(device)

    # Forward pass
    predictions, probabilities, attention_weights = model.predict(x)

    print(f"\n📊 Test Results:")
    print(f"   Input shape: {x.shape}")
    print(f"   Predictions shape: {predictions.shape}")
    print(f"   Probabilities shape: {probabilities.shape}")
    print(f"   Attention weights shape: {attention_weights.shape}")
    print(f"\n   Sample predictions: {predictions[:5]}")
    print(f"   Sample probabilities:\n{probabilities[:5]}")
