"""
Standard LSTM Model Implementation
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional

class StandardLSTM(nn.Module):
    """
    Standard LSTM Model for Time Series Prediction
    """
    
    def __init__(self, input_size: int = 150, hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 3)  # 3 classes: BUY, SELL, HOLD
        
        # Activation and dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor (batch_size, seq_len, input_size)
            
        Returns:
            output: Prediction tensor (batch_size, 3)
        """
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Use the last time step output
        last_output = lstm_out[:, -1, :]
        
        # Fully connected layers
        x = self.fc1(last_output)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        output = self.softmax(x)
        
        return output

def create_lstm_model(input_size: int = 150, hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.2) -> StandardLSTM:
    """
    Create and return a Standard LSTM model
    
    Args:
        input_size: Number of input features
        hidden_size: LSTM hidden layer size
        num_layers: Number of LSTM layers
        dropout: Dropout rate
        
    Returns:
        StandardLSTM: Initialized LSTM model
    """
    model = StandardLSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout
    )
    
    return model