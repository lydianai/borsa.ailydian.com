"""
Standard CNN Model Implementation for Pattern Recognition
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional

class StandardCNN(nn.Module):
    """
    Standard CNN Model for Pattern Recognition in Time Series
    """
    
    def __init__(self, input_channels: int = 1, seq_len: int = 60, num_features: int = 150):
        super().__init__()
        self.input_channels = input_channels
        self.seq_len = seq_len
        self.num_features = num_features
        
        # CNN layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1)
        
        # Pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Calculate flattened size for fully connected layer
        # After 3 pooling operations: (seq_len / 2^3) * (num_features / 2^3)
        self.flattened_size = (seq_len // 8) * (num_features // 8) * 128
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.flattened_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 3)  # 3 classes: BUY, SELL, HOLD
        
        # Activation and dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor (batch_size, seq_len, num_features)
            
        Returns:
            output: Prediction tensor (batch_size, 3)
        """
        # Reshape for CNN: (batch_size, channels, height, width)
        x = x.unsqueeze(1)  # Add channel dimension
        
        # CNN layers
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        
        # Flatten
        x = x.view(-1, self.flattened_size)
        
        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        
        x = self.fc3(x)
        output = self.softmax(x)
        
        return output

def create_cnn_model(input_channels: int = 1, seq_len: int = 60, num_features: int = 150) -> StandardCNN:
    """
    Create and return a Standard CNN model
    
    Args:
        input_channels: Number of input channels
        seq_len: Sequence length
        num_features: Number of features
        
    Returns:
        StandardCNN: Initialized CNN model
    """
    model = StandardCNN(
        input_channels=input_channels,
        seq_len=seq_len,
        num_features=num_features
    )
    
    return model