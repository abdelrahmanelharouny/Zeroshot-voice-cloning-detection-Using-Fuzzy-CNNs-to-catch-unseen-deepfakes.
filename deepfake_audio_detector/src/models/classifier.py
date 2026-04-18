"""
Classifier module for the deepfake detection model.
"""

import torch
import torch.nn as nn
from typing import Dict, Any


class Classifier(nn.Module):
    """
    Fully connected classifier for deepfake detection.
    
    Architecture:
        128 -> 64 -> 1
    With ReLU activation and Sigmoid output.
    
    Attributes:
        feature_dim (int): Dimension of input features.
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the Classifier.
        
        Args:
            config: Configuration dictionary containing model parameters.
        """
        super(Classifier, self).__init__()
        
        self.feature_dim = config.get("feature_dim", 128)
        
        # First fully connected layer: 128 -> 64
        self.fc1 = nn.Linear(self.feature_dim, 64)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        
        # Second fully connected layer: 64 -> 1
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the classifier.
        
        Args:
            x: Input tensor of shape (batch, feature_dim).
            
        Returns:
            Output tensor of shape (batch, 1) with classification probabilities.
        """
        # First layer
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        # Second layer
        x = self.fc2(x)
        x = self.sigmoid(x)
        
        return x.squeeze(-1)
