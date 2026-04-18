"""
CNN Encoder module for extracting features from audio spectrograms.
"""

import torch
import torch.nn as nn
from typing import Dict, Any


class ConvBlock(nn.Module):
    """
    A convolutional block consisting of Conv2D, ReLU, BatchNorm, and MaxPool.
    
    Attributes:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
    """
    
    def __init__(self, in_channels: int, out_channels: int) -> None:
        """
        Initialize the ConvBlock.
        
        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
        """
        super(ConvBlock, self).__init__()
        
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
        )
        self.relu = nn.ReLU()
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the convolutional block.
        
        Args:
            x: Input tensor of shape (batch, in_channels, height, width).
            
        Returns:
            Output tensor of shape (batch, out_channels, height/2, width/2).
        """
        x = self.conv(x)
        x = self.relu(x)
        x = self.batch_norm(x)
        x = self.max_pool(x)
        return x


class CNNEncoder(nn.Module):
    """
    CNN Encoder for extracting features from audio spectrograms.
    
    Consists of 3 convolutional blocks followed by adaptive average pooling.
    
    Attributes:
        input_channels (int): Number of input channels.
        feature_dim (int): Dimension of the output feature vector.
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the CNNEncoder.
        
        Args:
            config: Configuration dictionary containing model parameters.
        """
        super(CNNEncoder, self).__init__()
        
        self.input_channels = config.get("input_channels", 1)
        self.feature_dim = config.get("feature_dim", 128)
        
        # Three convolutional blocks
        self.conv_block1 = ConvBlock(self.input_channels, 32)
        self.conv_block2 = ConvBlock(32, 64)
        self.conv_block3 = ConvBlock(64, 128)
        
        # Adaptive average pooling to get fixed-size output
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layer to project to feature_dim
        self.fc = nn.Linear(128, self.feature_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the CNN encoder.
        
        Args:
            x: Input tensor of shape (batch, input_channels, 128, 128).
            
        Returns:
            Output tensor of shape (batch, feature_dim).
        """
        # Pass through convolutional blocks
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        
        # Adaptive average pooling
        x = self.adaptive_pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Project to feature dimension
        x = self.fc(x)
        
        return x
