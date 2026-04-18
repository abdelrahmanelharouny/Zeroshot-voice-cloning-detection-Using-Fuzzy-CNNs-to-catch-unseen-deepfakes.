"""
Full Model module combining CNN Encoder, Neuro-Fuzzy Inference System, and Classifier.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Tuple

from src.models.cnn_encoder import CNNEncoder
from src.fuzzy.fuzzy_layer import NeuroFuzzyInferenceSystem
from src.models.classifier import Classifier


class DeepfakeDetector(nn.Module):
    """
    Complete Deepfake Detection Model with Neuro-Fuzzy Inference System.
    
    Architecture:
        CNNEncoder -> NeuroFuzzyInferenceSystem -> Classifier
    
    The model returns a dictionary containing:
        - prediction: Classification probability
        - uncertainty: Uncertainty estimate from fuzzy layer
    
    Attributes:
        cnn_encoder (CNNEncoder): CNN feature extractor.
        fuzzy_layer (NeuroFuzzyInferenceSystem): Advanced fuzzy inference system.
        classifier (Classifier): Final classification layer.
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the DeepfakeDetector.
        
        Args:
            config: Configuration dictionary containing all model parameters.
        """
        super(DeepfakeDetector, self).__init__()
        
        self.config = config
        
        # Get configuration values
        model_config = config.get("model", {})
        fuzzy_config = config.get("fuzzy", {})
        
        feature_dim = model_config.get("feature_dim", 128)
        num_memberships = fuzzy_config.get("num_memberships", 3)
        num_rules = fuzzy_config.get("num_rules", 32)
        sigma_init = fuzzy_config.get("sigma_init", 0.5)
        
        # CNN Encoder: (batch, 1, 128, 128) -> (batch, feature_dim)
        self.cnn_encoder = CNNEncoder(config)
        
        # Neuro-Fuzzy Inference System
        self.fuzzy_layer = NeuroFuzzyInferenceSystem(
            feature_dim=feature_dim,
            num_memberships=num_memberships,
            num_rules=num_rules,
            sigma_init=sigma_init,
        )
        
        # Classifier: (batch, feature_dim) -> (batch,)
        self.classifier = Classifier(config)
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the complete model.
        
        Args:
            x: Input tensor of shape (batch, input_channels, 128, 128).
            
        Returns:
            Dictionary containing:
                - prediction: Tensor of shape (batch,) with classification probabilities
                - uncertainty: Tensor of shape (batch_size, 1) with uncertainty estimates
        """
        # Extract features using CNN
        features = self.cnn_encoder(x)  # (batch, feature_dim)
        
        # Apply neuro-fuzzy inference system
        fuzzy_out = self.fuzzy_layer(features)
        
        # Get refined features and uncertainty
        refined_features = fuzzy_out["refined_features"]  # (batch, feature_dim)
        uncertainty = fuzzy_out["uncertainty"]  # (batch, 1)
        
        # Classify using refined features
        prediction = self.classifier(refined_features)  # (batch,)
        
        return {
            "prediction": prediction,
            "uncertainty": uncertainty,
        }
    
    def get_feature_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get feature embeddings from the CNN encoder.
        
        Args:
            x: Input tensor of shape (batch, input_channels, 128, 128).
            
        Returns:
            Feature embeddings of shape (batch, feature_dim).
        """
        return self.cnn_encoder(x)
    
    def get_fuzzy_output(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get full fuzzy layer output including refined features, uncertainty, and rule activations.
        
        Args:
            x: Input tensor of shape (batch, input_channels, 128, 128).
            
        Returns:
            Dictionary containing:
                - refined_features: Tensor of shape (batch, feature_dim)
                - uncertainty: Tensor of shape (batch, 1)
                - rule_activations: Tensor of shape (batch, num_rules)
        """
        features = self.cnn_encoder(x)
        return self.fuzzy_layer(features)
    
    def explain_rules(self) -> Dict[str, Any]:
        """
        Get interpretable explanations of the learned fuzzy rules.
        
        Returns:
            Dictionary containing rule explanations from the fuzzy layer.
        """
        return self.fuzzy_layer.explain_rules()


def build_model(config: Dict[str, Any]) -> DeepfakeDetector:
    """
    Build the deepfake detection model from configuration.
    
    Args:
        config: Configuration dictionary.
        
    Returns:
        Initialized DeepfakeDetector model.
    """
    return DeepfakeDetector(config)


def count_parameters(model: nn.Module) -> int:
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model.
        
    Returns:
        Number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
