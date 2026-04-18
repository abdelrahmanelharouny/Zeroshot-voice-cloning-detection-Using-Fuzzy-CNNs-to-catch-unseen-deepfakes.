"""
Loss functions for training the deepfake detection model with uncertainty awareness.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional


class BinaryCrossEntropyLoss(nn.Module):
    """
    Binary Cross Entropy Loss for binary classification.
    
    This is a wrapper around PyTorch's BCELoss with optional label smoothing.
    
    Attributes:
        reduction (str): Reduction method ('mean', 'sum', or 'none').
    """
    
    def __init__(self, config: Dict[str, Any] = None) -> None:
        """
        Initialize the BinaryCrossEntropyLoss.
        
        Args:
            config: Configuration dictionary (unused for this loss).
        """
        super(BinaryCrossEntropyLoss, self).__init__()
        
        self.loss_fn = nn.BCELoss()
        
    def forward(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the binary cross entropy loss.
        
        Args:
            predictions: Predicted probabilities of shape (batch,).
            targets: Ground truth labels of shape (batch,).
            
        Returns:
            Scalar loss value.
        """
        # Ensure predictions are in valid range for numerical stability
        predictions_clamped = torch.clamp(predictions, 1e-7, 1 - 1e-7)
        
        loss = self.loss_fn(predictions_clamped, targets)
        
        return loss


class UncertaintyAwareBCELoss(nn.Module):
    """
    Uncertainty-aware Binary Cross Entropy Loss.
    
    This loss function incorporates the model's uncertainty estimates into
    the training objective. It can penalize high uncertainty or use uncertainty
    to weight the classification loss.
    
    The total loss is computed as:
        total_loss = bce_loss + uncertainty_weight * uncertainty_penalty
    
    Where the uncertainty penalty encourages lower uncertainty for confident predictions.
    
    Attributes:
        uncertainty_weight (float): Weight for the uncertainty regularization term.
        eps (float): Small constant for numerical stability.
    """
    
    def __init__(
        self,
        uncertainty_weight: float = 0.01,
        eps: float = 1e-8,
    ) -> None:
        """
        Initialize the UncertaintyAwareBCELoss.
        
        Args:
            uncertainty_weight: Weight for the uncertainty regularization term.
            eps: Small constant for numerical stability.
        """
        super(UncertaintyAwareBCELoss, self).__init__()
        
        self.uncertainty_weight = uncertainty_weight
        self.eps = eps
        self.bce_loss = nn.BCELoss()
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        uncertainty: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute the uncertainty-aware binary cross entropy loss.
        
        Args:
            predictions: Predicted probabilities of shape (batch,).
            targets: Ground truth labels of shape (batch,).
            uncertainty: Optional uncertainty estimates of shape (batch, 1) or (batch,).
                        If None, falls back to standard BCE loss.
            
        Returns:
            Scalar loss value combining BCE and uncertainty regularization.
        """
        # Ensure predictions are in valid range for numerical stability
        predictions_clamped = torch.clamp(predictions, self.eps, 1 - self.eps)
        
        # Compute standard BCE loss
        bce_loss = self.bce_loss(predictions_clamped, targets)
        
        # If uncertainty is provided, add uncertainty regularization
        if uncertainty is not None:
            # Squeeze uncertainty to match batch dimension if needed
            if uncertainty.dim() == 2:
                uncertainty = uncertainty.squeeze(-1)
            
            # Uncertainty penalty: encourage lower uncertainty
            # Higher uncertainty leads to higher loss
            uncertainty_penalty = uncertainty.mean()
            
            # Total loss
            total_loss = bce_loss + self.uncertainty_weight * uncertainty_penalty
        else:
            total_loss = bce_loss
        
        return total_loss


def build_loss(config: Dict[str, Any] = None) -> nn.Module:
    """
    Build the loss function from configuration.
    
    Args:
        config: Configuration dictionary.
        
    Returns:
        Initialized loss function.
    """
    if config is None:
        config = {}
    
    training_config = config.get("training", {})
    uncertainty_weight = training_config.get("uncertainty_weight", 0.01)
    
    return UncertaintyAwareBCELoss(uncertainty_weight=uncertainty_weight)
