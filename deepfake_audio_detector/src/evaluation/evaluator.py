"""
Evaluator module for evaluating the deepfake detection model.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Tuple
import numpy as np

from src.utils.metrics import compute_metrics


class Evaluator:
    """
    Evaluator class for evaluating the deepfake detection model.
    
    Attributes:
        model (nn.Module): The model to evaluate.
        device (torch.device): Device to use for evaluation.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device = None,
    ) -> None:
        """
        Initialize the Evaluator.
        
        Args:
            model: Model to evaluate.
            device: Device to use for evaluation (default: cuda if available).
        """
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Move model to device
        self.model.to(self.device)
        self.model.eval()
        
    def evaluate(self, data_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate the model on a dataset.
        
        Args:
            data_loader: DataLoader for evaluation data.
            
        Returns:
            Dictionary containing evaluation metrics.
        """
        all_predictions = []
        all_targets = []
        all_uncertainties = []
        
        with torch.no_grad():
            for features, labels in data_loader:
                # Move data to device
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass - model returns dict with prediction and uncertainty
                outputs = self.model(features)
                
                # Extract predictions
                predictions = outputs["prediction"]
                
                # Store predictions and targets
                all_predictions.extend(predictions.cpu().detach().numpy())
                all_targets.extend(labels.cpu().detach().numpy())
                
                # Store uncertainties if available
                if "uncertainty" in outputs:
                    all_uncertainties.extend(outputs["uncertainty"].cpu().detach().numpy())
        
        # Convert to numpy arrays
        predictions = np.array(all_predictions)
        targets = np.array(all_targets)
        
        # Compute metrics
        metrics = compute_metrics(predictions, targets)
        
        # Add average uncertainty if available
        if len(all_uncertainties) > 0:
            metrics["avg_uncertainty"] = float(np.mean(all_uncertainties))
        
        return metrics
    
    def predict(self, features: torch.Tensor) -> np.ndarray:
        """
        Make predictions on input features.
        
        Args:
            features: Input features tensor of shape (batch, channels, height, width).
            
        Returns:
            Predicted probabilities as numpy array.
        """
        self.model.eval()
        
        with torch.no_grad():
            features = features.to(self.device)
            outputs = self.model(features)
            predictions = outputs["prediction"].cpu().detach().numpy()
        
        return predictions
    
    def predict_class(
        self, features: torch.Tensor, threshold: float = 0.5
    ) -> np.ndarray:
        """
        Make class predictions on input features.
        
        Args:
            features: Input features tensor of shape (batch, channels, height, width).
            threshold: Classification threshold.
            
        Returns:
            Predicted classes as numpy array (0 or 1).
        """
        probabilities = self.predict(features)
        classes = (probabilities >= threshold).astype(int)
        return classes
    
    def predict_with_uncertainty(self, features: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with uncertainty estimates.
        
        Args:
            features: Input features tensor of shape (batch, channels, height, width).
            
        Returns:
            Tuple of (predictions, uncertainties) as numpy arrays.
        """
        self.model.eval()
        
        with torch.no_grad():
            features = features.to(self.device)
            outputs = self.model(features)
            predictions = outputs["prediction"].cpu().detach().numpy()
            uncertainties = outputs.get("uncertainty", None)
            if uncertainties is not None:
                uncertainties = uncertainties.cpu().detach().numpy()
            else:
                uncertainties = np.zeros_like(predictions)
        
        return predictions, uncertainties
