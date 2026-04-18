"""
Trainer module for training the deepfake detection model with uncertainty-aware loss.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Tuple, Optional
import numpy as np

from src.training.loss import UncertaintyAwareBCELoss
from src.utils.metrics import compute_metrics


class Trainer:
    """
    Trainer class for training and validating the deepfake detection model.
    
    Supports uncertainty-aware training where the loss can be weighted by
    the model's uncertainty estimates from the fuzzy layer.
    
    Attributes:
        model (nn.Module): The model to train.
        config (Dict[str, Any]): Configuration dictionary.
        device (torch.device): Device to use for training.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        criterion (nn.Module): Loss function.
        uncertainty_weight (float): Weight for uncertainty regularization.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Initialize the Trainer.
        
        Args:
            model: Model to train.
            config: Configuration dictionary.
            device: Device to use for training (default: cuda if available).
        """
        self.model = model
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Training parameters
        training_config = config.get("training", {})
        self.lr = training_config.get("lr", 0.0001)
        self.weight_decay = training_config.get("weight_decay", 0.0001)
        self.epochs = training_config.get("epochs", 30)
        self.uncertainty_weight = training_config.get("uncertainty_weight", 0.01)
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        
        # Initialize loss function (supports uncertainty-aware training)
        self.criterion = UncertaintyAwareBCELoss(uncertainty_weight=self.uncertainty_weight)
        
        # Move model to device
        self.model.to(self.device)
        
        # Best validation metrics
        self.best_val_loss = float("inf")
        self.best_val_accuracy = 0.0
        
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        Train the model for one epoch.
        
        Args:
            train_loader: DataLoader for training data.
            
        Returns:
            Dictionary containing average training loss and accuracy.
        """
        self.model.train()
        
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        for batch_idx, (features, labels) in enumerate(train_loader):
            # Move data to device
            features = features.to(self.device)
            labels = labels.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass - model now returns dict with prediction and uncertainty
            outputs = self.model(features)
            predictions = outputs["prediction"]
            
            # Compute loss (includes uncertainty regularization if available)
            uncertainty = outputs.get("uncertainty", None)
            loss = self.criterion(predictions, labels, uncertainty)
            
            # Backward pass
            loss.backward()
            
            # Update weights
            self.optimizer.step()
            
            # Accumulate loss
            total_loss += loss.item()
            
            # Store predictions and targets for metrics
            all_predictions.extend(predictions.cpu().detach().numpy())
            all_targets.extend(labels.cpu().detach().numpy())
        
        # Compute metrics
        avg_loss = total_loss / len(train_loader)
        metrics = compute_metrics(
            np.array(all_predictions),
            np.array(all_targets),
        )
        
        return {
            "loss": avg_loss,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
        }
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate the model on validation data.
        
        Args:
            val_loader: DataLoader for validation data.
            
        Returns:
            Dictionary containing validation loss and metrics.
        """
        self.model.eval()
        
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_idx, (features, labels) in enumerate(val_loader):
                # Move data to device
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(features)
                predictions = outputs["prediction"]
                uncertainty = outputs.get("uncertainty", None)
                
                # Compute loss
                loss = self.criterion(predictions, labels, uncertainty)
                
                # Accumulate loss
                total_loss += loss.item()
                
                # Store predictions and targets
                all_predictions.extend(predictions.cpu().detach().numpy())
                all_targets.extend(labels.cpu().detach().numpy())
        
        # Compute metrics
        avg_loss = total_loss / len(val_loader)
        metrics = compute_metrics(
            np.array(all_predictions),
            np.array(all_targets),
        )
        
        return {
            "loss": avg_loss,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
        }
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        checkpoint_dir: str = "checkpoints",
    ) -> Dict[str, Any]:
        """
        Full training loop with checkpoint saving.
        
        Args:
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data.
            checkpoint_dir: Directory to save checkpoints.
            
        Returns:
            Dictionary containing training history.
        """
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Training history
        history = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
        }
        
        print(f"Starting training for {self.epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Uncertainty weight: {self.uncertainty_weight}")
        
        for epoch in range(self.epochs):
            # Train
            train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Update history
            history["train_loss"].append(train_metrics["loss"])
            history["train_accuracy"].append(train_metrics["accuracy"])
            history["val_loss"].append(val_metrics["loss"])
            history["val_accuracy"].append(val_metrics["accuracy"])
            
            # Print progress
            print(
                f"Epoch {epoch + 1}/{self.epochs} | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Train Acc: {train_metrics['accuracy']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val Acc: {val_metrics['accuracy']:.4f}"
            )
            
            # Save best checkpoint
            if val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["loss"]
                self.best_val_accuracy = val_metrics["accuracy"]
                
                checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")
                self.save_checkpoint(checkpoint_path)
                print(f"  -> Saved new best model (Val Loss: {self.best_val_loss:.4f})")
        
        print(f"\nTraining completed!")
        print(f"Best Validation Loss: {self.best_val_loss:.4f}")
        print(f"Best Validation Accuracy: {self.best_val_accuracy:.4f}")
        
        return history
    
    def save_checkpoint(self, path: str) -> None:
        """
        Save a model checkpoint.
        
        Args:
            path: Path to save the checkpoint.
        """
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "best_val_accuracy": self.best_val_accuracy,
            "config": self.config,
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str) -> None:
        """
        Load a model checkpoint.
        
        Args:
            path: Path to the checkpoint file.
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.best_val_loss = checkpoint["best_val_loss"]
        self.best_val_accuracy = checkpoint["best_val_accuracy"]
