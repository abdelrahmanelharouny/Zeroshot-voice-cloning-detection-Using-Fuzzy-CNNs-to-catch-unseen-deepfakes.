"""
Metrics module for computing evaluation metrics.
"""

import numpy as np
from typing import Dict, Any


def compute_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute classification metrics.
    
    Args:
        predictions: Predicted probabilities.
        targets: Ground truth labels.
        threshold: Classification threshold.
        
    Returns:
        Dictionary containing accuracy, precision, recall, and f1-score.
    """
    # Convert probabilities to binary predictions
    pred_binary = (predictions >= threshold).astype(int)
    
    # Compute true positives, false positives, true negatives, false negatives
    tp = np.sum((pred_binary == 1) & (targets == 1))
    fp = np.sum((pred_binary == 1) & (targets == 0))
    tn = np.sum((pred_binary == 0) & (targets == 0))
    fn = np.sum((pred_binary == 0) & (targets == 1))
    
    # Compute accuracy
    total = tp + fp + tn + fn
    accuracy = (tp + tn) / (total + 1e-6)
    
    # Compute precision
    precision = tp / (tp + fp + 1e-6)
    
    # Compute recall
    recall = tp / (tp + fn + 1e-6)
    
    # Compute F1-score
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0
    
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def compute_confusion_matrix(
    predictions: np.ndarray,
    targets: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Compute confusion matrix.
    
    Args:
        predictions: Predicted probabilities.
        targets: Ground truth labels.
        threshold: Classification threshold.
        
    Returns:
        Confusion matrix as 2x2 numpy array:
        [[TN, FP],
         [FN, TP]]
    """
    pred_binary = (predictions >= threshold).astype(int)
    
    tp = np.sum((pred_binary == 1) & (targets == 1))
    fp = np.sum((pred_binary == 1) & (targets == 0))
    tn = np.sum((pred_binary == 0) & (targets == 0))
    fn = np.sum((pred_binary == 0) & (targets == 1))
    
    return np.array([[tn, fp], [fn, tp]])


def print_metrics(metrics: Dict[str, float]) -> None:
    """
    Print metrics in a formatted way.
    
    Args:
        metrics: Dictionary of metrics to print.
    """
    print("\n" + "=" * 40)
    print("Evaluation Metrics")
    print("=" * 40)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1']:.4f}")
    print("-" * 40)
    print(f"True Positives:  {metrics['tp']}")
    print(f"False Positives: {metrics['fp']}")
    print(f"True Negatives:  {metrics['tn']}")
    print(f"False Negatives: {metrics['fn']}")
    
    # Print uncertainty if available
    if "avg_uncertainty" in metrics:
        print(f"Avg Uncertainty: {metrics['avg_uncertainty']:.4f}")
    
    print("=" * 40 + "\n")
