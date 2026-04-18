"""
Evaluation script for the deepfake audio detector.

This script:
1. Loads saved model checkpoint
2. Evaluates on test set
3. Prints all metrics
"""

import os
import sys
import yaml
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import kagglehub

from src.preprocessing.dataset import DeepfakeAudioDataset
from src.models.full_model import build_model
from src.evaluation.evaluator import Evaluator
from src.utils.metrics import print_metrics


def load_config(config_path: str) -> dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the configuration file.
        
    Returns:
        Configuration dictionary.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    """
    Main evaluation function.
    """
    # Get project directories
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    config_path = os.path.join(project_root, "configs", "config.yaml")
    
    # Parse command line arguments for checkpoint path
    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]
    else:
        checkpoint_path = os.path.join(project_root, "checkpoints", "best_model.pth")
    
    # Load configuration
    print("Loading configuration...")
    config = load_config(config_path)
    
    # Set random seed
    seed = config.get("training", {}).get("random_seed", 42)
    set_seed(seed)
    
    # Download dataset (if not already downloaded)
    print("\nDownloading/loading dataset from Kaggle...")
    dataset_path = kagglehub.dataset_download("birdy654/deep-voice-deepfake-voice-recognition")
    print(f"Dataset location: {dataset_path}")
    
    # Get audio parameters from config
    audio_config = config.get("audio", {})
    sample_rate = audio_config.get("sample_rate", 16000)
    duration = audio_config.get("duration", 3.0)
    
    feature_config = config.get("features", {})
    n_mels = feature_config.get("n_mels", 128)
    n_fft = feature_config.get("n_fft", 2048)
    hop_length = feature_config.get("hop_length", 512)
    
    # Create full dataset
    print("\nCreating dataset...")
    full_dataset = DeepfakeAudioDataset(
        dataset_path=dataset_path,
        sample_rate=sample_rate,
        duration=duration,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    
    print(f"Total samples found: {len(full_dataset)}")
    
    if len(full_dataset) == 0:
        print("Error: No audio files found in the dataset!")
        sys.exit(1)
    
    # For evaluation, we'll use the full dataset
    # In a real scenario, you might have a separate test set
    batch_size = config.get("training", {}).get("batch_size", 32)
    num_workers = config.get("training", {}).get("num_workers", 4)
    
    test_loader = DataLoader(
        full_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    # Build model
    print("\nBuilding model...")
    model = build_model(config)
    
    # Load checkpoint
    print(f"\nLoading checkpoint from: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print("Please train the model first using: python scripts/train.py")
        sys.exit(1)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print("Checkpoint loaded successfully!")
    
    # Create evaluator
    evaluator = Evaluator(model, device)
    
    # Evaluate
    print("\nEvaluating model...")
    metrics = evaluator.evaluate(test_loader)
    
    # Print metrics
    print_metrics(metrics)
    
    print(f"Evaluation completed on {len(full_dataset)} samples.")


if __name__ == "__main__":
    main()
