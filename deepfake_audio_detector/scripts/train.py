"""
Training script for the deepfake audio detector.

This script:
1. Downloads the dataset using kagglehub
2. Builds file paths dynamically
3. Creates labels
4. Performs train/test split (80/20, stratified)
5. Initializes Dataset and DataLoader
6. Initializes model
7. Trains model
8. Saves best checkpoint
"""

import os
import sys
import yaml
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import kagglehub

from src.preprocessing.dataset import DeepfakeAudioDataset
from src.models.full_model import build_model, count_parameters
from src.training.trainer import Trainer


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
    Main training function.
    """
    # Get project directories
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    config_path = os.path.join(project_root, "configs", "config.yaml")
    
    # Load configuration
    print("Loading configuration...")
    config = load_config(config_path)
    
    # Set random seed
    seed = config.get("training", {}).get("random_seed", 42)
    set_seed(seed)
    
    # Download dataset
    print("\nDownloading dataset from Kaggle...")
    dataset_path = kagglehub.dataset_download("birdy654/deep-voice-deepfake-voice-recognition")
    print(f"Dataset downloaded to: {dataset_path}")
    
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
    
    # Get labels for stratified split
    labels = full_dataset.labels
    
    # Perform stratified train/test split
    val_split = config.get("training", {}).get("val_split", 0.2)
    
    indices = list(range(len(full_dataset)))
    train_indices, val_indices = train_test_split(
        indices,
        test_size=val_split,
        stratify=labels,
        random_state=seed,
    )
    
    print(f"Training samples: {len(train_indices)}")
    print(f"Validation samples: {len(val_indices)}")
    
    # Create subsets
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    
    # Create data loaders
    batch_size = config.get("training", {}).get("batch_size", 32)
    num_workers = config.get("training", {}).get("num_workers", 4)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    # Build model
    print("\nBuilding model...")
    model = build_model(config)
    num_params = count_parameters(model)
    print(f"Model built with {num_params:,} trainable parameters")
    
    # Create trainer
    trainer = Trainer(model, config)
    
    # Get checkpoint directory
    checkpoint_dir = os.path.join(project_root, config.get("paths", {}).get("checkpoint_dir", "checkpoints"))
    
    # Train model
    print("\nStarting training...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        checkpoint_dir=checkpoint_dir,
    )
    
    print(f"\nTraining completed! Best model saved to: {os.path.join(checkpoint_dir, 'best_model.pth')}")


if __name__ == "__main__":
    main()
