"""
Feature extraction module for audio processing.
"""

import numpy as np
import librosa
from typing import Dict, Any


def extract_mel_spectrogram(
    audio: np.ndarray,
    sr: int,
    n_mels: int = 128,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    """
    Extract Mel spectrogram from audio waveform.
    
    Args:
        audio: Input audio waveform.
        sr: Sample rate of the audio.
        n_mels: Number of Mel frequency bands.
        n_fft: FFT window size.
        hop_length: Hop length for STFT computation.
        
    Returns:
        Log-scale Mel spectrogram.
    """
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    
    # Convert to log scale (dB)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    return mel_spec_db


def extract_features(
    audio: np.ndarray,
    sr: int,
    config: Dict[str, Any],
) -> np.ndarray:
    """
    Extract features from audio based on configuration.
    
    Args:
        audio: Input audio waveform.
        sr: Sample rate of the audio.
        config: Configuration dictionary containing feature parameters.
        
    Returns:
        Extracted features.
    """
    feature_type = config.get("type", "mel_spectrogram")
    n_mels = config.get("n_mels", 128)
    n_fft = config.get("n_fft", 2048)
    hop_length = config.get("hop_length", 512)
    
    if feature_type == "mel_spectrogram":
        return extract_mel_spectrogram(
            audio=audio,
            sr=sr,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
        )
    else:
        raise ValueError(f"Unknown feature type: {feature_type}")


def normalize_features(features: np.ndarray) -> np.ndarray:
    """
    Normalize features using mean and standard deviation.
    
    Args:
        features: Input features.
        
    Returns:
        Normalized features.
    """
    mean = np.mean(features)
    std = np.std(features)
    normalized = (features - mean) / (std + 1e-6)
    return normalized


def resize_features(features: np.ndarray, target_shape: tuple) -> np.ndarray:
    """
    Resize features to target shape.
    
    Args:
        features: Input features.
        target_shape: Target shape (height, width).
        
    Returns:
        Resized features.
    """
    current_shape = features.shape
    
    if current_shape == target_shape:
        return features
    
    # Create output array
    output = np.zeros(target_shape, dtype=np.float32)
    
    # Copy overlapping region
    h = min(current_shape[0], target_shape[0])
    w = min(current_shape[1], target_shape[1])
    output[:h, :w] = features[:h, :w]
    
    return output
