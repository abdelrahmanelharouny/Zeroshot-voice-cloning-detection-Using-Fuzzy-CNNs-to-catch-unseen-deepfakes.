"""
Audio utilities for loading and processing audio files.
"""

import numpy as np
import librosa
from typing import Tuple


def load_audio(file_path: str, sample_rate: int = 16000) -> Tuple[np.ndarray, int]:
    """
    Load an audio file from disk.
    
    Args:
        file_path: Path to the audio file.
        sample_rate: Target sample rate for resampling.
        
    Returns:
        Tuple of (audio waveform, sample rate).
    """
    audio, sr = librosa.load(file_path, sr=sample_rate, mono=True)
    return audio, sr


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """
    Resample audio to a different sample rate.
    
    Args:
        audio: Input audio waveform.
        orig_sr: Original sample rate.
        target_sr: Target sample rate.
        
    Returns:
        Resampled audio waveform.
    """
    resampled = librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
    return resampled


def fix_duration(
    audio: np.ndarray, sample_rate: int, duration: float
) -> np.ndarray:
    """
    Fix audio length to exactly the specified duration.
    
    If shorter: zero-pad
    If longer: truncate
    
    Args:
        audio: Input audio waveform.
        sample_rate: Sample rate of the audio.
        duration: Target duration in seconds.
        
    Returns:
        Audio waveform with fixed duration.
    """
    target_samples = int(sample_rate * duration)
    
    if len(audio) < target_samples:
        padded = np.zeros(target_samples, dtype=np.float32)
        padded[:len(audio)] = audio
        return padded
    else:
        return audio[:target_samples]


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """
    Normalize audio waveform to have zero mean and unit variance.
    
    Args:
        audio: Input audio waveform.
        
    Returns:
        Normalized audio waveform.
    """
    mean = np.mean(audio)
    std = np.std(audio)
    normalized = (audio - mean) / (std + 1e-6)
    return normalized
