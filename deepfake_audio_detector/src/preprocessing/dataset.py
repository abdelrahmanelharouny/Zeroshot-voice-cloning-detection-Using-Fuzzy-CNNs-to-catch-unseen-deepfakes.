"""
Dataset module for loading and preprocessing audio files.
"""

import os
from typing import Tuple, List
import numpy as np
import librosa
import torch
from torch.utils.data import Dataset


class DeepfakeAudioDataset(Dataset):
    """
    PyTorch Dataset for loading and preprocessing deepfake audio files.
    
    This dataset scans directories for real and fake audio files,
    loads them using librosa, and extracts Mel spectrograms.
    
    Attributes:
        dataset_path (str): Path to the root dataset directory.
        sample_rate (int): Target sample rate for resampling.
        duration (float): Duration in seconds to fix audio length.
        n_mels (int): Number of Mel bands for spectrogram extraction.
    """
    
    def __init__(
        self,
        dataset_path: str,
        sample_rate: int = 16000,
        duration: float = 3.0,
        n_mels: int = 128,
        n_fft: int = 2048,
        hop_length: int = 512,
    ) -> None:
        """
        Initialize the DeepfakeAudioDataset.
        
        Args:
            dataset_path: Root path containing 'real' and 'fake' subdirectories.
            sample_rate: Target sample rate for audio resampling.
            duration: Fixed duration in seconds for all audio samples.
            n_mels: Number of Mel frequency bands.
            n_fft: FFT window size for spectrogram computation.
            hop_length: Hop length for STFT computation.
        """
        self.dataset_path = dataset_path
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        self.file_paths: List[str] = []
        self.labels: List[int] = []
        
        self._scan_directories()
        
    def _scan_directories(self) -> None:
        """
        Scan the real and fake directories to collect file paths and labels.
        
        Labels:
            real = 0
            fake = 1
        """
        real_path = os.path.join(self.dataset_path, "real")
        fake_path = os.path.join(self.dataset_path, "fake")
        
        if os.path.exists(real_path):
            for filename in os.listdir(real_path):
                if filename.endswith((".wav", ".mp3", ".flac", ".ogg")):
                    self.file_paths.append(os.path.join(real_path, filename))
                    self.labels.append(0)
                    
        if os.path.exists(fake_path):
            for filename in os.listdir(fake_path):
                if filename.endswith((".wav", ".mp3", ".flac", ".ogg")):
                    self.file_paths.append(os.path.join(fake_path, filename))
                    self.labels.append(1)
                    
    def _load_audio(self, file_path: str) -> np.ndarray:
        """
        Load audio file and resample to target sample rate.
        
        Args:
            file_path: Path to the audio file.
            
        Returns:
            Audio waveform as numpy array.
        """
        audio, sr = librosa.load(file_path, sr=self.sample_rate, mono=True)
        return audio
    
    def _fix_duration(self, audio: np.ndarray) -> np.ndarray:
        """
        Fix audio length to exactly the specified duration.
        
        If shorter: zero-pad
        If longer: truncate
        
        Args:
            audio: Input audio waveform.
            
        Returns:
            Audio waveform with fixed duration.
        """
        target_samples = int(self.sample_rate * self.duration)
        
        if len(audio) < target_samples:
            padded = np.zeros(target_samples, dtype=np.float32)
            padded[:len(audio)] = audio
            return padded
        else:
            return audio[:target_samples]
    
    def _extract_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract Mel spectrogram from audio waveform.
        
        Args:
            audio: Input audio waveform.
            
        Returns:
            Log-scale Mel spectrogram.
        """
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )
        
        # Convert to log scale (dB)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        return mel_spec_db
    
    def _normalize(self, spectrogram: np.ndarray) -> np.ndarray:
        """
        Normalize spectrogram using mean and standard deviation.
        
        Args:
            spectrogram: Input spectrogram.
            
        Returns:
            Normalized spectrogram.
        """
        mean = np.mean(spectrogram)
        std = np.std(spectrogram)
        normalized = (spectrogram - mean) / (std + 1e-6)
        return normalized
    
    def __len__(self) -> int:
        """
        Return the total number of samples in the dataset.
        
        Returns:
            Number of samples.
        """
        return len(self.file_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Index of the sample.
            
        Returns:
            Tuple of (feature_tensor, label_tensor).
            feature_tensor shape: (1, 128, 128)
            label_tensor shape: ()
        """
        file_path = self.file_paths[idx]
        label = self.labels[idx]
        
        # Load and preprocess audio
        audio = self._load_audio(file_path)
        audio = self._fix_duration(audio)
        
        # Extract Mel spectrogram
        mel_spec = self._extract_mel_spectrogram(audio)
        
        # Resize to 128x128 if needed
        if mel_spec.shape[1] != 128:
            mel_spec = librosa.resample(
                mel_spec,
                orig_sr=self.sample_rate // self.hop_length,
                target_sr=128,
            )
        
        # Ensure exact shape (128, 128)
        if mel_spec.shape[0] > 128:
            mel_spec = mel_spec[:128, :]
        if mel_spec.shape[1] > 128:
            mel_spec = mel_spec[:, :128]
        
        # Pad if necessary
        mel_spec_padded = np.zeros((128, 128), dtype=np.float32)
        h = min(mel_spec.shape[0], 128)
        w = min(mel_spec.shape[1], 128)
        mel_spec_padded[:h, :w] = mel_spec[:h, :w]
        
        # Normalize
        mel_spec_normalized = self._normalize(mel_spec_padded)
        
        # Convert to tensor and add channel dimension
        feature_tensor = torch.FloatTensor(mel_spec_normalized).unsqueeze(0)
        label_tensor = torch.FloatTensor([label]).squeeze()
        
        return feature_tensor, label_tensor
