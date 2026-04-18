# Audio Deepfake Detection using CNN + Fuzzy Logic

A production-grade deep learning system for detecting audio deepfakes using Convolutional Neural Networks (CNN) combined with Fuzzy Logic.

## Project Description

This project implements a robust audio deepfake detection system that:
- Uses CNN for feature extraction from Mel spectrograms
- Incorporates a Fuzzy Logic layer with learnable Gaussian membership functions
- Provides binary classification (real vs fake audio)

The system is fully modular, configurable, and reproducible.

## Dataset

This project uses the **birdy654/deep-voice-deepfake-voice-recognition** dataset from Kaggle.

Dataset structure:
```
dataset_path/
├── real/    # Real audio samples (label = 0)
└── fake/    # Fake/deepfake audio samples (label = 1)
```

The dataset is automatically downloaded using `kagglehub` when running the training or evaluation scripts.

## Project Structure

```
deepfake_audio_detector/
│
├── configs/
│   └── config.yaml          # Configuration file
│
├── src/
│   ├── preprocessing/
│   │   ├── dataset.py       # PyTorch Dataset class
│   │   └── audio_utils.py   # Audio utilities
│   │
│   ├── features/
│   │   └── feature_extractor.py  # Feature extraction functions
│   │
│   ├── models/
│   │   ├── cnn_encoder.py   # CNN feature extractor
│   │   ├── classifier.py    # Fully connected classifier
│   │   └── full_model.py    # Complete model architecture
│   │
│   ├── fuzzy/
│   │   └── fuzzy_layer.py   # Fuzzy logic layer
│   │
│   ├── training/
│   │   ├── trainer.py       # Training loop
│   │   └── loss.py          # Loss functions
│   │
│   ├── evaluation/
│   │   └── evaluator.py     # Evaluation utilities
│   │
│   └── utils/
│       └── metrics.py       # Metrics computation
│
├── scripts/
│   ├── train.py             # Training script
│   └── evaluate.py          # Evaluation script
│
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

## Architecture

### Model Components

1. **CNN Encoder**: 3 convolutional blocks (Conv2D → ReLU → BatchNorm → MaxPool) followed by AdaptiveAvgPool2d
2. **Fuzzy Layer**: Learnable Gaussian membership functions with centers and sigmas
3. **Classifier**: Fully connected layers (128 → 64 → 1) with Sigmoid activation

### Audio Processing Pipeline

1. Load audio using librosa
2. Resample to 16000 Hz
3. Fix duration to exactly 3 seconds (zero-pad or truncate)
4. Extract Mel Spectrogram (n_mels=128)
5. Convert to log scale (dB)
6. Normalize: (x - mean) / (std + 1e-6)
7. Output tensor shape: (1, 128, 128)

## Setup Instructions

### Prerequisites

- Python 3.10+
- pip package manager

### Installation

1. Clone or navigate to the project directory:
```bash
cd deepfake_audio_detector
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure Kaggle API credentials (required for dataset download):
   - Go to https://www.kaggle.com/account
   - Create new API token
   - Download kaggle.json
   - Place it in ~/.kaggle/kaggle.json (Linux/Mac) or C:\Users\<YourUsername>\.kaggle\kaggle.json (Windows)
   - Set permissions: `chmod 600 ~/.kaggle/kaggle.json`

## Training

To train the model:

```bash
cd deepfake_audio_detector
python scripts/train.py
```

The training script will:
1. Download the dataset from Kaggle
2. Preprocess audio files
3. Split data into train/validation (80/20, stratified)
4. Train the model for configured epochs
5. Save the best checkpoint to `checkpoints/best_model.pth`

### Configuration

Edit `configs/config.yaml` to modify hyperparameters:

```yaml
model:
  input_channels: 1
  feature_dim: 128

training:
  batch_size: 32
  lr: 0.0001
  epochs: 30

audio:
  sample_rate: 16000
  duration: 3

features:
  type: mel_spectrogram
  n_mels: 128
```

## Evaluation

To evaluate the trained model:

```bash
python scripts/evaluate.py
```

Or specify a custom checkpoint path:

```bash
python scripts/evaluate.py /path/to/checkpoint.pth
```

The evaluation script will:
1. Load the saved model checkpoint
2. Evaluate on the dataset
3. Print accuracy, precision, recall, and F1-score

## Output Metrics

The evaluation outputs:
- **Accuracy**: Overall classification accuracy
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- Confusion matrix values (TP, FP, TN, FN)

## Requirements

- torch >= 2.0.0
- librosa >= 0.10.0
- numpy >= 1.24.0
- scikit-learn >= 1.3.0
- pyyaml >= 6.0
- kagglehub >= 0.1.0

## License

This project is provided as-is for educational and research purposes.

## Citation

If you use this code in your research, please cite the dataset:
- Dataset: birdy654/deep-voice-deepfake-voice-recognition on Kaggle
