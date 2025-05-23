# Line-Level Handwritten Text Recognition (HTR)

A deep learning-based offline handwritten text recognition system using CRNN (Convolutional Recurrent Neural Network) with CTC (Connectionist Temporal Classification) loss for line-level transcription.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Table of Contents

- [Features](#-features)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Dataset Setup](#-dataset-setup)
- [Usage](#-usage)
  - [Training](#training)
  - [Prediction](#prediction)
  - [Evaluation](#evaluation)
- [Model Architecture](#-model-architecture)
- [Data Processing](#-data-processing)
- [Evaluation Metrics](#-metrics)
- [Results](#-results)

##  Features

- **CRNN Architecture**: Convolutional Recurrent Neural Network optimized for handwritten text
- **CTC Loss**: Connectionist Temporal Classification for sequence transcription without alignment
- **Data Augmentation**: Comprehensive preprocessing pipeline for robust training
- **CLI Interface**: Easy-to-use command-line interface for training, prediction, and evaluation
- **Visualization Tools**: Built-in tools for dataset analysis and model interpretation
- **IAM Dataset Support**: Pre-configured for the IAM Handwriting Database

##  Project Structure

```
./project/
├── HTR.py                   # CLI entry point
├── datasets/
│   ├── lines/              # IAM line images
│   └── lines.txt           # IAM line text annotations
└── src/
    ├── HTR_Model.py        # CRNN-CTC architecture and logic
    ├── data_handler.py     # Dataset loader and label encoding/decoding
    ├── preprocessing.py    # Preprocessing and transformation utilities
    └── visualization.py    # Tools for visualizing data
```

## Installation

### Prerequisites
- Python 3.8 or newer
- CUDA-compatible GPU (recommended)

### Setup Steps

1. **Clone the Repository**
   ```bash
   git clone https://github.com/nickcyran/Handwritten-Text-Recognition.git
   cd Handwritten-Text-Recognition
   ```

2. **Create Virtual Environment** (Recommended)
   ```bash
   python -m venv .venv
   
   # On Linux/macOS
   source .venv/bin/activate
   
   # On Windows
   .venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Required Dependencies
- `torch` - PyTorch deep learning framework
- `torchvision` - Computer vision utilities
- `tqdm` - Progress bars
- `Pillow` - Image processing
- `editdistance` - Edit distance calculations
- `matplotlib` - Plotting and visualization

## Dataset Setup

This project uses the **IAM Handwriting Database**. Follow these steps to set it up:

### 1. Register and Download
- Register at the [IAM Handwriting Database website](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database)
- Download the following files:
  - **`lines.tgz`** - Individual handwritten line images
  - **`ascii.tgz`** - Ground truth text transcriptions

### 2. Extract and Organize
```bash
# Extract line images to datasets/lines/
tar -xzf lines.tgz -C datasets/

# Extract ascii files and move lines.txt to datasets/
tar -xzf ascii.tgz
mv ascii/lines.txt datasets/
```

### 3. Verify Structure
Your `datasets/` directory should look like this:
```
datasets/
├── lines/
│   ├── a01/
│   │   ├── a01-000u/
│   │   │   ├── a01-000u-00.png
│   │   │   └── ...
│   │   └── ...
│   └── ...
└── lines.txt
```

## Usage

The main interface is through `HTR.py` with the following command structure:

```bash
python HTR.py --mode [train|predict|eval] [options]
```

### Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--mode` | Operation mode: `train`, `predict`, or `eval` | **Required** |
| `--model` | Path to model checkpoint (`.pth` file) | `best_model.pth` |
| `--save_path` | Path to save trained model | Same as `--model` |
| `--image_path` | Image or directory path for prediction | **Required for predict** |
| `--epochs` | Number of training epochs | `75` |
| `--visualize` | Enable visualizations and plots | `False` |
| `--eval` | Evaluate model on validation data | `False` |

### Training

Train a new model from scratch:
```bash
python HTR.py --mode train --epochs 60 --save_path my_htr_model.pth
```

Resume training from checkpoint:
```bash
python HTR.py --mode train --model checkpoint.pth --epochs 50
```

Train with visualizations:
```bash
python HTR.py --mode train --visualize --epochs 75
```

### Prediction

Predict single image:
```bash
python HTR.py --mode predict --image_path sample.png --model my_htr_model.pth
```

Batch prediction on folder:
```bash
python HTR.py --mode predict --image_path ./test_images/ --model my_htr_model.pth
```

### Evaluation

Evaluate model performance:
```bash
python HTR.py --eval --model my_htr_model.pth
```

## Model Architecture

The HTR system uses a **CRNN (Convolutional Recurrent Neural Network)** architecture:

### Components

1. **CNN Feature Extractor**
   - Series of Conv2D → BatchNorm → ReLU → MaxPool layers
   - Extracts visual features from input line images

2. **Linear Projection**
   - Projects CNN features into RNN-compatible sequence format

3. **Bidirectional LSTM**
   - Multi-layer bidirectional LSTM network
   - Captures contextual information from both directions

4. **Classification Layer**
   - Fully connected layer mapping LSTM outputs to character vocabulary

### Training Strategy
- **CTC Loss**: Enables training on unsegmented sequence data
- **No explicit alignment** required between input and target sequences

## Data Processing

### Dataset Handling
- **`IAMLinesDataset`**: Loads line images and transcriptions from IAM dataset
- **`LabelConverter`**: Handles text ↔ numerical label conversion for CTC loss

### Preprocessing Pipeline
- **Grayscale Conversion**: Reduces computational complexity
- **ResizeAndPad**: 
  - Resize to fixed height (64px) maintaining aspect ratio
  - Pad to target width (1024px) for consistent input size
- **Augmentation Options**:
  - Median filtering and Gaussian blur
  - Random rotation and perspective distortion
  - Brightness/contrast adjustments
  - Noise addition

## Metrics
- **CER (Character Error Rate)**: Character-level accuracy
- **WER (Word Error Rate)**: Word-level accuracy

### Generated Outputs
- `loss_over_epochs.png` - Training progress visualization
- `error_rates_over_epochs.png` - Validation metrics over time

## Results

### Best Model Performance (`BestModel.pth`)
- **Training Loss**: 0.0607
- **Validation Loss**: 0.0573
- **Character Error Rate (CER)**: ~1.6%
- **Word Error Rate (WER)**: ~6.5%
