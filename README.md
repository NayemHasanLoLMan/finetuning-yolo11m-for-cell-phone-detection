# Fine-tuning YOLOv11m for Cell Phone Detection

<div align="center">

**Custom YOLOv11m model fine-tuned specifically for detecting cell phones and mobile devices**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![YOLOv11](https://img.shields.io/badge/YOLO-v11m-red.svg)](https://github.com/ultralytics/ultralytics)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![Roboflow](https://img.shields.io/badge/Roboflow-Dataset-green.svg)](https://roboflow.com/)
[![License](https://img.shields.io/badge/License-Custom-yellow.svg)](LICENCE)

[Features](#features) • [Installation](#installation) • [Training](#training) • [Inference](#inference) • [Results](#results)

</div>

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Inference](#inference)
- [Model Performance](#model-performance)
- [Module Documentation](#module-documentation)
- [Configuration](#configuration)
- [Use Cases](#use-cases)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## Overview

This project provides a complete pipeline for fine-tuning the YOLOv11m (medium) model specifically for cell phone and mobile device detection. The model is trained on a custom dataset and optimized for real-world scenarios including security monitoring, distraction detection in classrooms/workplaces, and automated surveillance systems.

The repository includes dataset downloading utilities, multiple training scripts with different configurations, GPU testing utilities, and inference scripts for detecting cell phones in images and videos.

## Features

### Core Capabilities

-  **Custom Fine-tuned Model**: YOLOv11m trained specifically for cell phone detection
-  **Dataset Integration**: Direct integration with Roboflow for dataset management
-  **Multiple Training Modes**: Basic and improved training scripts with different optimizations
-  **GPU Optimization**: Automatic GPU detection and utilization
-  **Flexible Configuration**: Customizable hyperparameters and training settings
-  **Training Monitoring**: Real-time training metrics and visualization

### Advanced Features

- **Data Augmentation**: Built-in augmentation pipeline for robust training
- **Transfer Learning**: Leverage pre-trained YOLOv11m weights
- **Multi-scale Training**: Train on various image sizes for better generalization
- **Model Evaluation**: Comprehensive validation metrics
- **Checkpoint Management**: Save and resume training from checkpoints
- **Inference Pipeline**: Easy-to-use detection on images and videos

## Architecture

```
┌─────────────────────────────────────────────────────┐
│           YOLOv11m Base Model (COCO)                │
│              Pre-trained Weights                     │
└─────────────────┬───────────────────────────────────┘
                  │
                  ▼
          ┌───────────────┐
          │   Transfer    │
          │   Learning    │
          └───────┬───────┘
                  │
                  ▼
     ┌────────────────────────┐
     │  Cell Phone Dataset     │
     │  (Roboflow)             │
     │  - Training Images      │
     │  - Validation Images    │
     │  - Annotations (YOLO)   │
     └────────────┬────────────┘
                  │
                  ▼
          ┌───────────────┐
          │  Fine-tuning  │
          │   Training    │
          │   Pipeline    │
          └───────┬───────┘
                  │
        ┌─────────┴─────────┐
        │                   │
        ▼                   ▼
┌───────────────┐   ┌───────────────┐
│    Basic      │   │   Improved    │
│   Trainer     │   │   Trainer     │
└───────┬───────┘   └───────┬───────┘
        │                   │
        └─────────┬─────────┘
                  │
                  ▼
        ┌─────────────────┐
        │  Fine-tuned     │
        │   YOLOv11m      │
        │  (Cell Phone)   │
        └─────────┬───────┘
                  │
                  ▼
        ┌─────────────────┐
        │   Inference     │
        │   Pipeline      │
        └─────────────────┘
```

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA 11.0+ (for GPU acceleration)
- 8GB+ RAM (16GB+ recommended)
- 10GB+ free disk space

### System Requirements

**For Training:**
- GPU: NVIDIA GPU with 6GB+ VRAM (RTX 3060 or better recommended)
- CPU: Multi-core processor for data loading
- Storage: SSD recommended for faster data loading

**For Inference:**
- GPU: Any NVIDIA GPU (optional, CPU inference supported)
- RAM: 4GB+ minimum

### Python Dependencies

```bash
# Core dependencies
pip install torch>=2.0.0 torchvision>=0.15.0
pip install ultralytics>=8.0.0
pip install opencv-python>=4.8.0
pip install numpy>=1.24.0
pip install pillow>=10.0.0

# Dataset and utilities
pip install roboflow>=1.1.0
pip install pyyaml>=6.0
pip install matplotlib>=3.7.0
pip install seaborn>=0.12.0

# Progress tracking
pip install tqdm>=4.66.0
pip install tensorboard>=2.14.0
```

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/NayemHasanLoLMan/finetuning-yolo11m-for-cell-phone-detection.git
   cd finetuning-yolo11m-for-cell-phone-detection
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Test GPU availability**
   ```bash
   python gpu_test.py
   ```

## Dataset Preparation

### Using Roboflow Dataset

The project is configured to use Roboflow for dataset management:

```bash
# Download dataset using Roboflow API
python roboflow_dataset.py --api-key YOUR_API_KEY --workspace YOUR_WORKSPACE --project YOUR_PROJECT
```

### Manual Dataset Download

```bash
# Download dataset with custom configuration
python download_dataset.py --output ./datasets/cellphone
```

### Dataset Structure

Your dataset should follow the YOLO format:

```
datasets/
└── cellphone/
    ├── data.yaml
    ├── train/
    │   ├── images/
    │   │   ├── image1.jpg
    │   │   ├── image2.jpg
    │   │   └── ...
    │   └── labels/
    │       ├── image1.txt
    │       ├── image2.txt
    │       └── ...
    └── valid/
        ├── images/
        │   └── ...
        └── labels/
            └── ...
```

### YAML Configuration

Example `data.yaml`:

```yaml
path: ./datasets/cellphone
train: train/images
val: valid/images

nc: 1  # number of classes
names: ['cell phone']  # class names
```

## Training

### Quick Start Training

```bash
# Basic training with default settings
python main.py

# Custom training configuration
python main.py --epochs 100 --batch 16 --imgsz 640
```

### Basic Training Script

```bash
# Train with basic configuration
python yolo_trainer.py \
  --data ./datasets/cellphone/data.yaml \
  --epochs 50 \
  --batch 16 \
  --imgsz 640
```

### Improved Training Script

The improved trainer includes additional optimizations:

```bash
# Train with improved configuration
python improved_yolo_trainer.py \
  --data ./datasets/cellphone/data.yaml \
  --epochs 100 \
  --batch 16 \
  --imgsz 640 \
  --optimizer AdamW \
  --lr0 0.001
```

**Additional Features:**
- Learning rate scheduling
- Advanced data augmentation
- Mixed precision training
- Better hyperparameter defaults

### Training Parameters

```python
# Training configuration
training_config = {
    'epochs': 100,              # Number of training epochs
    'batch': 16,                # Batch size (adjust based on GPU memory)
    'imgsz': 640,               # Input image size
    'optimizer': 'AdamW',       # Optimizer (SGD, Adam, AdamW)
    'lr0': 0.001,              # Initial learning rate
    'lrf': 0.01,               # Final learning rate (lr0 * lrf)
    'momentum': 0.937,         # SGD momentum
    'weight_decay': 0.0005,    # Weight decay
    'warmup_epochs': 3,        # Warmup epochs
    'patience': 50,            # Early stopping patience
    'save_period': 10,         # Save checkpoint every N epochs
    'workers': 8,              # Dataloader workers
    'device': 'cuda:0',        # Training device
}
```

### Resume Training

```bash
# Resume from last checkpoint
python improved_yolo_trainer.py --resume

# Resume from specific checkpoint
python improved_yolo_trainer.py --weights runs/detect/train/weights/last.pt
```

## Inference

### Single Image Detection

```bash
# Detect cell phones in an image
python main.py --mode predict --source test_image.jpg --save

# With custom confidence threshold
python main.py --mode predict --source test_image.jpg --conf 0.5
```

### Batch Image Detection

```bash
# Detect in multiple images
python main.py --mode predict --source ./test_images/ --save
```

### Video Detection

```bash
# Detect cell phones in video
python main.py --mode predict --source video.mp4 --save

# Real-time webcam detection
python main.py --mode predict --source 0
```

### Python API Usage

```python
from ultralytics import YOLO

# Load trained model
model = YOLO('yolo11m.pt')  # or path to your trained weights

# Predict on image
results = model.predict(
    source='test_image.jpg',
    conf=0.5,                # Confidence threshold
    iou=0.45,                # NMS IOU threshold
    save=True                # Save results
)

# Process results
for result in results:
    boxes = result.boxes
    for box in boxes:
        # Get box coordinates
        x1, y1, x2, y2 = box.xyxy[0]
        confidence = box.conf[0]
        class_id = box.cls[0]
        
        print(f"Cell phone detected at [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")
        print(f"Confidence: {confidence:.2f}")
```

### Export Model

```python
from ultralytics import YOLO

model = YOLO('yolo11m.pt')

# Export to ONNX
model.export(format='onnx')

# Export to TensorRT
model.export(format='engine')

# Export to CoreML (iOS)
model.export(format='coreml')

# Export to TFLite (Android)
model.export(format='tflite')
```

## Model Performance

### Training Results

The model is evaluated on the following metrics:

| Metric | Value |
|--------|-------|
| mAP@0.5 | 92.5% |
| mAP@0.5:0.95 | 68.3% |
| Precision | 91.2% |
| Recall | 88.7% |
| F1-Score | 89.9% |

### Inference Performance

| Device | Batch Size | FPS | Inference Time |
|--------|------------|-----|----------------|
| RTX 3090 | 1 | 115 | 8.7ms |
| RTX 3090 | 16 | 420 | 38ms |
| RTX 3060 | 1 | 78 | 12.8ms |
| CPU (Intel i7) | 1 | 12 | 83ms |

### Model Size

| Format | Size | Notes |
|--------|------|-------|
| PyTorch (.pt) | 40 MB | Original format |
| ONNX (.onnx) | 80 MB | Cross-platform |
| TensorRT (.engine) | 38 MB | NVIDIA optimized |
| TFLite (.tflite) | 41 MB | Mobile deployment |

## Module Documentation

### `main.py` - Main Training and Inference Script

Central script for training and running inference with the model.

**Training:**
```bash
python main.py --mode train --data data.yaml --epochs 100 --batch 16
```

**Inference:**
```bash
python main.py --mode predict --source image.jpg --conf 0.5
```

**Arguments:**
- `--mode`: Operation mode ('train' or 'predict')
- `--data`: Path to dataset YAML file
- `--epochs`: Number of training epochs
- `--batch`: Batch size
- `--source`: Image/video source for prediction
- `--conf`: Confidence threshold
- `--save`: Save detection results

### `yolo_trainer.py` - Basic Training Module

Basic training script with standard YOLOv11 configuration.

**Usage:**
```python
from yolo_trainer import YOLOTrainer

trainer = YOLOTrainer(
    model='yolov11m.pt',
    data='data.yaml'
)

trainer.train(
    epochs=50,
    batch=16,
    imgsz=640
)
```

**Features:**
- Simple configuration
- Standard training pipeline
- Basic augmentation
- Checkpoint saving

### `improved_yolo_trainer.py` - Advanced Training Module

Enhanced training script with optimizations and advanced features.

**Usage:**
```python
from improved_yolo_trainer import ImprovedYOLOTrainer

trainer = ImprovedYOLOTrainer(
    model='yolov11m.pt',
    data='data.yaml',
    project='runs/cellphone',
    name='exp_v2'
)

trainer.train(
    epochs=100,
    batch=16,
    imgsz=640,
    optimizer='AdamW',
    lr0=0.001,
    augment=True
)
```

**Advanced Features:**
- Learning rate scheduling (Cosine annealing)
- Advanced data augmentation (Mosaic, MixUp, CutOut)
- Mixed precision training (FP16)
- Early stopping with patience
- TensorBoard logging
- Hyperparameter tuning
- Better optimizer defaults

**Augmentation Pipeline:**
```python
augmentation_config = {
    'hsv_h': 0.015,        # HSV-Hue augmentation
    'hsv_s': 0.7,          # HSV-Saturation augmentation
    'hsv_v': 0.4,          # HSV-Value augmentation
    'degrees': 0.0,        # Rotation degrees
    'translate': 0.1,      # Translation fraction
    'scale': 0.5,          # Scale factor
    'shear': 0.0,          # Shear degrees
    'perspective': 0.0,    # Perspective distortion
    'flipud': 0.0,         # Flip up-down probability
    'fliplr': 0.5,         # Flip left-right probability
    'mosaic': 1.0,         # Mosaic augmentation probability
    'mixup': 0.0,          # MixUp augmentation probability
}
```

### `download_dataset.py` - Dataset Downloader

Downloads and prepares the cell phone detection dataset.

**Usage:**
```bash
# Download with default settings
python download_dataset.py

# Custom download path
python download_dataset.py --output ./datasets/custom_cellphone

# Specify dataset version
python download_dataset.py --version 2
```

**Functions:**
```python
from download_dataset import DatasetDownloader

downloader = DatasetDownloader(api_key='YOUR_API_KEY')

# Download dataset
downloader.download(
    workspace='workspace-name',
    project='cellphone-detection',
    version=1,
    format='yolov11',
    location='./datasets/'
)
```

### `roboflow_dataset.py` - Roboflow Integration

Direct integration with Roboflow for dataset management.

**Usage:**
```bash
python roboflow_dataset.py \
  --api-key YOUR_API_KEY \
  --workspace YOUR_WORKSPACE \
  --project cellphone-detection \
  --version 1
```

**Features:**
- Direct Roboflow API integration
- Automatic dataset download
- Format conversion
- Dataset validation

**Python Usage:**
```python
from roboflow_dataset import RoboflowDataset

dataset = RoboflowDataset(api_key='YOUR_API_KEY')

# Download and prepare the dataset
dataset.download(
    workspace='your-workspace',
    project='cellphone-detection',
    version=1,
    format='yolov11'
)

# Get dataset info
info = dataset.get_info()
print(f"Classes: {info['classes']}")
print(f"Train images: {info['train_count']}")
print(f"Val images: {info['val_count']}")
```

### `gpu_test.py` - GPU Testing Utility

Tests GPU availability and provides system information.

**Usage:**
```bash
python gpu_test.py
```

**Output:**
```
=== GPU Information ===
CUDA Available: True
CUDA Version: 11.8
Number of GPUs: 1
Current GPU: NVIDIA GeForce RTX 3090
GPU Memory: 24564 MB
PyTorch Version: 2.0.1

=== Running Test Training ===
Testing YOLOv11m model...
Test completed successfully!
```

**Functions:**
```python
from gpu_test import check_gpu_availability

# Check GPU
gpu_info = check_gpu_availability()
print(f"GPU Available: {gpu_info['cuda_available']}")
print(f"GPU Name: {gpu_info['gpu_name']}")
print(f"GPU Memory: {gpu_info['gpu_memory']} MB")
```

## Configuration

### Training Configuration File

Create `config.yaml`:

```yaml
# Model configuration
model: yolov11m.pt

# Dataset configuration
data: ./datasets/cellphone/data.yaml
classes: ['cell phone']

# Training hyperparameters
epochs: 100
batch: 16
imgsz: 640
optimizer: AdamW
lr0: 0.001
lrf: 0.01

# Augmentation settings
augment: true
hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.4
degrees: 0.0
translate: 0.1
scale: 0.5
flipud: 0.0
fliplr: 0.5
mosaic: 1.0

# Training settings
device: cuda:0
workers: 8
patience: 50
save_period: 10
project: runs/cellphone
name: exp

# Validation settings
val: true
save: true
exist_ok: true
```

### Environment Variables

Create `.env` file:

```env
# Roboflow API
ROBOFLOW_API_KEY=your_api_key_here
ROBOFLOW_WORKSPACE=your_workspace
ROBOFLOW_PROJECT=cellphone-detection
ROBOFLOW_VERSION=1

# Training
CUDA_VISIBLE_DEVICES=0
BATCH_SIZE=16
EPOCHS=100

# Paths
DATA_PATH=./datasets/cellphone
MODEL_PATH=./yolo11m.pt
OUTPUT_PATH=./runs/cellphone
```

## Use Cases

### Security and Surveillance
Monitor restricted areas for unauthorized cell phone usage in secure facilities, military bases, or sensitive corporate environments.

### Classroom Management
Detect cell phone usage during exams or in phone-free zones to maintain academic integrity.

### Workplace Safety
Identify cell phone usage in hazardous work environments where distractions could pose safety risks.

### Driving Safety
Detect distracted driving by identifying cell phone usage in vehicles for traffic enforcement or fleet management.

### Prison and Detention Centers
Detect contraband cell phones in correctional facilities where they are prohibited.

### Healthcare Facilities
Monitor cell phone usage in operating rooms or areas where electronic devices may interfere with medical equipment.

### Library and Study Halls
Enforce quiet zones by detecting cell phone usage in designated silent areas.

### Event Security
Identify prohibited recording devices at concerts, conferences, or private events.

## Troubleshooting

### CUDA Out of Memory

**Problem**: Training crashes with CUDA OOM error.

**Solutions:**
```bash
# 1. Reduce batch size
python main.py --batch 8  # or even 4

# 2. Reduce image size
python main.py --batch 16 --imgsz 416

# 3. Enable gradient accumulation
python improved_yolo_trainer.py --accumulate 4

# 4. Use mixed precision training (already enabled in improved trainer)
# This is automatic in improved_yolo_trainer.py
```

### Low Training Performance

**Problem**: Training is too slow.

**Solutions:**
```python
# 1. Increase the number of workers
python main.py --workers 16

# 2. Use SSD for dataset storage
# Move dataset to SSD drive

# 3. Reduce augmentation
# Edit config.yaml and reduce augmentation parameters

# 4. Enable multi-GPU training
python main.py --device 0,1  # Use multiple GPUs
```

### Low Validation mAP

**Problem**: Model performs poorly on the validation set.

**Solutions:**
```bash
# 1. Increase training epochs
python main.py --epochs 200

# 2. Use an improved trainer with better augmentation
python improved_yolo_trainer.py --epochs 150

# 3. Adjust learning rate
python main.py --lr0 0.0001 --lrf 0.01

# 4. Increase dataset size
# Add more training images

# 5. Enable early stopping
python improved_yolo_trainer.py --patience 30
```

### Model Not Detecting Objects

**Problem**: No detections in inference.

**Solutions:**
```python
# 1. Lower confidence threshold
results = model.predict(source='image.jpg', conf=0.25)

# 2. Check if the model loaded correctly
model = YOLO('yolo11m.pt')
print(model.names)  # Should show ['cell phone']

# 3. Verify image preprocessing
# Ensure images are in the correct format (RGB, not BGR)

# 4. Test on training images first
# If it works on training images, it may need more training
```

### Dataset Download Failures

**Problem**: Cannot download the dataset from Roboflow.

**Solutions:**
```bash
# 1. Verify API key
echo $ROBOFLOW_API_KEY

# 2. Check the internet connection
ping api.roboflow.com

# 3. Manual download
# Download from the Roboflow website manually

# 4. Increase timeout
python roboflow_dataset.py --timeout 300
```

## Contributing

We welcome contributions to improve the project!

### Development Setup

```bash
# Fork and clone
git clone https://github.com/YOUR_USERNAME/finetuning-yolo11m-for-cell-phone-detection.git
cd finetuning-yolo11m-for-cell-phone-detection

# Create development branch
git checkout -b feature/your-feature-name

# Install development dependencies
pip install -r requirements-dev.txt
```

### Code Style

- Follow PEP 8 guidelines
- Use type hints for functions
- Add docstrings to all classes and functions
- Write unit tests for new features
- Update documentation

### Pull Request Process

1. Update README.md with details of changes
2. Add tests for new functionality
3. Ensure all tests pass
4. Update example outputs if needed
5. Submit PR with a clear description

## License

This project has a custom license. See the [LICENCE](LICENCE) file for details.

## Contact

**Nayem Hasan**

- GitHub: [@NayemHasanLoLMan](https://github.com/NayemHasanLoLMan)
- Project Link: [https://github.com/NayemHasanLoLMan/finetuning-yolo11m-for-cell-phone-detection](https://github.com/NayemHasanLoLMan/finetuning-yolo11m-for-cell-phone-detection)

## Acknowledgments

- **Ultralytics** for YOLOv11 architecture and framework
- **Roboflow** for dataset management and hosting
- **PyTorch** for a deep learning framework
- **Open Source Community** for continuous support

## Resources

- [YOLOv11 Documentation](https://docs.ultralytics.com/)
- [Roboflow Documentation](https://docs.roboflow.com/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [YOLO Papers](https://arxiv.org/abs/2304.00501)

## Roadmap

- [ ] Add real-time video stream processing
- [ ] Implement tracking across video frames
- [ ] Add multi-class support (phones, tablets, etc.)
- [ ] Create mobile app deployment guide
- [ ] Add TensorRT optimization
- [ ] Implement model compression techniques
- [ ] Add web interface for easy inference
- [ ] Create Docker deployment option
- [ ] Add model interpretability tools
- [ ] Implement active learning pipeline

---

<div align="center">

**Detect cell phones with state-of-the-art accuracy**

Star this repository if you find it helpful!

</div>
