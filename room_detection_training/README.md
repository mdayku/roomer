# Room Detection Training Pipeline

Mask R-CNN training pipeline for room boundary detection using polygon segmentation.

## Overview

This directory contains the complete training pipeline for the Room Detection AI project. It uses Mask R-CNN with polygon segmentation to detect room boundaries in architectural blueprints.

### Key Features

- ✅ **Polygon Segmentation**: Preserves exact room boundaries (not just bounding boxes)
- ✅ **Mask R-CNN**: State-of-the-art instance segmentation
- ✅ **COCO Format**: ML-ready data format with established tooling
- ✅ **Performance Optimized**: Targets <30s inference per blueprint
- ✅ **Data Augmentation**: Robust training with augmentation

## Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended for training)
- 8GB+ RAM
- 50GB+ disk space for models and data

## Quick Start

```bash
# One-command setup and training
python run_training.py

# Or step-by-step:
python setup.py                    # Install dependencies
python test_environment.py         # Verify setup
python train.py                    # Start training
python inference.py                # Test trained model
```

## Setup

```bash
# Install Python dependencies
python setup.py

# Or manually:
pip install -r requirements.txt
```

### Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

## Data Requirements

The pipeline expects COCO-formatted data in `../../room_detection_dataset_coco/`:

```
room_detection_dataset_coco/
├── train/
│   └── annotations.json
├── val/
│   └── annotations.json
└── test/
    └── annotations.json
```

Each `annotations.json` contains:
- Image metadata (dimensions, IDs)
- Room annotations with polygon segmentation
- Bounding boxes and area calculations

## Training

## Training Options

### 🚀 **Recommended: Cloud YOLO Training (2-4 hours total)**
Since Gauntlet AI covers AWS costs, cloud training is now the clear winner!

```bash
# First, check your AWS credentials
python check_aws.py

# Then start training
python sagemaker_train.py  # YOLOv8s-seg on AWS SageMaker (recommended)
```

**Why Cloud YOLO?**
- ✅ **4x faster** than local RTX 4050 (2-4h vs 8-12h)
- ✅ **Better GPUs** (T4/A10G vs your RTX 4050)
- ✅ **Cost covered** by Gauntlet AI
- ✅ **Spot pricing** (~70% discount)
- ✅ **Enterprise infrastructure** (no local setup needed)

## 🔐 AWS Credentials Setup

**Before training, you need AWS credentials.** Since Gauntlet AI is paying, ask your team for:

### Quick Setup Options:

**Option 1 - .env File (Easiest - Recommended):**
```bash
# Copy the template and edit with your credentials
cp env_template.txt .env

# Edit .env with your actual keys from Gauntlet AI team
# AWS_ACCESS_KEY_ID=your_actual_key_from_team
# AWS_SECRET_ACCESS_KEY=your_actual_secret_from_team
# AWS_DEFAULT_REGION=us-east-1
```

**Sample .env file:**
```bash
# AWS Credentials for Room Detection Training
AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE
AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
AWS_DEFAULT_REGION=us-east-1
```

**Option 2 - AWS CLI:**
```bash
# Install AWS CLI first: https://aws.amazon.com/cli/
aws configure
# Enter your Access Key ID, Secret Key, region (us-east-1)
```

**Option 3 - Environment Variables:**
```bash
export AWS_ACCESS_KEY_ID=your_key_here
export AWS_SECRET_ACCESS_KEY=your_secret_here
export AWS_DEFAULT_REGION=us-east-1
```

### Verify Setup:
```bash
python check_aws.py  # Tests all credentials and permissions
```

### 💻 **Alternative: Local Training**
If you prefer local development or want to compare results:

```bash
python train_yolo.py       # YOLO11-seg on local RTX 4050 (2-4 hours)
python train.py           # Mask R-CNN on local RTX 4050 (8-12 hours)
```

## Model Comparison

| Model | Training Time | Inference Speed | Accuracy | Cost | Best For |
|-------|---------------|----------------|----------|------|----------|
| **YOLO11-seg (Cloud)** | **2-4h** ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | **FREE** | **🏆 Recommended: Speed + Cost** |
| **Mask R-CNN (Cloud)** | 4-8h ⭐⭐⭐ | Moderate | ⭐⭐⭐⭐⭐ | FREE | Complex shapes, highest accuracy |
| **YOLO11-seg (Local)** | 2-4h ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | FREE | Local development |
| **Mask R-CNN (Local)** | 8-12h ⭐ | Moderate | ⭐⭐⭐⭐⭐ | FREE | Local development |

### Training Configuration

Key parameters in `config.py`:
- `NUM_EPOCHS = 30` - Training epochs (with early stopping)
- `BATCH_SIZE = 2` - RTX 4050 optimized
- `LEARNING_RATE = 0.001` - Initial learning rate
- `PATIENCE = 5` - Early stopping patience
- `IMAGE_SIZE = (800, 800)` - Input resolution

### Monitoring Training

Training logs are saved to `outputs/logs/training_log.json`:

```json
{
  "config": {...},
  "epochs": [
    {
      "epoch": 1,
      "train_losses": {
        "loss": 1.234,
        "loss_classifier": 0.456,
        "loss_mask": 0.789
      },
      "learning_rate": 0.001,
      "epoch_time": 120.5
    }
  ]
}
```

### Model Checkpoints

Models are saved to `outputs/models/`:
- `best_model.pth` - Best validation loss
- `model_epoch_N.pth` - Every 5 epochs
- `final_model.pth` - After training completion

## Inference

### Test Trained Model

```bash
python inference.py
```

### Custom Inference

```python
from inference import RoomDetector

# Load trained model
detector = RoomDetector(model_path="outputs/models/best_model.pth")

# Predict on image
image = np.zeros((800, 800, 3), dtype=np.uint8)  # Your blueprint image
predictions = detector.predict(image)

print(f"Found {len(predictions['boxes'])} rooms")
print(f"Inference time: {predictions['inference_time']:.3f}s")

# Visualize results
detector.visualize_prediction(image, predictions, save_path="room_detection.png")
```

### Performance Benchmarking

The inference script includes automatic benchmarking:

```
Benchmarking inference on 5 images...
Average: 1250.34ms ± 45.67ms
Target: <30000ms per image
✅ Meets latency requirement!
```

## Model Architecture

### Mask R-CNN Components

- **Backbone**: ResNet-50 with FPN
- **RPN**: Region Proposal Network with custom anchor sizes
- **ROI Heads**: Separate heads for classification, bounding boxes, and masks
- **Loss Functions**: Multi-task loss (classification + bbox + mask)

### Customization for Rooms

```python
# Optimized anchor sizes for typical room dimensions
ANCHOR_SIZES = ((32,), (64,), (128,), (256,), (512,))

# Multi-scale ROI alignment for better mask quality
box_roi_pool = MultiScaleRoIAlign(featmap_names=['0', '1', '2', '3'], output_size=7)
mask_roi_pool = MultiScaleRoIAlign(featmap_names=['0', '1', '2', '3'], output_size=14)
```

## Evaluation Metrics

### Training Metrics
- **Total Loss**: Combined loss from all heads
- **Component Losses**:
  - Classification loss
  - Bounding box regression loss
  - Mask segmentation loss
  - Objectness loss
  - RPN box regression loss

### Validation Metrics (Future)
- **mAP@0.5**: Mean Average Precision at IoU=0.5
- **mAP@0.75**: Mean Average Precision at IoU=0.75 (stricter)
- **Mask mAP**: Segmentation quality
- **Inference Latency**: <30 seconds per blueprint

## Troubleshooting

### Common Issues

**CUDA Out of Memory**
```python
# Reduce batch size in config.py
BATCH_SIZE = 1
```

**Slow Training**
- Use GPU: `DEVICE = "cuda"`
- Reduce image size: `IMAGE_SIZE = (600, 600)`
- Use smaller backbone: `BACKBONE = "resnet34"`

**Poor Convergence**
- Increase epochs: `NUM_EPOCHS = 30`
- Adjust learning rate: `LEARNING_RATE = 0.0005`
- Add more augmentation

**High Inference Latency**
- Use smaller model
- Optimize with ONNX/TensorRT
- Reduce input resolution

### Debug Mode

Add debug prints to training:

```python
# In train.py, add after model predictions
print(f"Predictions: {len(predictions)}")
for i, pred in enumerate(predictions):
    print(f"  Image {i}: {len(pred['boxes'])} boxes, {len(pred['masks'])} masks")
```

## Production Deployment

### Model Optimization

1. **Quantization**:
```python
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
torch.quantization.prepare(model, inplace=True)
```

2. **ONNX Export**:
```python
torch.onnx.export(model, dummy_input, "model.onnx", opset_version=11)
```

3. **TensorRT** (NVIDIA GPUs):
```python
# Convert to TensorRT engine for maximum speed
```

### AWS Integration

For production deployment to AWS:

1. **SageMaker Training**: Use built-in Mask R-CNN containers
2. **SageMaker Inference**: Deploy as endpoint with auto-scaling
3. **Lambda**: For serverless inference (if latency allows)
4. **Batch Processing**: For offline blueprint processing

### Performance Targets

- **Training**: 20 epochs in 4-8 hours (GPU)
- **Inference**: <30 seconds per blueprint
- **Accuracy**: >90% room detection, >75% IoU for complex shapes
- **Model Size**: <500MB for deployment

## File Structure

```
room_detection_training/
├── config.py              # Training configuration
├── dataset.py             # COCO dataset loader
├── model.py               # Mask R-CNN implementation
├── train.py               # Main training script
├── inference.py           # Inference and testing
├── setup.py               # Environment setup
├── requirements.txt       # Python dependencies
├── outputs/               # Training outputs
│   ├── models/           # Saved model checkpoints
│   ├── logs/            # Training logs
│   └── visualizations/  # Prediction visualizations
└── README.md             # This file
```

## Next Steps

1. **Run Training**: `python train.py`
2. **Monitor Progress**: Check `outputs/logs/training_log.json`
3. **Test Inference**: `python inference.py`
4. **Optimize Performance**: Adjust config for speed/accuracy tradeoffs
5. **Deploy**: Export to ONNX/TensorRT for production

The pipeline is designed to scale from local development to AWS production deployment while maintaining the polygon segmentation accuracy needed for complex room shapes.
