# DINO Training on SageMaker

## Overview

This script (`sagemaker_train_dino.py`) sets up DINO training on AWS SageMaker using your existing COCO dataset. **No dataset conversion needed** - DINO uses COCO format directly!

## Quick Start

```bash
cd room_detection_training
python sagemaker_train_dino.py
```

## What It Does

1. **Checks COCO Dataset** - Verifies `room_detection_dataset_coco/train/annotations.json` exists
2. **Uploads to S3** - Uploads COCO annotations and images to SageMaker
3. **Creates Training Script** - Generates `train_dino_sagemaker.py` for the container
4. **Launches Training** - Starts SageMaker job on A100 GPU

## Configuration

### Instance Type
- **Default:** `ml.g5.2xlarge` (A10G x2 GPUs, 48GB VRAM total)
- **Cost:** ~$2.41/hour
- **Why:** Cost-effective! A100 would be $40.96/hr (17x more expensive)
- **Trade-off:** Slightly slower training (~15 hours vs 12 hours) but saves $460+

### Hyperparameters
- **Epochs:** 50 (DINO converges faster than YOLO)
- **Batch Size:** 4 (can increase if memory allows)
- **Learning Rate:** 0.0001 (lower for transformers)
- **Classes:** 1 (room detection)

## Current Status

⚠️ **Template Script Created** - The training script (`train_dino_sagemaker.py`) is a template that:
- ✅ Sets up DINO repository
- ✅ Installs dependencies
- ✅ Configures COCO dataset paths
- ⚠️ **Needs:** Actual DINO training loop integration

## Next Steps to Complete DINO Training

1. **Clone DINO Repository Locally:**
   ```bash
   git clone https://github.com/IDEA-Research/DINO.git
   cd DINO
   ```

2. **Study DINO's main.py:**
   - Understand their training loop
   - See how they load COCO datasets
   - Check their config system

3. **Adapt for SageMaker:**
   - Update paths to use `/opt/ml/input/data/training`
   - Configure for single-class detection (`num_classes=1`)
   - Save checkpoints to `/opt/ml/model`
   - Export final model for inference

4. **Update `train_dino_sagemaker.py`:**
   - Replace placeholder code with actual DINO training loop
   - Use DINO's `build_model()`, `build_dataset()`, `train_one_epoch()` functions

## Cost Estimate

- **Instance:** ml.g5.2xlarge (A10G x2) - **Cost-optimized!**
- **Hourly Rate:** $2.41
- **Estimated Time:** ~15 hours for 50 epochs
- **Total Cost:** ~$36 (vs $490 for A100 - **93% cheaper!**)

## Comparison: YOLO vs DINO

| Feature | YOLO (Current) | DINO (Future) |
|---------|---------------|---------------|
| **Format** | YOLO (needs conversion) | COCO (direct use!) |
| **Instance** | ml.g5.2xlarge ($2.41/hr) | ml.g5.2xlarge ($2.41/hr) ✅ Same! |
| **Training Time** | 8 hours (200 epochs) | 15 hours (50 epochs) |
| **Cost** | ~$20 | ~$36 (much more reasonable!) |
| **Accuracy** | Good (73% F1) | Better (expected 80%+ F1) |
| **Inference Speed** | Fast (~600ms) | Slower (~1500ms) |

## Files Created

- `sagemaker_train_dino.py` - Main launcher script
- `train_dino_sagemaker.py` - Training script (generated, needs adaptation)
- `requirements_dino.txt` - Python dependencies
- `DINO_SAGEMAKER_README.md` - This file

## Notes

- The script will clone DINO repo inside the SageMaker container
- COCO dataset is uploaded once and reused
- Training can be monitored in SageMaker console
- Model artifacts are saved to S3 automatically

