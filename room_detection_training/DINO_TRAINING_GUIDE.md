# Training DINO Model for Room Detection

## Overview

DINO (DETR with Improved Denoising Anchor Boxes) is a state-of-the-art object detection model based on Vision Transformers. It offers potentially better accuracy than YOLO, especially for complex scenes, but requires more computational resources and training time.

## What's Involved

### 1. **Dataset Conversion**

**Current Format:** YOLO format (`.txt` files with normalized bbox coordinates)
**Required Format:** COCO format (JSON with annotations)

**Steps:**
- Convert YOLO labels to COCO format
- Create `annotations.json` with:
  - `images`: List of image metadata
  - `annotations`: List of bounding box annotations
  - `categories`: Room class definitions

**Script Needed:**
```python
# Convert YOLO → COCO
# Your existing convert_geojson_to_coco.ts can be adapted
```

### 2. **DINO Framework Setup**

**Repository:** https://github.com/IDEA-Research/DINO

**Dependencies:**
- PyTorch (1.13+)
- CUDA-capable GPU (16GB+ VRAM recommended)
- Transformers library
- Additional dependencies from DINO repo

**Installation:**
```bash
git clone https://github.com/IDEA-Research/DINO.git
cd DINO
pip install -r requirements.txt
```

### 3. **Configuration**

**Key Files to Modify:**
- `configs/DINO/DINO_4scale.py` - Main config file
- Update:
  - `num_classes`: 1 (single class: "room")
  - `dataset_file`: Path to your COCO dataset
  - `data_root`: Path to images
  - `backbone`: Choose backbone (ResNet50, Swin-T, etc.)

**Example Config Changes:**
```python
# In config file
model = dict(
    num_classes=1,  # Single class: room
    # ... other settings
)

data = dict(
    train=dict(
        ann_file='path/to/train_annotations.json',
        img_prefix='path/to/train/images',
    ),
    val=dict(
        ann_file='path/to/val_annotations.json',
        img_prefix='path/to/val/images',
    ),
)
```

### 4. **Training Process**

**Command:**
```bash
python tools/train.py configs/DINO/DINO_4scale.py \
    --work-dir ./work_dirs/dino_room_detection \
    --gpu-ids 0
```

**Training Time Estimate:**
- **Small dataset (<1000 images):** 4-8 hours on A100
- **Medium dataset (1000-5000 images):** 12-24 hours on A100
- **Large dataset (5000+ images):** 2-3 days on A100

**Resource Requirements:**
- **Minimum:** 16GB GPU (A10G, V100)
- **Recommended:** 24GB+ GPU (A100, A10G x2)
- **Batch size:** 2-4 per GPU (depends on GPU memory)

### 5. **Advantages vs YOLO**

**DINO Advantages:**
- ✅ Better accuracy on complex scenes
- ✅ Better handling of small objects
- ✅ End-to-end training (no anchor boxes)
- ✅ More interpretable attention maps

**DINO Disadvantages:**
- ❌ Slower inference (~2-3x slower than YOLO)
- ❌ More memory intensive
- ❌ Longer training time
- ❌ More complex setup

### 6. **Inference Integration**

**Model Export:**
```python
# Export to ONNX or TorchScript
torch.onnx.export(model, dummy_input, "dino_room_detection.onnx")
```

**Integration Steps:**
1. Load DINO model in `inference.py`
2. Preprocess image (resize, normalize)
3. Run inference
4. Post-process outputs (convert to bounding boxes)
5. Format as required JSON array

**Code Changes Needed:**
- Update `load_model()` to support DINO
- Add DINO-specific preprocessing
- Convert DINO outputs to `[x_min, y_min, x_max, y_max]` format

### 7. **Cost Comparison**

**Training Costs (AWS SageMaker):**
- **YOLO Large (200 epochs):** ~$50-100 on ml.g5.2xlarge
- **DINO Base (100 epochs):** ~$150-300 on ml.p4d.24xlarge (A100)
- **DINO Large (100 epochs):** ~$300-500 on ml.p4d.24xlarge

**Inference Costs:**
- **YOLO:** Lower (faster, less memory)
- **DINO:** Higher (slower, more memory)

### 8. **Recommended Approach**

**Option A: Hybrid Approach**
- Use YOLO for fast inference (current setup)
- Train DINO as a "premium" model option
- Let users choose based on speed vs accuracy needs

**Option B: Full DINO Migration**
- Replace YOLO entirely with DINO
- Accept slower inference for better accuracy
- Requires infrastructure upgrades

**Option C: Ensemble**
- Run both models
- Combine results (weighted average)
- Best accuracy but 2x inference time

## Quick Start Checklist

- [ ] Convert YOLO dataset to COCO format
- [ ] Clone DINO repository
- [ ] Install dependencies
- [ ] Configure DINO for single-class detection
- [ ] Test training on small subset (10-20 images)
- [ ] Full training run
- [ ] Evaluate on test set
- [ ] Export model for inference
- [ ] Integrate into inference pipeline
- [ ] Update frontend to support DINO model option

## Estimated Timeline

- **Dataset conversion:** 1-2 days
- **Setup & configuration:** 1-2 days
- **Initial training run:** 1-2 days
- **Hyperparameter tuning:** 2-3 days
- **Integration:** 2-3 days
- **Total:** ~1-2 weeks

## Resources

- DINO Paper: https://arxiv.org/abs/2203.03605
- DINO GitHub: https://github.com/IDEA-Research/DINO
- COCO Format Guide: https://cocodataset.org/#format-data

