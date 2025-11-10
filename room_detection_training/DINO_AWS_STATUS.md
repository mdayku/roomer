# DINO AWS Training Status & Requirements

## Current Status: NOT READY ❌

### What We Have ✅
1. **COCO Dataset Prepared** (`coco_room_dino/`)
   - Train/val/test splits (70/20/10, seed=42)
   - Annotations in COCO format
   - 2-class setup (wall + room)

2. **SageMaker Training Script** (`train_dino_sagemaker.py`)
   - Handles DINO repo cloning in container
   - Compiles custom CUDA ops (MultiScaleDeformableAttention)
   - Configures hyperparameters

3. **Launch Script** (`sagemaker_train_dino.py`)
   - IAM role setup
   - S3 upload logic
   - PyTorch estimator configuration

4. **Custom Config** (`DINO/config/DINO/DINO_4scale_room_wall.py`)
   - 2-class detection config
   - Hyperparameters tuned for room detection

### Critical Blockers 🚨

#### 1. **NO IMAGES UPLOADED TO S3**
**Problem:**
```python
# Line 92-94 in sagemaker_train_dino.py
print("[INFO] Skipping image upload - images will be loaded from local paths or existing S3 location")
```

The script only uploads `annotations.json` files, **NO IMAGES**.

**Impact:** Training will fail immediately when DINO tries to load images from COCO annotations.

**Fix Required:**
- Upload all 4,194 images to S3 (structure: `s3://bucket/dino-data/{split}/images/*.png`)
- Update COCO annotations to reference S3 paths or relative paths
- Estimated upload time: 5-10 minutes (4200 images @ ~200KB each = ~840MB)

#### 2. **IMAGE PATHS IN COCO ANNOTATIONS**
**Problem:** COCO annotations likely contain local Kaggle paths:
```json
{
  "file_name": "/kaggle/input/cubicasa5k/.../F1_original.png"
}
```

**Impact:** DINO will look for images at wrong paths in SageMaker container.

**Fix Required:**
- Update `prepare_coco_for_dino.py` to write relative paths
- Or: Copy images to COCO directory structure before uploading
- Structure should be: `coco_room_dino/{split}/images/{img_id:06d}.png`

#### 3. **CUDA OPS COMPILATION UNCERTAINTY**
**Problem:** DINO requires custom CUDA operators to be compiled in container.

**Status:** Script includes compilation logic, but **UNTESTED**.

**Risk:** Medium - SageMaker Deep Learning Containers have nvcc/gcc, but compilation could fail.

**Mitigation:** Training script handles it, but first run might fail.

---

## Estimated Work to Launch Training

### Option A: Quick Fix (Minimal Changes)
**Time: ~30-45 minutes**

1. **Copy images to COCO directory structure** (10 min)
   ```bash
   # Copy from yolo_room_only to coco_room_dino
   python copy_images_for_dino.py
   ```

2. **Update upload script to include images** (5 min)
   ```python
   # Modify sagemaker_train_dino.py line 82-94
   # Add image upload loop for each split
   ```

3. **Upload to S3** (5-10 min)
   - 4,194 images + annotations = ~850MB
   - Parallel upload with boto3

4. **Test launch** (5 min)
   - Run `python sagemaker_train_dino.py`
   - Monitor first 10 minutes for errors

5. **Debug & relaunch if needed** (10-15 min)

**Total: 30-45 minutes**

### Option B: Robust Solution (Production-Ready)
**Time: ~1-2 hours**

1. **Refactor `prepare_coco_for_dino.py`** (20 min)
   - Copy images to output directory
   - Write relative paths in annotations
   - Create proper COCO dataset structure

2. **Update SageMaker script** (15 min)
   - Remove "skip image upload" logic
   - Add robust image upload with progress
   - Verify annotations reference correct paths

3. **Test locally with small subset** (15 min)
   - Verify COCO structure is valid
   - Test DINO can load dataset

4. **Upload and launch** (10 min)

5. **Monitor & debug** (variable)

---

## Cost Estimate

**Instance:** `ml.g5.2xlarge` (A10G x2 GPUs)
- **Hourly rate:** $2.41/hour
- **5 epochs (test run):** ~1.5 hours = **$3.62**
- **50 epochs (full training):** ~15 hours = **$36.15**

**Storage (S3):**
- Dataset: ~850MB
- Model artifacts: ~1GB
- Cost: <$0.10/month

---

## Priority Decision

**Question:** Is DINO worth the effort right now?

**Pros:**
- Transformer-based (potentially better than YOLO)
- COCO format (no conversion needed)
- Could handle complex scenes better

**Cons:**
- 30-45 min setup work required
- Untested in SageMaker environment
- More expensive than YOLO ($2.41/hr vs $1.21/hr)
- Longer training time (15 hours vs 3 hours)
- You already have 2 working YOLO models at 89.5% F1

**Recommendation:**
1. **Short term:** Focus on demo with existing YOLO models
2. **After demo:** Come back to DINO if you need better accuracy
3. **Low priority:** YOLO is performing well, DINO is exploratory

---

## Next Steps (If Pursuing DINO)

### Immediate Actions:
1. Create `copy_images_for_dino.py` script
2. Run it to populate `coco_room_dino/{split}/images/`
3. Update `sagemaker_train_dino.py` to upload images
4. Test with 5-epoch run
5. Monitor for CUDA compilation issues

### Success Criteria:
- Training job runs without errors for >10 minutes
- Loss decreases (not NaN or constant)
- No CUDA OOM errors
- Can download model artifacts after completion

---

## Files to Create/Modify

### New Files:
- [ ] `copy_images_for_dino.py` - Copy images from yolo dataset to COCO structure

### Files to Modify:
- [ ] `sagemaker_train_dino.py` (lines 82-94) - Add image upload logic
- [ ] `prepare_coco_for_dino.py` - Update to write relative image paths
- [ ] `coco_room_dino/{split}/annotations.json` - Fix image paths

---

## Summary

**Status:** DINO training is 80% ready, but has a critical blocker (no images in S3).

**Effort to fix:** 30-45 minutes of focused work.

**ROI:** Questionable - you already have excellent YOLO models.

**Recommendation:** Defer DINO until after demo, focus on segmentation model results first.

