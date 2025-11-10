# DINO AWS Training - READY TO LAUNCH ✅

## Status: FIXED! All blockers resolved

### What We Fixed 🔧

1. ✅ **Image Upload** - `sagemaker_train_dino.py` now uploads both images and annotations
   - Uses parallel uploads (10 threads)
   - Checks for existing files to avoid re-uploads
   - Properly handles both `.png` and `.jpg` files

2. ✅ **ID-Based Naming** - No more fuzzy matching!
   - Images use `{img_id:06d}.png` format (e.g., `000123.png`)
   - COCO annotations reference these ID-based filenames
   - Reuses images from `yolo_room_wall_2class` (already have correct IDs)

3. ✅ **Image Copying** - `prepare_coco_for_dino.py` updated
   - Copies images from YOLO dataset (fast!)
   - Updates COCO `file_name` to match
   - Creates proper directory structure: `{split}/images/{img_id}.png`

---

## Launch Commands

### Step 1: Prepare DINO Dataset (10-15 minutes)

```bash
cd room_detection_training

# Copy images from YOLO dataset and create COCO structure
python prepare_coco_for_dino.py --yolo-dataset ./yolo_room_wall_2class --output-dir ./coco_room_dino
```

**What this does:**
- Loads CubiCasa5K COCO annotations
- Creates 70/20/10 split (seed=42, same as YOLO)
- Copies 4,194 images from `yolo_room_wall_2class` (reuses existing files!)
- Updates COCO annotations to reference ID-based filenames
- Outputs to `coco_room_dino/{train,val,test}/{images,annotations.json}`

**Expected output:**
```
[TRAIN] Creating COCO annotations...
  [OK] 2935 images, XXXX annotations
       Images copied: 2935, skipped: 0
       
[VAL] Creating COCO annotations...
  [OK] 838 images, XXXX annotations
       Images copied: 838, skipped: 0
       
[TEST] Creating COCO annotations...
  [OK] 421 images, XXXX annotations  
       Images copied: 421, skipped: 0
```

---

### Step 2: Launch DINO Training on SageMaker (5 minutes)

```bash
# Test run (5 epochs, ~1.5 hours, ~$3.62)
python sagemaker_train_dino.py
```

**What this does:**
- Uploads `coco_room_dino/` to S3 (annotations + images)
- Creates SageMaker PyTorch estimator
- Launches training job on `ml.g5.2xlarge` (A10G x2 GPUs)
- Training script clones DINO repo in container
- Compiles custom CUDA operators
- Runs DINO training with room detection config

**Expected output:**
```
Found 4197 files to upload
Using parallel uploads (10 threads)...
  Progress: 200/4197 files processed (4.8%)
  ...
  Progress: 4197/4197 files processed (100.0%)

[OK] Data sync complete: s3://bucket/room-detection-dino/data
  - Uploaded: 4197 new files
  - Skipped: 0 existing files

[OK] DINO training job launched: room-detection-dino-1731234567
```

---

## Cost Breakdown

**Instance:** `ml.g5.2xlarge` (A10G x2 GPUs)

| Run Type | Epochs | Time | Cost |
|----------|--------|------|------|
| Test | 5 | ~1.5 hrs | **$3.62** |
| Full | 50 | ~15 hrs | **$36.15** |

**Storage:** ~$0.10/month (850MB dataset + 1GB model)

---

## Expected Training Behavior

### Success Indicators ✅
- Job runs >10 minutes without errors
- Loss decreases smoothly
- No `CUDA out of memory` errors
- No `ModuleNotFoundError` for DINO modules
- Custom ops compile successfully

### Potential Issues ⚠️

1. **CUDA ops compilation fails**
   - Symptom: `ModuleNotFoundError: No module named 'MultiScaleDeformableAttention'`
   - Fix: Check container has nvcc and gcc (should be pre-installed in DLC)

2. **COCO format issues**
   - Symptom: `KeyError: 'images'` or `FileNotFoundError` for images
   - Fix: Verify `coco_room_dino/{split}/images/` contains images
   - Fix: Check annotations reference `images/{img_id}.png`

3. **OOM (Out of Memory)**
   - Symptom: `CUDA out of memory`
   - Fix: Reduce batch size in `sagemaker_train_dino.py` (currently 2)
   - Fix: Reduce `hidden_dim` from 256 to 192

---

## Verify Dataset Before Upload

Run this quick check:

```bash
cd room_detection_training

# Check structure
ls coco_room_dino/train/images | wc -l  # Should be 2935
ls coco_room_dino/val/images | wc -l    # Should be 838
ls coco_room_dino/test/images | wc -l   # Should be 421

# Check annotations reference correct paths
python -c "import json; data=json.load(open('coco_room_dino/train/annotations.json')); print(data['images'][0]['file_name'])"
# Should output: images/XXXXXX.png
```

---

## Monitor Training

**CloudWatch Logs:**
```
https://console.aws.amazon.com/cloudwatch/home?region=us-east-1#logsV2:log-groups/log-group/aws/sagemaker/TrainingJobs
```

**SageMaker Console:**
```
https://console.aws.amazon.com/sagemaker/home?region=us-east-1#/jobs
```

**Watch for:**
- `[OK] DINO ops compiled successfully`
- `[OK] Starting DINO training with config`
- `Epoch [X/5] ...`
- Decreasing loss values

---

## After Training Completes

### Download Model

```bash
aws s3 cp s3://sagemaker-us-east-1-ACCOUNT/room-detection-dino-TIMESTAMP/output/model.tar.gz ./dino_model.tar.gz
tar -xzf dino_model.tar.gz
```

### Test Inference (TODO)

Create inference script similar to YOLO comprehensive_test.py

---

## Summary

**Status:** ✅ READY TO LAUNCH

**Blockers Fixed:**
- ✅ Images now uploaded to S3
- ✅ ID-based naming (no fuzzy matching)
- ✅ COCO annotations reference correct paths
- ✅ Parallel uploads for speed

**Time to Launch:** 15-20 minutes
1. Prepare dataset: 10-15 min
2. Upload to S3: 2-5 min (parallel)
3. Launch job: 1 min

**Recommendation:**
- Run test (5 epochs) first to verify everything works
- If successful, launch full training (50 epochs)
- Compare results with YOLO models (89.5% F1 to beat!)

