# S3 Data Organization for YOLO Training

## S3 Bucket Structure

All training data is stored in: `s3://sagemaker-us-east-1-971422717446/room-detection-training/`

### Data Prefixes

| S3 Prefix | Local Dataset | Classes | Format | Notes |
|-----------|---------------|---------|--------|-------|
| `data-detect-1class` | `yolo_room_only` | 1 (room) | Bounding boxes | **CORRECTED** - Uses category_id=2 (room), proper 70/20/10 split |
| `data-detect-2class` | `yolo_room_wall_2class` | 2 (wall, room) | Bounding boxes | For experimentation - can suppress walls at inference |
| `data-segment` | `yolo_room_seg` | 1 (room) | Polygons | Full segmentation masks from GeoJSON data |
| `data-detect` | ❌ OLD/INCORRECT | - | - | **DO NOT USE** - Wrong split (all 4200 in train) |

## Image Reuse Optimization

When training 2-class or segmentation models:
- The script checks if images exist in `data-detect-1class`
- If found, performs S3 server-side copy for images (much faster!)
- Only uploads new label files
- Saves significant upload time (images are ~4GB, labels are ~50MB)

## Training Commands

### 1-Class Detection (Room Only)
```bash
python sagemaker_train.py --task detect --model-size l --imgsz 800
# Auto-detects: data-detect-1class
```

### 2-Class Detection (Wall + Room)
```bash
python sagemaker_train.py --task detect --model-size l --imgsz 800 --data-dir ./yolo_room_wall_2class
# Auto-detects: data-detect-2class
```

### Segmentation (Room Polygons)
```bash
python sagemaker_train.py --task segment --model-size l --imgsz 800
# Auto-detects: data-segment
```

### Custom S3 Prefix
```bash
python sagemaker_train.py --task detect --s3-prefix my-custom-prefix/data
```

## Dataset Splits

All datasets use **seed=42** for reproducible 70/20/10 splits:
- Train: 70% (~2940 images)
- Val: 20% (~840 images)
- Test: 10% (~420 images)
- **Total: 4194 images** (from CubiCasa5K with room annotations)

## Key Fix: Wall vs Room

Original datasets incorrectly trained on **category_id=1 (wall)** instead of **category_id=2 (room)**.

All new datasets (`*-1class`, `*-2class`, `*-segment`) correctly use **category_id=2 (room)**.

