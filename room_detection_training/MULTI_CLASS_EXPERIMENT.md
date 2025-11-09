## Multi-Class Training Experiment: Walls + Rooms

### The Idea

Train YOLO on **both walls and rooms** (2 classes), but suppress wall detections at inference time. This allows the model to learn the spatial relationship between walls and rooms, which could improve room boundary detection.

### When This Might Help

**For DETECTION (bbox):**
- ✅ Could improve room boundary accuracy
- ✅ Model learns "rooms are defined by walls"
- ✅ More training data (173K annotations vs 49K)
- ✅ Walls provide spatial context

**For SEGMENTATION:**
- ❌ **NOT recommended** - room polygons already encode wall boundaries
- ❌ Walls are redundant with room edges
- ❌ No additional benefit from wall masks

### Dataset Comparison

| Dataset | Classes | Annotations | Use Case |
|---------|---------|-------------|----------|
| `yolo_room_only` | 1 (room) | 49K rooms | Standard detection |
| `yolo_room_wall_2class` | 2 (wall+room) | 123K walls + 49K rooms | Experimental detection |
| `yolo_room_seg` | 1 (room) | 49K polygons | Segmentation |

### Building the 2-Class Dataset

```bash
cd room_detection_training
python rebuild_yolo_2class_dataset.py
```

This creates `yolo_room_wall_2class/` with:
- Class 0: wall
- Class 1: room

### Training on SageMaker

```bash
python sagemaker_train.py \
  --task detect \
  --model-size l \
  --imgsz 800 \
  --data-dir ./yolo_room_wall_2class
```

### Suppressing Walls at Inference

There are 3 ways to suppress wall detections:

#### Option 1: Confidence Threshold (Easiest)
Set wall confidence threshold to 1.0 (impossible to reach):

```python
from ultralytics import YOLO

model = YOLO('path/to/trained/model.pt')

# Predict with class-specific confidence thresholds
results = model.predict(
    image,
    conf=0.5,  # Base confidence
    classes=[1],  # Only keep room class (ignore walls)
)
```

#### Option 2: Post-Processing Filter

```python
results = model.predict(image, conf=0.5)

for result in results:
    # Filter out wall detections (class 0)
    room_boxes = result.boxes[result.boxes.cls == 1]  # Only class 1 (rooms)
    
    # Use room_boxes for downstream tasks
    for box in room_boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        confidence = box.conf[0].cpu().numpy()
        # ... process room detection
```

#### Option 3: Modify Inference Script

Update `apps/server/src/services/inference.py`:

```python
def load_model(model_id):
    # ... existing code ...
    
    # For 2-class models, configure to suppress walls
    if model_id.endswith('_2class'):
        # Will filter in detection processing
        model.class_filter = [1]  # Only room class
    
    return model

def detectRooms(self, image_path: str, model_id: str):
    # ... existing code ...
    
    results = model(img, conf=0.5)
    
    detections = []
    for r in results:
        boxes = r.boxes
        
        # Filter for room class only (class 1)
        for box in boxes:
            if box.cls == 1:  # Only rooms, skip walls (class 0)
                detections.append({
                    'bbox': box.xyxy[0].tolist(),
                    'confidence': float(box.conf),
                    'class': 'room'
                })
```

### Expected Results

**Potential Improvements:**
- Better room boundary detection (+2-5% IoU)
- More accurate room shape estimation
- Improved detection of rooms with complex layouts

**Potential Issues:**
- Slightly slower training (more data)
- Need to handle 2-class model at inference
- May not show significant improvement

### Recommendation

1. **Start with single-class** (`yolo_room_only`) - simpler, cleaner
2. **If room boundaries are imprecise**, try 2-class as an experiment
3. **Compare metrics** on test set to see if it helps
4. **Only use for detection**, not segmentation

### Testing the Benefit

After training both models, compare them:

```python
# Test both models on same images
from ultralytics import YOLO

model_1class = YOLO('yolo_room_only/best.pt')
model_2class = YOLO('yolo_room_wall_2class/best.pt')

# Run validation
metrics_1 = model_1class.val()
metrics_2 = model_2class.val()

print(f"1-class mAP: {metrics_1.box.map:.4f}")
print(f"2-class mAP: {metrics_2.box.map:.4f}")  # Filter rooms only
```

If 2-class shows >2-3% improvement, it's worth the added complexity. Otherwise, stick with single-class.

