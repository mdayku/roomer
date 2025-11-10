#!/usr/bin/env python3
"""
Build YOLO-SEG Dataset - FIXED VERSION
Maps GeoJSON files to images using proper image IDs from filenames
"""
import json
import shutil
from pathlib import Path
from tqdm import tqdm
import random
import re

def extract_image_id_from_geojson(geojson_filename):
    """
    Extract numeric ID from GeoJSON filename
    Example: 'F1_original_1234.geo.json' -> 1234
    """
    stem = Path(geojson_filename).stem
    if stem.endswith('.geo'):
        stem = stem[:-4]
    
    # Extract the trailing number
    match = re.search(r'_(\d+)$', stem)
    if match:
        return int(match.group(1))
    return None

def load_geojson_polygons(geojson_path):
    """Load room polygons from a GeoJSON file"""
    with open(geojson_path, 'r') as f:
        data = json.load(f)
    
    polygons = []
    for feature in data.get('features', []):
        if feature['geometry']['type'] == 'Polygon':
            # Get outer ring, already in normalized [0,1] coordinates
            ring = feature['geometry']['coordinates'][0]
            if len(ring) >= 3:  # Valid polygon
                polygons.append(ring)
    
    return polygons

def polygon_to_yolo_seg(polygon):
    """Convert normalized polygon to YOLO-seg format"""
    coords = []
    for x, y in polygon:
        # Already normalized, just clamp
        x = max(0.0, min(1.0, float(x)))
        y = max(0.0, min(1.0, float(y)))
        coords.append(f"{x:.6f}")
        coords.append(f"{y:.6f}")
    
    return "0 " + " ".join(coords)

print("\n" + "="*60)
print("REBUILDING YOLO SEGMENTATION DATASET - FIXED")
print("Using proper image ID matching")
print("="*60)

# Load COCO to get room images
print("\nLoading COCO data...")
coco_file = Path("../cubicasa5k_coco/train_coco_pt.json")
with open(coco_file) as f:
    coco_data = json.load(f)

# Find images with room annotations (category_id=2)
room_images = set()
for ann in coco_data['annotations']:
    if ann['category_id'] == 2:
        room_images.add(ann['image_id'])

print(f"Found {len(room_images)} images with room annotations")

# Create 70/20/10 split (seed=42)
print("\nCreating 70/20/10 split (seed=42)...")
random.seed(42)
all_img_ids = sorted(list(room_images))
random.shuffle(all_img_ids)

total = len(all_img_ids)
train_size = int(0.7 * total)
val_size = int(0.2 * total)

train_ids = set(all_img_ids[:train_size])
val_ids = set(all_img_ids[train_size:train_size + val_size])
test_ids = set(all_img_ids[train_size + val_size:])

print(f"Train: {len(train_ids)} | Val: {len(val_ids)} | Test: {len(test_ids)}")

# Build GeoJSON index by image ID
print("\nIndexing GeoJSON files by image ID...")
geojson_dir = Path("../cubicasa5k_geojson")
geojson_by_id = {}
skipped_geojson = 0

for geojson_file in geojson_dir.glob("*.json"):
    img_id = extract_image_id_from_geojson(geojson_file.name)
    if img_id is not None:
        geojson_by_id[img_id] = geojson_file
    else:
        skipped_geojson += 1

print(f"Indexed {len(geojson_by_id)} GeoJSON files by image ID")
if skipped_geojson > 0:
    print(f"  (Skipped {skipped_geojson} files with no extractable ID)")

# Process each split
output_dir = Path("./yolo_room_seg")
detection_dataset = Path("./yolo_room_only")

total_images = 0
total_labels = 0

for split_name, split_ids in [('train', train_ids), ('val', val_ids), ('test', test_ids)]:
    print(f"\n[INFO] Processing {split_name} split...")
    
    split_images_dir = output_dir / split_name / "images"
    split_labels_dir = output_dir / split_name / "labels"
    split_images_dir.mkdir(parents=True, exist_ok=True)
    split_labels_dir.mkdir(parents=True, exist_ok=True)
    
    source_images_dir = detection_dataset / split_name / "images"
    
    processed = 0
    labels_created = 0
    skipped = 0
    
    for img_id in tqdm(split_ids, desc=f"  {split_name}"):
        # Find corresponding GeoJSON file by image ID
        if img_id not in geojson_by_id:
            skipped += 1
            continue
        
        geojson_file = geojson_by_id[img_id]
        
        # Load polygons from GeoJSON
        try:
            polygons = load_geojson_polygons(geojson_file)
        except Exception as e:
            skipped += 1
            continue
        
        if not polygons:
            skipped += 1
            continue
        
        # Copy image from detection dataset
        image_filename = f"{img_id:06d}.png"
        source_image = source_images_dir / image_filename
        
        if not source_image.exists():
            skipped += 1
            continue
        
        dest_image = split_images_dir / image_filename
        if not dest_image.exists():
            shutil.copy(source_image, dest_image)
        processed += 1
        
        # Create YOLO-seg label file
        label_filename = f"{img_id:06d}.txt"
        label_path = split_labels_dir / label_filename
        
        with open(label_path, 'w') as f:
            for polygon in polygons:
                yolo_line = polygon_to_yolo_seg(polygon)
                if yolo_line:
                    f.write(yolo_line + '\n')
                    labels_created += 1
    
    print(f"  [OK] {split_name}: {processed} images, {labels_created} polygon labels (skipped {skipped})")
    total_images += processed
    total_labels += labels_created

# Create data.yaml
yaml_path = output_dir / 'data.yaml'
yaml_content = f"""# YOLO Segmentation Dataset - Room Polygons from GeoJSON
# 1 class: room (with full polygon masks)
# Fixed image ID matching

path: {output_dir.absolute()}
train: train/images
val: val/images
test: test/images

nc: 1
names: ['room']

task: segment
"""

with open(yaml_path, 'w') as f:
    f.write(yaml_content)

print(f"\n{'='*60}")
print("SEGMENTATION DATASET BUILD COMPLETE!")
print(f"{'='*60}")
print(f"Total images: {total_images}")
print(f"Total polygon labels: {total_labels}")
print(f"Output: {output_dir.absolute()}")
print(f"{'='*60}\n")

