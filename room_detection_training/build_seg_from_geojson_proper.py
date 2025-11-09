#!/usr/bin/env python3
"""
Build YOLO-SEG Dataset from GeoJSON files
Uses SAME split as detection dataset by matching image IDs
"""

import json
from pathlib import Path
import shutil
from tqdm import tqdm
import random

def load_geojson_polygons(geojson_path):
    """Load room polygons from a GeoJSON file"""
    with open(geojson_path, 'r') as f:
        data = json.load(f)
    
    polygons = []
    for feature in data.get('features', []):
        if feature['geometry']['type'] == 'Polygon':
            # Get outer ring, already in normalized [0,1] coordinates
            ring = feature['geometry']['coordinates'][0]
            polygons.append(ring)
    
    return polygons

def polygon_to_yolo_seg(polygon):
    """Convert normalized polygon to YOLO-seg format"""
    if len(polygon) < 3:
        return None
    
    coords = []
    for x, y in polygon:
        # Already normalized, just clamp
        x = max(0.0, min(1.0, x))
        y = max(0.0, min(1.0, y))
        coords.append(f"{x:.6f}")
        coords.append(f"{y:.6f}")
    
    return "0 " + " ".join(coords)

# Load COCO to get the same image set and split
print("Loading COCO data to match detection dataset split...")
coco_file = Path("../cubicasa5k_coco/train_coco_pt.json")
with open(coco_file, 'r') as f:
    coco_data = json.load(f)

# Filter for images with room annotations (category_id=2)
print("Finding images with room annotations...")
room_images = set()
for ann in coco_data['annotations']:
    if ann['category_id'] == 2:
        room_images.add(ann['image_id'])

print(f"Found {len(room_images)} images with room annotations")

# Build image filename mapping
image_id_to_filename = {}
for img in coco_data['images']:
    if img['id'] in room_images:
        # Extract clean filename from path
        filename = Path(img['file_name']).stem
        # Remove _original suffix if present
        if filename.endswith('_original'):
            filename = filename[:-9]
        image_id_to_filename[img['id']] = filename

# Create same 70/20/10 split (seed=42)
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

# Build mapping of all GeoJSON files by partial filename match
print("\nIndexing GeoJSON files...")
geojson_dir = Path("../cubicasa5k_geojson")
geojson_index = {}
for geojson_file in geojson_dir.glob("*.json"):
    stem = geojson_file.stem
    if stem.endswith('.geo'):
        stem = stem[:-4]
    # Key by the main part of filename for fuzzy matching
    geojson_index[stem] = geojson_file

print(f"Found {len(geojson_index)} GeoJSON files")

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
        # Find corresponding GeoJSON file
        if img_id not in image_id_to_filename:
            skipped += 1
            continue
        
        base_filename = image_id_to_filename[img_id]
        
        # Try to find matching GeoJSON file
        geojson_file = None
        for key in geojson_index:
            if base_filename in key or key in base_filename:
                geojson_file = geojson_index[key]
                break
        
        if not geojson_file:
            skipped += 1
            continue
        
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

