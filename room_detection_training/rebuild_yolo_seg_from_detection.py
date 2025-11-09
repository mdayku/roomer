#!/usr/bin/env python3
"""
Build YOLO-SEG Dataset by reusing images from detection dataset
and generating polygon labels from room_detection_dataset_coco

This is MUCH faster since:
1. Images are already copied (reuse from yolo_room_only)
2. We just generate new polygon label files
3. Uses the SAME train/val/test split (seed=42)
"""

import json
import shutil
from pathlib import Path
import argparse
from tqdm import tqdm
import random

def convert_polygon_to_yolo_seg(segmentation, img_width, img_height):
    """
    Convert COCO segmentation polygon to YOLO format
    COCO format: [x1, y1, x2, y2, ...] in pixels
    YOLO format: "0 x1 y1 x2 y2 ..." in normalized [0,1] coordinates
    """
    if not segmentation or len(segmentation) == 0:
        return None
    
    # Take first polygon (outer boundary)
    polygon = segmentation[0]
    if len(polygon) < 6:  # Need at least 3 points (6 coordinates)
        return None
    
    # Normalize coordinates
    normalized = []
    for i in range(0, len(polygon), 2):
        x = polygon[i] / img_width
        y = polygon[i+1] / img_height
        # Clamp to [0,1]
        x = max(0.0, min(1.0, x))
        y = max(0.0, min(1.0, y))
        normalized.append(f"{x:.6f}")
        normalized.append(f"{y:.6f}")
    
    return "0 " + " ".join(normalized)

def main():
    parser = argparse.ArgumentParser(description='Build YOLO-SEG dataset from detection dataset + COCO polygons')
    parser.add_argument('--detection-dataset', default='./yolo_room_only',
                       help='Detection dataset to reuse images from')
    parser.add_argument('--coco-train', default='../cubicasa5k_coco/train_coco_pt.json',
                       help='COCO file with image metadata')
    parser.add_argument('--output-dir', default='./yolo_room_seg',
                       help='Output directory for segmentation dataset')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (must match detection dataset split!)')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("BUILDING YOLO SEGMENTATION DATASET")
    print("Reusing images from detection dataset + generating polygon labels")
    print("="*60)
    
    detection_dataset = Path(args.detection_dataset)
    output_dir = Path(args.output_dir)
    
    # Load COCO to get image dimensions
    print(f"\nLoading COCO metadata from: {args.coco_train}")
    with open(args.coco_train, 'r') as f:
        coco_data = json.load(f)
    
    # Build image metadata index
    image_metadata = {img['id']: img for img in coco_data['images']}
    
    # Filter for ROOM annotations with segmentation
    print("Filtering for ROOM annotations with polygons...")
    room_annotations = {}
    room_images = set()
    
    for ann in coco_data['annotations']:
        if ann['category_id'] == 2 and 'segmentation' in ann and ann['segmentation']:  # room with polygons
            img_id = ann['image_id']
            room_images.add(img_id)
            if img_id not in room_annotations:
                room_annotations[img_id] = []
            room_annotations[img_id].append(ann)
    
    print(f"Found {len(room_images)} images with room polygon annotations")
    print(f"Total room polygons: {sum(len(anns) for anns in room_annotations.values())}")
    
    # Create same 70/20/10 split as detection dataset (using same seed!)
    random.seed(args.seed)
    all_img_ids = sorted(list(room_images))
    random.shuffle(all_img_ids)
    
    total = len(all_img_ids)
    train_size = int(0.7 * total)
    val_size = int(0.2 * total)
    
    train_ids = set(all_img_ids[:train_size])
    val_ids = set(all_img_ids[train_size:train_size + val_size])
    test_ids = set(all_img_ids[train_size + val_size:])
    
    print(f"\n70/20/10 Split (seed={args.seed}):")
    print(f"  Train: {len(train_ids)} images")
    print(f"  Val: {len(val_ids)} images")
    print(f"  Test: {len(test_ids)} images")
    
    # Process each split
    total_images = 0
    total_labels = 0
    
    for split_name, split_ids in [('train', train_ids), ('val', val_ids), ('test', test_ids)]:
        print(f"\n[INFO] Processing {split_name} split...")
        
        # Create directories
        split_images_dir = output_dir / split_name / "images"
        split_labels_dir = output_dir / split_name / "labels"
        split_images_dir.mkdir(parents=True, exist_ok=True)
        split_labels_dir.mkdir(parents=True, exist_ok=True)
        
        # Source images from detection dataset
        source_images_dir = detection_dataset / split_name / "images"
        
        processed = 0
        labels_created = 0
        
        for img_id in tqdm(split_ids, desc=f"  Processing {split_name}"):
            if img_id not in image_metadata or img_id not in room_annotations:
                continue
            
            img_info = image_metadata[img_id]
            img_width = img_info['width']
            img_height = img_info['height']
            
            # Image filename with unique ID (matching detection dataset)
            image_filename = f"{img_id:06d}.png"
            source_image = source_images_dir / image_filename
            
            if not source_image.exists():
                continue
            
            # Copy image
            dest_image = split_images_dir / image_filename
            if not dest_image.exists():
                shutil.copy(source_image, dest_image)
            processed += 1
            
            # Create polygon label file
            label_filename = f"{img_id:06d}.txt"
            label_path = split_labels_dir / label_filename
            
            with open(label_path, 'w') as f:
                for ann in room_annotations[img_id]:
                    yolo_line = convert_polygon_to_yolo_seg(
                        ann['segmentation'],
                        img_width,
                        img_height
                    )
                    if yolo_line:
                        f.write(yolo_line + '\n')
                        labels_created += 1
        
        print(f"  [OK] {split_name}: {processed} images, {labels_created} polygon labels")
        total_images += processed
        total_labels += labels_created
    
    # Create data.yaml
    yaml_path = output_dir / 'data.yaml'
    yaml_content = f"""# YOLO Segmentation Dataset - CubiCasa5K Room Polygons
# 1 class: room (with full polygon masks)
# Same 70/20/10 split as detection dataset (seed={args.seed})

path: {output_dir.absolute()}
train: train/images
val: val/images
test: test/images

nc: 1
names: ['room']

task: segment  # This is a segmentation dataset!
"""
    
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"\n[OK] Created data.yaml at: {yaml_path}")
    
    # Summary
    print(f"\n{'='*60}")
    print("YOLO SEGMENTATION DATASET BUILD COMPLETE!")
    print(f"{'='*60}")
    print(f"Total images: {total_images}")
    print(f"Total polygon labels: {total_labels}")
    print(f"Output: {output_dir.absolute()}")
    print(f"\nThis dataset uses FULL POLYGON segmentation!")
    print(f"Ready for YOLO-seg training (yolov8-seg models)")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    main()

