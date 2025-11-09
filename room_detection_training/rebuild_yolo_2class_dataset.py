#!/usr/bin/env python3
"""
Rebuild YOLO Dataset with BOTH rooms AND walls (2-class)
For experimentation: train on both, suppress walls at inference

This trains the model on the spatial relationship between walls and rooms,
which could improve room boundary detection in the detection (bbox) task.

NOT recommended for segmentation - room polygons already capture wall info.
"""

import json
import os
import shutil
from pathlib import Path
from PIL import Image
import argparse
from tqdm import tqdm
import random

def convert_coco_to_yolo_2class(coco_path, output_dir, images_base_path, split_name):
    """
    Convert COCO to YOLO format with BOTH wall (class 0) and room (class 1)
    """
    print(f"\n{'='*60}")
    print(f"Processing {split_name} split - 2-CLASS (wall + room)")
    print(f"{'='*60}")
    
    with open(coco_path, 'r') as f:
        coco = json.load(f)
    
    print(f"  Total images: {len(coco['images'])}")
    print(f"  Total annotations: {len(coco['annotations'])}")
    
    # Find wall and room category IDs
    wall_cat_id = None
    room_cat_id = None
    for cat in coco['categories']:
        if cat['name'].lower() == 'wall':
            wall_cat_id = cat['id']
        elif cat['name'].lower() == 'room':
            room_cat_id = cat['id']
    
    print(f"  [OK] Found WALL category with ID: {wall_cat_id}")
    print(f"  [OK] Found ROOM category with ID: {room_cat_id}")
    
    if not wall_cat_id or not room_cat_id:
        raise ValueError("Could not find both 'wall' and 'room' categories!")
    
    # Build image index
    image_index = {img['id']: img for img in coco['images']}
    
    # Group annotations by image
    print(f"  Collecting both WALL and ROOM annotations...")
    annotations_by_image = {}
    wall_count = 0
    room_count = 0
    
    for ann in coco['annotations']:
        if ann['category_id'] in [wall_cat_id, room_cat_id]:
            img_id = ann['image_id']
            if img_id not in annotations_by_image:
                annotations_by_image[img_id] = []
            annotations_by_image[img_id].append(ann)
            
            if ann['category_id'] == wall_cat_id:
                wall_count += 1
            else:
                room_count += 1
    
    print(f"  [OK] Found {wall_count} WALL annotations")
    print(f"  [OK] Found {room_count} ROOM annotations")
    print(f"  [OK] Total annotations: {wall_count + room_count} across {len(annotations_by_image)} images")
    
    return annotations_by_image, image_index, wall_cat_id, room_cat_id


def split_data_70_20_10(annotations_by_image):
    """Split into 70/20/10"""
    all_image_ids = list(annotations_by_image.keys())
    random.shuffle(all_image_ids)
    
    total = len(all_image_ids)
    train_size = int(0.7 * total)
    val_size = int(0.2 * total)
    
    train_ids = all_image_ids[:train_size]
    val_ids = all_image_ids[train_size:train_size + val_size]
    test_ids = all_image_ids[train_size + val_size:]
    
    print(f"\n{'='*60}")
    print("70/20/10 split:")
    print(f"  Train: {len(train_ids)}")
    print(f"  Val: {len(val_ids)}")
    print(f"  Test: {len(test_ids)}")
    print(f"{'='*60}")
    
    return {
        'train': train_ids,
        'val': val_ids,
        'test': test_ids
    }


def process_split(split_name, image_ids, annotations_by_image, image_index, 
                   output_dir, images_base_path, wall_cat_id, room_cat_id, reuse_images_from=None):
    """Process one split with 2-class labels"""
    split_dir = Path(output_dir) / split_name
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    processed_images = 0
    skipped_images = 0
    wall_labels = 0
    room_labels = 0
    
    # Check if we should reuse images from another dataset
    reuse_images = False
    if reuse_images_from and Path(reuse_images_from).exists():
        reuse_source = Path(reuse_images_from) / split_name / "images"
        if reuse_source.exists():
            reuse_images = True
            print(f"  [FAST MODE] Reusing images from: {reuse_source}")
    
    kaggle_prefix = "/kaggle/input/cubicasa5k/cubicasa5k/cubicasa5k/"
    images_base = Path(images_base_path)
    
    for img_id in tqdm(image_ids, desc=f"  Processing {split_name}"):
        if img_id not in image_index or img_id not in annotations_by_image:
            continue
        
        img_info = image_index[img_id]
        img_width = img_info['width']
        img_height = img_info['height']
        
        # Find local image
        coco_filename = img_info['file_name']
        if coco_filename.startswith(kaggle_prefix):
            relative_path = coco_filename[len(kaggle_prefix):]
        else:
            relative_path = Path(coco_filename).name
        
        local_img_path = images_base / relative_path.replace('/', os.sep)
        
        if not local_img_path.exists():
            filename_only = Path(relative_path).name
            found = False
            for root, dirs, files in os.walk(images_base):
                if filename_only in files:
                    local_img_path = Path(root) / filename_only
                    found = True
                    break
            if not found:
                skipped_images += 1
                continue
        
        # Create label file
        label_filename = f"{img_id:06d}.txt"
        image_filename = f"{img_id:06d}.png"
        label_path = labels_dir / label_filename
        new_image_path = images_dir / image_filename
        
        # Write YOLO labels with 2 classes
        with open(label_path, 'w') as f:
            for ann in annotations_by_image[img_id]:
                if 'bbox' not in ann:
                    continue
                
                x, y, w, h = ann['bbox']
                
                # Normalize to [0, 1]
                center_x = (x + w / 2) / img_width
                center_y = (y + h / 2) / img_height
                norm_w = w / img_width
                norm_h = h / img_height
                
                # Map to YOLO class IDs: wall=0, room=1
                if ann['category_id'] == wall_cat_id:
                    yolo_class = 0
                    wall_labels += 1
                elif ann['category_id'] == room_cat_id:
                    yolo_class = 1
                    room_labels += 1
                else:
                    continue
                
                f.write(f"{yolo_class} {center_x:.6f} {center_y:.6f} {norm_w:.6f} {norm_h:.6f}\n")
        
        # Copy or reuse image
        try:
            if reuse_images:
                # Fast mode: reuse from existing dataset
                source_image = reuse_source / image_filename
                if source_image.exists():
                    if not new_image_path.exists():
                        shutil.copy2(str(source_image), str(new_image_path))
                    processed_images += 1
                else:
                    # Fallback to original source
                    shutil.copy2(str(local_img_path), str(new_image_path))
                    processed_images += 1
            else:
                # Normal mode: copy from source
                shutil.copy2(str(local_img_path), str(new_image_path))
                processed_images += 1
        except Exception as e:
            print(f"    Warning: Failed to copy {local_img_path}: {e}")
            skipped_images += 1
            if label_path.exists():
                label_path.unlink()
    
    print(f"  [OK] {split_name}: {processed_images} images")
    print(f"       Wall labels: {wall_labels}, Room labels: {room_labels}")
    if skipped_images > 0:
        print(f"  [WARN] Skipped: {skipped_images} images")
    
    return processed_images, wall_labels, room_labels


def create_data_yaml(output_dir):
    """Create data.yaml for 2-class training"""
    data_yaml_content = f"""# YOLO 2-Class Data Configuration
# Classes: wall (0) and room (1)
# Strategy: Train on both, suppress walls at inference

path: {output_dir}
train: train/images
val: val/images
test: test/images

# Classes
nc: 2
names: ['wall', 'room']

# Task
task: detect

# Notes:
# - Train the model to understand wall-room relationships
# - At inference, set wall confidence threshold to 1.0 to suppress
# - Or post-process to remove class 0 (wall) detections
"""
    
    yaml_path = Path(output_dir) / "data.yaml"
    with open(yaml_path, 'w') as f:
        f.write(data_yaml_content)
    
    print(f"\n[OK] Created data.yaml at: {yaml_path}")
    return yaml_path


def main():
    parser = argparse.ArgumentParser(description='Build 2-class YOLO dataset (wall + room)')
    parser.add_argument('--coco-train', default='../cubicasa5k_coco/train_coco_pt.json')
    parser.add_argument('--images-base', default='../cubicasa5k/cubicasa5k')
    parser.add_argument('--output-dir', default='./yolo_room_wall_2class')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--reuse-images-from', default='./yolo_room_only',
                       help='Reuse images from existing dataset (much faster - only generates labels)')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("BUILDING 2-CLASS YOLO DATASET: WALL + ROOM")
    print("="*60)
    print("Purpose: Experimental - train on wall context")
    print("Use case: Detection only (not for segmentation)")
    print("Inference: Suppress wall detections (conf threshold = 1.0)")
    print("="*60)
    
    random.seed(args.seed)
    
    coco_train = Path(args.coco_train)
    output_dir = Path(args.output_dir)
    images_base = Path(args.images_base)
    
    if not coco_train.exists():
        raise FileNotFoundError(f"COCO file not found: {coco_train}")
    if not images_base.exists():
        raise FileNotFoundError(f"Images directory not found: {images_base}")
    
    output_dir.mkdir(exist_ok=True)
    
    # Load data
    annotations_by_image, image_index, wall_cat_id, room_cat_id = convert_coco_to_yolo_2class(
        coco_train, output_dir, images_base, 'all'
    )
    
    # Split data
    splits = split_data_70_20_10(annotations_by_image)
    
    # Check if reusing images
    reuse_images_from = args.reuse_images_from if hasattr(args, 'reuse_images_from') else None
    if reuse_images_from and Path(reuse_images_from).exists():
        print(f"\n[FAST MODE] Reusing images from: {reuse_images_from}")
        print("Only generating new label files (much faster!)")
    
    # Process splits
    total_images = 0
    total_walls = 0
    total_rooms = 0
    
    for split_name, image_ids in splits.items():
        processed, walls, rooms = process_split(
            split_name, image_ids, annotations_by_image, image_index,
            output_dir, images_base, wall_cat_id, room_cat_id, reuse_images_from
        )
        total_images += processed
        total_walls += walls
        total_rooms += rooms
    
    # Create data.yaml
    create_data_yaml(output_dir)
    
    print("\n" + "="*60)
    print("2-CLASS DATASET COMPLETE!")
    print("="*60)
    print(f"Total images: {total_images}")
    print(f"Wall labels: {total_walls}")
    print(f"Room labels: {total_rooms}")
    print(f"Total labels: {total_walls + total_rooms}")
    print(f"\nOutput: {output_dir.absolute()}")
    
    print("\n" + "="*60)
    print("USAGE NOTES")
    print("="*60)
    print("Train with:")
    print(f"  python sagemaker_train.py --task detect --data-dir {output_dir}")
    print("\nAt inference, suppress walls by:")
    print("  1. Setting wall confidence threshold = 1.0")
    print("  2. Or filtering out class 0 in post-processing")
    print("\nExpected benefit:")
    print("  Model learns wall-room spatial relationships")
    print("  May improve room boundary detection")
    print("\nNOT recommended for segmentation task!")
    print("="*60)


if __name__ == "__main__":
    main()

