#!/usr/bin/env python3
"""
Rebuild YOLO-SEG Dataset from CubiCasa5K COCO - ROOM POLYGONS (Category 2)
Creates segmentation dataset with full polygon masks, not just bounding boxes
70/20/10 train/val/test split
"""

import json
import os
import shutil
from pathlib import Path
from PIL import Image
import argparse
from tqdm import tqdm
import random

def convert_coco_to_yolo_seg(coco_path, output_dir, images_base_path, split_name):
    """
    Convert COCO annotations to YOLO SEGMENTATION format with polygons
    """
    print(f"\n{'='*60}")
    print(f"Processing {split_name} split - SEGMENTATION FORMAT")
    print(f"{'='*60}")
    
    # Load COCO data
    print(f"Loading COCO annotations from: {coco_path}")
    with open(coco_path, 'r') as f:
        coco = json.load(f)
    
    print(f"  Total images: {len(coco['images'])}")
    print(f"  Total annotations: {len(coco['annotations'])}")
    
    # Find ROOM category ID (should be 1 in room_detection_dataset_coco)
    room_cat_id = None
    for cat in coco['categories']:
        if cat['name'].lower() == 'room':
            room_cat_id = cat['id']
            print(f"  [OK] Found ROOM category with ID: {room_cat_id}")
            break
    
    if room_cat_id is None:
        raise ValueError("Could not find 'room' category in COCO data!")
    
    # Build image index
    image_index = {img['id']: img for img in coco['images']}
    
    # Group annotations by image, filtering for ROOM only with segmentation
    print(f"  Filtering for ROOM annotations with segmentation polygons...")
    annotations_by_image = {}
    room_annotation_count = 0
    skipped_no_seg = 0
    
    for ann in coco['annotations']:
        if ann['category_id'] == room_cat_id:
            if 'segmentation' in ann and ann['segmentation']:
                img_id = ann['image_id']
                if img_id not in annotations_by_image:
                    annotations_by_image[img_id] = []
                annotations_by_image[img_id].append(ann)
                room_annotation_count += 1
            else:
                skipped_no_seg += 1
    
    print(f"  [OK] Found {room_annotation_count} ROOM segmentation annotations across {len(annotations_by_image)} images")
    if skipped_no_seg > 0:
        print(f"  [WARN] Skipped {skipped_no_seg} room annotations without segmentation data")
    
    return annotations_by_image, image_index, room_cat_id


def split_data_70_20_10(annotations_by_image):
    """
    Split data into 70% train, 20% val, 10% test
    """
    print(f"\n{'='*60}")
    print("Creating 70/20/10 train/val/test split")
    print(f"{'='*60}")
    
    all_image_ids = list(annotations_by_image.keys())
    random.shuffle(all_image_ids)
    
    total = len(all_image_ids)
    train_size = int(0.7 * total)
    val_size = int(0.2 * total)
    # test_size is the remainder
    
    train_ids = all_image_ids[:train_size]
    val_ids = all_image_ids[train_size:train_size + val_size]
    test_ids = all_image_ids[train_size + val_size:]
    
    print(f"  Total images: {total}")
    print(f"  Train: {len(train_ids)} ({len(train_ids)/total*100:.1f}%)")
    print(f"  Val: {len(val_ids)} ({len(val_ids)/total*100:.1f}%)")
    print(f"  Test: {len(test_ids)} ({len(test_ids)/total*100:.1f}%)")
    
    return {
        'train': train_ids,
        'val': val_ids,
        'test': test_ids
    }


def process_split(split_name, image_ids, annotations_by_image, image_index, output_dir, images_base_path, reuse_images_from=None):
    """
    Process one split (train/val/test) and create YOLO-seg format files
    """
    # Create output directories
    split_dir = Path(output_dir) / split_name
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    processed_images = 0
    skipped_images = 0
    total_room_labels = 0
    
    # Check if we should reuse images from another dataset
    reuse_images = False
    if reuse_images_from and Path(reuse_images_from).exists():
        reuse_source = Path(reuse_images_from) / split_name / "images"
        if reuse_source.exists():
            reuse_images = True
            print(f"  [FAST MODE] Reusing images from: {reuse_source}")
    
    # Build local path mapping
    kaggle_prefix = "/kaggle/input/cubicasa5k/cubicasa5k/cubicasa5k/"
    images_base = Path(images_base_path)
    
    for img_id in tqdm(image_ids, desc=f"  Processing {split_name}"):
        if img_id not in image_index or img_id not in annotations_by_image:
            continue
        
        img_info = image_index[img_id]
        img_width = img_info['width']
        img_height = img_info['height']
        
        # Convert Kaggle path to local path
        coco_filename = img_info['file_name']
        if coco_filename.startswith(kaggle_prefix):
            relative_path = coco_filename[len(kaggle_prefix):]
        else:
            relative_path = Path(coco_filename).name
        
        # Find local image file
        local_img_path = images_base / relative_path.replace('/', os.sep)
        
        if not local_img_path.exists():
            # Try finding by filename only
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
        
        # Create YOLO label file with unique filename
        label_filename = f"{img_id:06d}.txt"
        image_filename = f"{img_id:06d}.png"
        
        label_path = labels_dir / label_filename
        new_image_path = images_dir / image_filename
        
        # Write YOLO SEGMENTATION format labels
        with open(label_path, 'w') as f:
            for ann in annotations_by_image[img_id]:
                # Get segmentation polygon
                if 'segmentation' not in ann or not ann['segmentation']:
                    continue
                
                segmentation = ann['segmentation']
                
                # Handle different segmentation formats
                if isinstance(segmentation, list) and len(segmentation) > 0:
                    # Polygon format: [[x1, y1, x2, y2, ...]]
                    polygon = segmentation[0] if isinstance(segmentation[0], list) else segmentation
                    
                    # Normalize coordinates
                    normalized_coords = []
                    for i in range(0, len(polygon), 2):
                        if i + 1 < len(polygon):
                            x = polygon[i] / img_width
                            y = polygon[i + 1] / img_height
                            normalized_coords.extend([x, y])
                    
                    # YOLO segmentation format: class_id x1 y1 x2 y2 x3 y3 ...
                    # class_id = 0 for single-class (room)
                    if normalized_coords:
                        coords_str = ' '.join([f"{coord:.6f}" for coord in normalized_coords])
                        f.write(f"0 {coords_str}\n")
                        total_room_labels += 1
        
        # Copy or symlink image
        try:
            if reuse_images:
                # Fast mode: create symlink to existing image
                source_image = reuse_source / image_filename
                if source_image.exists():
                    if not new_image_path.exists():
                        # On Windows, copy is more reliable than symlinks
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
            # Remove label file if image copy failed
            if label_path.exists():
                label_path.unlink()
    
    print(f"  [OK] {split_name}: {processed_images} images, {total_room_labels} polygon labels")
    if skipped_images > 0:
        print(f"  [WARN] {split_name}: Skipped {skipped_images} images (not found locally)")
    
    return processed_images, total_room_labels


def create_data_yaml(output_dir):
    """Create data.yaml for YOLO segmentation training"""
    data_yaml_content = f"""# YOLO Segmentation Data Configuration for Room Detection
# Generated from CubiCasa5K COCO dataset - category_id=2 (room)
# 70/20/10 train/val/test split

path: {output_dir}
train: train/images
val: val/images
test: test/images

# Classes
nc: 1
names: ['room']

# Task
task: segment
"""
    
    yaml_path = Path(output_dir) / "data.yaml"
    with open(yaml_path, 'w') as f:
        f.write(data_yaml_content)
    
    print(f"\n[OK] Created data.yaml at: {yaml_path}")
    return yaml_path


def main():
    parser = argparse.ArgumentParser(description='Rebuild YOLO-SEG dataset - ROOM POLYGONS')
    parser.add_argument('--coco-train', default='../room_detection_dataset_coco/train/annotations.json',
                       help='Path to train COCO annotations (use room_detection_dataset_coco with polygons!)')
    parser.add_argument('--images-base', default='../cubicasa5k/cubicasa5k',
                       help='Base path to cubicasa5k images')
    parser.add_argument('--output-dir', default='./yolo_room_seg',
                       help='Output directory for YOLO segmentation dataset')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducible splits (use 42 to match other datasets)')
    parser.add_argument('--reuse-images-from', default='./yolo_room_only',
                       help='Reuse images from existing dataset (much faster - only generates labels)')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("REBUILDING YOLO SEGMENTATION DATASET - ROOM POLYGONS")
    print("Source: room_detection_dataset_coco (GeoJSON-derived with polygons)")
    print("Format: YOLO-seg with full polygon masks")
    print("Split: 70% train / 20% val / 10% test")
    print("="*60)
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    coco_train = Path(args.coco_train)
    output_dir = Path(args.output_dir)
    images_base = Path(args.images_base)
    
    # Check paths
    if not coco_train.exists():
        raise FileNotFoundError(f"COCO file not found: {coco_train}")
    if not images_base.exists():
        raise FileNotFoundError(f"Images directory not found: {images_base}")
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    # Load and filter COCO data
    annotations_by_image, image_index, room_cat_id = convert_coco_to_yolo_seg(
        coco_train, 
        output_dir, 
        images_base,
        'all'
    )
    
    # Create 70/20/10 split
    splits = split_data_70_20_10(annotations_by_image)
    
    # Check if reusing images
    reuse_images_from = args.reuse_images_from if hasattr(args, 'reuse_images_from') else None
    if reuse_images_from and Path(reuse_images_from).exists():
        print(f"\n[FAST MODE] Reusing images from: {reuse_images_from}")
        print("Only generating new label files (much faster!)")
    
    # Process each split
    total_images = 0
    total_labels = 0
    
    for split_name, image_ids in splits.items():
        processed, labels = process_split(
            split_name,
            image_ids,
            annotations_by_image,
            image_index,
            output_dir,
            images_base,
            reuse_images_from
        )
        total_images += processed
        total_labels += labels
    
    # Create data.yaml
    create_data_yaml(output_dir)
    
    print("\n" + "="*60)
    print("YOLO SEGMENTATION DATASET BUILD COMPLETE!")
    print("="*60)
    print(f"Total images processed: {total_images}")
    print(f"Total ROOM polygon labels: {total_labels}")
    print(f"Output directory: {output_dir.absolute()}")
    print(f"\nSplit summary:")
    print(f"  Train: {len(splits['train'])} images")
    print(f"  Val: {len(splits['val'])} images")
    print(f"  Test: {len(splits['test'])} images")
    print(f"\nThis dataset uses FULL POLYGON segmentation, not just bboxes!")
    print(f"Ready for YOLO-seg training (yolov8-seg models)")
    print("="*60)


if __name__ == "__main__":
    main()

