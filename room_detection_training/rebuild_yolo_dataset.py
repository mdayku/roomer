#!/usr/bin/env python3
"""
Rebuild YOLO Dataset from CubiCasa5K COCO - ROOM ONLY (Category 2)
This script properly filters for ROOM annotations, not walls
"""

import json
import os
import shutil
from pathlib import Path
from PIL import Image
import argparse
from tqdm import tqdm
import random

def convert_coco_to_yolo_room_only(coco_path, output_dir, images_base_path, split_name):
    """
    Convert COCO annotations to YOLO format, filtering for ROOM category only
    """
    print(f"\n{'='*60}")
    print(f"Processing {split_name} split")
    print(f"{'='*60}")
    
    # Load COCO data
    print(f"Loading COCO annotations from: {coco_path}")
    with open(coco_path, 'r') as f:
        coco = json.load(f)
    
    print(f"  Total images: {len(coco['images'])}")
    print(f"  Total annotations: {len(coco['annotations'])}")
    print(f"  Categories: {[(c['id'], c['name']) for c in coco['categories']]}")
    
    # Find ROOM category ID
    room_cat_id = None
    for cat in coco['categories']:
        if cat['name'].lower() == 'room':
            room_cat_id = cat['id']
            print(f"  [OK] Found ROOM category with ID: {room_cat_id}")
            break
    
    if room_cat_id is None:
        raise ValueError("Could not find 'room' category in COCO data!")
    
    # Create output directories
    output_dir = Path(output_dir) / split_name
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Build image index
    image_index = {img['id']: img for img in coco['images']}
    
    # Group annotations by image, filtering for ROOM only
    print(f"  Filtering for ROOM annotations only (category_id={room_cat_id})...")
    annotations_by_image = {}
    room_annotation_count = 0
    
    for ann in coco['annotations']:
        if ann['category_id'] == room_cat_id:  # ONLY ROOM annotations
            img_id = ann['image_id']
            if img_id not in annotations_by_image:
                annotations_by_image[img_id] = []
            annotations_by_image[img_id].append(ann)
            room_annotation_count += 1
    
    print(f"  [OK] Found {room_annotation_count} ROOM annotations across {len(annotations_by_image)} images")
    
    # Process each image
    processed_images = 0
    skipped_images = 0
    total_room_labels = 0
    
    # Build local path mapping
    kaggle_prefix = "/kaggle/input/cubicasa5k/cubicasa5k/cubicasa5k/"
    images_base = Path(images_base_path)
    
    print(f"  Searching for images in: {images_base.absolute()}")
    
    for img_id, anns in tqdm(annotations_by_image.items(), desc=f"  Processing {split_name}"):
        if img_id not in image_index:
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
            # Search in subdirectories
            found = False
            for root, dirs, files in os.walk(images_base):
                if filename_only in files:
                    local_img_path = Path(root) / filename_only
                    found = True
                    break
            
            if not found:
                skipped_images += 1
                continue
        
        # Create YOLO label file
        # Use image_id to create unique filename
        label_filename = f"{img_id:06d}.txt"
        image_filename = f"{img_id:06d}.png"
        
        label_path = labels_dir / label_filename
        new_image_path = images_dir / image_filename
        
        # Write YOLO format labels
        with open(label_path, 'w') as f:
            for ann in anns:
                # Convert COCO bbox [x, y, width, height] to YOLO [center_x, center_y, width, height] normalized
                bbox = ann['bbox']
                x, y, w, h = bbox
                
                # Normalize
                center_x = (x + w / 2) / img_width
                center_y = (y + h / 2) / img_height
                norm_w = w / img_width
                norm_h = h / img_height
                
                # YOLO format: class_id center_x center_y width height
                # class_id = 0 for single-class (room)
                f.write(f"0 {center_x:.6f} {center_y:.6f} {norm_w:.6f} {norm_h:.6f}\n")
                total_room_labels += 1
        
        # Copy image
        try:
            shutil.copy2(str(local_img_path), str(new_image_path))
            processed_images += 1
        except Exception as e:
            print(f"    Warning: Failed to copy {local_img_path}: {e}")
            skipped_images += 1
            # Remove label file if image copy failed
            if label_path.exists():
                label_path.unlink()
    
    print(f"  [OK] Processed: {processed_images} images")
    print(f"  [OK] Created: {total_room_labels} ROOM labels")
    print(f"  [WARN] Skipped: {skipped_images} images (not found locally)")
    
    return processed_images, total_room_labels


def process_split_images(split_img_ids, image_metadata, room_annotations, images_base, images_dir, labels_dir):
    """
    Process images for a given split
    """
    from tqdm import tqdm
    import shutil
    
    processed_count = 0
    label_count = 0
    
    for img_id in tqdm(split_img_ids, desc="  Processing"):
        if img_id not in image_metadata:
            continue
        
        img_info = image_metadata[img_id]
        img_width = img_info['width']
        img_height = img_info['height']
        img_filename = img_info['file_name']
        
        # Extract relative path from absolute Kaggle path
        # e.g., /kaggle/input/.../high_quality_architectural/6044/F1_original.png
        # -> high_quality_architectural/6044/F1_original.png
        if 'cubicasa5k/' in img_filename:
            # Split on last occurrence of 'cubicasa5k/' and take the part after
            relative_path = img_filename.split('cubicasa5k/')[-1]
        else:
            relative_path = img_filename
        
        # Find source image
        source_image_path = images_base / relative_path
        if not source_image_path.exists():
            continue
        
        # Create unique filename using image ID to avoid collisions
        # Many images have same names (F1_original.png) in different directories
        file_extension = source_image_path.suffix
        unique_filename = f"{img_id:06d}{file_extension}"  # e.g., 000123.png
        
        # Copy image with unique name
        dest_image_path = images_dir / unique_filename
        shutil.copy(source_image_path, dest_image_path)
        processed_count += 1
        
        # Create label file with same unique name
        label_filename = f"{img_id:06d}.txt"
        label_path = labels_dir / label_filename
        
        with open(label_path, 'w') as f:
            if img_id in room_annotations:
                for ann in room_annotations[img_id]:
                    if 'bbox' in ann:
                        x, y, w, h = ann['bbox']
                        
                        # Convert to YOLO format (center x, center y, width, height - normalized)
                        center_x = (x + w/2) / img_width
                        center_y = (y + h/2) / img_height
                        norm_w = w / img_width
                        norm_h = h / img_height
                        
                        # YOLO format: class_id center_x center_y width height
                        f.write(f"0 {center_x:.6f} {center_y:.6f} {norm_w:.6f} {norm_h:.6f}\n")
                        label_count += 1
    
    print(f"  [OK] Processed {processed_count} images, {label_count} labels")
    return processed_count, label_count


def create_data_yaml(output_dir):
    """Create data.yaml for YOLO training"""
    data_yaml_content = f"""# YOLO Data Configuration for Room Detection (ROOM CLASS ONLY)
# Generated from CubiCasa5K COCO dataset - filtering category_id=2 (room)
# PROPER 70/20/10 split with seed=42 (no image duplication!)

path: {output_dir}
train: train/images
val: val/images
test: test/images

# Classes
nc: 1
names: ['room']

# Task
task: detect
"""
    
    yaml_path = Path(output_dir) / "data.yaml"
    with open(yaml_path, 'w') as f:
        f.write(data_yaml_content)
    
    print(f"\n[OK] Created data.yaml at: {yaml_path}")
    return yaml_path


def main():
    parser = argparse.ArgumentParser(description='Rebuild YOLO dataset from CubiCasa5K COCO - ROOM ONLY')
    parser.add_argument('--coco-dir', default='../cubicasa5k_coco', 
                       help='Directory containing train_coco_pt.json, val_coco_pt.json, test_coco_pt.json')
    parser.add_argument('--images-base', default='../cubicasa5k/cubicasa5k', 
                       help='Base path to cubicasa5k images')
    parser.add_argument('--output-dir', default='./yolo_room_only', 
                       help='Output directory for YOLO dataset')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("REBUILDING YOLO DATASET - ROOM CLASS ONLY")
    print("Filtering for category_id=2 (room) from CubiCasa5K COCO")
    print("="*60)
    
    coco_dir = Path(args.coco_dir)
    output_dir = Path(args.output_dir)
    images_base = Path(args.images_base)
    
    # Check paths
    if not coco_dir.exists():
        raise FileNotFoundError(f"COCO directory not found: {coco_dir}")
    if not images_base.exists():
        raise FileNotFoundError(f"Images directory not found: {images_base}")
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    # Load ALL images from train COCO file (which contains most/all images)
    # Then re-split 70/20/10 ourselves for proper validation
    print("\n[INFO] Loading all images and creating 70/20/10 split...")
    print("[INFO] This ensures no image appears in multiple splits!")
    
    # Load train COCO (contains 4200 images)
    train_coco_file = coco_dir / "train_coco_pt.json"
    if not train_coco_file.exists():
        raise FileNotFoundError(f"Train COCO file not found: {train_coco_file}")
    
    with open(train_coco_file, 'r') as f:
        coco_data = json.load(f)
    
    # Filter for ROOM annotations only (category_id=2)
    print(f"[INFO] Filtering for category_id=2 (room)...")
    room_images = set()
    room_annotations = {}
    
    for ann in coco_data['annotations']:
        if ann['category_id'] == 2:  # room
            img_id = ann['image_id']
            room_images.add(img_id)
            if img_id not in room_annotations:
                room_annotations[img_id] = []
            room_annotations[img_id].append(ann)
    
    # Get image metadata
    image_metadata = {img['id']: img for img in coco_data['images'] if img['id'] in room_images}
    
    print(f"[INFO] Found {len(room_images)} images with room annotations")
    
    # Create 70/20/10 split with seed=42 for reproducibility
    import random
    random.seed(42)
    all_img_ids = sorted(list(room_images))
    random.shuffle(all_img_ids)
    
    total = len(all_img_ids)
    train_size = int(0.7 * total)
    val_size = int(0.2 * total)
    
    train_ids = set(all_img_ids[:train_size])
    val_ids = set(all_img_ids[train_size:train_size + val_size])
    test_ids = set(all_img_ids[train_size + val_size:])
    
    print(f"[INFO] Split: Train={len(train_ids)} ({len(train_ids)/total*100:.1f}%), Val={len(val_ids)} ({len(val_ids)/total*100:.1f}%), Test={len(test_ids)} ({len(test_ids)/total*100:.1f}%)")
    
    # Process each split
    total_images = 0
    total_labels = 0
    
    for split_name, split_ids in [('train', train_ids), ('val', val_ids), ('test', test_ids)]:
        # Create split directories
        split_images_dir = output_dir / split_name / "images"
        split_labels_dir = output_dir / split_name / "labels"
        split_images_dir.mkdir(parents=True, exist_ok=True)
        split_labels_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n[INFO] Processing {split_name} split ({len(split_ids)} images)...")
        
        processed, labels = process_split_images(
            split_ids,
            image_metadata,
            room_annotations,
            images_base,
            split_images_dir,
            split_labels_dir
        )
        total_images += processed
        total_labels += labels
    
    # Create data.yaml
    create_data_yaml(output_dir)
    
    print("\n" + "="*60)
    print("DATASET REBUILD COMPLETE!")
    print("="*60)
    print(f"Total images processed: {total_images}")
    print(f"Total ROOM labels created: {total_labels}")
    print(f"Output directory: {output_dir.absolute()}")
    print(f"\nNext steps:")
    print(f"1. Review the dataset: {output_dir}")
    print(f"2. Update your training script to use: {output_dir}/data.yaml")
    print(f"3. Retrain your model with the correct ROOM annotations!")
    print("="*60)


if __name__ == "__main__":
    main()

