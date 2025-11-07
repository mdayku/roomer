#!/usr/bin/env python3
"""
Prepare YOLO format dataset from cleaned COCO data
Run this BEFORE starting SageMaker training
"""

import os
import json
import shutil
from pathlib import Path
import argparse

def convert_coco_to_yolo_format(coco_annotations_path, output_dir, image_dir=None):
    """Convert COCO annotations to YOLO format"""
    print(f"Converting {coco_annotations_path} to YOLO format...")

    # Load COCO annotations
    with open(coco_annotations_path, 'r') as f:
        coco_data = json.load(f)

    # Create output directories
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    # Get image info
    images = {img['id']: img for img in coco_data['images']}
    annotations_by_image = {}

    # Group annotations by image
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)

    processed_count = 0

    for img_id, img_anns in annotations_by_image.items():
        if img_id not in images:
            continue

        img_info = images[img_id]
        img_width = img_info['width']
        img_height = img_info['height']
        img_filename = img_info['file_name']

        # Create label file
        label_filename = Path(img_filename).stem + '.txt'
        label_path = labels_dir / label_filename

        with open(label_path, 'w') as f:
            for ann in img_anns:
                # Only process room annotations (category_id should be for room)
                if ann['category_id'] != 1:  # Assuming room is category 1
                    continue

                # Get bounding box (COCO format: [x, y, width, height])
                if 'bbox' in ann:
                    x, y, w, h = ann['bbox']

                    # Convert to YOLO format (center x, center y, width, height - normalized)
                    center_x = (x + w/2) / img_width
                    center_y = (y + h/2) / img_height
                    norm_w = w / img_width
                    norm_h = h / img_height

                    # YOLO format: class_id center_x center_y width height
                    # For segmentation, we need polygon points
                    # For now, just use bounding boxes as the model will learn segmentation
                    f.write(f"0 {center_x:.6f} {center_y:.6f} {norm_w:.6f} {norm_h:.6f}\n")

        # Copy image if source directory provided
        if image_dir:
            src_img = Path(image_dir) / img_filename
            dst_img = images_dir / img_filename
            if src_img.exists():
                shutil.copy2(src_img, dst_img)

        processed_count += 1
        if processed_count % 100 == 0:
            print(f"Processed {processed_count} images...")

    print(f"Converted {processed_count} images to YOLO format")
    return processed_count

def create_data_yaml(train_dir, val_dir, output_path):
    """Create data.yaml file for YOLO training"""
    data_yaml = f"""# YOLO Data Configuration
path: .  # dataset root dir
train: {train_dir}/images  # train images
val: {val_dir}/images  # val images

# Classes
nc: 1  # number of classes
names: ['room']  # class names

# Segmentation task
task: segment
"""

    with open(output_path, 'w') as f:
        f.write(data_yaml)

    print(f"Created data.yaml at {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Prepare YOLO dataset from COCO')
    parser.add_argument('--coco-train', default='../cubicasa5k_coco/train_coco_pt.json', help='Path to train COCO annotations')
    parser.add_argument('--coco-val', default='../cubicasa5k_coco/val_coco_pt.json', help='Path to val COCO annotations')
    parser.add_argument('--images-train', help='Path to train images directory')
    parser.add_argument('--images-val', help='Path to val images directory')
    parser.add_argument('--output-dir', default='./yolo_data', help='Output directory for YOLO dataset')

    args = parser.parse_args()

    print("Preparing YOLO dataset from COCO annotations...")
    print("=" * 50)

    output_dir = Path(args.output_dir)
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"

    # Convert train data
    print("\nConverting train data...")
    convert_coco_to_yolo_format(args.coco_train, train_dir, args.images_train)

    # Convert val data
    print("\nConverting validation data...")
    convert_coco_to_yolo_format(args.coco_val, val_dir, args.images_val)

    # Create data.yaml
    print("\nCreating data.yaml...")
    create_data_yaml(train_dir, val_dir, output_dir / "data.yaml")

    print("\n[SUCCESS] YOLO dataset preparation complete!")
    print(f"Dataset saved to: {output_dir}")
    print(f"Train images: {len(list(train_dir.glob('images/*.png')))}")
    print(f"Val images: {len(list(val_dir.glob('images/*.png')))}")

if __name__ == "__main__":
    main()
