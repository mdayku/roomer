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

def convert_coco_to_yolo_format(coco_annotations_path, output_dir, images_dict=None):
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

        # Create label file with unique name matching the image
        unique_stem = f"{img_id:06d}_{Path(img_filename).stem}"
        label_filename = unique_stem + '.txt'
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

                    # YOLO detection format: class_id center_x center_y width height
                    # Note: For segmentation training, we should use polygons, but since
                    # the COCO data only has bounding boxes, we'll use those for now
                    f.write(f"0 {center_x:.6f} {center_y:.6f} {norm_w:.6f} {norm_h:.6f}\n")

        # Copy image if found in COCO-to-local mapping
        if images_dict and img_filename in images_dict:
            src_img = images_dict[img_filename]
            # Create unique filename using image ID to avoid collisions
            dst_filename = f"{img_id:06d}_{os.path.basename(img_filename)}"
            dst_img = images_dir / dst_filename

            # Debug: print first few copy operations
            if processed_count <= 5:
                print(f"Copying: {img_filename}")
                print(f"  From: {src_img}")
                print(f"  To: {dst_img}")

            try:
                shutil.copy2(src_img, dst_img)
                if processed_count <= 5:
                    print(f"  [SUCCESS] Copied {dst_filename}")
            except Exception as e:
                if processed_count <= 5:
                    print(f"  [ERROR] Failed to copy: {e}")
        elif processed_count <= 5:
            print(f"No mapping found for: {img_filename}")

        processed_count += 1
        if processed_count % 100 == 0:
            print(f"Processed {processed_count} images...")

    print(f"Converted {processed_count} images to YOLO format")
    return processed_count

def create_data_yaml(train_dir, val_dir, test_dir, output_path):
    """Create data.yaml file for YOLO training"""
    test_path = f"{test_dir}/images" if test_dir else f"{val_dir}/images"

    data_yaml = f"""# YOLO Data Configuration
path: .  # dataset root dir
train: {train_dir}/images  # train images
val: {val_dir}/images  # val images
test: {test_path}  # test images

# Classes
nc: 1  # number of classes
names: ['room']  # class names

# Detection task (using bounding boxes since segmentation data not available)
task: detect
"""

    with open(output_path, 'w') as f:
        f.write(data_yaml)

    print(f"Created data.yaml at {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Prepare YOLO dataset from COCO')
    parser.add_argument('--coco-train', default='../cubicasa5k_coco/train_coco_pt.json', help='Path to train COCO annotations')
    parser.add_argument('--coco-val', default='../cubicasa5k_coco/val_coco_pt.json', help='Path to val COCO annotations')
    parser.add_argument('--images-base', default='../cubicasa5k/cubicasa5k', help='Base path to cubicasa5k images')
    parser.add_argument('--output-dir', default='./yolo_data', help='Output directory for YOLO dataset')

    args = parser.parse_args()

    print("Preparing YOLO dataset from COCO annotations...")
    print("=" * 50)

    output_dir = Path(args.output_dir)
    images_base = Path(args.images_base)

    # Find all image files in the cubicasa5k directory
    print("Finding all image files...")
    print(f"Searching in: {images_base.absolute()}")
    all_images = {}  # full_local_path -> full_local_path (for existence check)
    png_count = 0
    for root, dirs, files in os.walk(str(images_base)):
        for file in files:
            if file.endswith('.png'):
                png_count += 1
                full_path = os.path.join(root, file)
                # Store the local path for quick existence checking
                all_images[full_path] = full_path

    print(f"Walked through directory tree, found {png_count} PNG files")

    # Build a mapping from COCO paths to local paths
    coco_to_local = {}
    kaggle_prefix = "/kaggle/input/cubicasa5k/cubicasa5k/cubicasa5k/"
    local_prefix = str(images_base)

    for local_path in all_images.keys():
        # Convert local path to COCO-style path for matching
        relative_path = os.path.relpath(local_path, local_prefix)
        coco_path = kaggle_prefix + relative_path.replace('\\', '/')
        coco_to_local[coco_path] = local_path

    print(f"Mapped {len(coco_to_local)} COCO paths to local paths")
    print("Sample mappings:")
    for i, (coco_path, local_path) in enumerate(list(coco_to_local.items())[:3]):
        print(f"  {coco_path} -> {local_path}")

    train_dir = output_dir / "train"
    val_dir = output_dir / "val"

    # Convert train data
    print("\nConverting train data...")
    convert_coco_to_yolo_format(args.coco_train, train_dir, coco_to_local)

    # Convert val data
    print("\nConverting validation data...")
    convert_coco_to_yolo_format(args.coco_val, val_dir, coco_to_local)

    # Convert test data if available
    test_coco_path = args.coco_val.replace('val_coco_pt.json', 'test_coco_pt.json')
    if os.path.exists(test_coco_path):
        test_dir = output_dir / "test"
        print("\nConverting test data...")
        convert_coco_to_yolo_format(test_coco_path, test_dir, coco_to_local)

    # Create data.yaml
    print("\nCreating data.yaml...")
    test_dir = output_dir / "test" if (output_dir / "test").exists() else None
    create_data_yaml(train_dir, val_dir, test_dir, output_dir / "data.yaml")

    test_count = len(list((output_dir / "test").glob('images/*.png'))) if (output_dir / "test").exists() else 0

    print("\n[SUCCESS] YOLO dataset preparation complete!")
    print(f"Dataset saved to: {output_dir}")
    print(f"Train images: {len(list(train_dir.glob('images/*.png')))}")
    print(f"Val images: {len(list(val_dir.glob('images/*.png')))}")
    if test_count > 0:
        print(f"Test images: {test_count}")

if __name__ == "__main__":
    main()
