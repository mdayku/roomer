#!/usr/bin/env python3
"""
Build YOLO-SEG Dataset from CubiCasa5K GeoJSON - ROOM POLYGONS
Directly converts 4200 GeoJSON files to YOLO segmentation format
70/20/10 train/val/test split (seed=42 for reproducibility)
"""

import json
import os
import shutil
from pathlib import Path
from PIL import Image
import argparse
from tqdm import tqdm
import random

def load_geojson_file(geojson_path):
    """
    Load a GeoJSON file and extract room polygons
    Returns: List of normalized polygons (each polygon is a list of [x,y] pairs)
    """
    with open(geojson_path, 'r') as f:
        data = json.load(f)
    
    polygons = []
    for feature in data.get('features', []):
        if feature['geometry']['type'] == 'Polygon':
            # Get the outer ring (first element of coordinates)
            ring = feature['geometry']['coordinates'][0]
            # ring is already in normalized [0,1] coordinates
            polygons.append(ring)
    
    return polygons, data.get('image', {})

def convert_polygon_to_yolo_seg(polygon):
    """
    Convert a GeoJSON polygon to YOLO segmentation format
    Input: List of [x,y] pairs in normalized [0,1] coordinates
    Output: String in YOLO format: "0 x1 y1 x2 y2 x3 y3 ..."
    """
    if len(polygon) < 3:
        return None
    
    # YOLO seg format: class_id x1 y1 x2 y2 ...
    # All coordinates should already be normalized [0,1]
    coords = []
    for x, y in polygon:
        # Clamp to [0,1] just in case
        x = max(0.0, min(1.0, x))
        y = max(0.0, min(1.0, y))
        coords.append(f"{x:.6f}")
        coords.append(f"{y:.6f}")
    
    return "0 " + " ".join(coords)

def process_split(geojson_files, split_name, output_dir, images_base_path, reuse_images_from=None):
    """
    Process a single split (train/val/test)
    """
    print(f"\n  Processing {split_name}...")
    
    images_dir = output_dir / split_name / "images"
    labels_dir = output_dir / split_name / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    image_count = 0
    label_count = 0
    skipped_count = 0
    
    # Check if we're reusing images
    reuse_images = False
    if reuse_images_from and Path(reuse_images_from).exists():
        reuse_images = True
        reuse_images_dir = Path(reuse_images_from) / split_name / "images"
        print(f"  [FAST MODE] Reusing images from: {reuse_images_dir}")
    
    for geojson_path in tqdm(geojson_files, desc=f"  Processing {split_name}"):
        # Extract image filename from GeoJSON filename
        # Format: {baseName}_{imageId}.geo.json -> {baseName}_{imageId}.png
        geojson_stem = Path(geojson_path).stem  # removes .json
        if geojson_stem.endswith('.geo'):
            geojson_stem = geojson_stem[:-4]  # remove .geo
        
        image_filename = geojson_stem + '.png'
        
        # Load polygons from GeoJSON
        try:
            polygons, img_meta = load_geojson_file(geojson_path)
        except Exception as e:
            print(f"    [WARN] Failed to load {geojson_path}: {e}")
            skipped_count += 1
            continue
        
        if not polygons:
            skipped_count += 1
            continue
        
        # Find source image
        # cubicasa5k structure: cubicasa5k/XX/Fxx_original.png
        # We need to locate the matching image
        image_id_parts = geojson_stem.split('_')
        if len(image_id_parts) >= 2:
            # Try to construct the path: {baseName}_F{floor}_original
            possible_paths = [
                images_base_path / geojson_stem.replace('_', '/') / f"{image_id_parts[-1]}_original.png",
                images_base_path / f"{image_id_parts[0]}" / f"{image_id_parts[1]}_original.png",
            ]
            
            # Also check the direct filename mapping from GeoJSON
            # The convert script used img.file_name from COCO
            source_image_path = None
            
            # Try direct search in cubicasa5k
            for search_path in [images_base_path]:
                for img_path in Path(search_path).rglob(f"*{image_id_parts[0]}*/*{image_id_parts[1]}*original.png"):
                    source_image_path = img_path
                    break
                if source_image_path:
                    break
            
            if not source_image_path:
                # Try reusing from existing dataset
                if reuse_images:
                    reuse_src = reuse_images_dir / image_filename
                    if reuse_src.exists():
                        source_image_path = reuse_src
                    else:
                        # Try alternate extension
                        for ext in ['.png', '.jpg', '.jpeg']:
                            alt_src = reuse_images_dir / (geojson_stem + ext)
                            if alt_src.exists():
                                source_image_path = alt_src
                                image_filename = alt_src.name
                                break
            
            if not source_image_path or not Path(source_image_path).exists():
                skipped_count += 1
                continue
            
            # Copy image
            dest_image_path = images_dir / image_filename
            if not dest_image_path.exists():
                shutil.copy(source_image_path, dest_image_path)
            image_count += 1
            
            # Create YOLO label file
            label_filename = Path(image_filename).stem + '.txt'
            label_path = labels_dir / label_filename
            
            with open(label_path, 'w') as f:
                for polygon in polygons:
                    yolo_line = convert_polygon_to_yolo_seg(polygon)
                    if yolo_line:
                        f.write(yolo_line + '\n')
                        label_count += 1
    
    print(f"  [OK] {split_name}: {image_count} images, {label_count} polygon labels")
    if skipped_count > 0:
        print(f"  [WARN] {split_name}: Skipped {skipped_count} files (not found or no polygons)")
    
    return image_count, label_count

def create_data_yaml(output_dir):
    """
    Create data.yaml for YOLO segmentation training
    """
    yaml_path = output_dir / 'data.yaml'
    
    yaml_content = f"""# YOLO Segmentation Dataset - CubiCasa5K Room Polygons
# Generated from GeoJSON room polygons
# 1 class: room (with full polygon masks)

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
    return yaml_path

def main():
    parser = argparse.ArgumentParser(description='Build YOLO-SEG dataset from GeoJSON files')
    parser.add_argument('--geojson-dir', default='../cubicasa5k_geojson',
                       help='Directory containing GeoJSON files')
    parser.add_argument('--images-base', default='../cubicasa5k/cubicasa5k',
                       help='Base path to cubicasa5k images')
    parser.add_argument('--output-dir', default='./yolo_room_seg',
                       help='Output directory for YOLO segmentation dataset')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducible splits')
    parser.add_argument('--reuse-images-from', default='./yolo_room_only',
                       help='Reuse images from existing dataset (much faster)')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("BUILDING YOLO SEGMENTATION DATASET FROM GEOJSON")
    print("Source: cubicasa5k_geojson (4200 GeoJSON files)")
    print("Format: YOLO-seg with full polygon masks")
    print("Split: 70% train / 20% val / 10% test")
    print("="*60)
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    geojson_dir = Path(args.geojson_dir)
    images_base_path = Path(args.images_base)
    output_dir = Path(args.output_dir)
    
    # Check paths
    if not geojson_dir.exists():
        raise FileNotFoundError(f"GeoJSON directory not found: {geojson_dir}")
    if not images_base_path.exists():
        raise FileNotFoundError(f"Images base path not found: {images_base_path}")
    
    # Find all GeoJSON files
    geojson_files = sorted(geojson_dir.glob('*.json'))
    if not geojson_files:
        raise FileNotFoundError(f"No GeoJSON files found in: {geojson_dir}")
    
    print(f"\nFound {len(geojson_files)} GeoJSON files")
    
    # Create 70/20/10 split
    random.shuffle(geojson_files)
    total = len(geojson_files)
    train_size = int(0.7 * total)
    val_size = int(0.2 * total)
    
    train_files = geojson_files[:train_size]
    val_files = geojson_files[train_size:train_size + val_size]
    test_files = geojson_files[train_size + val_size:]
    
    print(f"\n{'='*60}")
    print("Creating 70/20/10 train/val/test split")
    print(f"{'='*60}")
    print(f"  Total files: {total}")
    print(f"  Train: {len(train_files)} ({len(train_files)/total*100:.1f}%)")
    print(f"  Val: {len(val_files)} ({len(val_files)/total*100:.1f}%)")
    print(f"  Test: {len(test_files)} ({len(test_files)/total*100:.1f}%)")
    
    # Check if we're reusing images
    if args.reuse_images_from and Path(args.reuse_images_from).exists():
        print(f"\n[FAST MODE] Reusing images from: {args.reuse_images_from}")
        print("Only generating new label files (much faster!)")
    
    # Process each split
    train_imgs, train_labels = process_split(train_files, 'train', output_dir, images_base_path, args.reuse_images_from)
    val_imgs, val_labels = process_split(val_files, 'val', output_dir, images_base_path, args.reuse_images_from)
    test_imgs, test_labels = process_split(test_files, 'test', output_dir, images_base_path, args.reuse_images_from)
    
    # Create data.yaml
    create_data_yaml(output_dir)
    
    # Summary
    print(f"\n{'='*60}")
    print("YOLO SEGMENTATION DATASET BUILD COMPLETE!")
    print(f"{'='*60}")
    print(f"Total images processed: {train_imgs + val_imgs + test_imgs}")
    print(f"Total ROOM polygon labels: {train_labels + val_labels + test_labels}")
    print(f"Output directory: {output_dir.absolute()}")
    print(f"\nSplit summary:")
    print(f"  Train: {train_imgs} images, {train_labels} labels")
    print(f"  Val: {val_imgs} images, {val_labels} labels")
    print(f"  Test: {test_imgs} images, {test_labels} labels")
    print(f"\nThis dataset uses FULL POLYGON segmentation, not just bboxes!")
    print(f"Ready for YOLO-seg training (yolov8-seg models)")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    main()

