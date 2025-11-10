#!/usr/bin/env python3
"""
Prepare CubiCasa5K COCO dataset for DINO training
- Keep both wall + room classes (category_id=1,2)
- Fix image paths to be relative
- Create train/val/test splits matching YOLO dataset
- Use same 70/20/10 split with seed=42 for consistency
"""
import json
from pathlib import Path
import random
import argparse

def prepare_dino_coco(coco_file, cubicasa_root, output_dir, single_class=False, yolo_dataset_dir=None):
    """Prepare COCO dataset for DINO - copies images with ID-based naming"""
    print("="*60)
    print("PREPARING COCO FOR DINO TRAINING (2-CLASS: WALL + ROOM)")
    print("="*60)
    
    coco_file = Path(coco_file)
    cubicasa_root = Path(cubicasa_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"\nLoading COCO data from: {coco_file}")
    with open(coco_file, 'r') as f:
        coco_data = json.load(f)
    
    print(f"  Images: {len(coco_data['images'])}")
    print(f"  Annotations: {len(coco_data['annotations'])}")
    print(f"  Categories: {len(coco_data['categories'])}")
    
    # Keep both categories (wall=1, room=2)
    wall_category_id = None
    room_category_id = None
    
    for cat in coco_data['categories']:
        if cat['name'] == 'wall':
            wall_category_id = cat['id']
        elif cat['name'] == 'room':
            room_category_id = cat['id']
    
    print(f"\n[OK] Found categories:")
    print(f"  Wall: ID={wall_category_id}")
    print(f"  Room: ID={room_category_id}")
    
    if single_class:
        # Filter for room annotations only
        print("\nFiltering for ROOM annotations only (1-class)...")
        target_categories = {room_category_id}
    else:
        # Keep both wall + room
        print("\nKeeping BOTH wall + room annotations (2-class)...")
        target_categories = {wall_category_id, room_category_id}
    
    relevant_image_ids = set()
    relevant_annotations = []
    
    for ann in coco_data['annotations']:
        if ann['category_id'] in target_categories:
            relevant_image_ids.add(ann['image_id'])
            relevant_annotations.append(ann)
    
    print(f"  Total annotations: {len(relevant_annotations)}")
    print(f"  Images with annotations: {len(relevant_image_ids)}")
    
    # Get images with relevant annotations
    relevant_images = []
    image_id_map = {}
    
    for img in coco_data['images']:
        if img['id'] in relevant_image_ids:
            relevant_images.append(img)
            image_id_map[img['id']] = img
    
    # Create 70/20/10 split (same as YOLO dataset)
    print("\nCreating 70/20/10 split (seed=42)...")
    random.seed(42)
    all_img_ids = sorted(list(relevant_image_ids))
    random.shuffle(all_img_ids)
    
    total = len(all_img_ids)
    train_size = int(0.7 * total)
    val_size = int(0.2 * total)
    
    train_ids = set(all_img_ids[:train_size])
    val_ids = set(all_img_ids[train_size:train_size + val_size])
    test_ids = set(all_img_ids[train_size + val_size:])
    
    print(f"  Train: {len(train_ids)}")
    print(f"  Val: {len(val_ids)}")
    print(f"  Test: {len(test_ids)}")
    
    # Setup image source (reuse from YOLO dataset if available)
    import shutil
    if yolo_dataset_dir:
        yolo_dataset_dir = Path(yolo_dataset_dir)
        print(f"\n[FAST MODE] Reusing images from YOLO dataset: {yolo_dataset_dir}")
    
    # Create split datasets
    for split_name, split_ids in [('train', train_ids), ('val', val_ids), ('test', test_ids)]:
        print(f"\n[{split_name.upper()}] Creating COCO annotations...")
        
        # Create output directories
        split_dir = output_dir / split_name
        split_dir.mkdir(exist_ok=True)
        images_dir = split_dir / 'images'
        images_dir.mkdir(exist_ok=True)
        
        split_images = []
        split_annotations = []
        images_copied = 0
        images_skipped = 0
        
        # Filter images and copy them
        for img_id in split_ids:
            img = image_id_map[img_id]
            
            # Use ID-based filename (same as YOLO datasets) - NO FUZZY MATCHING!
            new_filename = f"{img_id:06d}.png"
            dest_path = images_dir / new_filename
            
            # Copy image if not already present
            if not dest_path.exists():
                # Try to copy from YOLO dataset first (already has correct IDs)
                if yolo_dataset_dir:
                    yolo_source = yolo_dataset_dir / split_name / 'images' / new_filename
                    if yolo_source.exists():
                        shutil.copy(yolo_source, dest_path)
                        images_copied += 1
                    else:
                        print(f"  [WARN] Image not found in YOLO dataset: {new_filename}")
                        images_skipped += 1
                else:
                    # Fallback: try to find in cubicasa5k (slower)
                    original_path = Path(img['file_name'])
                    # Search for the image in cubicasa_root
                    found = False
                    for img_file in cubicasa_root.rglob(original_path.name):
                        shutil.copy(img_file, dest_path)
                        images_copied += 1
                        found = True
                        break
                    if not found:
                        images_skipped += 1
            
            # Update COCO annotation to reference ID-based filename
            split_images.append({
                **img,
                'file_name': f"images/{new_filename}"  # Relative path from split directory
            })
        
        # Filter annotations for this split
        for ann in relevant_annotations:
            if ann['image_id'] in split_ids:
                split_annotations.append(ann)
        
        # Create COCO structure for this split
        if single_class:
            categories = [
                {"id": 1, "name": "room", "supercategory": "none"}
            ]
            # Re-map room (ID 2) to ID 1
            category_map = {room_category_id: 1}
        else:
            categories = [
                {"id": 1, "name": "wall", "supercategory": "none"},
                {"id": 2, "name": "room", "supercategory": "none"}
            ]
            # Keep original IDs (wall=1, room=2)
            category_map = {wall_category_id: 1, room_category_id: 2}
        
        split_coco = {
            "images": split_images,
            "annotations": split_annotations,
            "categories": categories
        }
        
        # Re-map category IDs if needed
        for ann in split_coco['annotations']:
            ann['category_id'] = category_map[ann['category_id']]
        
        # Save annotations
        ann_file = split_dir / 'annotations.json'
        with open(ann_file, 'w') as f:
            json.dump(split_coco, f, indent=2)
        
        print(f"  [OK] {len(split_images)} images, {len(split_annotations)} annotations")
        print(f"       Images copied: {images_copied}, skipped: {images_skipped}")
        print(f"       Saved to: {ann_file}")
    
    # Create data.yaml for reference
    if single_class:
        classes_info = "nc: 1\nnames: ['room']"
    else:
        classes_info = "nc: 2\nnames: ['wall', 'room']"
    
    data_yaml = f"""# DINO COCO Dataset for Wall + Room Detection
# Converted from CubiCasa5K

path: {output_dir.absolute()}
train: train/annotations.json
val: val/annotations.json
test: test/annotations.json

# Image root (relative to annotations)
img_root: {cubicasa_root.absolute()}

# Classes
{classes_info}
"""
    
    yaml_path = output_dir / 'dataset_info.yaml'
    with open(yaml_path, 'w') as f:
        f.write(data_yaml)
    
    print(f"\n[OK] Created dataset info: {yaml_path}")
    
    print("\n" + "="*60)
    print("COCO DATASET READY FOR DINO!")
    print("="*60)
    print(f"Output: {output_dir.absolute()}")
    print(f"Images root: {cubicasa_root.absolute()}")
    print(f"\nNext: Configure DINO to use:")
    print(f"  --coco_path {output_dir.absolute()}")
    print(f"  --coco_img_root {cubicasa_root.absolute()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--coco-file', default='../cubicasa5k_coco/train_coco_pt.json')
    parser.add_argument('--cubicasa-root', default='../cubicasa5k')
    parser.add_argument('--output-dir', default='./coco_room_dino')
    parser.add_argument('--single-class', action='store_true', help='Use room only (default: wall + room)')
    parser.add_argument('--yolo-dataset', default='./yolo_room_wall_2class',
                       help='Reuse images from YOLO dataset (much faster!)')
    args = parser.parse_args()
    
    prepare_dino_coco(
        args.coco_file, 
        args.cubicasa_root, 
        args.output_dir, 
        args.single_class,
        yolo_dataset_dir=args.yolo_dataset
    )

