#!/usr/bin/env python3
"""
Convert YOLO dataset to COCO format for DINO training
"""
import json
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import argparse

def yolo_to_coco(yolo_dataset_dir, output_dir):
    """Convert YOLO format dataset to COCO format"""
    yolo_dir = Path(yolo_dataset_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("="*60)
    print("CONVERTING YOLO → COCO FOR DINO")
    print("="*60)
    
    for split in ['train', 'val', 'test']:
        print(f"\n[{split.upper()}] Converting...")
        
        images_dir = yolo_dir / split / 'images'
        labels_dir = yolo_dir / split / 'labels'
        
        if not images_dir.exists():
            print(f"  [SKIP] {split} images directory not found")
            continue
        
        # COCO format structure
        coco_data = {
            "images": [],
            "annotations": [],
            "categories": [
                {"id": 1, "name": "room", "supercategory": "none"}
            ]
        }
        
        image_files = sorted(images_dir.glob('*.png'))
        annotation_id = 1
        
        print(f"  Processing {len(image_files)} images...")
        
        for img_id, img_path in enumerate(tqdm(image_files, desc=f"  {split}"), start=1):
            # Get image info
            img = Image.open(img_path)
            width, height = img.size
            
            # Add image to COCO
            coco_data["images"].append({
                "id": img_id,
                "file_name": img_path.name,
                "width": width,
                "height": height
            })
            
            # Read YOLO label
            label_path = labels_dir / f"{img_path.stem}.txt"
            if not label_path.exists():
                continue
            
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    
                    class_id = int(parts[0])
                    x_center, y_center, bbox_width, bbox_height = map(float, parts[1:5])
                    
                    # Convert YOLO (normalized) to COCO (absolute pixels)
                    x_center_px = x_center * width
                    y_center_px = y_center * height
                    bbox_width_px = bbox_width * width
                    bbox_height_px = bbox_height * height
                    
                    # COCO uses top-left corner + width/height
                    x_min = x_center_px - (bbox_width_px / 2)
                    y_min = y_center_px - (bbox_height_px / 2)
                    
                    # Add annotation
                    coco_data["annotations"].append({
                        "id": annotation_id,
                        "image_id": img_id,
                        "category_id": class_id + 1,  # COCO uses 1-indexed
                        "bbox": [x_min, y_min, bbox_width_px, bbox_height_px],
                        "area": bbox_width_px * bbox_height_px,
                        "iscrowd": 0
                    })
                    annotation_id += 1
        
        # Save COCO annotations
        split_output_dir = output_dir / split
        split_output_dir.mkdir(exist_ok=True)
        
        ann_file = split_output_dir / 'annotations.json'
        with open(ann_file, 'w') as f:
            json.dump(coco_data, f, indent=2)
        
        print(f"  [OK] {split}: {len(coco_data['images'])} images, {len(coco_data['annotations'])} annotations")
        print(f"       Saved to: {ann_file}")
    
    print("\n" + "="*60)
    print("CONVERSION COMPLETE!")
    print("="*60)
    print(f"Output directory: {output_dir.absolute()}")
    print("\nNext steps:")
    print("1. Train DINO locally: cd DINO && python main.py ...")
    print("2. Or train on SageMaker: python sagemaker_train_dino.py")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-dir', default='./yolo_room_only', help='YOLO dataset directory')
    parser.add_argument('--output-dir', default='./coco_room_detection', help='Output COCO directory')
    args = parser.parse_args()
    
    yolo_to_coco(args.yolo_dir, args.output_dir)

