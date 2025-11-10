#!/usr/bin/env python3
"""Restructure COCO dataset to match DINO's expected format"""

from pathlib import Path
import json
import shutil

def restructure():
    src = Path('coco_room_dino')
    dst = Path('coco_room_dino_restructured')
    
    # Process each split
    for split in ['train', 'val']:
        # Load annotations
        ann_file = src / split / 'annotations.json'
        with open(ann_file, 'r') as f:
            coco_data = json.load(f)
        
        # Update image paths: images/000001.png -> 000001.png (relative to train2017/)
        for img in coco_data['images']:
            old_path = img['file_name']  # "images/000001.png"
            img['file_name'] = Path(old_path).name  # "000001.png"
        
        # Save to DINO's expected path
        output_name = f"instances_{split}2017.json"
        output_path = dst / 'annotations' / output_name
        with open(output_path, 'w') as f:
            json.dump(coco_data, f)
        
        print(f"[OK] {split}: {len(coco_data['images'])} images, {len(coco_data['annotations'])} annotations")
        print(f"     Saved to: {output_path}")

if __name__ == "__main__":
    restructure()

