#!/usr/bin/env python3
"""
Check COCO annotation structure
"""

import json

def main():
    with open('../cubicasa5k_coco/train_coco_pt.json', 'r') as f:
        data = json.load(f)

    print('Total images:', len(data['images']))
    print('Total annotations:', len(data['annotations']))

    # Check first few annotations
    for i, ann in enumerate(data['annotations'][:3]):
        print(f'Annotation {i}:')
        print(f'  ID: {ann["id"]}')
        print(f'  Image ID: {ann["image_id"]}')
        print(f'  Category: {ann["category_id"]}')
        print(f'  Has segmentation: {"segmentation" in ann}')
        if 'segmentation' in ann:
            seg = ann['segmentation']
            print(f'  Segmentation type: {type(seg)}')
            if isinstance(seg, list) and len(seg) > 0:
                if isinstance(seg[0], list):
                    print(f'  First polygon length: {len(seg[0])}')
                    print(f'  First 10 coords: {seg[0][:10]}')
                else:
                    print(f'  Segmentation length: {len(seg)}')
                    print(f'  First 10 coords: {seg[:10]}')
        print()

if __name__ == "__main__":
    main()
