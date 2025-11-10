"""Remap COCO category IDs from [1,2] to [0,1] for DINO training"""
import json
import shutil
from pathlib import Path

def remap_coco_file(input_file, output_file):
    """Remap category_ids: 1->0, 2->1"""
    print(f"Processing: {input_file}")
    
    # Load COCO data
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Remap categories
    for cat in data['categories']:
        old_id = cat['id']
        cat['id'] = old_id - 1  # 1->0, 2->1
        print(f"  Category '{cat['name']}': {old_id} -> {cat['id']}")
    
    # Remap annotations
    remapped_count = 0
    for ann in data['annotations']:
        old_id = ann['category_id']
        ann['category_id'] = old_id - 1  # 1->0, 2->1
        remapped_count += 1
    
    print(f"  Remapped {remapped_count} annotations")
    
    # Save remapped data
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(data, f)
    
    print(f"  Saved to: {output_file}")

def main():
    base_dir = Path('coco_room_dino_restructured')
    
    # Backup original if not already backed up
    backup_dir = Path('coco_room_dino_restructured_BACKUP')
    if not backup_dir.exists():
        print(f"Creating backup: {backup_dir}")
        shutil.copytree(base_dir, backup_dir)
        print("[OK] Backup created")
    else:
        print(f"[INFO] Backup already exists: {backup_dir}")
    
    # Remap train and val annotations
    for split in ['train', 'val']:
        ann_file = base_dir / 'annotations' / f'instances_{split}2017.json'
        if ann_file.exists():
            # Remap in place
            remap_coco_file(ann_file, ann_file)
        else:
            print(f"[WARNING] File not found: {ann_file}")
    
    print("\n[OK] Category remapping complete!")
    print("Categories now: [0=wall, 1=room]")
    print("Ready for DINO training!")

if __name__ == '__main__':
    main()

