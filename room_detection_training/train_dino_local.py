#!/usr/bin/env python3
"""
Local DINO Training Script for Testing
Tests DINO training with room detection dataset before SageMaker deployment
"""
import os
import sys
from pathlib import Path

# Add DINO to path
dino_dir = Path(__file__).parent / 'DINO'
sys.path.insert(0, str(dino_dir))

from main import main as dino_main
from main import get_args_parser

def main():
    print("=" * 60)
    print("Local DINO Training Test - Room Detection")
    print("=" * 60)
    
    # Setup paths
    coco_data_dir = Path(__file__).parent.parent / 'room_detection_dataset_coco'
    output_dir = Path(__file__).parent / 'local_training_output' / 'dino_test'
    config_file = dino_dir / 'config' / 'DINO' / 'DINO_4scale_room_detection.py'
    
    # Verify paths
    if not coco_data_dir.exists():
        print(f"[ERROR] COCO dataset not found: {coco_data_dir}")
        sys.exit(1)
    
    train_ann = coco_data_dir / 'train' / 'annotations.json'
    val_ann = coco_data_dir / 'val' / 'annotations.json'
    
    if not train_ann.exists() or not val_ann.exists():
        print(f"[ERROR] COCO annotations not found!")
        print(f"  Expected: {train_ann}")
        print(f"  Expected: {val_ann}")
        sys.exit(1)
    
    if not config_file.exists():
        print(f"[ERROR] Config file not found: {config_file}")
        print("Please create DINO_4scale_room_detection.py config first")
        sys.exit(1)
    
    print(f"[OK] COCO dataset: {coco_data_dir}")
    print(f"[OK] Config file: {config_file}")
    print(f"[OK] Output directory: {output_dir}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse arguments
    parser = get_args_parser()
    # Build command line args
    cmd_args = [
        '--config_file', str(config_file),
        '--coco_path', str(coco_data_dir),
        '--output_dir', str(output_dir),
        '--dataset_file', 'coco',
        '--options',
        'num_classes=1',
        'batch_size=2',  # Small batch for local GPU
        'epochs=5',  # Test with 5 epochs
        'lr=0.0001',
        'lr_backbone=1e-05',
        'lr_drop=4',  # Drop LR at epoch 4
        'weight_decay=0.0001',
        'clip_max_norm=0.1',
        'num_queries=300',
        'dn_labelbook_size=2',  # num_classes + 1
    ]
    args = parser.parse_args(cmd_args)
    
    # Single GPU for local test
    args.world_size = 1
    args.dist_url = 'env://'
    args.rank = 0
    args.local_rank = 0
    args.device = 'cuda' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu'
    args.num_workers = 4
    args.amp = False
    args.find_unused_params = False
    
    print("\n[INFO] Training configuration:")
    print(f"  Epochs: 5 (test run)")
    print(f"  Batch size: 2")
    print(f"  Num classes: 1 (room)")
    print(f"  Device: {args.device}")
    print(f"  Output: {args.output_dir}")
    print("\n[INFO] Starting training...\n")
    
    try:
        # Run DINO training
        dino_main(args)
        print("\n[OK] Local DINO training test completed successfully!")
        print(f"[INFO] Checkpoints saved to: {output_dir}")
        
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()

