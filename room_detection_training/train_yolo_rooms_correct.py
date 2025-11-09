#!/usr/bin/env python3
"""
YOLO Training - CORRECTED for ROOM CLASS
Now training on actual ROOM annotations (category_id=2) instead of walls!
"""

import torch
from ultralytics import YOLO
from pathlib import Path
import sys

def main():
    print("\n" + "="*60)
    print("YOLO ROOM DETECTION TRAINING - CORRECTED DATASET")
    print("Training on ROOM class (not walls!)")
    print("="*60)
    
    # Check CUDA
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("WARNING: No GPU detected. Training will be very slow.")
    
    # Data configuration
    data_yaml = Path('./yolo_room_only/data.yaml')
    if not data_yaml.exists():
        print(f"\nERROR: data.yaml not found at {data_yaml.absolute()}")
        print("Please run rebuild_yolo_dataset.py first!")
        sys.exit(1)
    
    print(f"\nData config: {data_yaml.absolute()}")
    
    # Model selection
    print("\nSelect YOLO model:")
    print("  1. YOLOv8n (nano - fastest, least accurate)")
    print("  2. YOLOv8s (small - balanced)")
    print("  3. YOLOv8m (medium)")
    print("  4. YOLOv8l (large - slower, more accurate)")
    print("  5. YOLOv8x (extra large - best accuracy, slowest)")
    
    model_choice = input("Enter choice (1-5, default 2): ").strip() or "2"
    model_map = {
        "1": "yolov8n.pt",
        "2": "yolov8s.pt",
        "3": "yolov8m.pt",
        "4": "yolov8l.pt",
        "5": "yolov8x.pt",
    }
    model_name = model_map.get(model_choice, "yolov8s.pt")
    
    print(f"\nLoading {model_name}...")
    model = YOLO(model_name)
    
    # Training configuration
    print("\nTraining Configuration:")
    epochs = int(input("  Epochs (default 50): ").strip() or "50")
    batch_size = int(input(f"  Batch size (default 16): ").strip() or "16")
    img_size = int(input("  Image size (default 640): ").strip() or "640")
    
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    print(f"Model: {model_name}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Image size: {img_size}")
    print(f"Device: {device}")
    print("="*60 + "\n")
    
    # Train
    try:
        results = model.train(
            data=str(data_yaml),
            epochs=epochs,
            batch=batch_size,
            imgsz=img_size,
            device=device,
            workers=4 if device == 'cuda' else 2,
            project='./local_training_output',
            name='room_detection_corrected',
            save=True,
            save_period=10,  # Save checkpoint every 10 epochs
            patience=15,  # Early stopping patience
            verbose=True,
            
            # Optimizer settings
            optimizer='AdamW',
            lr0=0.01,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3.0,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            
            # Loss weights
            box=7.5,
            cls=0.5,
            dfl=1.5,
            
            # Augmentation (YOLO handles this automatically)
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=0.0,
            translate=0.1,
            scale=0.5,
            shear=0.0,
            perspective=0.0,
            flipud=0.0,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.0,
            copy_paste=0.0,
        )
        
        print("\n" + "="*60)
        print("TRAINING COMPLETED!")
        print("="*60)
        
        # Run validation
        print("\nRunning validation on test set...")
        metrics = model.val()
        
        print(f"\nValidation Results:")
        print(f"  mAP50: {metrics.box.map50:.4f}")
        print(f"  mAP50-95: {metrics.box.map:.4f}")
        print(f"  Precision: {metrics.box.mp:.4f}")
        print(f"  Recall: {metrics.box.mr:.4f}")
        
        # Save final model info
        output_dir = Path('./local_training_output/room_detection_corrected')
        print(f"\nModel saved to: {output_dir.absolute()}")
        print(f"Best weights: {output_dir / 'weights' / 'best.pt'}")
        print(f"Last weights: {output_dir / 'weights' / 'last.pt'}")
        
        print("\n" + "="*60)
        print("SUCCESS! Your model is now trained on ROOM annotations!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

