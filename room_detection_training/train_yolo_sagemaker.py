#!/usr/bin/env python3
"""
YOLO Training Script for SageMaker Container
This runs inside the SageMaker training environment
"""

import os
import sys
import traceback

def main():
    print("Room Detection YOLO Training in SageMaker")
    print("=" * 60)
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")

    try:
        # Install ultralytics with more robust installation
        print("\n1. Installing ultralytics...")
        import subprocess
        result = subprocess.run([
            sys.executable, "-m", "pip", "install",
            "ultralytics>=8.1.0", "torch", "torchvision", "torchaudio",
            "--quiet", "--no-cache-dir"
        ], capture_output=True, text=True, timeout=300)

        if result.returncode != 0:
            print(f"Installation failed: {result.stderr}")
            return

        print("Ultralytics installed successfully")

        # Import libraries
        print("\n2. Importing libraries...")
        import argparse
        from ultralytics import YOLO
        import torch
        print("Libraries imported successfully")

        # Check GPU
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

        # Parse arguments
        parser = argparse.ArgumentParser()
        parser.add_argument('--epochs', type=int, default=10)  # Start small for testing
        parser.add_argument('--batch-size', type=int, default=4)  # Smaller batch
        parser.add_argument('--imgsz', type=int, default=416)  # Smaller images
        parser.add_argument('--data', type=str, default='data.yaml')
        parser.add_argument('--weights', type=str, default='yolov8s-seg.pt')
        args = parser.parse_args()

        print(f"\n3. Training config: {args.epochs} epochs, batch {args.batch_size}, size {args.imgsz}")

        # Check data path
        print("\n4. Checking data paths...")
        data_path = '/opt/ml/input/data/training/data.yaml'
        if os.path.exists(data_path):
            args.data = data_path
            print(f"✓ Using SageMaker data: {data_path}")

            # Check if data.yaml content is valid
            with open(data_path, 'r') as f:
                content = f.read()
                print(f"Data.yaml content preview: {content[:200]}...")

        else:
            print("✗ ERROR: SageMaker data not found!")
            print("Available files in /opt/ml/input/:")
            for root, dirs, files in os.walk('/opt/ml/input'):
                for file in files[:10]:  # Show first 10 files
                    print(f"  {os.path.join(root, file)}")
            return

        # Load model
        print("\n5. Loading YOLO model...")
        model = YOLO(args.weights)
        print("Model loaded successfully")

        # Training with error handling
        print("\n6. Starting training...")
        results = model.train(
            data=args.data,
            epochs=args.epochs,
            batch=args.batch_size,
            imgsz=args.imgsz,
            device=device,
            workers=2,  # Minimal workers
            project='/opt/ml/model',
            name='room_detection',
            save=True,
            patience=5,  # Shorter patience
        )

        print("Training completed successfully!")

        # Save model
        print("\n7. Saving model...")
        model_path = '/opt/ml/model/final_model.pt'
        model.save(model_path)
        print(f"✓ Model saved to {model_path}")

        # Try validation
        print("\n8. Running validation...")
        try:
            metrics = model.val()
            print(f"✓ Validation mAP: {getattr(metrics.box, 'map', 'N/A')}")
        except Exception as val_e:
            print(f"⚠ Validation failed (non-critical): {val_e}")

        print("\n🎉 SageMaker training completed successfully!")

    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
        print("Full traceback:")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
