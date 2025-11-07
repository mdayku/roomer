#!/usr/bin/env python3
"""
Local YOLO Training for Debugging
Run this locally to debug issues before deploying to SageMaker
"""

import os
import sys
import traceback
from pathlib import Path

def main():
    print("Room Detection YOLO Training (Local)")
    print("=" * 60)
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")

    try:
        # Install ultralytics locally
        print("\n1. Installing ultralytics...")
        import subprocess
        result = subprocess.run([
            sys.executable, "-m", "pip", "install",
            "ultralytics>=8.1.0", "--quiet", "--no-cache-dir"
        ], capture_output=True, text=True, timeout=300)

        if result.returncode != 0:
            print(f"Installation failed: {result.stderr}")
            return

        print("Ultralytics installed successfully")

        # Import libraries
        print("\n2. Importing libraries...")
        from ultralytics import YOLO
        import torch
        print("Libraries imported successfully")

        # Check GPU and force usage if available
        if torch.cuda.is_available():
            device = 0  # Use GPU device 0
            print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            print(f"CUDA Version: {torch.version.cuda}")
        else:
            device = 'cpu'
            print("⚠️ No GPU available, using CPU (this will be slow)")

        # Check data
        print("\n3. Checking data paths...")
        data_yaml = Path("./yolo_data/data.yaml")
        if not data_yaml.exists():
            print(f"❌ ERROR: data.yaml not found at {data_yaml.absolute()}")
            return

        with open(data_yaml, 'r') as f:
            content = f.read()
            print(f"Data.yaml content:\n{content}")

        # Verify dataset files exist
        train_images = len(list(Path("./yolo_data/train/images").glob("*.png")))
        val_images = len(list(Path("./yolo_data/val/images").glob("*.png")))
        train_labels = len(list(Path("./yolo_data/train/labels").glob("*.txt")))
        val_labels = len(list(Path("./yolo_data/val/labels").glob("*.txt")))

        print(f"Dataset stats: Train={train_images} images/{train_labels} labels, Val={val_images} images/{val_labels} labels")

        if train_images == 0 or val_images == 0:
            print("❌ ERROR: No images found in dataset!")
            return

        # Load model
        print("\n4. Loading YOLO model...")
        weights_path = Path("./yolo_data/yolov8s.pt")
        if not weights_path.exists():
            print(f"Downloading YOLO weights to {weights_path}...")
            import urllib.request
            weights_url = "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8s.pt"
            urllib.request.urlretrieve(weights_url, str(weights_path))
            print("Weights downloaded")

        model = YOLO(str(weights_path))
        print("Model loaded successfully")

        # Training with proper config for detection model
        print("\n5. Starting training...")
        results = model.train(
            data=str(data_yaml),
            epochs=20,  # Proper training epochs
            batch=8,    # Reasonable batch size for GPU
            imgsz=640,  # Standard YOLO size
            device=device,
            workers=2,   # Use multiprocessing
            project='./local_training_output',
            name='room_detection_v1',
            save=True,
            verbose=True,
            patience=5,  # Early stopping
            lr0=0.01,   # Learning rate
            optimizer='AdamW'  # Better optimizer
        )

        print("Training completed successfully!")

        # Try validation
        print("\n6. Running validation...")
        try:
            metrics = model.val()
            print(f"✓ Validation completed")
        except Exception as val_e:
            print(f"⚠ Validation failed (non-critical): {val_e}")

        print("\n🎉 Local training test completed successfully!")

    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
        print("Full traceback:")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
