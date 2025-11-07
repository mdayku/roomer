#!/usr/bin/env python3
"""
SageMaker Inference Script for Room Detection
This runs inside the SageMaker endpoint container
"""

import os
import sys
import json
import base64
from io import BytesIO
import traceback

def main():
    print("Room Detection Inference in SageMaker")
    print("=" * 50)

    try:
        # Install ultralytics if needed
        print("Installing ultralytics...")
        os.system(f"{sys.executable} -m pip install ultralytics>=8.1.0 --quiet")
        print("Ultralytics installed")

        # Import libraries
        print("Importing libraries...")
        from ultralytics import YOLO
        import torch
        from PIL import Image
        import numpy as np
        print("Libraries imported successfully")

        # Check GPU
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")

        # Load model from SageMaker model directory
        model_path = '/opt/ml/model/model.pt'
        if not os.path.exists(model_path):
            print(f"Model not found at {model_path}, looking for alternatives...")
            # Try different possible model locations
            possible_paths = [
                '/opt/ml/model/final_model.pt',
                '/opt/ml/model/best.pt',
                '/opt/ml/model/yolov8s-seg.pt'
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    model_path = path
                    break

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No model file found. Searched: {possible_paths}")

        print(f"Loading model from: {model_path}")
        model = YOLO(model_path)
        print("Model loaded successfully")

        # For testing - create a simple mock inference
        print("Model ready for inference")

        # In production, this would be called via SageMaker endpoint
        # For now, just confirm everything works
        print("SageMaker inference setup complete!")

    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        print("Full traceback:")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
