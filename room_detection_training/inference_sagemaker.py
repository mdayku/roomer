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

def model_fn(model_dir):
    """
    Load the model for inference
    """
    print("Loading model from:", model_dir)
    print("Contents of model_dir:", os.listdir(model_dir) if os.path.exists(model_dir) else "Directory does not exist")

    try:
        # Try importing ultralytics first (might already be installed)
        try:
            from ultralytics import YOLO
            import torch
            print("ultralytics already installed")
        except ImportError:
            # Install ultralytics if not already installed
            print("Installing ultralytics...")
            import subprocess
            result = subprocess.run([
                sys.executable, "-m", "pip", "install",
                "ultralytics>=8.1.0", "--quiet", "--no-cache-dir"
            ], capture_output=True, text=True, timeout=300)

            if result.returncode != 0:
                print(f"Installation failed: {result.stderr}")
                raise Exception("Failed to install ultralytics")
            
            from ultralytics import YOLO
            import torch
            print("ultralytics installed successfully")

        # Check GPU
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")

        # Load model - try different possible model files
        model_files = ['best.pt', 'last.pt', 'yolov8s.pt', 'model.pt']
        model = None

        for model_file in model_files:
            model_path = os.path.join(model_dir, model_file)
            if os.path.exists(model_path):
                print(f"Loading model: {model_path}")
                model = YOLO(model_path)
                break

        if model is None:
            raise Exception(f"No model file found in {model_dir}")

        print("Model loaded successfully")
        return model

    except Exception as e:
        print(f"Error loading model: {e}")
        traceback.print_exc()
        raise

def input_fn(request_body, request_content_type):
    """
    Parse input data
    """
    print(f"Input content type: {request_content_type}")

    try:
        if request_content_type == 'application/json':
            input_data = json.loads(request_body)
            print("Received JSON input")
            return input_data
        else:
            raise ValueError(f"Unsupported content type: {request_content_type}")
    except Exception as e:
        print(f"Error parsing input: {e}")
        raise

def predict_fn(input_data, model):
    """
    Make prediction
    """
    print("Starting prediction...")

    try:
        # Extract image data
        if 'image' not in input_data:
            raise ValueError("No 'image' field in input data")

        image_data = input_data['image']

        # Handle different image formats
        if isinstance(image_data, str):
            # Assume base64 encoded image
            if image_data.startswith('data:image'):
                # Remove data URL prefix
                header, encoded = image_data.split(',', 1)
                image_bytes = base64.b64decode(encoded)
            else:
                image_bytes = base64.b64decode(image_data)
        elif isinstance(image_data, bytes):
            image_bytes = image_data
        else:
            raise ValueError("Unsupported image data format")

        print(f"Image size: {len(image_bytes)} bytes")

        # Save image to temporary file
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            tmp_file.write(image_bytes)
            tmp_path = tmp_file.name

        print(f"Saved image to: {tmp_path}")

        # Run inference
        results = model(tmp_path, conf=0.25, iou=0.45)

        # Clean up temp file
        os.unlink(tmp_path)

        # Process results
        predictions = []

        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get box coordinates (xyxy format)
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())

                    prediction = {
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'confidence': float(confidence),
                        'class_id': class_id,
                        'class_name': 'room'
                    }
                    predictions.append(prediction)

        print(f"Found {len(predictions)} predictions")

        return {
            'predictions': predictions,
            'image_width': results[0].orig_shape[1] if results else None,
            'image_height': results[0].orig_shape[0] if results else None
        }

    except Exception as e:
        print(f"Error during prediction: {e}")
        traceback.print_exc()
        raise

def output_fn(prediction, response_content_type):
    """
    Format output
    """
    print(f"Output content type: {response_content_type}")

    try:
        if response_content_type == 'application/json':
            return json.dumps(prediction, indent=2)
        else:
            return json.dumps(prediction)
    except Exception as e:
        print(f"Error formatting output: {e}")
        raise

# This script is designed to be used with SageMaker inference
# The functions above will be called by SageMaker automatically
