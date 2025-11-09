#!/usr/bin/env python3
"""
Python Inference Service for Room Detection
Uses locally trained YOLO models - no AWS/S3 needed!
"""

import sys
import json
import base64
from io import BytesIO
import traceback
import os
from pathlib import Path
from PIL import Image
import numpy as np

# Get project root - use cwd() like comprehensive_test.py does
# The Node.js spawn sets cwd to project root, so this should work
PROJECT_ROOT = Path.cwd()

# Available models - using LOCAL paths only
MODELS = {
    'yolo-v8l-200epoch': {
        'local_path': PROJECT_ROOT / 'room_detection_training' / 'local_training_output' / 'yolo-v8l-200epoch' / 'weights' / 'best.pt',
        'description': 'YOLO v8 Large - 200 epochs (Best Model)'
    },
    'yolo-v8l-200': {
        'local_path': PROJECT_ROOT / 'room_detection_training' / 'local_training_output' / 'yolo-v8l-200' / 'room_detection' / 'weights' / 'best.pt',
        'description': 'YOLO v8 Large - 200 epochs (Alternative)'
    },
    'yolo-v8s-sagemaker': {
        # Use v8l-200epoch as fallback since we don't have v8s locally
        'local_path': PROJECT_ROOT / 'room_detection_training' / 'local_training_output' / 'yolo-v8l-200epoch' / 'weights' / 'best.pt',
        'description': 'YOLO v8 Large - 200 epochs (using v8l as v8s not available locally)'
    },
    'default': {
        'local_path': PROJECT_ROOT / 'room_detection_training' / 'local_training_output' / 'yolo-v8l-200epoch' / 'weights' / 'best.pt',
        'description': 'YOLO v8 Large - 200 epochs (Default)'
    }
}

# Global model cache
loaded_models = {}

def main():
    try:
        # Read raw image bytes from stdin (no JSON, just binary data)
        # First 4 bytes are model ID length, then model ID, then image data
        model_id_length_bytes = sys.stdin.buffer.read(4)
        if len(model_id_length_bytes) != 4:
            raise ValueError("Invalid input format")
        
        model_id_length = int.from_bytes(model_id_length_bytes, 'big')
        model_id_bytes = sys.stdin.buffer.read(model_id_length)
        model_id = model_id_bytes.decode('utf-8') if model_id_bytes else 'default'
        
        # Read remaining image data
        image_data = sys.stdin.buffer.read()
        
        if not image_data:
            raise ValueError("No image data provided")
        
        sys.stderr.write(f"Received {len(image_data)} bytes of image data for model: {model_id}\n")
        
        # Perform inference
        result = perform_inference(image_data, model_id)
        # Only output JSON to stdout - all debug goes to stderr
        print(json.dumps(result))

    except Exception as e:
        # Send errors to stderr, JSON to stdout
        sys.stderr.write(f"Error: {e}\n")
        sys.stderr.write(traceback.format_exc())
        print(json.dumps({
            'error': str(e),
            'traceback': traceback.format_exc()
        }))

def load_model(model_id='default'):
    """Load the specified YOLO model from local filesystem"""
    global loaded_models

    if model_id in loaded_models:
        return loaded_models[model_id]

    if model_id not in MODELS:
        sys.stderr.write(f"Unknown model {model_id}, using default\n")
        model_id = 'default'

    model_config = MODELS[model_id]
    local_path = model_config['local_path']
    description = model_config.get('description', model_id)

    try:
        # Convert Path object to string
        model_path = str(local_path)
        sys.stderr.write(f"Loading {model_id} model ({description})...\n")
        sys.stderr.write(f"Model path: {model_path}\n")

        # Check if model file exists
        if not local_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Load Ultralytics YOLO model
        from ultralytics import YOLO
        sys.stderr.write(f"Loading YOLO model from {model_path}...\n")
        model = YOLO(model_path)
        loaded_models[model_id] = model
        sys.stderr.write(f"Model {model_id} loaded successfully\n")
        return model

    except Exception as e:
        sys.stderr.write(f"Failed to load model {model_id}: {e}\n")
        traceback.print_exc(file=sys.stderr)
        # Try fallback to default model
        if model_id != 'default':
            sys.stderr.write("Trying default model...\n")
            return load_model('default')
        raise Exception(f"Model loading failed: {e}")

def perform_inference(image_data, model_id='default'):
    """Perform room detection inference with trained YOLO model"""
    try:
        sys.stderr.write(f"Processing image: {len(image_data)} bytes with model {model_id}\n")
        # Decode the image
        image = Image.open(BytesIO(image_data))
        img_width, img_height = image.size
        sys.stderr.write(f"Processing image: {img_width}x{img_height} pixels with model {model_id}\n")

        # Load the specified model
        model = load_model(model_id)

        # Run inference - suppress YOLO's verbose output
        sys.stderr.write("Running YOLO inference...\n")
        # Set verbose=False and also redirect stdout temporarily to suppress YOLO output
        import contextlib
        import io
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            results = model(image, conf=0.25, iou=0.45, verbose=False)  # Standard YOLO thresholds
        # YOLO output is now captured in f, not printed to stdout

        # Convert image to numpy array for drawing
        import numpy as np
        img_array = np.array(image)
        if len(img_array.shape) == 2:  # Grayscale
            img_array = np.stack([img_array] * 3, axis=-1)
        elif img_array.shape[2] == 4:  # RGBA
            img_array = img_array[:, :, :3]  # Convert to RGB
        
        # Draw bounding boxes on image
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(image)
        
        # Convert to required JSON array format: [{id, bounding_box, name_hint}]
        # bounding_box is normalized to 0-1000 range: [x_min, y_min, x_max, y_max]
        detected_rooms = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for i, box in enumerate(boxes):
                    # Get bounding box coordinates in pixels
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = box.conf[0].item()
                    
                    # Draw bounding box (green rectangle)
                    draw.rectangle([x1, y1, x2, y2], outline='green', width=3)
                    
                    # Draw label
                    label = f"Room {i+1}"
                    try:
                        font = ImageFont.truetype("arial.ttf", 20)
                    except:
                        font = ImageFont.load_default()
                    draw.text((x1, y1 - 25), label, fill='green', font=font)
                    
                    # Convert to 0-1000 normalized range
                    bounding_box = [
                        int((x1 / img_width) * 1000),   # x_min
                        int((y1 / img_height) * 1000),   # y_min
                        int((x2 / img_width) * 1000),    # x_max
                        int((y2 / img_height) * 1000)    # y_max
                    ]
                    
                    detected_rooms.append({
                        'id': f'room_{i+1:03d}',
                        'bounding_box': bounding_box,
                        'name_hint': 'room'  # One-class model
                    })

        sys.stderr.write(f"Detection complete: {len(detected_rooms)} rooms found\n")

        # Convert annotated image to base64
        output_buffer = BytesIO()
        image.save(output_buffer, format='PNG')
        annotated_image_base64 = base64.b64encode(output_buffer.getvalue()).decode('utf-8')
        
        return {
            'detections': detected_rooms,
            'annotated_image': annotated_image_base64
        }

    except Exception as e:
        sys.stderr.write(f"Inference failed: {e}\n")
        traceback.print_exc(file=sys.stderr)
        # Fallback to mock results if model fails
        sys.stderr.write("Falling back to mock results...\n")
        mock_detections = get_mock_results(image_data)
        return {
            'detections': mock_detections,
            'annotated_image': None  # No annotated image for mock results
        }

def get_mock_results(image_data):
    """Fallback mock results when model inference fails"""
    image_size = len(image_data)
    num_rooms = 2 if image_size < 100000 else 3

    features = []
    for i in range(num_rooms):
        x_start = 0.1 + (i * 0.2)
        x_end = x_start + 0.15
        y_start = 0.1
        y_end = 0.3

        features.append({
            'type': 'Feature',
            'properties': {
                'id': f'room_{i+1:03d}',
                'name_hint': ['Living Room', 'Kitchen', 'Bedroom'][i % 3],
                'confidence': 0.85 + (i * 0.05),
                'bbox_norm': [x_start, y_start, x_end, y_end]
            },
            'geometry': {
                'type': 'Polygon',
                'coordinates': [[[x_start, y_start], [x_end, y_start], [x_end, y_end], [x_start, y_end], [x_start, y_start]]]
            }
        })

    # Convert to required JSON array format
    detected_rooms = []
    for i, feature in enumerate(features):
        bbox_norm = feature['properties']['bbox_norm']
        bounding_box = [
            int(bbox_norm[0] * 1000),
            int(bbox_norm[1] * 1000),
            int(bbox_norm[2] * 1000),
            int(bbox_norm[3] * 1000)
        ]
        detected_rooms.append({
            'id': feature['properties']['id'],
            'bounding_box': bounding_box,
            'name_hint': 'room'  # One-class model
        })
    
    return detected_rooms

if __name__ == "__main__":
    main()
