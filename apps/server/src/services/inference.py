#!/usr/bin/env python3
"""
Python Inference Service for Room Detection
Called from Node.js backend
"""

import sys
import json
import base64
from io import BytesIO
import traceback
import os

def main():
    try:
        # Read input from stdin (JSON)
        input_data = json.loads(sys.stdin.read())

        if input_data.get('action') == 'detect':
            # Perform inference
            image_data = base64.b64decode(input_data['image'])
            result = perform_inference(image_data)
            print(json.dumps(result))
        else:
            print(json.dumps({'error': 'Unknown action'}))

    except Exception as e:
        print(json.dumps({
            'error': str(e),
            'traceback': traceback.format_exc()
        }))

def perform_inference(image_base64):
    """Perform room detection inference"""
    try:
        # Decode the base64 image
        image_data = base64.b64decode(image_base64)

        # Mock inference for now - in production this would:
        # 1. Load the trained YOLO model from S3/local path
        # 2. Preprocess the image (PIL Image from BytesIO)
        # 3. Run inference
        # 4. Convert results to GeoJSON

        # For now, return mock results based on image size
        # In production, we'd analyze the actual image
        image_size = len(image_data)
        print(f"Processing image of size: {image_size} bytes")

        # Vary results based on "image complexity" (mock)
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

        return {
            'type': 'FeatureCollection',
            'image': {'normalized': True},
            'features': features
        }

    except Exception as e:
        raise Exception(f"Inference failed: {e}")

if __name__ == "__main__":
    main()
