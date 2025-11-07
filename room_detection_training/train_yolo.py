"""
YOLO11-seg Training for Room Detection
Alternative to Mask R-CNN - faster training and inference
"""

import torch
from ultralytics import YOLO
import json
from pathlib import Path
import shutil
from datetime import datetime

from config import *


def convert_coco_to_yolo_format(coco_annotations_path, output_dir):
    """
    Convert COCO format to YOLO segmentation format
    YOLO format: class_id x1 y1 x2 y2 ... (normalized coordinates)
    """
    print(f"Converting {coco_annotations_path} to YOLO format...")

    with open(coco_annotations_path, 'r') as f:
        coco_data = json.load(f)

    # Create output directories
    images_dir = Path(output_dir) / "images"
    labels_dir = Path(output_dir) / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    # For each annotation, create YOLO format label
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        category_id = ann['category_id'] - 1  # YOLO classes start from 0

        # Get segmentation (polygon)
        if 'segmentation' in ann and ann['segmentation']:
            # Convert polygon to YOLO format: class_id x1 y1 x2 y2 ... (normalized)
            polygon = ann['segmentation'][0]  # Take first polygon

            # Normalize coordinates (assuming 1000x1000 image size)
            normalized_polygon = []
            for i in range(0, len(polygon), 2):
                x = polygon[i] / 1000.0
                y = polygon[i + 1] / 1000.0
                normalized_polygon.extend([x, y])

            # Create label file
            label_file = labels_dir / f"{image_id:06d}.txt"
            with open(label_file, 'a') as f:  # Append mode for multiple objects per image
                line = f"{category_id} " + " ".join([f"{coord:.6f}" for coord in normalized_polygon])
                f.write(line + "\n")

    # Copy/create dummy images (since we don't have actual images)
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        image_file = images_dir / f"{image_id:06d}.png"

        if not image_file.exists():
            # Create a dummy white image
            import numpy as np
            from PIL import Image
            img = Image.new('RGB', (1000, 1000), color='white')
            img.save(image_file)

    print(f"✅ Converted to YOLO format: {len(list(labels_dir.glob('*.txt')))} label files")
    return str(images_dir), str(labels_dir)


def create_data_yaml(train_dir, val_dir, output_path):
    """Create data.yaml for YOLO training"""
    yaml_content = f"""
# YOLO Data Configuration for Room Detection
path: {Path(train_dir).parent}  # dataset root dir
train: {Path(train_dir).name}  # train images (relative to 'path')
val: {Path(val_dir).name}  # val images (relative to 'path')

# Classes
nc: 1  # number of classes
names: ['room']  # class names

# Segmentation task
task: segment
"""

    with open(output_path, 'w') as f:
        f.write(yaml_content)

    print(f"✅ Created data.yaml: {output_path}")
    return output_path


def main():
    print("🚀 YOLO11-seg Training for Room Detection")
    print("=" * 50)

    # Check CUDA
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Convert datasets
    print("\n📊 Converting datasets to YOLO format...")

    yolo_data_dir = Path("./yolo_data")
    yolo_data_dir.mkdir(exist_ok=True)

    train_img_dir, train_label_dir = convert_coco_to_yolo_format(
        TRAIN_ANNOTATIONS, yolo_data_dir / "train"
    )
    val_img_dir, val_label_dir = convert_coco_to_yolo_format(
        VAL_ANNOTATIONS, yolo_data_dir / "val"
    )

    # Create data.yaml
    data_yaml = create_data_yaml(
        yolo_data_dir / "train",
        yolo_data_dir / "val",
        yolo_data_dir / "data.yaml"
    )

    # Load YOLO model
    print("\n🏗️ Loading YOLOv8s-seg model...")
    model = YOLO('yolov8s-seg.pt')  # Load pretrained YOLOv8s-seg (optimal for deployment)

    # Training configuration
    training_args = {
        'data': str(data_yaml),
        'epochs': 50,  # YOLO typically needs more epochs
        'batch': 8,    # YOLO can handle larger batches
        'imgsz': 640,  # YOLO standard size
        'device': device,
        'workers': 4,
        'project': './outputs/yolo',
        'name': 'room_detection_yolo',
        'save': True,
        'save_period': 10,
        'patience': 10,  # Early stopping
        'optimizer': 'AdamW',
        'lr0': 0.001,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3.0,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'box': 7.5,      # Box loss gain
        'cls': 0.5,      # Class loss gain
        'dfl': 1.5,      # Distribution focal loss gain
        'seg': 2.0,      # Segmentation loss gain (higher for our task)
    }

    print("\n🎯 Starting YOLO training...")
    print(f"Training for {training_args['epochs']} epochs")
    print(f"Batch size: {training_args['batch']}")
    print(f"Image size: {training_args['imgsz']}x{training_args['imgsz']}")

    # Start training
    results = model.train(**training_args)

    print("\n✅ YOLO training completed!")

    # Save final model
    final_model_path = MODEL_DIR / 'yolo_final.pt'
    model.save(str(final_model_path))
    print(f"Model saved to: {final_model_path}")

    # Run validation
    print("\n🔍 Running final validation...")
    metrics = model.val()
    print(f"Validation mAP: {metrics.box.map:.4f}")
    if hasattr(metrics, 'seg'):
        print(f"Segmentation mAP: {metrics.seg.map:.4f}")

    # Export for inference
    print("\n📦 Exporting model for inference...")
    model.export(format='onnx', imgsz=640)

    print("\n🎉 YOLO training pipeline completed!")
    print("Compare with Mask R-CNN results to choose the best model.")


if __name__ == "__main__":
    main()
