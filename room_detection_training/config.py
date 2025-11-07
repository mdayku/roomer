"""
Configuration for Room Detection Training Pipeline
"""

import os
from pathlib import Path

# Dataset paths
DATA_ROOT = Path("../../room_detection_dataset_coco")
TRAIN_ANNOTATIONS = DATA_ROOT / "train" / "annotations.json"
VAL_ANNOTATIONS = DATA_ROOT / "val" / "annotations.json"
TEST_ANNOTATIONS = DATA_ROOT / "test" / "annotations.json"

# Training parameters
NUM_CLASSES = 2  # background + room
IMAGE_SIZE = (800, 800)  # Input image size for model
BATCH_SIZE = 2  # Adjust based on GPU memory
NUM_WORKERS = 4  # Data loading workers

# Model configuration
BACKBONE = "resnet50"  # resnet50, resnet101
HIDDEN_LAYERS = 256
ANCHOR_SIZES = ((32,), (64,), (128,), (256,), (512,))

# Training hyperparameters
LEARNING_RATE = 0.001
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
STEP_SIZE = 3  # Learning rate scheduler step size
GAMMA = 0.1    # Learning rate decay factor

# Training settings
NUM_EPOCHS = 30  # Increased for better convergence
PRINT_FREQ = 10  # Print training stats every N iterations
SAVE_FREQ = 5    # Save model checkpoint every N epochs

# Early stopping
PATIENCE = 5     # Stop if no improvement for 5 epochs
MIN_DELTA = 0.001 # Minimum improvement threshold

# Paths
OUTPUT_DIR = Path("./outputs")
MODEL_DIR = OUTPUT_DIR / "models"
LOG_DIR = OUTPUT_DIR / "logs"
VISUALIZATION_DIR = OUTPUT_DIR / "visualizations"

# Create directories
for dir_path in [OUTPUT_DIR, MODEL_DIR, LOG_DIR, VISUALIZATION_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Device configuration
DEVICE = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"

# Data augmentation
DATA_AUGMENTATION = {
    "horizontal_flip": True,
    "vertical_flip": False,
    "rotation": 15,  # degrees
    "scale": (0.8, 1.2),
    "brightness": 0.1,
    "contrast": 0.1,
    "saturation": 0.1,
    "hue": 0.05
}

# Evaluation metrics
EVALUATION_METRICS = [
    "bbox",      # Bounding box AP
    "segm",      # Segmentation AP (what we care about most)
    "bbox_per_class",
    "segm_per_class"
]

# PRD Requirements
TARGET_LATENCY_MS = 30000  # 30 seconds max per blueprint
MIN_IOU_THRESHOLD = 0.75   # Target IoU for complex room shapes
CONFIDENCE_THRESHOLD = 0.5 # Minimum confidence for predictions
