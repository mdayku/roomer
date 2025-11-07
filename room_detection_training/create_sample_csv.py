#!/usr/bin/env python3
"""
Create a sample detailed predictions CSV to show the user the format
"""

import pandas as pd
from pathlib import Path

# Sample data showing the format you requested
sample_data = [
    # True Positives
    {
        "image": "000343_F1_original.png",
        "class": "room",
        "pred_confidence": 0.679,
        "pred_bbox": [100.5, 200.3, 150.8, 250.7],
        "gt_bbox": [98.2, 198.5, 152.1, 248.9],
        "iou": 0.85,
        "match_type": "TP",
        "pred_area": 1015.5,
        "pred_pct": 0.027,
        "gt_area": 1089.2,
        "gt_pct": 0.029
    },
    {
        "image": "000343_F1_original.png",
        "class": "room",
        "pred_confidence": 0.654,
        "pred_bbox": [300.2, 180.5, 380.9, 220.1],
        "gt_bbox": [298.7, 178.3, 382.4, 218.8],
        "iou": 0.92,
        "match_type": "TP",
        "pred_area": 1524.3,
        "pred_pct": 0.041,
        "gt_area": 1587.6,
        "gt_pct": 0.042
    },
    # False Positives
    {
        "image": "000343_F1_original.png",
        "class": "room",
        "pred_confidence": 0.423,
        "pred_bbox": [50.1, 400.2, 80.5, 430.8],
        "gt_bbox": None,
        "iou": 0.0,
        "match_type": "FP",
        "pred_area": 742.8,
        "pred_pct": 0.020,
        "gt_area": None,
        "gt_pct": None
    },
    # False Negatives
    {
        "image": "000343_F1_original.png",
        "class": "room",
        "pred_confidence": None,
        "pred_bbox": None,
        "gt_bbox": [500.3, 150.7, 550.9, 200.2],
        "iou": 0.0,
        "match_type": "FN",
        "pred_area": None,
        "pred_pct": None,
        "gt_area": 1205.4,
        "gt_pct": 0.032
    },
    # Data from another image
    {
        "image": "000322_F1_original.png",
        "class": "room",
        "pred_confidence": 0.573,
        "pred_bbox": [75.4, 120.8, 125.9, 170.3],
        "gt_bbox": [72.1, 118.5, 128.4, 172.7],
        "iou": 0.78,
        "match_type": "TP",
        "pred_area": 1225.8,
        "pred_pct": 0.033,
        "gt_area": 1298.7,
        "gt_pct": 0.035
    }
]

df = pd.DataFrame(sample_data)
csv_path = Path("./testing/sample_detailed_predictions.csv")
df.to_csv(csv_path, index=False, float_format='%.6f')

print(f"Sample detailed predictions CSV created: {csv_path}")
print(f"Contains {len(sample_data)} sample predictions")
print("\nColumns include:")
print("- image: filename")
print("- class: object class")
print("- pred_confidence: model confidence (0-1)")
print("- pred_bbox: predicted bounding box [x1,y1,x2,y2]")
print("- gt_bbox: ground truth bounding box [x1,y1,x2,y2]")
print("- iou: Intersection over Union score")
print("- match_type: TP/FP/FN classification")
print("- pred_area: prediction area in pixels")
print("- pred_pct: prediction area as % of image")
print("- gt_area: ground truth area in pixels")
print("- gt_pct: ground truth area as % of image")

print(f"\nFull CSV will contain ~15,000+ rows for all 400 test images!")
