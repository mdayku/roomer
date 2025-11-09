#!/usr/bin/env python3
"""
Debug CSV generation
"""

from pathlib import Path
import torch
from ultralytics import YOLO

def debug_csv():
    # Load model
    model_path = Path("./local_training_output/room_detection_v2/weights/best.pt")
    model = YOLO(str(model_path))

    # Test on one image
    test_images_dir = Path("./yolo_data/test/images")
    test_labels_dir = Path("./yolo_data/test/labels")

    img_path = list(test_images_dir.glob("*.png"))[0]  # First image
    print(f"Testing with: {img_path}")

    # Load ground truth
    label_path = test_labels_dir / f"{img_path.stem}.txt"
    print(f"Label path: {label_path}")
    print(f"Label exists: {label_path.exists()}")

    # Load ground truth boxes
    gt_boxes = []
    if label_path.exists():
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])

                    img_width, img_height = 640, 640
                    x1 = (x_center - width/2) * img_width
                    y1 = (y_center - height/2) * img_height
                    x2 = (x_center + width/2) * img_width
                    y2 = (y_center + height/2) * img_height

                    gt_boxes.append([x1, y1, x2, y2])

    print(f"Ground truth boxes: {len(gt_boxes)}")
    for i, box in enumerate(gt_boxes):
        print(f"  GT {i}: {box}")

    # Run inference
    results = model(img_path, conf=0.3, iou=0.5, verbose=False)
    print(f"Results: {results}")

    if results and len(results) > 0 and results[0].boxes is not None:
        boxes = results[0].boxes
        pred_boxes = boxes.xyxy.cpu().numpy()
        pred_confs = boxes.conf.cpu().numpy()
        pred_classes = boxes.cls.cpu().numpy()

        print(f"Predictions: {len(pred_boxes)}")
        for i, (box, conf) in enumerate(zip(pred_boxes, pred_confs)):
            print(f"  Pred {i}: {box} conf={conf:.3f}")

if __name__ == "__main__":
    debug_csv()


