#!/usr/bin/env python3
"""
Get the top 50 images by recall (TP / GT) - rooms detected / total rooms
"""

import torch
from ultralytics import YOLO
from pathlib import Path
import numpy as np

def load_ground_truth_boxes(label_path):
    """Load ground truth boxes from YOLO format label file"""
    gt_boxes = []
    if not label_path.exists():
        return gt_boxes

    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])

                # Convert to [x1, y1, x2, y2] format
                img_width, img_height = 640, 640
                x1 = (x_center - width/2) * img_width
                y1 = (y_center - height/2) * img_height
                x2 = (x_center + width/2) * img_width
                y2 = (y_center + height/2) * img_height

                gt_boxes.append([x1, y1, x2, y2])

    return gt_boxes

def calculate_iou(box1, box2):
    """Calculate IoU between two boxes [x1, y1, x2, y2]"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection

    return intersection / union if union > 0 else 0

def calculate_image_recall(predictions, ground_truth_boxes, iou_threshold=0.5):
    """Calculate recall for a single image: TP / (TP + FN)"""
    if len(ground_truth_boxes) == 0:
        return 0.0, 0

    if predictions is None or len(predictions) == 0:
        return 0.0, 0

    pred_boxes = predictions[:, :4]
    tp_count = 0
    matched_gt = set()

    # Sort predictions by confidence (highest first)
    sorted_indices = torch.argsort(predictions[:, 4], descending=True)

    for pred_idx in sorted_indices:
        pred_box = pred_boxes[pred_idx]

        best_iou = 0
        best_gt_idx = -1

        for gt_idx, gt_box in enumerate(ground_truth_boxes):
            if gt_idx in matched_gt:
                continue

            iou = calculate_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_threshold and best_gt_idx != -1:
            tp_count += 1
            matched_gt.add(best_gt_idx)

    recall = tp_count / len(ground_truth_boxes) if len(ground_truth_boxes) > 0 else 0
    return recall, tp_count

def apply_iou_suppression(boxes, confidences, classes, iou_threshold=0.2):
    """Apply IoU suppression to remove overlapping detections"""
    if len(boxes) == 0:
        return torch.empty(0, 6)

    indices = np.argsort(confidences)[::-1]
    boxes = boxes[indices]
    confidences = confidences[indices]
    classes = classes[indices]

    keep_indices = []
    for i, box1 in enumerate(boxes):
        should_keep = True
        for j in keep_indices:
            box2 = boxes[j]
            if calculate_iou(box1, box2) > iou_threshold:
                should_keep = False
                break
        if should_keep:
            keep_indices.append(i)

    if keep_indices:
        kept_boxes = boxes[keep_indices]
        kept_confidences = confidences[keep_indices]
        kept_classes = classes[keep_indices]

        predictions = torch.cat([
            torch.tensor(kept_boxes),
            torch.tensor(kept_confidences).unsqueeze(1),
            torch.tensor(kept_classes).unsqueeze(1)
        ], dim=1)
    else:
        predictions = torch.empty(0, 6)

    return predictions

def main():
    print("🔍 Finding top 50 images by RECALL (TP / GT ratio)")
    print("=" * 60)

    # Load model
    model_path = Path("./local_training_output/room_detection_v2/weights/best.pt")
    if not model_path.exists():
        print("❌ Model not found")
        return

    model = YOLO(str(model_path))
    print("✅ Model loaded")

    # Get test images
    test_images_dir = Path("./yolo_data/test/images")
    test_labels_dir = Path("./yolo_data/test/labels")

    if not test_images_dir.exists():
        print("❌ Test directories not found")
        return

    test_images = list(test_images_dir.glob("*.png"))
    print(f"📊 Processing {len(test_images)} test images...")

    # Process each image
    results = []
    for i, img_path in enumerate(test_images):
        if i % 50 == 0:
            print(f"Processing {i+1}/{len(test_images)}...")

        try:
            # Load ground truth
            label_path = test_labels_dir / f"{img_path.stem}.txt"
            gt_boxes = load_ground_truth_boxes(label_path)

            if len(gt_boxes) == 0:
                continue  # Skip images with no ground truth

            # Run inference
            inference_result = model(img_path, conf=0.3, iou=0.5, verbose=False)

            # Get predictions with IoU suppression
            predictions = torch.empty(0, 6)
            if inference_result and len(inference_result) > 0 and inference_result[0].boxes is not None:
                boxes = inference_result[0].boxes
                pred_boxes = boxes.xyxy.cpu().numpy()
                pred_confs = boxes.conf.cpu().numpy()
                pred_classes = boxes.cls.cpu().numpy()

                predictions = apply_iou_suppression(pred_boxes, pred_confs, pred_classes, iou_threshold=0.2)

            # Calculate recall
            recall, tp_count = calculate_image_recall(predictions, gt_boxes)

            results.append({
                'image': img_path.name,
                'gt_count': len(gt_boxes),
                'detected_count': len(predictions),
                'tp_count': tp_count,
                'recall': recall
            })

        except Exception as e:
            print(f"⚠️ Error processing {img_path.name}: {e}")
            continue

    # Sort by recall (highest first) and take top 50
    results.sort(key=lambda x: x['recall'], reverse=True)
    top_50 = results[:50] if len(results) >= 50 else results

    if len(top_50) == 0:
        print("❌ No images were successfully processed")
        return

    # Save results
    output_file = Path("./testing/top_50_by_recall.txt")
    with open(output_file, 'w') as f:
        f.write("TOP 50 IMAGES BY RECALL (TP / GT)\n")
        f.write("=" * 60 + "\n\n")
        f.write("Rank  Image Name                Recall    TP    GT    Detected\n")
        f.write("----  ------------------------  --------  ----  ----  --------\n")

        for rank, result in enumerate(top_50, 1):
            f.write(f"{rank:3d}   {result['image']:<25} {result['recall']:.3f}  {result['tp_count']:3d}  {result['gt_count']:3d}  {result['detected_count']:3d}\n")

        f.write("\n" + "=" * 60 + "\n")
        f.write("SUMMARY:\n")
        f.write(f"Total images processed: {len(results)}\n")
        avg_recall = sum(r['recall'] for r in top_50) / len(top_50) if len(top_50) > 0 else 0
        f.write(f"Average recall in top 50: {avg_recall:.1%}\n")
        # Distribution
        recall_ranges = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
        f.write("\nRECALL DISTRIBUTION:\n")
        for min_r, max_r in recall_ranges:
            count = sum(1 for r in top_50 if min_r <= r['recall'] < max_r)
            f.write(f"{min_r:.1f}-{max_r:.1f}: {count} images\n")

    print(f"\n✅ Results saved to: {output_file}")
    print(f"\n🎯 TOP {min(5, len(top_50))} BY RECALL:")
    for i, result in enumerate(top_50[:5], 1):
        print(f"{i}. {result['image']} - {result['recall']:.1%} recall ({result['tp_count']}/{result['gt_count']} rooms detected)")

if __name__ == "__main__":
    main()
