#!/usr/bin/env python3
"""
Test script for trained YOLO room detection model
Runs inference on test images and shows results
"""

import os
import sys
from pathlib import Path
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages
import cv2
from PIL import Image
import numpy as np
from typing import List, Tuple

def main():
    print("Room Detection YOLO Model Testing")
    print("=" * 50)

    # Check for GPU
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load the trained model
    model_path = Path("./local_training_output/room_detection_v2/weights/best.pt")
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return

    print(f"Loading model from: {model_path}")
    model = YOLO(str(model_path))
    print("Model loaded successfully!")

    # Test on validation images
    test_images_dir = Path("./yolo_data/val/images")
    if not test_images_dir.exists():
        print(f"❌ Test images not found: {test_images_dir}")
        return

    # Get some test images
    test_images = list(test_images_dir.glob("*.png"))[:5]  # Test first 5 images
    print(f"Testing on {len(test_images)} images...")

    # Run inference
    results = []
    for img_path in test_images:
        print(f"Processing: {img_path.name}")
        result = model(img_path, conf=0.3, iou=0.5)  # Lower confidence for testing
        results.append((img_path, result))

    # Display results
    print("\n" + "=" * 50)
    print("RESULTS SUMMARY")
    print("=" * 50)

    total_rooms = 0
    for img_path, result in results:
        boxes = result[0].boxes
        num_rooms = len(boxes) if boxes is not None else 0
        total_rooms += num_rooms
        print(f"{img_path.name}: {num_rooms} rooms detected")

        # Show confidence scores
        if boxes is not None and len(boxes) > 0:
            confs = boxes.conf.cpu().numpy()
            print(f"  Confidences: {confs}")
    print(f"\nTotal rooms detected across {len(results)} images: {total_rooms}")
    print(f"Average rooms per image: {total_rooms/len(results):.1f}")
    # Run TEST SET evaluation (not validation set)
    print("\n" + "=" * 50)
    print("RUNNING TEST SET EVALUATION")
    print("=" * 50)

    try:
        # Create testing output directory
        import os
        test_output_dir = Path("./testing")
        test_output_dir.mkdir(exist_ok=True)

        test_results = model.val(data="./yolo_data/data.yaml", split='test', project=str(test_output_dir), name='test_evaluation')
        print("✅ Test set evaluation completed!")
        print(f"mAP50: {test_results.box.map50:.3f}")
        print(f"mAP50-95: {test_results.box.map:.3f}")
        print(f"Precision: {test_results.box.mp:.3f}")
        print(f"Recall: {test_results.box.mr:.3f}")
        print(f"Test results saved to: {test_output_dir}/test_evaluation/")
    except Exception as e:
        print(f"⚠️ Test evaluation failed: {e}")
        print("Falling back to validation set evaluation...")

        try:
            val_results = model.val(data="./yolo_data/data.yaml", split='val')
            print("✅ Validation completed (as fallback)!")
            print(f"mAP50: {val_results.box.map50:.3f}")
            print(f"mAP50-95: {val_results.box.map:.3f}")
            print(f"Precision: {val_results.box.mp:.3f}")
            print(f"Recall: {val_results.box.mr:.3f}")
        except Exception as e2:
            print(f"⚠️ Both test and validation failed: {e2}")

    # Generate detailed CSV with all predictions
    print("\n" + "=" * 50)
    print("GENERATING DETAILED PREDICTIONS CSV")
    print("=" * 50)

    csv_path = None
    try:
        csv_path = generate_detailed_predictions_csv(model, test_output_dir)
        print(f"✅ Detailed predictions CSV: {csv_path}")
    except Exception as e:
        print(f"⚠️ CSV generation failed: {e}")

    # Generate Top 50 Results Report
    print("\n" + "=" * 50)
    print("GENERATING TOP 50 RESULTS REPORT")
    print("=" * 50)

    try:
        generate_top_50_report(model, test_output_dir)
    except Exception as e:
        print(f"⚠️ Report generation failed: {e}")

    print("\n🎉 Testing completed!")

def calculate_image_recall(predictions, ground_truth_boxes, iou_threshold=0.5):
    """
    Calculate recall for a single image: TP / (TP + FN)

    Args:
        predictions: tensor of predicted boxes [x1, y1, x2, y2, conf, class]
        ground_truth_boxes: list of ground truth boxes [x1, y1, x2, y2]
        iou_threshold: IoU threshold for matching

    Returns:
        recall: float (TP / (TP + FN))
        tp_confidences: list of confidence scores for true positives
    """
    if len(ground_truth_boxes) == 0:
        return 0.0, []

    if predictions is None or len(predictions) == 0:
        return 0.0, []

    pred_boxes = predictions[:, :4]  # [x1, y1, x2, y2]
    pred_confs = predictions[:, 4]  # confidence scores

    gt_boxes = torch.tensor(ground_truth_boxes, device=pred_boxes.device)

    tp_count = 0
    tp_confidences = []
    used_gt = set()

    # Sort predictions by confidence (highest first)
    sorted_indices = torch.argsort(pred_confs, descending=True)

    for pred_idx in sorted_indices:
        pred_box = pred_boxes[pred_idx]
        pred_conf = pred_confs[pred_idx]

        best_iou = 0
        best_gt_idx = -1

        # Find best matching ground truth box
        for gt_idx, gt_box in enumerate(gt_boxes):
            if gt_idx in used_gt:
                continue

            # Calculate IoU
            iou = calculate_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        # If IoU > threshold and GT not used, count as TP
        if best_iou >= iou_threshold and best_gt_idx != -1:
            tp_count += 1
            tp_confidences.append(float(pred_conf))
            used_gt.add(best_gt_idx)

    # Calculate recall: TP / (TP + FN)
    recall = tp_count / len(ground_truth_boxes) if len(ground_truth_boxes) > 0 else 0

    return recall, tp_confidences

def calculate_iou(box1, box2):
    """Calculate IoU between two boxes [x1, y1, x2, y2]"""
    # Intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    # Union
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection

    return intersection / union if union > 0 else 0

def apply_iou_suppression(boxes, confidences, classes, iou_threshold=0.2):
    """
    Apply IoU-based suppression to remove overlapping detections.
    Keeps higher confidence detections when boxes overlap by more than threshold.

    Args:
        boxes: numpy array of [x1, y1, x2, y2]
        confidences: numpy array of confidence scores
        classes: numpy array of class ids
        iou_threshold: IoU threshold for suppression

    Returns:
        torch tensor of filtered predictions [x1, y1, x2, y2, conf, class]
    """
    if len(boxes) == 0:
        return torch.empty(0, 6)

    # Sort by confidence (highest first)
    indices = np.argsort(confidences)[::-1]
    boxes = boxes[indices]
    confidences = confidences[indices]
    classes = classes[indices]

    keep_indices = []

    for i, box1 in enumerate(boxes):
        should_keep = True

        # Check against all previously kept boxes
        for j in keep_indices:
            box2 = boxes[j]
            iou = calculate_iou(box1, box2)

            if iou > iou_threshold:
                # Overlap too much, don't keep this box
                should_keep = False
                break

        if should_keep:
            keep_indices.append(i)

    # Keep only the selected boxes
    if keep_indices:
        kept_boxes = boxes[keep_indices]
        kept_confidences = confidences[keep_indices]
        kept_classes = classes[keep_indices]

        # Combine into predictions array: [x1, y1, x2, y2, conf, class]
        predictions = torch.cat([
            torch.tensor(kept_boxes),
            torch.tensor(kept_confidences).unsqueeze(1),
            torch.tensor(kept_classes).unsqueeze(1)
        ], dim=1)
    else:
        predictions = torch.empty(0, 6)

    return predictions

def load_ground_truth_boxes(label_path):
    """Load ground truth boxes from YOLO format label file"""
    gt_boxes = []

    if not label_path.exists():
        return gt_boxes

    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                # YOLO format: class x_center y_center width height
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])

                # Convert to [x1, y1, x2, y2] format (assuming 640x640 image size)
                img_width, img_height = 640, 640
                x1 = (x_center - width/2) * img_width
                y1 = (y_center - height/2) * img_height
                x2 = (x_center + width/2) * img_width
                y2 = (y_center + height/2) * img_height

                gt_boxes.append([x1, y1, x2, y2])

    return gt_boxes

def generate_top_50_report(model, output_dir):
    """Generate report with top 50 test images ranked by weighted recall-confidence score"""

    # Get all test images
    test_images_dir = Path("./yolo_data/test/images")
    test_labels_dir = Path("./yolo_data/test/labels")

    if not test_images_dir.exists():
        print("❌ Test images directory not found")
        return

    test_images = list(test_images_dir.glob("*.png"))
    print(f"Evaluating {len(test_images)} test images for top 50 report...")

    # Evaluate each image
    image_scores = []
    for i, img_path in enumerate(test_images):
        if i % 50 == 0:  # Progress indicator
            print(f"Processing image {i+1}/{len(test_images)}...")

        try:
            # Load ground truth
            label_path = test_labels_dir / f"{img_path.stem}.txt"
            gt_boxes = load_ground_truth_boxes(label_path)

            # Run inference
            results = model(img_path, conf=0.3, iou=0.5, verbose=False)

            # Get predictions in the right format
            if results and len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                pred_boxes = boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
                pred_confs = boxes.conf.cpu().numpy()  # confidence scores
                pred_classes = boxes.cls.cpu().numpy()  # class ids

                # Apply IoU suppression to remove overlapping boxes
                predictions = apply_iou_suppression(pred_boxes, pred_confs, pred_classes, iou_threshold=0.2)
            else:
                predictions = torch.empty(0, 6)

            # Calculate recall and TP confidences
            recall, tp_confidences = calculate_image_recall(predictions, gt_boxes)

            # Calculate weighted score: recall * mean_TP_confidence
            if tp_confidences:
                mean_tp_confidence = sum(tp_confidences) / len(tp_confidences)
                weighted_score = recall * mean_tp_confidence
            else:
                mean_tp_confidence = 0.0
                weighted_score = 0.0

            image_scores.append({
                'image_path': img_path,
                'image_name': img_path.name,
                'ground_truth_rooms': len(gt_boxes),
                'detected_rooms': len(predictions),
                'recall': recall,
                'mean_tp_confidence': mean_tp_confidence,
                'weighted_score': weighted_score,
                'tp_confidences': tp_confidences
            })

        except Exception as e:
            print(f"⚠️ Error processing {img_path.name}: {e}")
            continue

    # Sort by weighted score (descending) and take top 50
    image_scores.sort(key=lambda x: x['weighted_score'], reverse=True)
    top_50 = image_scores[:50]

    # Generate report
    report_path = output_dir / "top_50_test_results_report.txt"
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("TOP 50 TEST IMAGES - RANKED BY WEIGHTED RECALL-CONFIDENCE SCORE\n")
        f.write("=" * 80 + "\n\n")
        f.write("Metric: weighted_score = (TP / (TP + FN)) × mean_TP_confidence\n")
        f.write("Higher scores = Better recall with higher confidence in correct detections\n\n")

        f.write(f"Total test images evaluated: {len(image_scores)}\n")
        f.write(f"Report generated: {Path.cwd()}\n\n")

        f.write("-" * 90 + "\n")
        f.write(f"{'Rank':<4} {'Image Name':<25} {'GT':<3} {'Det':<3} {'Recall':<6} {'TP Conf':<7} {'Score':<8}\n")
        f.write("-" * 90 + "\n")

        for rank, result in enumerate(top_50, 1):
            f.write(f"{rank:<4} {result['image_name']:<25} {result['ground_truth_rooms']:<3} "
                   f"{result['detected_rooms']:<3} {result['recall']:<6.3f} "
                   f"{result['mean_tp_confidence']:<7.3f} {result['weighted_score']:<8.3f}\n")

        f.write("-" * 90 + "\n\n")

        # Summary statistics
        all_scores = [r['weighted_score'] for r in image_scores]
        all_recalls = [r['recall'] for r in image_scores]
        all_tp_confs = [r['mean_tp_confidence'] for r in image_scores if r['tp_confidences']]

        f.write("SUMMARY STATISTICS:\n")
        f.write(f"Average weighted score across all images: {sum(all_scores)/len(all_scores):.3f}\n")
        f.write(f"Average recall across all images: {sum(all_recalls)/len(all_recalls):.3f}\n")
        f.write(f"Average TP confidence across all images: {sum(all_tp_confs)/len(all_tp_confs):.3f}\n")
        f.write(f"Top 50 average score: {sum(r['weighted_score'] for r in top_50)/len(top_50):.3f}\n")
        f.write(f"Score range in top 50: {top_50[0]['weighted_score']:.3f} - {top_50[-1]['weighted_score']:.3f}\n")

        # Best performance examples
        best_recall = max(image_scores, key=lambda x: x['recall'])
        f.write(f"\nBEST RECALL: {best_recall['image_name']} ({best_recall['recall']:.1%} recall)\n")

        best_tp_conf = max(image_scores, key=lambda x: x['mean_tp_confidence'] if x['tp_confidences'] else 0)
        f.write(f"BEST TP CONFIDENCE: {best_tp_conf['image_name']} (avg conf: {best_tp_conf['mean_tp_confidence']:.3f})\n")

        best_score = top_50[0]
        f.write(f"BEST WEIGHTED SCORE: {best_score['image_name']} (score: {best_score['weighted_score']:.3f})\n")

    print(f"✅ Top 50 text report generated: {report_path}")

    # Generate PDF visualization report
    pdf_path = output_dir / "top_50_visualizations.pdf"
    print("🎨 Generating PDF visualization report...")
    generate_visualization_pdf(model, top_50, pdf_path)

    print(f"📊 Best image: {top_50[0]['image_name']} (score: {top_50[0]['weighted_score']:.3f})")
    print(f"📄 Text report: {report_path}")
    print(f"📖 PDF visualizations: {pdf_path}")
    print(f"📊 Detailed predictions CSV: {csv_path}")

def generate_detailed_predictions_csv(model, output_dir):
    """Generate detailed CSV with all predictions, ground truths, and metrics"""

    import pandas as pd

    # Get all test images and labels
    test_images_dir = Path("./yolo_data/test/images")
    test_labels_dir = Path("./yolo_data/test/labels")

    rows = []

    print("Processing all test images for detailed predictions...")

    for i, img_path in enumerate(test_images_dir.glob("*.png")):
        if i % 50 == 0:
            print(f"Processing image {i+1}/400...")

        # Load ground truth boxes
        label_path = test_labels_dir / f"{img_path.stem}.txt"
        gt_boxes = load_ground_truth_boxes(label_path)

        # Run inference
        results = model(img_path, conf=0.3, iou=0.5, verbose=False)

        # Get predictions
        predictions = torch.empty(0, 6)
        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            pred_boxes = boxes.xyxy.cpu().numpy()
            pred_confs = boxes.conf.cpu().numpy()
            pred_classes = boxes.cls.cpu().numpy()

            # Apply IoU suppression
            predictions = apply_iou_suppression(pred_boxes, pred_confs, pred_classes, iou_threshold=0.2)

        # Classify predictions and find matches
        tp_boxes, fp_boxes, fn_boxes = classify_predictions(predictions, gt_boxes)

        # Add TP rows
        for pred_box, conf in tp_boxes:
            # Find the matched ground truth box
            best_iou = 0
            best_gt_idx = -1
            for gt_idx, gt_box in enumerate(gt_boxes):
                iou = calculate_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_gt_idx != -1:
                gt_box = gt_boxes[best_gt_idx]
                # Calculate areas
                pred_area = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
                gt_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
                img_area = 640 * 640  # Assuming 640x640 images
                pred_pct = pred_area / img_area
                gt_pct = gt_area / img_area

                rows.append({
                    "image": img_path.name,
                    "class": "room",  # We only have one class
                    "pred_confidence": conf,
                    "pred_bbox": pred_box.tolist(),
                    "gt_bbox": gt_box.tolist(),
                    "iou": best_iou,
                    "match_type": "TP",
                    "pred_area": pred_area,
                    "pred_pct": pred_pct,
                    "gt_area": gt_area,
                    "gt_pct": gt_pct
                })

        # Add FP rows
        for pred_box, conf in fp_boxes:
            pred_area = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
            img_area = 640 * 640
            pred_pct = pred_area / img_area

            rows.append({
                "image": img_path.name,
                "class": "room",
                "pred_confidence": conf,
                "pred_bbox": pred_box.tolist(),
                "gt_bbox": None,
                "iou": 0.0,
                "match_type": "FP",
                "pred_area": pred_area,
                "pred_pct": pred_pct,
                "gt_area": None,
                "gt_pct": None
            })

        # Add FN rows
        for fn_box in fn_boxes:
            gt_area = (fn_box[2] - fn_box[0]) * (fn_box[3] - fn_box[1])
            img_area = 640 * 640
            gt_pct = gt_area / img_area

            rows.append({
                "image": img_path.name,
                "class": "room",
                "pred_confidence": None,
                "pred_bbox": None,
                "gt_bbox": fn_box.tolist(),
                "iou": 0.0,
                "match_type": "FN",
                "pred_area": None,
                "pred_pct": None,
                "gt_area": gt_area,
                "gt_pct": gt_pct
            })

    # Save to CSV
    df = pd.DataFrame(rows)
    csv_path = output_dir / "detailed_predictions.csv"
    df.to_csv(csv_path, index=False, float_format='%.6f')

    print(f"✅ Saved detailed predictions CSV with {len(rows)} rows to: {csv_path}")

    # Print summary
    tp_count = sum(1 for r in rows if r['match_type'] == 'TP')
    fp_count = sum(1 for r in rows if r['match_type'] == 'FP')
    fn_count = sum(1 for r in rows if r['match_type'] == 'FN')

    print(f"Summary: {tp_count} TP, {fp_count} FP, {fn_count} FN across {len(set(r['image'] for r in rows))} images")

    return csv_path

def generate_visualization_pdf(model, top_50_images, pdf_path):
    """Generate PDF with detailed visualizations showing TP, FP, FN, and GT boxes"""

    # Define colors for different box types
    COLORS = {
        'GT': ('cyan', 2, 'dotted'),      # Ground truth - dotted cyan (lowest layer)
        'TP': ('lime', 3, 'solid'),       # True positives - solid lime (highest layer)
        'FP': ('red', 2, 'solid'),        # False positives - solid red (2nd layer)
        'FN': ('yellow', 2, 'solid')      # False negatives - solid yellow (3rd layer)
    }

    with PdfPages(str(pdf_path)) as pdf:
        for i, result in enumerate(top_50_images):
            if i % 10 == 0:
                print(f"Processing visualization {i+1}/50...")

            try:
                img_path = result['image_path']

                # Load ground truth boxes
                label_path = Path("./yolo_data/test/labels") / f"{img_path.stem}.txt"
                gt_boxes = load_ground_truth_boxes(label_path)

                # Run inference
                results = model(img_path, conf=0.3, iou=0.5, verbose=False)

                # Load and display image
                img = cv2.imread(str(img_path))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Create figure with higher DPI for better quality
                fig, ax = plt.subplots(1, 1, figsize=(16, 12))
                ax.imshow(img)
                ax.axis('off')

                # Get predictions
                predictions = torch.empty(0, 6)
                if results and len(results) > 0 and results[0].boxes is not None:
                    boxes = results[0].boxes
                    pred_boxes = boxes.xyxy.cpu().numpy()
                    pred_confs = boxes.conf.cpu().numpy()
                    pred_classes = boxes.cls.cpu().numpy()

                    # Apply IoU suppression
                    predictions = apply_iou_suppression(pred_boxes, pred_confs, pred_classes, iou_threshold=0.2)

                # Classify predictions and find FNs
                tp_boxes, fp_boxes, fn_boxes = classify_predictions(predictions, gt_boxes)

                # Draw ground truth boxes first (dotted, lowest layer)
                for gt_box in gt_boxes:
                    x1, y1, x2, y2 = gt_box
                    draw_box(ax, x1, y1, x2, y2, COLORS['GT'], "GT")

                # Draw false negatives (solid, 3rd layer)
                for fn_box in fn_boxes:
                    x1, y1, x2, y2 = fn_box
                    draw_box(ax, x1, y1, x2, y2, COLORS['FN'], "FN")

                # Draw false positives (solid, 2nd layer)
                for fp_box in fp_boxes:
                    x1, y1, x2, y2, conf = fp_box
                    draw_box(ax, x1, y1, x2, y2, COLORS['FP'], f"FP {conf:.2f}")

                # Draw true positives last (solid, highest layer)
                for tp_box in tp_boxes:
                    x1, y1, x2, y2, conf = tp_box
                    draw_box(ax, x1, y1, x2, y2, COLORS['TP'], f"TP {conf:.2f}")

                # Calculate precision for display
                precision = len(tp_boxes)/(len(tp_boxes)+len(fp_boxes)) if (len(tp_boxes)+len(fp_boxes)) > 0 else 0

                # Add comprehensive metrics overlay
                metrics_text = f"""Rank #{i+1}: {result['image_name']}
Ground Truth Rooms: {result['ground_truth_rooms']}
Total Detections: {len(predictions)}
True Positives: {len(tp_boxes)}
False Positives: {len(fp_boxes)}
False Negatives: {len(fn_boxes)}
Recall: {result['recall']:.1%}
Precision: {precision:.1%}
TP Confidence: {result['mean_tp_confidence']:.3f}
Weighted Score: {result['weighted_score']:.3f}"""

                # Position text in top-left with semi-transparent background
                ax.text(0.02, 0.98, metrics_text,
                       transform=ax.transAxes,
                       fontsize=11,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round,pad=0.7',
                               facecolor='white',
                               alpha=0.9,
                               edgecolor='black',
                               linewidth=1),
                       family='monospace',
                       fontweight='bold')

                # Add legend
                legend_text = """Box Legend:
● GT: Ground Truth (Cyan Dotted)
● TP: True Positive (Lime Solid)
● FP: False Positive (Red Solid)
● FN: False Negative (Yellow Solid)"""

                ax.text(0.02, 0.02, legend_text,
                       transform=ax.transAxes,
                       fontsize=10,
                       verticalalignment='bottom',
                       bbox=dict(boxstyle='round,pad=0.5',
                               facecolor='lightgray',
                               alpha=0.8),
                       family='monospace')

                # Set title
                ax.set_title(f'Top {i+1}: {result["image_name"]} (Score: {result["weighted_score"]:.3f})',
                           fontsize=14, pad=20, fontweight='bold')

                plt.tight_layout()

                # Save to PDF with high quality
                pdf.savefig(fig, bbox_inches='tight', dpi=200)
                plt.close(fig)

            except Exception as e:
                print(f"⚠️ Error generating visualization for {result['image_name']}: {e}")
                continue

    print(f"✅ PDF visualization report saved: {pdf_path}")

def draw_box(ax, x1, y1, x2, y2, color_style, label=""):
    """Draw a bounding box with specified color and style"""
    color, linewidth, style = color_style

    if style == 'dotted':
        # Draw dotted box using matplotlib's linestyle
        import matplotlib.patches as patches
        rect = patches.Rectangle(
            (x1, y1), x2-x1, y2-y1,
            linewidth=linewidth,
            edgecolor=color,
            facecolor='none',
            alpha=0.9,
            linestyle='--'  # Dotted line
        )
        ax.add_patch(rect)
    else:
        # Solid rectangle
        import matplotlib.patches as patches
        rect = patches.Rectangle(
            (x1, y1), x2-x1, y2-y1,
            linewidth=linewidth,
            edgecolor=color,
            facecolor='none',
            alpha=0.9
        )
        ax.add_patch(rect)

    # Add label if provided
    if label:
        ax.text(x1, y1-8, label,
               bbox=dict(facecolor=color, alpha=0.8, edgecolor='black', linewidth=1),
               fontsize=9, color='white', fontweight='bold')

def classify_predictions(predictions, ground_truth_boxes, iou_threshold=0.5):
    """
    Classify predictions into TP, FP, and identify FN
    Returns: (tp_boxes, fp_boxes, fn_boxes)
    """
    if len(ground_truth_boxes) == 0:
        # All predictions are FP if no ground truth
        fp_boxes = [(pred[0], pred[1], pred[2], pred[3], pred[4]) for pred in predictions]
        return [], fp_boxes, []

    if len(predictions) == 0:
        # All ground truth are FN if no predictions
        return [], [], ground_truth_boxes

    tp_boxes = []
    fp_boxes = []
    matched_gt = set()

    # Sort predictions by confidence (highest first)
    sorted_indices = torch.argsort(predictions[:, 4], descending=True)

    for pred_idx in sorted_indices:
        pred = predictions[pred_idx]
        pred_box = pred[:4]  # [x1, y1, x2, y2]
        pred_conf = pred[4]  # confidence

        best_iou = 0
        best_gt_idx = -1

        # Find best matching ground truth
        for gt_idx, gt_box in enumerate(ground_truth_boxes):
            if gt_idx in matched_gt:
                continue

            iou = calculate_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_threshold and best_gt_idx != -1:
            # True positive
            tp_boxes.append((pred_box[0], pred_box[1], pred_box[2], pred_box[3], pred_conf))
            matched_gt.add(best_gt_idx)
        else:
            # False positive
            fp_boxes.append((pred_box[0], pred_box[1], pred_box[2], pred_box[3], pred_conf))

    # False negatives are unmatched ground truth
    fn_boxes = [ground_truth_boxes[i] for i in range(len(ground_truth_boxes)) if i not in matched_gt]

    return tp_boxes, fp_boxes, fn_boxes

if __name__ == "__main__":
    main()
