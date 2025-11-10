#!/usr/bin/env python3
"""
Comprehensive DINO Model Testing Script
Mirrors the YOLO comprehensive_test.py but adapted for DINO inference

Tests DINO models and generates comparable metrics to YOLO
"""

import torch
import torch.nn as nn
from pathlib import Path
import pandas as pd
import cv2
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import json
from typing import List, Tuple, Dict

start_time = time.time()

# Configuration
VID_W = VID_H = 640  # Fixed square output for consistent video
fps = 10

# Dataset paths
root = Path.cwd()
dataset_dir = root / "yolo_room_only"  # Test on room dataset for consistency
test_images_dir = dataset_dir / "test/images"
test_labels_dir = dataset_dir / "test/labels"

# Output directory with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
batch_tag = f"dino_detection_test_{timestamp}"
output_dir = root / batch_tag
output_dir.mkdir(parents=True, exist_ok=True)

print(f"DINO Model Comprehensive Testing")
print(f"Output directory: {batch_tag}")
print(f"Test images: {len(list(test_images_dir.glob('*.png')))}")
print(f"Test labels: {len(list(test_labels_dir.glob('*.txt')))}")

# === DINO MODEL DISCOVERY ===
# Look for DINO models in local_training_output/*/dino_checkpoint.pth
all_model_paths = []
all_model_paths.extend(list((root / "local_training_output").glob("*/dino_checkpoint.pth")))
all_model_paths.extend(list((root / "local_training_output").glob("*/checkpoint*.pth")))
all_model_paths.extend(list((root / "local_training_output").glob("*/best.pth")))

print(f"Found {len(all_model_paths)} DINO model(s)")

if not all_model_paths:
    print(f"\nNo DINO models found!")
    print(f"   Expected location: local_training_output/dino-*/dino_checkpoint.pth")
    print(f"   Please download trained DINO model from S3")
    exit(1)

# Sort by modification time (newest first)
model_paths = sorted(all_model_paths, key=lambda p: p.stat().st_mtime, reverse=True)

print(f"Testing {len(model_paths)} DINO model(s):")
for i, path in enumerate(model_paths, 1):
    model_name = path.parent.name
    print(f"  {i}. {model_name}")

# === DRAW CONFIG ===
COLORS = {"TP": (0,255,0), "FP": (0,0,255), "FN": (255,0,0)}  # TP=Green, FP=Red, FN=Blue
THICK = 2
conf_threshold = 0.3  # Lower threshold for room detection
IoU_TH = 0.5  # Standard IoU threshold
FP_SUPPRESS_IOU = 0.2  # Suppress overlapping FPs

# === DINO SETUP ===
def setup_dino():
    """Setup DINO environment and imports"""
    # Add DINO to path if it exists
    dino_dir = root / "DINO"
    if not dino_dir.exists():
        print(f"[ERROR] DINO directory not found at {dino_dir}")
        print(f"[INFO] Please clone DINO: git clone https://github.com/IDEA-Research/DINO.git")
        sys.exit(1)
    
    sys.path.insert(0, str(dino_dir))
    
    # Import DINO modules
    try:
        from models import build_model
        from util.slconfig import SLConfig
        from datasets import build_dataset
        from util.visualizer import COCOVisualizer
        from util import box_ops
        return build_model, SLConfig, box_ops
    except ImportError as e:
        print(f"[ERROR] Failed to import DINO modules: {e}")
        print(f"[INFO] Make sure DINO is properly installed")
        sys.exit(1)

build_model, SLConfig, box_ops = setup_dino()

def load_dino_model(checkpoint_path: Path, config_path: Path = None):
    """Load DINO model from checkpoint"""
    
    # If no config provided, look for it in the same directory or use default
    if config_path is None:
        # Look for config in same directory
        config_candidates = [
            checkpoint_path.parent / "dino_config.py",
            checkpoint_path.parent / "config.py",
            root / "DINO/config/DINO/DINO_4scale_room_detection.py",
            root / "DINO/config/DINO/DINO_4scale.py",
        ]
        
        for candidate in config_candidates:
            if candidate.exists():
                config_path = candidate
                break
        
        if config_path is None:
            print(f"[ERROR] Could not find DINO config file")
            print(f"[INFO] Looked in: {[str(c) for c in config_candidates]}")
            sys.exit(1)
    
    print(f"  Loading config from: {config_path}")
    
    # Load config
    args = SLConfig.fromfile(str(config_path))
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Build model
    model, criterion, postprocessors = build_model(args)
    
    # Load checkpoint
    checkpoint = torch.load(str(checkpoint_path), map_location='cpu')
    
    # Load model state
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    model.to(args.device)
    model.eval()
    
    print(f"  Model loaded: {checkpoint_path.name}")
    print(f"  Device: {args.device}")
    
    return model, postprocessors, args

def preprocess_image(image_path: Path, device: str = 'cuda'):
    """Preprocess image for DINO inference"""
    import torchvision.transforms as T
    
    # Load image
    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Get original size
    h, w = img.shape[:2]
    orig_size = torch.tensor([h, w])
    
    # Transform
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    return img_tensor, orig_size, img

def run_dino_inference(model, image_tensor, orig_size, postprocessors, args):
    """Run DINO inference on image"""
    with torch.no_grad():
        outputs = model(image_tensor)
        
        # Post-process outputs
        orig_target_sizes = orig_size.unsqueeze(0).to(args.device)
        results = postprocessors['bbox'](outputs, orig_target_sizes)[0]
        
        # Extract predictions
        scores = results['scores']
        labels = results['labels']
        boxes = results['boxes']  # [x1, y1, x2, y2] format
        
        return scores, labels, boxes

# === UTILS (Same as YOLO script) ===
def iou_tensor(b1, b2):
    inter_x1 = torch.max(b1[0], b2[0]); inter_y1 = torch.max(b1[1], b2[1])
    inter_x2 = torch.min(b1[2], b2[2]); inter_y2 = torch.min(b1[3], b2[3])
    inter = torch.clamp(inter_x2-inter_x1, 0) * torch.clamp(inter_y2-inter_y1, 0)
    a1 = (b1[2]-b1[0])*(b1[3]-b1[1]); a2 = (b2[2]-b2[0])*(b2[3]-b2[1])
    union = a1 + a2 - inter
    return (inter/union).item() if union > 0 else 0

def metrics(box, img_area):
    x1, y1, x2, y2 = box
    w, h = x2-x1, y2-y1
    area = w*h
    pct = area / img_area if img_area > 0 else 0
    return x1, y1, w, h, area, pct

def dotted(img, x1, y1, x2, y2, color, thk=1, gap=5):
    """Draw dotted rectangle for FN visualization"""
    for i in range(x1, x2, gap*2):
        cv2.line(img, (i, y1), (min(i+gap, x2), y1), color, thk)
        cv2.line(img, (i, y2), (min(i+gap, x2), y2), color, thk)
    for j in range(y1, y2, gap*2):
        cv2.line(img, (x1, j), (x1, min(j+gap, y2)), color, thk)
        cv2.line(img, (x2, j), (x2, min(j+gap, y2)), color, thk)

def create_top_10_pdf(model_name, summary_df, output_dir, batch_tag):
    """Create PDF with top 10 images by recall showing predictions"""
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.patches as patches

    # Load predictions and calculate per-image recall
    csv_path = output_dir / f"predictions_{model_name}.csv"
    df = pd.read_csv(csv_path)

    # Calculate per-image metrics
    image_stats = []
    for img_name in df['image'].unique():
        img_df = df[df['image'] == img_name]
        tp = (img_df['match_type'] == 'TP').sum()
        fn = (img_df['match_type'] == 'FN').sum()
        fp = (img_df['match_type'] == 'FP').sum()
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        gt_rooms = len(img_df[img_df['match_type'] == 'FN']) + tp

        image_stats.append({
            'image': img_name,
            'recall': recall,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'gt_rooms': gt_rooms
        })

    # Sort by recall and get top 10
    image_stats.sort(key=lambda x: x['recall'], reverse=True)
    top_10 = image_stats[:10]

    # Create PDF
    pdf_path = output_dir / f"top_10_images_{batch_tag}.pdf"

    with PdfPages(str(pdf_path)) as pdf:
        for i, img_stat in enumerate(top_10):
            img_name = img_stat['image']
            img_path = test_images_dir / img_name

            # Load image
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Get predictions for this image
            img_df = df[df['image'] == img_name]

            # Create figure
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            ax.imshow(img)
            ax.axis('off')

            # Draw predictions
            for _, row in img_df.iterrows():
                if row['match_type'] == 'TP':
                    color = 'lime'
                    label = f"TP {row['pred_confidence']:.2f}"
                elif row['match_type'] == 'FP':
                    color = 'red'
                    label = f"FP {row['pred_confidence']:.2f}"
                elif row['match_type'] == 'FN':
                    if pd.notna(row['gt_bbox']):
                        gt_box = eval(row['gt_bbox'])
                        x1, y1, x2, y2 = gt_box
                        for offset in range(0, int(x2-x1), 10):
                            ax.plot([x1+offset, min(x1+offset+5, x2)], [y1, y1], color='yellow', linewidth=2)
                            ax.plot([x1+offset, min(x1+offset+5, x2)], [y2, y2], color='yellow', linewidth=2)
                        for offset in range(0, int(y2-y1), 10):
                            ax.plot([x1, x1], [y1+offset, min(y1+offset+5, y2)], color='yellow', linewidth=2)
                            ax.plot([x2, x2], [y1+offset, min(y1+offset+5, y2)], color='yellow', linewidth=2)
                        ax.text(x1, y1-10, "FN", bbox=dict(facecolor='yellow', alpha=0.8), fontsize=8, color='black')
                    continue
                else:
                    continue

                # Draw TP and FP boxes
                if pd.notna(row['pred_bbox']):
                    pred_box = eval(row['pred_bbox'])
                    x1, y1, x2, y2 = pred_box
                    rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor=color, facecolor='none')
                    ax.add_patch(rect)
                    ax.text(x1, y1-10, label, bbox=dict(facecolor=color, alpha=0.8), fontsize=8, color='white')

            # Add title with metrics
            title = f"Top {i+1}: {img_name}\nRecall: {img_stat['recall']:.1%} | TP: {img_stat['tp']} | FP: {img_stat['fp']} | FN: {img_stat['fn']} | GT Rooms: {img_stat['gt_rooms']}"
            ax.set_title(title, fontsize=12, pad=20)

            # Add legend
            legend_text = "Legend:\n• Green: True Positives\n• Red: False Positives\n• Yellow Dotted: False Negatives"
            ax.text(0.02, 0.98, legend_text, transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))

            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight', dpi=150)
            plt.close()

    return pdf_path

# === MAIN PROCESSING LOOP ===
for model_path in model_paths:
    model_name = model_path.parent.name
    print(f"\nEvaluating {model_name}")

    # Output files for this model
    csv_path = output_dir / f"predictions_{model_name}.csv"
    video_path = output_dir / f"visualization_{model_name}.avi"

    # Skip if already processed
    if csv_path.exists() and video_path.exists():
        print(f"  Skipping {model_name}: output files already exist")
        continue

    # Load DINO model
    try:
        model, postprocessors, args = load_dino_model(model_path)
    except Exception as e:
        print(f"  [ERROR] Failed to load model: {e}")
        continue
    
    # Get test images list
    test_images = list(test_images_dir.glob("*.png"))
    
    # Determine class mapping (assume 2-class: 0=wall, 1=room)
    num_classes = 2  # DINO config
    room_class_id = 1  # Room is class 1
    
    print(f"  Model loaded: {model_name} (DINO Transformer)")
    print(f"  Classes: 2 (evaluating class {room_class_id} = room)")

    rows = []
    vw = None

    # Process all test images
    print(f"  Processing {len(test_images)} test images...")

    for img_idx, img_path in enumerate(test_images):
        if (img_idx + 1) % 50 == 0 or img_idx == 0:
            print(f"    Processed {img_idx+1}/{len(test_images)} images...")

        img_name = img_path.name

        # Run DINO inference
        try:
            img_tensor, orig_size, orig_img = preprocess_image(img_path, args.device)
            scores, labels, boxes = run_dino_inference(model, img_tensor, orig_size, postprocessors, args)
        except Exception as e:
            print(f"    [WARN] Failed to process {img_name}: {e}")
            continue

        # Original image dimensions and area
        ih, iw = orig_img.shape[:2]
        img_area = iw * ih

        # Initialize video writer on first frame
        if vw is None:
            vw = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (VID_W, VID_H))
            if not vw.isOpened():
                raise RuntimeError(f"VideoWriter failed to open: {video_path}")

        # Filter predictions by confidence and class (room only)
        preds = []
        for score, label, box in zip(scores, labels, boxes):
            if score >= conf_threshold and label == room_class_id:
                preds.append((box, score.item()))

        # Load ground truth boxes (YOLO format)
        gt_boxes = []
        label_path = test_labels_dir / f"{img_path.stem}.txt"
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        x_center, y_center, width, height = map(float, parts[1:5])
                        x1 = (x_center - width/2) * iw
                        y1 = (y_center - height/2) * ih
                        x2 = (x_center + width/2) * iw
                        y2 = (y_center + height/2) * ih
                        gt_boxes.append(torch.tensor([x1, y1, x2, y2], dtype=torch.float32))

        # Match predictions to ground truth (same logic as YOLO script)
        matched_gt = set()
        tp_boxes = []
        temp_tp_boxes = []
        temp_fp_boxes = []

        # Sort predictions by confidence
        preds.sort(key=lambda p: p[1], reverse=True)

        for box, conf in preds:
            best_iou = 0
            best_gt_idx = -1

            for gt_idx, gt_box in enumerate(gt_boxes):
                if gt_idx in matched_gt:
                    continue
                iou = iou_tensor(box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou >= IoU_TH and best_gt_idx != -1:
                matched_gt.add(best_gt_idx)
                temp_tp_boxes.append((box, conf, best_gt_idx, best_iou))
            else:
                temp_fp_boxes.append((box, conf))

        # === IoU OVERLAP SUPPRESSION (same as YOLO script) ===
        temp_tp_boxes.sort(key=lambda x: x[1], reverse=True)

        for i, (box, conf, gt_idx, iou_val) in enumerate(temp_tp_boxes):
            should_keep = True
            for kept_box, _, _, _ in tp_boxes:
                if iou_tensor(box, kept_box) > 0.20:
                    should_keep = False
                    break
            if should_keep:
                tp_boxes.append((box, conf, gt_idx, iou_val))

        # Suppress FPs overlapping with TPs
        filtered_fp_boxes = []
        for fp_box, fp_conf in temp_fp_boxes:
            should_keep = True
            for tp_box, _, _, _ in tp_boxes:
                if iou_tensor(fp_box, tp_box) > 0.10:
                    should_keep = False
                    break
            if should_keep:
                filtered_fp_boxes.append((fp_box, fp_conf))

        # Suppress overlapping FPs
        fp_boxes = []
        filtered_fp_boxes.sort(key=lambda x: x[1], reverse=True)

        for i, (box, conf) in enumerate(filtered_fp_boxes):
            should_keep = True
            for kept_box, _ in fp_boxes:
                if iou_tensor(box, kept_box) > 0.20:
                    should_keep = False
                    break
            if should_keep:
                fp_boxes.append((box, conf))

        # === GENERATE FINAL RESULTS ===
        # Add TP results
        for box, conf, gt_idx, iou_val in tp_boxes:
            bx = metrics(box.tolist(), img_area)
            gx = metrics(gt_boxes[gt_idx].tolist(), img_area)
            rows.append({
                "image": img_name,
                "pred_confidence": conf,
                "pred_bbox": box.tolist(),
                "gt_bbox": gt_boxes[gt_idx].tolist(),
                "iou": iou_val,
                "match_type": "TP",
                "pred_area": bx[4], "pred_pct": bx[5],
                "gt_area": gx[4], "gt_pct": gx[5]
            })

        # Add FP results
        for box, conf in fp_boxes:
            bx = metrics(box.tolist(), img_area)
            rows.append({
                "image": img_name,
                "pred_confidence": conf,
                "pred_bbox": box.tolist(),
                "gt_bbox": None,
                "iou": 0.0,
                "match_type": "FP",
                "pred_area": bx[4], "pred_pct": bx[5],
                "gt_area": None, "gt_pct": None
            })

        # False Negatives (unmatched ground truth)
        for gt_idx, gt_box in enumerate(gt_boxes):
            if gt_idx not in matched_gt:
                gx = metrics(gt_box.tolist(), img_area)
                rows.append({
                    "image": img_name,
                    "pred_confidence": None,
                    "pred_bbox": None,
                    "gt_bbox": gt_box.tolist(),
                    "iou": 0.0,
                    "match_type": "FN",
                    "pred_area": None, "pred_pct": None,
                    "gt_area": gx[4], "gt_pct": gx[5]
                })

        # === CREATE VISUALIZATION FRAME ===
        frame = cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR)
        frame_out = cv2.resize(frame, (VID_W, VID_H), interpolation=cv2.INTER_LINEAR)

        scale_x = VID_W / iw
        scale_y = VID_H / ih

        # Draw True Positives (Green)
        for box, conf, _, _ in tp_boxes:
            x1, y1, x2, y2 = box.tolist()
            X1, Y1 = int(x1 * scale_x), int(y1 * scale_y)
            X2, Y2 = int(x2 * scale_x), int(y2 * scale_y)
            cv2.rectangle(frame_out, (X1, Y1), (X2, Y2), COLORS["TP"], THICK)
            cv2.putText(frame_out, f"TP {conf:.2f}", (X1, max(Y1-5, 15)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLORS["TP"], 1)

        # Draw False Positives (Red)
        for box, conf in fp_boxes:
            x1, y1, x2, y2 = box.tolist()
            X1, Y1 = int(x1 * scale_x), int(y1 * scale_y)
            X2, Y2 = int(x2 * scale_x), int(y2 * scale_y)
            cv2.rectangle(frame_out, (X1, Y1), (X2, Y2), COLORS["FP"], THICK)
            cv2.putText(frame_out, f"FP {conf:.2f}", (X1, max(Y1-5, 15)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLORS["FP"], 1)

        # Draw False Negatives (Blue dotted)
        for gt_idx, gt_box in enumerate(gt_boxes):
            if gt_idx not in matched_gt:
                x1, y1, x2, y2 = gt_box.tolist()
                X1, Y1 = int(x1 * scale_x), int(y1 * scale_y)
                X2, Y2 = int(x2 * scale_x), int(y2 * scale_y)
                dotted(frame_out, X1, Y1, X2, Y2, COLORS["FN"], THICK)
                cv2.putText(frame_out, "FN", (X1, max(Y1-5, 15)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLORS["FN"], 1)

        # Add image name overlay
        cv2.putText(frame_out, img_name, (10, VID_H-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

        # Write frame to video
        vw.write(frame_out)

    # Save results for this model
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False, float_format='%.6f')
        print(f"  CSV saved: {csv_path.name} ({len(rows)} predictions)")

    if vw:
        vw.release()
        print(f"  Video saved: {video_path.name}")

# === CREATE SUMMARY ANALYSIS ===
print(f"\nCreating summary analysis...")

summary_rows = []

for csv_path in output_dir.glob("predictions_*.csv"):
    model_name = csv_path.stem.replace("predictions_", "")
    df = pd.read_csv(csv_path)

    # Calculate metrics
    tp_count = (df["match_type"] == "TP").sum()
    fp_count = (df["match_type"] == "FP").sum()
    fn_count = (df["match_type"] == "FN").sum()

    precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0
    recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Per-image statistics
    image_stats = []
    for img_name in df["image"].unique():
        img_df = df[df["image"] == img_name]
        img_tp = (img_df["match_type"] == "TP").sum()
        img_fn = (img_df["match_type"] == "FN").sum()
        img_recall = img_tp / (img_tp + img_fn) if (img_tp + img_fn) > 0 else 0
        image_stats.append(img_recall)

    avg_image_recall = np.mean(image_stats) if image_stats else 0
    median_image_recall = np.median(image_stats) if image_stats else 0

    summary_rows.append({
        "model": model_name,
        "total_images": df["image"].nunique(),
        "TP": tp_count,
        "FP": fp_count,
        "FN": fn_count,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "avg_image_recall": avg_image_recall,
        "median_image_recall": median_image_recall
    })
    
    # === PER-MODEL RECALL DISTRIBUTION CHART ===
    if image_stats:
        plt.figure(figsize=(10, 6))
        plt.hist(image_stats, bins=20, alpha=0.7, edgecolor='black')
        plt.axvline(avg_image_recall, color='red', linestyle='--', label=f'Mean: {avg_image_recall:.3f}')
        plt.axvline(median_image_recall, color='orange', linestyle='--', label=f'Median: {median_image_recall:.3f}')
        plt.xlabel('Per-Image Recall')
        plt.ylabel('Number of Images')
        plt.title(f'Per-Image Recall Distribution - {model_name} (DINO)')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        recall_plot = output_dir / f"recall_distribution_{model_name}.png"
        plt.savefig(recall_plot, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Recall distribution saved: {recall_plot.name}")
    
    # === PER-MODEL TOP 10 PDF ===
    try:
        pdf_path = create_top_10_pdf(model_name, df, output_dir, f"{model_name}")
        print(f"  Top 10 PDF saved: {pdf_path.name}")
    except Exception as e:
        print(f"  PDF generation failed for {model_name}: {e}")

# Save summary
summary_df = pd.DataFrame(summary_rows)
summary_csv = output_dir / f"summary_{batch_tag}.csv"
summary_df.to_csv(summary_csv, index=False, float_format='%.4f')

print(f"Summary saved: {summary_csv.name}")

# === CREATE VISUALIZATIONS ===
if len(summary_df) > 0:
    plt.figure(figsize=(12, 8))

    models = summary_df["model"]
    x = np.arange(len(models))
    width = 0.25

    plt.bar(x - width, summary_df["precision"], width, label='Precision', alpha=0.8)
    plt.bar(x, summary_df["recall"], width, label='Recall', alpha=0.8)
    plt.bar(x + width, summary_df["f1"], width, label='F1 Score', alpha=0.8)

    plt.xlabel('Models')
    plt.ylabel('Score')
    plt.title(f'DINO Performance Comparison - {batch_tag}')
    plt.xticks(x, models, rotation=45, ha='right')
    plt.legend()
    plt.ylim(0, 1.0)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    perf_plot = output_dir / f"performance_comparison_{batch_tag}.png"
    plt.savefig(perf_plot, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Performance plot saved: {perf_plot.name}")

# Print final summary
print(f"\nTesting completed in {time.time() - start_time:.1f} seconds!")
print(f"Results saved to: {output_dir}")
print(f"\nSUMMARY:")
print(summary_df.to_string(index=False, float_format='%.3f'))

# Show top performing models
if len(summary_df) > 0:
    print(f"\nTOP PERFORMING MODELS:")
    top_models = summary_df.sort_values("f1", ascending=False).head(3)
    for _, row in top_models.iterrows():
        print(f"  {row['model']}: F1={row['f1']:.3f}, Precision={row['precision']:.3f}, Recall={row['recall']:.3f}")

print(f"\n[INFO] DINO model evaluation complete!")
print(f"[INFO] Compare results with YOLO models from comprehensive_test.py")

