#!/usr/bin/env python3
"""
Generate PDF with bottom 10 images by recall showing predictions
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as patches
from pathlib import Path

# Dataset paths
test_images_dir = Path("yolo_data/test/images")

def main():
    # Find the most recent test results
    results_dirs = list(Path(".").glob("room_detection_test_*"))
    if not results_dirs:
        print("No test results found!")
        return

    latest_dir = max(results_dirs, key=lambda x: x.stat().st_mtime)
    print(f"Using results from: {latest_dir}")

    # Load predictions
    csv_files = list(latest_dir.glob("predictions_*.csv"))
    if not csv_files:
        print("No prediction files found!")
        return

    csv_path = csv_files[0]  # Use first CSV file
    print(f"Loading predictions from: {csv_path}")

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

    # Sort by recall (ascending) and get bottom 10
    image_stats.sort(key=lambda x: x['recall'])
    bottom_10 = image_stats[:10]

    # Create PDF
    pdf_path = latest_dir / f"bottom_10_images_{latest_dir.name}.pdf"
    print(f"Creating PDF: {pdf_path}")

    with PdfPages(str(pdf_path)) as pdf:
        for i, img_stat in enumerate(bottom_10):
            img_name = img_stat['image']
            img_path = test_images_dir / img_name

            if not img_path.exists():
                print(f"Warning: Image not found: {img_path}")
                continue

            # Load image
            import cv2
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
                    # Draw FN as dotted rectangle
                    if pd.notna(row['gt_bbox']):
                        gt_box = eval(row['gt_bbox'])
                        x1, y1, x2, y2 = gt_box
                        # Draw dotted rectangle for FN (orange for better contrast)
                        for offset in range(0, int(x2-x1), 10):
                            ax.plot([x1+offset, min(x1+offset+5, x2)], [y1, y1], color='orange', linewidth=2)
                            ax.plot([x1+offset, min(x1+offset+5, x2)], [y2, y2], color='orange', linewidth=2)
                        for offset in range(0, int(y2-y1), 10):
                            ax.plot([x1, x1], [y1+offset, min(y1+offset+5, y2)], color='orange', linewidth=2)
                            ax.plot([x2, x2], [y1+offset, min(y1+offset+5, y2)], color='orange', linewidth=2)
                        ax.text(x1, y1-10, "FN", bbox=dict(facecolor='orange', alpha=0.8), fontsize=8, color='black')
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
            title = f"Bottom {i+1}: {img_name}\nRecall: {img_stat['recall']:.1%} | TP: {img_stat['tp']} | FP: {img_stat['fp']} | FN: {img_stat['fn']} | GT Rooms: {img_stat['gt_rooms']}"
            ax.set_title(title, fontsize=12, pad=20)

            # Add legend (moved to bottom-right, smaller text)
            legend_text = "Legend:\n• Green: True Positives\n• Red: False Positives\n• Orange Dotted: False Negatives"
            ax.text(0.98, 0.02, legend_text, transform=ax.transAxes, fontsize=8,
                   verticalalignment='bottom', horizontalalignment='right',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))

            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight', dpi=150)
            plt.close()
            print(f"  Added page {i+1}/10: {img_name}")

    print(f"PDF created successfully: {pdf_path}")

if __name__ == "__main__":
    main()





