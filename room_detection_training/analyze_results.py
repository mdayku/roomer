#!/usr/bin/env python3
"""
Analyze the comprehensive test results
"""

import pandas as pd
import numpy as np

# Load the predictions
df = pd.read_csv('room_detection_test_20251107_154615/predictions_room_detection_v2.csv')

# Calculate per-image metrics
image_stats = []
for img_name in df['image'].unique():
    img_df = df[df['image'] == img_name]
    tp = (img_df['match_type'] == 'TP').sum()
    fn = (img_df['match_type'] == 'FN').sum()
    fp = (img_df['match_type'] == 'FP').sum()
    total_pred = tp + fp

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    gt_boxes = len(img_df[img_df['match_type'] == 'FN']) + tp

    image_stats.append({
        'image': img_name,
        'recall': recall,
        'precision': precision,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'gt_rooms': gt_boxes,
        'detected': total_pred
    })

# Sort by recall and show top 10
image_stats.sort(key=lambda x: x['recall'], reverse=True)
print('TOP 10 IMAGES BY RECALL (room_detection_v2 model):')
print('=' * 70)
print('Rank Image Name                 Recall Prec  F1    TP  GT  Det')
print('-' * 70)
for i, stat in enumerate(image_stats[:10], 1):
    print(f'{i:<4} {stat["image"]:<25} {stat["recall"]:<6.3f} {stat["precision"]:<6.3f} {stat["f1"]:<6.3f} {stat["tp"]:<3} {stat["gt_rooms"]:<3} {stat["detected"]:<3}')

print()
print('OVERALL MODEL PERFORMANCE:')
print(f'Total Images: {len(image_stats)}')
print(f'Average Recall: {np.mean([s["recall"] for s in image_stats]):.3f}')
print(f'Average Precision: {np.mean([s["precision"] for s in image_stats]):.3f}')
print(f'Average F1: {np.mean([s["f1"] for s in image_stats]):.3f}')

# Show the worst performing images too
print()
print('BOTTOM 10 IMAGES BY RECALL:')
print('-' * 70)
image_stats.sort(key=lambda x: x['recall'])
for i, stat in enumerate(image_stats[:10], 1):
    print(f'{i:<4} {stat["image"]:<25} {stat["recall"]:<6.3f} {stat["precision"]:<6.3f} {stat["f1"]:<6.3f} {stat["tp"]:<3} {stat["gt_rooms"]:<3} {stat["detected"]:<3}')
