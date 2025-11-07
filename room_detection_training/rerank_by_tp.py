#!/usr/bin/env python3
"""
Rerank the top 50 images by different metrics to show alternatives
"""

import pandas as pd
from pathlib import Path
import json

def load_top_50_data():
    """Load the current top 50 data"""
    report_file = Path("./testing/top_50_test_results_report.txt")

    # Parse the text report to extract data
    data = []
    with open(report_file, 'r') as f:
        lines = f.readlines()

    # Find the data lines (skip headers)
    in_data = False
    for line in lines:
        line = line.strip()

        # Skip header lines
        if not line or line.startswith('=') or line.startswith('TOP') or line.startswith('Metric') or line.startswith('Total') or line.startswith('Rank Image Name') or line.startswith('SUMMARY'):
            continue

        # Skip separator lines
        if line.startswith('-'):
            continue

        # Check if this looks like data (starts with number)
        parts = line.split()
        if parts and parts[0].isdigit():
            try:
                rank = int(parts[0])
                image = parts[1]
                gt = int(parts[2])
                det = int(parts[3])
                recall = float(parts[4])
                tp_conf = float(parts[5])
                score = float(parts[6])

                # Calculate TP count
                tp_count = int(gt * recall)  # TP = recall * (TP + FN) = recall * GT

                data.append({
                    'rank': rank,
                    'image': image,
                    'gt_rooms': gt,
                    'detected': det,
                    'recall': recall,
                    'tp_confidence': tp_conf,
                    'weighted_score': score,
                    'tp_count': tp_count,
                    'precision': tp_count / det if det > 0 else 0
                })
            except (ValueError, IndexError):
                continue

    return pd.DataFrame(data)

def rerank_and_save(df, metric, ascending=False, output_name=""):
    """Rerank by metric and save"""
    df_sorted = df.sort_values(metric, ascending=ascending)
    df_sorted['new_rank'] = range(1, len(df_sorted) + 1)

    output_file = Path(f"./testing/top_50_reranked_by_{output_name}.txt")

    with open(output_file, 'w') as f:
        f.write(f"TOP 50 IMAGES - RERANKED BY {output_name.upper()}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Original ranking based on: (TP/(TP+FN)) × mean_TP_confidence\n")
        f.write(f"New ranking based on: {output_name}\n\n")

        f.write("-" * 90 + "\n")
        f.write("New  Orig  Image Name                GT  Det Recall TP Conf TP#  Prec  Score\n")
        f.write("-" * 90 + "\n")

        for _, row in df_sorted.iterrows():
            f.write(f"{row['new_rank']:2d}   {row['rank']:2d}   {row['image']:<25} {row['gt_rooms']:2d}  {row['detected']:2d} {row['recall']:.3f} {row['tp_confidence']:.3f} {row['tp_count']:2d}  {row['precision']:.3f} {row['weighted_score']:.3f}\n")

        f.write("-" * 90 + "\n\n")
        f.write("LEGEND:\n")
        f.write("- New: New ranking by this metric\n")
        f.write("- Orig: Original weighted score ranking\n")
        f.write("- GT: Ground truth rooms\n")
        f.write("- Det: Rooms detected\n")
        f.write("- TP#: True positive count\n")
        f.write("- Prec: Precision (TP/Detected)\n")

    print(f"Saved: {output_file}")
    return df_sorted

def main():
    df = load_top_50_data()
    print(f"Loaded {len(df)} images from top 50")

    # Rerank by different metrics
    rerank_and_save(df, 'tp_count', ascending=False, output_name="tp_count")
    rerank_and_save(df, 'recall', ascending=False, output_name="recall")
    rerank_and_save(df, 'precision', ascending=False, output_name="precision")

    # Show top 5 for each ranking
    print("\n" + "="*60)
    print("TOP 5 BY DIFFERENT METRICS:")
    print("="*60)

    print("\n1. BY TRUE POSITIVE COUNT (raw performance):")
    tp_sorted = df.sort_values('tp_count', ascending=False)
    for i, (_, row) in enumerate(tp_sorted.head(5).iterrows(), 1):
        print(f"{i}. {row['image']} - {row['tp_count']} TPs (GT: {row['gt_rooms']}, Recall: {row['recall']:.1%})")

    print("\n2. BY RECALL (detection rate):")
    recall_sorted = df.sort_values('recall', ascending=False)
    for i, (_, row) in enumerate(recall_sorted.head(5).iterrows(), 1):
        print(f"{i}. {row['image']} - {row['recall']:.1%} recall ({row['tp_count']}/{row['gt_rooms']} TPs)")

    print("\n3. BY PRECISION (accuracy of detections):")
    prec_sorted = df.sort_values('precision', ascending=False)
    for i, (_, row) in enumerate(prec_sorted.head(5).iterrows(), 1):
        print(f"{i}. {row['image']} - {row['precision']:.1%} precision ({row['tp_count']}/{row['detected']} correct detections)")
if __name__ == "__main__":
    main()
