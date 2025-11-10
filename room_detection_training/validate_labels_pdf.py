#!/usr/bin/env python3
"""
Validate YOLO Labels by Overlaying on Images
Creates a PDF with 10 random samples to verify labels are correct
"""
import argparse
from pathlib import Path
import random
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as patches
import numpy as np

def read_yolo_label(label_path):
    """
    Read YOLO label file (detection or segmentation format)
    Returns list of (class_id, coords) tuples
    """
    labels = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            class_id = int(parts[0])
            coords = [float(x) for x in parts[1:]]
            labels.append((class_id, coords))
    return labels

def is_segmentation_label(coords):
    """Check if label is segmentation (>4 coords) or detection (4 coords)"""
    return len(coords) > 4

def draw_bbox_on_image(ax, img_array, class_id, coords, class_names):
    """Draw bounding box on image"""
    h, w = img_array.shape[:2]
    
    # YOLO bbox format: x_center, y_center, width, height (normalized)
    x_center, y_center, width, height = coords[:4]
    
    # Convert to pixel coordinates
    x_center_px = x_center * w
    y_center_px = y_center * h
    width_px = width * w
    height_px = height * h
    
    # Top-left corner
    x1 = x_center_px - width_px / 2
    y1 = y_center_px - height_px / 2
    
    # Create rectangle
    rect = patches.Rectangle(
        (x1, y1), width_px, height_px,
        linewidth=2, edgecolor='red', facecolor='none'
    )
    ax.add_patch(rect)
    
    # Add label
    class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
    ax.text(x1, y1 - 5, class_name, 
            color='red', fontsize=8, 
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='red'))

def draw_polygon_on_image(ax, img_array, class_id, coords, class_names):
    """Draw segmentation polygon on image"""
    h, w = img_array.shape[:2]
    
    # YOLO seg format: x1 y1 x2 y2 ... (normalized)
    points = []
    for i in range(0, len(coords), 2):
        x = coords[i] * w
        y = coords[i+1] * h
        points.append([x, y])
    
    # Create polygon
    polygon = patches.Polygon(
        points, closed=True,
        linewidth=2, edgecolor='blue', facecolor='blue', alpha=0.3
    )
    ax.add_patch(polygon)

def create_validation_pdf(dataset_dir, output_pdf, split='val', num_samples=10):
    """
    Create PDF with sample images and their labels overlaid
    """
    dataset_path = Path(dataset_dir)
    
    # Read class names from data.yaml
    yaml_path = dataset_path / 'data.yaml'
    class_names = ['room']  # Default
    if yaml_path.exists():
        with open(yaml_path, 'r') as f:
            for line in f:
                if line.strip().startswith('names:'):
                    # Parse: names: ['room'] or names: ['wall', 'room']
                    names_str = line.split('names:')[1].strip()
                    class_names = eval(names_str)
                    break
    
    # Get all images from split
    images_dir = dataset_path / split / 'images'
    labels_dir = dataset_path / split / 'labels'
    
    if not images_dir.exists():
        print(f"Error: Images directory not found: {images_dir}")
        return
    
    image_files = list(images_dir.glob('*.png')) + list(images_dir.glob('*.jpg'))
    
    if not image_files:
        print(f"Error: No images found in {images_dir}")
        return
    
    # Sample random images
    num_samples = min(num_samples, len(image_files))
    sample_images = random.sample(image_files, num_samples)
    
    print(f"\n{'='*60}")
    print(f"VALIDATING YOLO LABELS")
    print(f"{'='*60}")
    print(f"Dataset: {dataset_path}")
    print(f"Split: {split}")
    print(f"Class names: {class_names}")
    print(f"Total images: {len(image_files)}")
    print(f"Sampling: {num_samples} random images")
    print(f"Output PDF: {output_pdf}")
    print(f"{'='*60}\n")
    
    # Create PDF
    with PdfPages(output_pdf) as pdf:
        for idx, img_path in enumerate(sample_images, 1):
            # Load image
            img = Image.open(img_path)
            img_array = np.array(img)
            
            # Load corresponding label
            label_path = labels_dir / (img_path.stem + '.txt')
            
            if not label_path.exists():
                print(f"[WARN] No label found for {img_path.name}")
                continue
            
            labels = read_yolo_label(label_path)
            
            # Determine format
            if labels:
                first_coords = labels[0][1]
                is_seg = is_segmentation_label(first_coords)
                format_type = "SEGMENTATION" if is_seg else "DETECTION"
            else:
                format_type = "EMPTY"
            
            # Create figure
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            ax.imshow(img_array)
            ax.axis('off')
            
            # Draw labels
            for class_id, coords in labels:
                if is_segmentation_label(coords):
                    draw_polygon_on_image(ax, img_array, class_id, coords, class_names)
                else:
                    draw_bbox_on_image(ax, img_array, class_id, coords, class_names)
            
            # Add title with info
            title = f"Sample {idx}/{num_samples}: {img_path.name}\n"
            title += f"Format: {format_type} | Labels: {len(labels)} | "
            title += f"Image size: {img_array.shape[1]}x{img_array.shape[0]}"
            ax.set_title(title, fontsize=12, fontweight='bold')
            
            # Add label file hash for duplicate detection
            import hashlib
            with open(label_path, 'rb') as f:
                label_hash = hashlib.md5(f.read()).hexdigest()[:8]
            
            fig.text(0.99, 0.01, f"Label hash: {label_hash}", 
                    ha='right', va='bottom', fontsize=8, color='gray')
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
            print(f"  [{idx}/{num_samples}] {img_path.name} - {len(labels)} labels ({format_type}) - hash: {label_hash}")
    
    print(f"\n[OK] Validation PDF created: {output_pdf}")
    print(f"{'='*60}\n")

def main():
    parser = argparse.ArgumentParser(
        description='Validate YOLO labels by creating a PDF with sample images'
    )
    parser.add_argument('--dataset', required=True, help='Path to YOLO dataset directory')
    parser.add_argument('--split', default='val', choices=['train', 'val', 'test'],
                       help='Which split to sample from')
    parser.add_argument('--num-samples', type=int, default=10,
                       help='Number of random samples to include')
    parser.add_argument('--output', default='label_validation.pdf',
                       help='Output PDF filename')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducible sampling')
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    create_validation_pdf(
        args.dataset,
        args.output,
        split=args.split,
        num_samples=args.num_samples
    )

if __name__ == '__main__':
    main()

