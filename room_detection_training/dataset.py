"""
COCO Dataset Loader for Room Detection
Handles polygon segmentation masks for instance segmentation
"""

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image
import numpy as np
import json
from pathlib import Path
import cv2
from pycocotools.coco import COCO
from pycocotools.mask import decode as mask_decode
import albumentations as A
from albumentations.pytorch import ToTensorV2


class RoomDetectionDataset(Dataset):
    """
    COCO Dataset for room detection with polygon segmentation
    """

    def __init__(self, annotations_file, transforms=None):
        self.coco = COCO(annotations_file)
        self.image_ids = list(self.coco.imgs.keys())

        # Filter images that have annotations
        self.image_ids = [img_id for img_id in self.image_ids
                         if len(self.coco.getAnnIds(imgIds=img_id)) > 0]

        self.transforms = transforms

        # Get category mapping
        self.cat_ids = self.coco.getCatIds()
        self.cat_names = {cat['id']: cat['name'] for cat in self.coco.cats.values()}

        print(f"Loaded {len(self.image_ids)} images with annotations")
        print(f"Categories: {self.cat_names}")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.coco.loadImgs(image_id)[0]

        # Create a blank image (since we don't have actual image files)
        # Use the dimensions from our COCO data
        width, height = image_info['width'], image_info['height']
        image = np.zeros((height, width, 3), dtype=np.uint8)
        image.fill(255)  # White background

        # Get annotations for this image
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        annotations = self.coco.loadAnns(ann_ids)

        # Convert annotations to target format
        boxes = []
        masks = []
        labels = []

        for ann in annotations:
            # Skip crowd annotations
            if ann.get('iscrowd', 0):
                continue

            # Get bounding box
            bbox = ann['bbox']  # [x, y, width, height]
            boxes.append(bbox)

            # Convert COCO polygon to binary mask
            if 'segmentation' in ann and ann['segmentation']:
                # Handle polygon segmentation
                if isinstance(ann['segmentation'], list) and len(ann['segmentation']) > 0:
                    # Convert polygon to mask
                    rle = self.coco.annToRLE(ann)
                    mask = mask_decode(rle)
                    masks.append(mask)
                    labels.append(ann['category_id'])
                else:
                    # Handle RLE directly
                    mask = mask_decode(ann['segmentation'])
                    masks.append(mask)
                    labels.append(ann['category_id'])

        # Convert to tensors
        if boxes:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            masks = torch.as_tensor(np.array(masks), dtype=torch.uint8)
            image_id = torch.tensor([image_id])
            area = (boxes[:, 3] * boxes[:, 2])  # width * height
            iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

            target = {
                'boxes': boxes,
                'labels': labels,
                'masks': masks,
                'image_id': image_id,
                'area': area,
                'iscrowd': iscrowd
            }
        else:
            # No annotations for this image
            target = {
                'boxes': torch.zeros((0, 4), dtype=torch.float32),
                'labels': torch.zeros((0,), dtype=torch.int64),
                'masks': torch.zeros((0, height, width), dtype=torch.uint8),
                'image_id': torch.tensor([image_id]),
                'area': torch.zeros((0,), dtype=torch.float32),
                'iscrowd': torch.zeros((0,), dtype=torch.int64)
            }

        # Convert image to tensor
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        return image, target


def get_transforms(train=True):
    """Get data augmentation transforms"""
    if train:
        return A.Compose([
            A.Resize(800, 800),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']))
    else:
        return A.Compose([
            A.Resize(800, 800),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']))


def collate_fn(batch):
    """Custom collate function for variable-sized masks"""
    return tuple(zip(*batch))


# Utility functions for data analysis
def analyze_dataset(annotations_file):
    """Analyze dataset statistics"""
    coco = COCO(annotations_file)

    print(f"Dataset Statistics for {annotations_file}:")
    print(f"Images: {len(coco.imgs)}")
    print(f"Annotations: {len(coco.anns)}")

    # Category distribution
    cat_ids = coco.getCatIds()
    for cat_id in cat_ids:
        ann_ids = coco.getAnnIds(catIds=[cat_id])
        print(f"Category {cat_id}: {len(ann_ids)} annotations")

    # Image size distribution
    widths = [img['width'] for img in coco.imgs.values()]
    heights = [img['height'] for img in coco.imgs.values()]
    print(f"Image sizes: {min(widths)}x{min(heights)} to {max(widths)}x{max(heights)}")

    return coco


if __name__ == "__main__":
    # Quick test
    from config import TRAIN_ANNOTATIONS

    print("Testing dataset loading...")
    analyze_dataset(str(TRAIN_ANNOTATIONS))

    # Test data loading
    dataset = RoomDetectionDataset(str(TRAIN_ANNOTATIONS))
    image, target = dataset[0]

    print("
Sample data:")
    print(f"Image shape: {image.shape}")
    print(f"Boxes: {target['boxes'].shape}")
    print(f"Masks: {target['masks'].shape}")
    print(f"Labels: {target['labels']}")
