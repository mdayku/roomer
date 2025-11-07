"""
Main training script for Room Detection Mask R-CNN
"""

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
import time
import os
from tqdm import tqdm
import json
from datetime import datetime

from config import *
import math
from dataset import RoomDetectionDataset, get_transforms, collate_fn
from model import get_model, get_optimizer_and_scheduler, save_model
from pycocotools.cocoeval import COCOeval


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10):
    """Train for one epoch"""
    model.train()
    metric_logger = {
        'loss': [],
        'loss_classifier': [],
        'loss_box_reg': [],
        'loss_mask': [],
        'loss_objectness': [],
        'loss_rpn_box_reg': []
    }

    header = f'Epoch: [{epoch}]'

    for batch_idx, (images, targets) in enumerate(tqdm(data_loader, desc=header)):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Forward pass
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # Backward pass
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        # Log losses
        for key, value in loss_dict.items():
            if key in metric_logger:
                metric_logger[key].append(value.item())
        metric_logger['loss'].append(losses.item())

        # Print progress
        if batch_idx % print_freq == 0:
            current_losses = {k: np.mean(v[-print_freq:]) for k, v in metric_logger.items()}
            tqdm.write(f"{header} Batch {batch_idx}/{len(data_loader)} "
                      f"Loss: {current_losses['loss']:.4f} "
                      f"Classifier: {current_losses['loss_classifier']:.4f} "
                      f"Mask: {current_losses['loss_mask']:.4f}")

    # Return average losses
    return {k: np.mean(v) for k, v in metric_logger.items()}


def evaluate(model, data_loader, device):
    """Evaluate model on validation set"""
    model.eval()
    coco_results = []

    print("Running evaluation...")
    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Evaluating"):
            images = [img.to(device) for img in images]

            # Get predictions
            predictions = model(images)

            # Convert predictions to COCO format
            for prediction, target in zip(predictions, targets):
                image_id = target['image_id'].item()

                # Filter by confidence threshold
                keep = prediction['scores'] > CONFIDENCE_THRESHOLD
                boxes = prediction['boxes'][keep].cpu().numpy()
                scores = prediction['scores'][keep].cpu().numpy()
                labels = prediction['labels'][keep].cpu().numpy()
                masks = prediction['masks'][keep].cpu().numpy()

                # Convert to COCO format
                for box, score, label, mask in zip(boxes, scores, labels, masks):
                    # Convert mask to RLE
                    mask_binary = (mask > 0.5).astype(np.uint8)
                    rle = mask_to_rle(mask_binary)

                    result = {
                        'image_id': image_id,
                        'category_id': int(label),
                        'bbox': [float(box[0]), float(box[1]),
                               float(box[2] - box[0]), float(box[3] - box[1])],
                        'score': float(score),
                        'segmentation': rle
                    }
                    coco_results.append(result)

    return coco_results


def mask_to_rle(mask):
    """Convert binary mask to RLE format"""
    # Simple RLE implementation for evaluation
    # In production, use pycocotools encode
    flat = mask.flatten()
    diff = np.diff(np.concatenate(([0], flat, [0])))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    lengths = ends - starts

    rle = {'counts': [], 'size': list(mask.shape)}
    for start, length in zip(starts, lengths):
        rle['counts'].extend([start, length])

    return rle


def save_training_log(log_data, path):
    """Save training log to JSON"""
    with open(path, 'w') as f:
        json.dump(log_data, f, indent=2)


def main():
    print("🚀 Starting Room Detection Training Pipeline")
    print(f"Device: {DEVICE}")
    print(f"Dataset: {DATA_ROOT}")
    print(f"Output: {OUTPUT_DIR}")

    # Set device
    device = torch.device(DEVICE)

    # Create datasets
    print("\n📊 Loading datasets...")
    train_dataset = RoomDetectionDataset(str(TRAIN_ANNOTATIONS), get_transforms(train=True))
    val_dataset = RoomDetectionDataset(str(VAL_ANNOTATIONS), get_transforms(train=False))

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, collate_fn=collate_fn
    )

    print(f"Train: {len(train_dataset)} samples")
    print(f"Val: {len(val_dataset)} samples")

    # Create model
    print("\n🏗️ Creating Mask R-CNN model...")
    model = get_model(num_classes=NUM_CLASSES)
    model.to(device)

    # Create optimizer and scheduler
    optimizer, scheduler = get_optimizer_and_scheduler(
        model, LEARNING_RATE, MOMENTUM, WEIGHT_DECAY, STEP_SIZE, GAMMA
    )

    # Training log
    training_log = {
        'config': {
            'num_classes': NUM_CLASSES,
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'num_epochs': NUM_EPOCHS,
            'device': str(device)
        },
        'epochs': []
    }

    # Training loop with early stopping
    print("\n🎯 Starting training...")
    best_loss = float('inf')
    patience_counter = 0
    early_stopping_triggered = False

    for epoch in range(NUM_EPOCHS):
        epoch_start = time.time()

        # Train
        train_losses = train_one_epoch(model, optimizer, train_loader, device, epoch, PRINT_FREQ)

        # Step scheduler
        scheduler.step()

        # Evaluate
        if epoch % 2 == 0 or epoch == NUM_EPOCHS - 1:  # Evaluate every 2 epochs
            val_results = evaluate(model, val_loader, device)
            # In production, compute mAP here
            print(f"Validation completed: {len(val_results)} predictions")

        epoch_time = time.time() - epoch_start

        # Log progress
        epoch_log = {
            'epoch': epoch + 1,
            'train_losses': train_losses,
            'learning_rate': optimizer.param_groups[0]['lr'],
            'epoch_time': epoch_time
        }
        training_log['epochs'].append(epoch_log)

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} completed in {epoch_time:.1f}s")
        print(f"  Train Loss: {train_losses['loss']:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Save checkpoint
        if (epoch + 1) % SAVE_FREQ == 0:
            checkpoint_path = MODEL_DIR / f'model_epoch_{epoch+1}.pth'
            save_model(model, optimizer, scheduler, epoch+1, train_losses['loss'], checkpoint_path)

        # Early stopping logic
        current_loss = train_losses['loss']
        if current_loss < (best_loss - MIN_DELTA):
            best_loss = current_loss
            patience_counter = 0
            best_model_path = MODEL_DIR / 'best_model.pth'
            save_model(model, optimizer, scheduler, epoch+1, best_loss, best_model_path)
            print(f"  ✅ New best model saved (loss: {best_loss:.4f})")
        else:
            patience_counter += 1
            print(f"  ⏸️ No improvement for {patience_counter}/{PATIENCE} epochs")

        # Check early stopping
        if patience_counter >= PATIENCE:
            print(f"  🛑 Early stopping triggered after {epoch+1} epochs")
            early_stopping_triggered = True
            break

    # Save final model
    actual_epochs = epoch + 1  # Account for 0-indexing
    final_model_path = MODEL_DIR / 'final_model.pth'
    save_model(model, optimizer, scheduler, actual_epochs, train_losses['loss'], final_model_path)

    # Update training log
    training_log['early_stopping'] = early_stopping_triggered
    training_log['actual_epochs'] = actual_epochs
    training_log['best_loss'] = best_loss

    # Save training log
    log_path = LOG_DIR / 'training_log.json'
    save_training_log(training_log, log_path)

    print("\n🎉 Training completed!")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Epochs trained: {actual_epochs}/{NUM_EPOCHS}")
    print(f"Early stopping: {'Yes' if early_stopping_triggered else 'No'}")
    print(f"Models saved to: {MODEL_DIR}")
    print(f"Training log: {log_path}")


if __name__ == "__main__":
    main()
