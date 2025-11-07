"""
Mask R-CNN Model for Room Detection
Based on PyTorch torchvision implementation with custom configuration
"""

import torch
import torchvision
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torch.nn as nn

from config import NUM_CLASSES, BACKBONE, HIDDEN_LAYERS, ANCHOR_SIZES


def get_model(num_classes=NUM_CLASSES, pretrained=True):
    """
    Create Mask R-CNN model with custom configuration for room detection

    Args:
        num_classes: Number of classes (including background)
        pretrained: Whether to use pretrained backbone
    """
    # Create backbone
    backbone = resnet_fpn_backbone(BACKBONE, pretrained=pretrained)

    # Create Mask R-CNN model
    model = MaskRCNN(
        backbone,
        num_classes=num_classes,
        # Custom anchor sizes optimized for room detection
        rpn_anchor_generator=torchvision.models.detection.rpn.AnchorGenerator(
            sizes=ANCHOR_SIZES,
            aspect_ratios=((0.5, 1.0, 2.0),) * len(ANCHOR_SIZES)
        ),
        # Custom ROI heads
        box_roi_pool=torchvision.ops.MultiScaleRoIAlign(
            featmap_names=['0', '1', '2', '3'],
            output_size=7,
            sampling_ratio=2
        ),
        mask_roi_pool=torchvision.ops.MultiScaleRoIAlign(
            featmap_names=['0', '1', '2', '3'],
            output_size=14,
            sampling_ratio=2
        ),
        # Custom head configurations
        box_head=nn.Sequential(
            nn.Linear(1024, HIDDEN_LAYERS),
            nn.ReLU(),
            nn.Linear(HIDDEN_LAYERS, HIDDEN_LAYERS),
            nn.ReLU()
        ),
        box_predictor=FastRCNNPredictor(HIDDEN_LAYERS, num_classes),
        mask_head=nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU()
        ),
        mask_predictor=MaskRCNNPredictor(256, HIDDEN_LAYERS, num_classes)
    )

    return model


def get_optimizer_and_scheduler(model, lr=0.001, momentum=0.9, weight_decay=0.0005, step_size=3, gamma=0.1):
    """
    Create optimizer and learning rate scheduler

    Args:
        model: PyTorch model
        lr: Learning rate
        momentum: SGD momentum
        weight_decay: Weight decay for regularization
        step_size: LR scheduler step size
        gamma: LR decay factor
    """
    # Separate backbone and head parameters for different learning rates
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params = [p for p in model.parameters() if p.requires_grad and p not in backbone_params]

    optimizer = torch.optim.SGD([
        {'params': backbone_params, 'lr': lr * 0.1},  # Lower LR for backbone
        {'params': head_params, 'lr': lr}  # Higher LR for heads
    ], momentum=momentum, weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    return optimizer, scheduler


def save_model(model, optimizer, scheduler, epoch, loss, path):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, path)
    print(f"Model saved to {path}")


def load_model(model, optimizer, scheduler, path, device):
    """Load model checkpoint"""
    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    print(f"Model loaded from {path} (epoch {epoch})")
    return epoch, loss


def calculate_model_size(model):
    """Calculate model size in MB"""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb


def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):
    """Learning rate warmup scheduler"""
    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


if __name__ == "__main__":
    # Test model creation
    print("Creating Mask R-CNN model...")
    model = get_model()
    print(f"Model created with {calculate_model_size(model):.2f} MB parameters")

    # Test optimizer
    optimizer, scheduler = get_optimizer_and_scheduler(model)
    print(f"Optimizer: {type(optimizer).__name__}")
    print(f"Scheduler: {type(scheduler).__name__}")

    # Test on dummy data
    model.eval()
    dummy_image = torch.randn(1, 3, 800, 800)
    with torch.no_grad():
        predictions = model(dummy_image)

    print(f"Dummy prediction keys: {list(predictions[0].keys())}")
    print(f"Boxes shape: {predictions[0]['boxes'].shape}")
    print(f"Masks shape: {predictions[0]['masks'].shape}")
    print("Model test successful!")
