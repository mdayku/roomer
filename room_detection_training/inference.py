"""
Inference script for testing trained Room Detection model
"""

import torch
import numpy as np
import cv2
import json
import time
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon

from config import *
from model import get_model


class RoomDetector:
    """Room Detection Inference Class"""

    def __init__(self, model_path=None, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model(model_path)
        self.confidence_threshold = CONFIDENCE_THRESHOLD

    def load_model(self, model_path):
        """Load trained model"""
        model = get_model(num_classes=NUM_CLASSES, pretrained=False)
        if model_path and Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model from {model_path}")
        else:
            print("Warning: No model path provided, using untrained model")

        model.to(self.device)
        model.eval()
        return model

    def predict(self, image, confidence_threshold=None):
        """
        Run inference on a single image

        Args:
            image: PIL Image or numpy array (H, W, 3) or tensor (C, H, W)
            confidence_threshold: Override default confidence threshold

        Returns:
            dict: Predictions with boxes, masks, scores, labels
        """
        if confidence_threshold is None:
            confidence_threshold = self.confidence_threshold

        # Preprocess image
        if isinstance(image, Image.Image):
            # Convert PIL to tensor
            transform = torch.nn.Sequential(
                torch.nn.ToTensor(),
                torch.nn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            )
            image_tensor = transform(image).unsqueeze(0)
        elif isinstance(image, np.ndarray):
            # Convert numpy to tensor
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            # Normalize
            image_tensor = torch.nn.functional.normalize(image_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            image_tensor = image_tensor.unsqueeze(0)
        elif isinstance(image, torch.Tensor):
            image_tensor = image.unsqueeze(0) if image.dim() == 3 else image
        else:
            raise ValueError("Unsupported image type")

        image_tensor = image_tensor.to(self.device)

        # Run inference
        with torch.no_grad():
            start_time = time.time()
            predictions = self.model(image_tensor)
            inference_time = time.time() - start_time

        prediction = predictions[0]  # Batch size 1

        # Filter by confidence
        keep = prediction['scores'] > confidence_threshold
        filtered_prediction = {
            'boxes': prediction['boxes'][keep].cpu().numpy(),
            'masks': prediction['masks'][keep].cpu().numpy(),
            'scores': prediction['scores'][keep].cpu().numpy(),
            'labels': prediction['labels'][keep].cpu().numpy(),
            'inference_time': inference_time
        }

        return filtered_prediction

    def predict_batch(self, images, confidence_threshold=None):
        """Run inference on multiple images"""
        results = []
        for image in images:
            result = self.predict(image, confidence_threshold)
            results.append(result)
        return results

    def visualize_prediction(self, image, prediction, save_path=None):
        """
        Visualize prediction results

        Args:
            image: Original image (PIL or numpy)
            prediction: Model prediction dict
            save_path: Path to save visualization
        """
        if isinstance(image, torch.Tensor):
            # Convert tensor back to numpy
            image = image.cpu().numpy().transpose(1, 2, 0)
            image = (image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255
            image = np.clip(image, 0, 255).astype(np.uint8)

        fig, ax = plt.subplots(1, figsize=(12, 8))
        ax.imshow(image)

        # Plot each detected room
        for i, (box, mask, score, label) in enumerate(zip(
            prediction['boxes'], prediction['masks'],
            prediction['scores'], prediction['labels']
        )):
            # Bounding box
            x1, y1, x2, y2 = box
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                   linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)

            # Mask contour
            mask_binary = (mask > 0.5).squeeze()
            contours, _ = cv2.findContours(mask_binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                if len(contour) > 2:  # Need at least 3 points for polygon
                    # Scale contour coordinates
                    contour_scaled = contour.squeeze() * (mask.shape[1] / mask_binary.shape[1])
                    poly = Polygon(contour_scaled, facecolor='red', alpha=0.3, edgecolor='red', linewidth=2)
                    ax.add_patch(poly)

            # Label
            ax.text(x1, y1-5, f'Room {i+1} ({score:.2f})',
                   bbox=dict(facecolor='red', alpha=0.7), fontsize=10, color='white')

        ax.set_title(f'Room Detection Results ({len(prediction["boxes"])} rooms detected)')
        ax.axis('off')

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            print(f"Visualization saved to {save_path}")
        else:
            plt.show()

        plt.close()


def benchmark_inference(detector, test_images, num_runs=10):
    """Benchmark inference performance"""
    print(f"Benchmarking inference on {len(test_images)} images...")

    times = []
    for _ in range(num_runs):
        start_time = time.time()
        for image in test_images:
            _ = detector.predict(image)
        end_time = time.time()
        times.append((end_time - start_time) / len(test_images))

    avg_time = np.mean(times) * 1000  # Convert to milliseconds
    std_time = np.std(times) * 1000

    print(".2f")
    print(".2f")
    print(f"Target: <{TARGET_LATENCY_MS}ms per image")

    if avg_time < TARGET_LATENCY_MS:
        print("✅ Meets latency requirement!")
    else:
        print(f"❌ Exceeds latency requirement by {(avg_time - TARGET_LATENCY_MS):.0f}ms")

    return avg_time, std_time


def main():
    """Main inference script"""
    print("🏠 Room Detection Inference")

    # Initialize detector
    model_path = MODEL_DIR / "best_model.pth"
    detector = RoomDetector(model_path=str(model_path) if model_path.exists() else None)

    # Test on sample data
    print("\n🧪 Running inference test...")

    # Create a dummy test image (representing a blueprint)
    test_image = np.zeros((800, 800, 3), dtype=np.uint8)
    test_image.fill(255)  # White background

    # Add some mock "walls" and "rooms"
    cv2.rectangle(test_image, (100, 100), (300, 200), (200, 200, 200), 2)  # Room 1
    cv2.rectangle(test_image, (400, 150), (600, 350), (200, 200, 200), 2)  # Room 2
    cv2.rectangle(test_image, (200, 400), (500, 600), (200, 200, 200), 2)  # Room 3

    # Run prediction
    prediction = detector.predict(test_image)

    print(f"Detected {len(prediction['boxes'])} rooms")
    print(".1f")

    # Visualize
    vis_path = VISUALIZATION_DIR / "inference_test.png"
    detector.visualize_prediction(test_image, prediction, save_path=str(vis_path))

    # Benchmark performance
    test_images = [test_image] * 5  # Test with 5 identical images
    benchmark_inference(detector, test_images)

    print("\n✅ Inference test completed!")


if __name__ == "__main__":
    main()
