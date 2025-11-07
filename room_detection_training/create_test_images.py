#!/usr/bin/env python3
"""
Create placeholder images for YOLO training testing
"""

import os
from pathlib import Path
from PIL import Image, ImageDraw
import random

def create_placeholder_image(width=640, height=480, filename="placeholder.png"):
    """Create a simple placeholder image with some basic shapes"""
    # Create a white background
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)

    # Add some random rectangles to simulate rooms
    for _ in range(random.randint(1, 3)):
        x1 = random.randint(10, width-100)
        y1 = random.randint(10, height-100)
        x2 = x1 + random.randint(50, width-x1-10)
        y2 = y1 + random.randint(50, height-y1-10)

        # Random color
        color = (random.randint(200, 255), random.randint(200, 255), random.randint(200, 255))
        draw.rectangle([x1, y1, x2, y2], fill=color, outline='black', width=2)

    img.save(filename)
    print(f"Created {filename}")

def main():
    base_dir = Path("./yolo_data")

    # Create placeholder images for training
    train_images_dir = base_dir / "train" / "images"
    train_images_dir.mkdir(parents=True, exist_ok=True)

    print("Creating training placeholder images...")
    for i in range(100):  # Create 100 training images
        filename = train_images_dir / f"{i+1:04d}.png"
        create_placeholder_image(filename=filename)

    # Create placeholder images for validation
    val_images_dir = base_dir / "val" / "images"
    val_images_dir.mkdir(parents=True, exist_ok=True)

    print("Creating validation placeholder images...")
    for i in range(20):  # Create 20 validation images
        filename = val_images_dir / f"{i+1:04d}.png"
        create_placeholder_image(filename=filename)

    print("Placeholder images created!")

if __name__ == "__main__":
    main()
