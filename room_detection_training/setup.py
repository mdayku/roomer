"""
Setup script for Room Detection Training Environment
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description):
    """Run shell command with error handling"""
    print(f"📦 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(e.stderr)
        return False

def main():
    print("🚀 Setting up Room Detection Training Environment")

    # Check Python version
    python_version = sys.version_info
    if python_version < (3, 8):
        print(f"❌ Python {python_version.major}.{python_version.minor} detected. Need Python 3.8+")
        sys.exit(1)

    print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.minor} detected")

    # Install requirements
    if not run_command("pip install -r requirements.txt", "Installing Python dependencies"):
        sys.exit(1)

    # Verify key packages
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__} installed")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name(0)}")

        import torchvision
        print(f"✅ TorchVision {torchvision.__version__} installed")

        import pycocotools
        print(f"✅ COCO tools installed")

    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        sys.exit(1)

    # Create necessary directories
    dirs = ["outputs", "outputs/models", "outputs/logs", "outputs/visualizations"]
    for dir_name in dirs:
        Path(dir_name).mkdir(parents=True, exist_ok=True)
        print(f"📁 Created directory: {dir_name}")

    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Ensure COCO dataset is available: ../../room_detection_dataset_coco/")
    print("2. Run training: python train.py")
    print("3. Run inference: python inference.py")

if __name__ == "__main__":
    main()
