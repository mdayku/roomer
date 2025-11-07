"""
Test script to validate training environment setup
"""

import sys
import torch
import numpy as np
import json
from pathlib import Path

def test_imports():
    """Test that all required packages can be imported"""
    print("🧪 Testing imports...")

    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")

        import torchvision
        print(f"✅ TorchVision {torchvision.__version__}")

        import cv2
        print(f"✅ OpenCV {cv2.__version__}")

        import numpy as np
        print(f"✅ NumPy {np.__version__}")

        import PIL
        print(f"✅ PIL {PIL.__version__}")

        from pycocotools.coco import COCO
        print("✅ pycocotools")

        import albumentations
        print(f"✅ Albumentations {albumentations.__version__}")

        import tqdm
        print("✅ tqdm")

        import matplotlib
        print(f"✅ Matplotlib {matplotlib.__version__}")

        return True

    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_cuda():
    """Test CUDA availability"""
    print("\n🖥️ Testing CUDA...")

    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        return True
    else:
        print("⚠️ CUDA not available, using CPU")
        return False

def test_data_loading():
    """Test that training data can be loaded"""
    print("\n📊 Testing data loading...")

    data_root = Path("../../room_detection_dataset_coco")
    train_annotations = data_root / "train" / "annotations.json"

    if not train_annotations.exists():
        print(f"❌ Training data not found: {train_annotations}")
        print("   Run data conversion scripts first:")
        print("   pnpm --filter @app/server convert:coco")
        print("   pnpm --filter @app/server prepare-dataset")
        print("   pnpm --filter @app/server convert-to-coco")
        return False

    try:
        with open(train_annotations, 'r') as f:
            data = json.load(f)

        num_images = len(data.get('images', []))
        num_annotations = len(data.get('annotations', []))
        categories = data.get('categories', [])

        print(f"✅ Training data loaded:")
        print(f"   Images: {num_images}")
        print(f"   Annotations: {num_annotations}")
        print(f"   Categories: {[cat['name'] for cat in categories]}")

        return True

    except Exception as e:
        print(f"❌ Failed to load training data: {e}")
        return False

def test_model_creation():
    """Test that Mask R-CNN model can be created"""
    print("\n🏗️ Testing model creation...")

    try:
        from model import get_model

        model = get_model(num_classes=2)  # background + room

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"✅ Model created successfully")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        print(f"   Model size: {total_params * 4 / (1024**2):.1f} MB (FP32)")

        return True

    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return False

def test_dummy_inference():
    """Test that model can run inference on dummy data"""
    print("\n🚀 Testing dummy inference...")

    try:
        from model import get_model

        model = get_model(num_classes=2)
        model.eval()

        # Create dummy image (batch_size=1, channels=3, height=800, width=800)
        dummy_image = torch.randn(1, 3, 800, 800)

        with torch.no_grad():
            predictions = model(dummy_image)

        # Check prediction structure
        pred = predictions[0]
        expected_keys = ['boxes', 'labels', 'scores', 'masks']

        for key in expected_keys:
            if key not in pred:
                print(f"❌ Missing prediction key: {key}")
                return False

        print("✅ Dummy inference successful")
        print(f"   Predicted boxes: {len(pred['boxes'])}")
        print(f"   Predicted masks: {len(pred['masks'])}")
        print(f"   Box tensor shape: {pred['boxes'].shape}")
        print(f"   Mask tensor shape: {pred['masks'].shape}")

        return True

    except Exception as e:
        print(f"❌ Dummy inference failed: {e}")
        return False

def main():
    """Run all environment tests"""
    print("🧪 Room Detection Training Environment Test")
    print("=" * 50)

    tests = [
        test_imports,
        test_cuda,
        test_data_loading,
        test_model_creation,
        test_dummy_inference
    ]

    results = []
    for test in tests:
        results.append(test())

    print("\n" + "=" * 50)
    print("📊 Test Results:")

    passed = sum(results)
    total = len(results)

    for i, (test_func, result) in enumerate(zip(tests, results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_func.__name__}: {status}")

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! Ready to train.")
        print("\nNext steps:")
        print("1. python train.py          # Start training")
        print("2. python inference.py      # Test trained model")
    else:
        print("\n⚠️ Some tests failed. Fix issues before training.")
        sys.exit(1)

if __name__ == "__main__":
    main()
