
#!/usr/bin/env python3
"""
YOLO Training Script for SageMaker Container
"""

import os
import sys
import argparse

def main():
    print("Room Detection YOLO Training in SageMaker")
    print("=" * 50)

    # Install ultralytics if not available
    try:
        import ultralytics
        print("Ultralytics already installed")
    except ImportError:
        print("Installing ultralytics...")
        import subprocess
        result = subprocess.run([
            sys.executable, "-m", "pip", "install",
            "ultralytics>=8.1.0", "torch", "torchvision", "torchaudio",
            "--quiet", "--no-cache-dir"
        ], capture_output=True, text=True, timeout=300)

        if result.returncode != 0:
            print(f"Installation failed: {result.stderr}")
            return

        print("Ultralytics installed successfully")

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--batch-size', type=int, default=16)  # Alternative format
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--data', type=str, default='data.yaml')
    parser.add_argument('--weights', type=str, default='yolov8s.pt')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--optimizer', type=str, default='AdamW')
    parser.add_argument('--lr0', type=float, default=0.01)
    parser.add_argument('--save_period', type=int, default=10)
    parser.add_argument('--lrf', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.937)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--warmup_epochs', type=float, default=3.0)
    parser.add_argument('--box', type=float, default=7.5)
    parser.add_argument('--cls', type=float, default=0.5)
    parser.add_argument('--dfl', type=float, default=1.5)
    args = parser.parse_args()

    print(f"Training config: {args.epochs} epochs, batch {args.batch_size}, size {args.imgsz}")

    # Import after installation
    from ultralytics import YOLO, settings
    import torch

    # Configure Ultralytics to not download datasets and use our paths
    settings.update({'datasets_dir': '/opt/ml/input/data/training'})
    print(f"Ultralytics settings updated: datasets_dir = {settings['datasets_dir']}")

    # Check GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Load model
    print("\nLoading YOLO detection model...")
    model = YOLO(args.weights)

    # Update data path for SageMaker
    data_path = '/opt/ml/input/data/training/data.yaml'
    if os.path.exists(data_path):
        args.data = data_path
        print(f"Using SageMaker data: {data_path}")
        # Verify the data.yaml can be loaded
        try:
            import yaml
            with open(data_path, 'r') as f:
                data_config = yaml.safe_load(f)
            print(f"Data config loaded: path={data_config.get('path')}, train={data_config.get('train')}, val={data_config.get('val')}")
        except Exception as e:
            print(f"Warning: Could not verify data.yaml: {e}")
    else:
        print("Warning: SageMaker data not found, using local data.yaml")

    # Training
    print("\nStarting training...")
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch_size,
        imgsz=args.imgsz,
        device=device,
        workers=args.workers,
        project='/opt/ml/model',
        name='room_detection',
        save=True,
        save_period=args.save_period,
        patience=args.patience,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        box=args.box,
        cls=args.cls,
        dfl=args.dfl,
        # Removed seg parameter for detection model
    )

    # Save final model
    model_path = '/opt/ml/model/final_model.pt'
    model.save(model_path)
    print(f"\n[OK] Model saved to {model_path}")

    # Export for inference
    onnx_path = '/opt/ml/model/model.onnx'
    model.export(format='onnx', imgsz=args.imgsz)
    print(f"[OK] ONNX model exported to {onnx_path}")

    # Run validation
    print("\nRunning final validation...")
    metrics = model.val()
    print(f"Validation mAP: {metrics.box.map:.4f}")

    print("\nSageMaker training completed!")

if __name__ == "__main__":
    main()
