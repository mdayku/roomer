"""
AWS SageMaker Training for YOLO11-seg Room Detection
Cloud training with enterprise GPUs for maximum speed
"""

import boto3
import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker import get_execution_role
import json
from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# SageMaker configuration
INSTANCE_TYPE = 'ml.g5.xlarge'  # A10G GPU - better performance/cost balance
# Alternatives:
# 'ml.g4dn.xlarge' - T4 GPU (cost-effective, our previous choice)
# 'ml.p3.2xlarge' - V100 GPU (faster but more expensive)
# 'ml.p4d.24xlarge' - A100 GPU (fastest, most expensive)

INSTANCE_COUNT = 1
FRAMEWORK_VERSION = '2.0.0'
PYTHON_VERSION = 'py38'

# Training hyperparameters
HYPERPARAMETERS = {
    'epochs': 50,
    'batch_size': 16,  # Larger batch size with cloud GPUs
    'imgsz': 640,
    'data': 'data.yaml',  # Will be created in container
    'weights': 'yolov8s-seg.pt',  # Small model: optimal for AWS deployment
    'device': 0,  # Use GPU
    'workers': 8,
    'optimizer': 'AdamW',
    'lr0': 0.001,
    'lrf': 0.01,
    'momentum': 0.937,
    'weight_decay': 0.0005,
    'warmup_epochs': 3.0,
    'patience': 10,
    'box': 7.5,
    'cls': 0.5,
    'dfl': 1.5,
    'seg': 2.0,  # Higher segmentation loss for our task
}


def upload_data_to_s3(s3_bucket, s3_prefix, local_data_dir):
    """Upload training data to S3"""
    print(f"Uploading data to s3://{s3_bucket}/{s3_prefix}")

    s3_client = boto3.client('s3')

    for root, dirs, files in os.walk(local_data_dir):
        for file in files:
            local_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_path, local_data_dir)
            s3_key = f"{s3_prefix}/{relative_path}"

            print(f"Uploading {relative_path}...")
            s3_client.upload_file(local_path, s3_bucket, s3_key)

    s3_data_path = f"s3://{s3_bucket}/{s3_prefix}"
    print(f"✅ Data uploaded to {s3_data_path}")
    return s3_data_path


def create_data_yaml_for_sagemaker():
    """Create data.yaml for SageMaker container"""
    yaml_content = """
# YOLO Data Configuration for SageMaker
path: /opt/ml/input/data/training  # SageMaker data channel
train: train/images
val: val/images

# Classes
nc: 1
names: ['room']

# Task
task: segment
"""

    # Save locally for upload
    with open('data.yaml', 'w') as f:
        f.write(yaml_content.strip())

    print("✅ Created data.yaml for SageMaker")
    return 'data.yaml'


def train_on_sagemaker():
    """Launch SageMaker training job"""
    print("🚀 Launching YOLO training on AWS SageMaker")
    print(f"Instance: {INSTANCE_TYPE} ({INSTANCE_COUNT}x)")
    print("=" * 60)

    # Get SageMaker session and role
    try:
        role = get_execution_role()
        print(f"[OK] Using IAM role: {role}")
    except:
        print("[ERROR] Could not get SageMaker execution role")
        print("Make sure you're running this in a SageMaker notebook or have proper AWS credentials")
        return

    sess = sagemaker.Session()
    bucket = sess.default_bucket()
    print(f"[OK] Using S3 bucket: {bucket}")

    # Prepare data
    print("\n📊 Preparing training data...")

    # Convert COCO to YOLO format (if not already done)
    yolo_data_dir = Path("./yolo_data")
    if not yolo_data_dir.exists():
        print("Converting COCO data to YOLO format...")
        from train_yolo import convert_coco_to_yolo_format, create_data_yaml

        convert_coco_to_yolo_format(
            "../../room_detection_dataset_coco/train/annotations.json",
            yolo_data_dir / "train"
        )
        convert_coco_to_yolo_format(
            "../../room_detection_dataset_coco/val/annotations.json",
            yolo_data_dir / "val"
        )

    # Create data.yaml
    data_yaml = create_data_yaml_for_sagemaker()

    # Upload data to S3
    s3_data_path = upload_data_to_s3(
        bucket,
        'room-detection-training/data',
        str(yolo_data_dir)
    )

    # Upload YOLO weights (if not using pretrained)
    # For now, we'll download in the container

    # Create PyTorch estimator
    print("\n🏗️ Creating SageMaker estimator...")

    estimator = PyTorch(
        entry_point='train_yolo_sagemaker.py',  # We'll create this
        source_dir='.',  # Current directory
        role=role,
        instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
        framework_version=FRAMEWORK_VERSION,
        py_version=PYTHON_VERSION,
        hyperparameters=HYPERPARAMETERS,
        # Spot training for cost savings (70% discount)
        use_spot_instances=True,
        max_wait=86400,  # 24 hours
        max_run=3600,    # 1 hour training time
        # Enable distributed training if using multiple instances
        distribution={'pytorch': {'enabled': True}} if INSTANCE_COUNT > 1 else None,
    )

    # Launch training
    print("\n🎯 Starting training job...")

    job_name = f'room-detection-yolo-{int(sess._current_time().timestamp())}'

    estimator.fit(
        inputs={'training': s3_data_path},
        job_name=job_name,
        wait=False  # Don't wait for completion
    )

    print(f"✅ Training job launched: {job_name}")
    print(f"Monitor progress at: https://{sess.boto_region_name}.console.aws.amazon.com/sagemaker/home?region={sess.boto_region_name}#/jobs/{job_name}")

    # Estimate costs and time
    hourly_rate = get_instance_price(INSTANCE_TYPE)
    spot_rate = hourly_rate * 0.3  # ~70% spot discount
    estimated_hours = 1.5  # G5 should be faster than T4
    estimated_cost = spot_rate * estimated_hours

    print(f"\n💰 Cost Estimate (G5 A10G GPU - Upgraded!):")
    print(f"- Instance: {INSTANCE_TYPE} (${hourly_rate}/hour → ${spot_rate:.2f}/hour spot)")
    print("- Spot discount: ~70% savings")
    print(f"- Estimated training time: {estimated_hours} hours (faster than T4!)")
    print(f"- Estimated cost: ${estimated_cost:.2f} (with spot pricing)")

    return job_name, estimator


def get_instance_price(instance_type):
    """Get approximate hourly price for instance type"""
    prices = {
        'ml.g4dn.xlarge': 0.736,   # T4 GPU
        'ml.g5.xlarge': 1.206,     # A10G GPU
        'ml.p3.2xlarge': 3.825,    # V100 GPU
        'ml.p4d.24xlarge': 40.96,  # A100 GPU
    }
    return prices.get(instance_type, 'Unknown')


def create_sagemaker_training_script():
    """Create the training script that runs in SageMaker container"""
    script_content = '''
#!/usr/bin/env python3
"""
YOLO Training Script for SageMaker Container
"""

import os
import sys
import argparse
from ultralytics import YOLO
import torch

def main():
    print("🏠 Room Detection YOLO Training in SageMaker")
    print("=" * 50)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--data', type=str, default='data.yaml')
    parser.add_argument('--weights', type=str, default='yolo11s-seg.pt')
    args = parser.parse_args()

    print(f"Training config: {args.epochs} epochs, batch {args.batch_size}, size {args.imgsz}")

    # Check GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Load model
    print("\\n🏗️ Loading YOLO model...")
    model = YOLO(args.weights)

    # Update data path for SageMaker
    data_path = '/opt/ml/input/data/training/data.yaml'
    if os.path.exists(data_path):
        args.data = data_path
        print(f"Using SageMaker data: {data_path}")
    else:
        print("Warning: SageMaker data not found, using local data.yaml")

    # Training
    print("\\n🎯 Starting training...")
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch_size,
        imgsz=args.imgsz,
        device=device,
        workers=8,
        project='/opt/ml/model',
        name='room_detection',
        save=True,
        save_period=10,
        patience=10,
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        seg=2.0,
    )

    # Save final model
    model_path = '/opt/ml/model/final_model.pt'
    model.save(model_path)
    print(f"\\n✅ Model saved to {model_path}")

    # Export for inference
    onnx_path = '/opt/ml/model/model.onnx'
    model.export(format='onnx', imgsz=args.imgsz)
    print(f"✅ ONNX model exported to {onnx_path}")

    # Run validation
    print("\\n🔍 Running final validation...")
    metrics = model.val()
    print(f"Validation mAP: {metrics.box.map:.4f}")
    if hasattr(metrics, 'seg'):
        print(f"Segmentation mAP: {metrics.seg.map:.4f}")

    print("\\n🎉 SageMaker training completed!")

if __name__ == "__main__":
    main()
'''

    with open('train_yolo_sagemaker.py', 'w') as f:
        f.write(script_content)

    print("✅ Created SageMaker training script")


if __name__ == "__main__":
    print("AWS SageMaker YOLO Training Setup")
    print("=" * 40)

    # Check AWS credentials and permissions
    print("Checking AWS credentials...")
    try:
        import boto3
        sts = boto3.client('sts')
        identity = sts.get_caller_identity()
        account_id = identity['Account']
        print(f"✅ AWS Account: {account_id}")

        # Test required services
        sm = boto3.client('sagemaker', region_name=REGION)
        s3 = boto3.client('s3', region_name=REGION)
        iam = boto3.client('iam')
        print("✅ All required AWS services accessible")

    except Exception as e:
        print(f"❌ AWS access issue: {e}")
        print("\n🔧 Setup Options:")
        print("1. AWS CLI: Run 'aws configure' with your access keys")
        print("2. Environment variables:")
        print("   export AWS_ACCESS_KEY_ID=your_key")
        print("   export AWS_SECRET_ACCESS_KEY=your_secret")
        print("   export AWS_DEFAULT_REGION=us-east-1")
        print("3. IAM role (if on EC2/SageMaker)")
        print("4. ~/.aws/credentials file")
        print("\n💡 Since Gauntlet AI is paying, ask your team for:")
        print("   - AWS console access")
        print("   - IAM user with SageMaker permissions")
        print("   - Budget approval confirmation")
        sys.exit(1)

    # Create training script
    create_sagemaker_training_script()

    # Launch training
    job_name, estimator = train_on_sagemaker()

    print("\\n📋 Next steps:")
    print("1. Monitor training at the SageMaker console URL above")
    print("2. Training will take 2-4 hours with cloud GPU")
    print("3. Download trained model from S3 when complete")
    print("4. Deploy model for inference testing")
