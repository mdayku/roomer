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
import sys
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Disable SageMaker config file loading to avoid hangs
os.environ['SAGEMAKER_DISABLE_CONFIG_FILE'] = 'true'

# SageMaker configuration - OPTIMIZED FOR YOLO LARGE + 200 EPOCHS
INSTANCE_TYPE = 'ml.g5.2xlarge'  # A10G x2 GPUs - best cost/performance for large models
# Why this choice for YOLO Large + 200 epochs:
# - 48GB VRAM total (24GB per GPU) → handles yolov8l/x models easily
# - Multi-GPU training → faster convergence, shorter training time
# - Cost-effective: ~$2.41/hr vs $9.60/hr for single A100
# Alternatives:
# 'ml.g5.xlarge' - A10G single GPU ($1.21/hr, 24GB VRAM)
# 'ml.p4d.24xlarge' - A100 single GPU ($40.96/hr, 80GB VRAM) - overkill
# 'ml.g4dn.xlarge' - T4 GPU ($0.74/hr) - too slow for 200 epochs

INSTANCE_COUNT = 1
FRAMEWORK_VERSION = '2.0.0'
PYTHON_VERSION = 'py310'

# Training hyperparameters - OPTIMIZED FOR YOLO LARGE MODEL + 200 EPOCHS
HYPERPARAMETERS = {
    'epochs': 200,  # Extended training for better convergence
    'batch_size': 4,  # Smaller batch size for stability with large model
    'imgsz': 640,  # Same as local training
    'data': 'data.yaml',  # Will be created in container
    'weights': 'yolov8l.pt',  # LARGE model for better accuracy
    'workers': 4,  # More workers for multi-GPU setup
    'patience': 20,  # More patience for longer training
    'optimizer': 'AdamW',  # Same as local training
    'lr0': 0.001,  # Lower learning rate for stability
    'save_period': 25,  # Save checkpoints every 25 epochs (fewer saves)
    'lrf': 0.01,  # Final learning rate
    'momentum': 0.937,  # Momentum
    'weight_decay': 0.0005,  # Weight decay
    'warmup_epochs': 5.0,  # Longer warmup for large model
    'box': 7.5,  # Box loss gain
    'cls': 0.5,  # Classification loss gain
    'dfl': 1.5,  # DFL loss gain
    # Removed 'seg' parameter since we're doing detection, not segmentation
}


def upload_data_to_s3(s3_bucket, s3_prefix, local_data_dir, force_upload=False):
    """Upload training data to S3"""
    print(f"Checking if data exists in s3://{s3_bucket}/{s3_prefix}")

    s3_client = boto3.client('s3')

    # Check if data.yaml already exists (indicating data was uploaded)
    try:
        s3_client.head_object(Bucket=s3_bucket, Key=f"{s3_prefix}/data.yaml")
        if force_upload:
            print("Data exists in S3 but force_upload=True, will re-upload...")
        else:
            print("Data already exists in S3, skipping upload")
            return f"s3://{s3_bucket}/{s3_prefix}"
    except s3_client.exceptions.NoSuchKey:
        print("Data not found in S3, uploading...")

    print(f"Uploading data to s3://{s3_bucket}/{s3_prefix}")

    for root, dirs, files in os.walk(local_data_dir):
        for file in files:
            local_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_path, local_data_dir)
            s3_key = f"{s3_prefix}/{relative_path}"

            print(f"Uploading {relative_path}...")
            s3_client.upload_file(local_path, s3_bucket, s3_key)

    s3_data_path = f"s3://{s3_bucket}/{s3_prefix}"
    print(f"[OK] Data uploaded to {s3_data_path}")
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

# Detection task (using bounding boxes)
task: detect
"""

    # Save locally for upload
    with open('data.yaml', 'w') as f:
        f.write(yaml_content.strip())

    print("[OK] Created data.yaml for SageMaker")
    return 'data.yaml'


def train_on_sagemaker():
    """Launch SageMaker training job"""
    print("Launching YOLO training on AWS SageMaker")
    print(f"Instance: {INSTANCE_TYPE} ({INSTANCE_COUNT}x)")
    print("=" * 60)

    # Get SageMaker session and role
    print("Setting up IAM role...")
    iam = boto3.client('iam')
    role_name = 'room-detection-sagemaker-role'

    try:
        # Try to get existing role
        response = iam.get_role(RoleName=role_name)
        role = response['Role']['Arn']
        print(f"Using existing IAM role: {role}")
    except iam.exceptions.NoSuchEntityException:
        # Create new role
        print(f"Creating IAM role: {role_name}")

        trust_policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {"Service": "sagemaker.amazonaws.com"},
                    "Action": "sts:AssumeRole"
                }
            ]
        }

        iam.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(trust_policy),
            Description='IAM role for Room Detection ML service'
        )

        # Attach policies
        policies = [
            'arn:aws:iam::aws:policy/AmazonSageMakerFullAccess',
            'arn:aws:iam::aws:policy/AmazonS3FullAccess',
            'arn:aws:iam::aws:policy/CloudWatchLogsFullAccess'
        ]

        for policy in policies:
            iam.attach_role_policy(RoleName=role_name, PolicyArn=policy)

        # Get the role ARN
        response = iam.get_role(RoleName=role_name)
        role = response['Role']['Arn']
        print(f"Created IAM role: {role}")

    sess = sagemaker.Session()
    bucket = sess.default_bucket()
    print(f"[OK] Using S3 bucket: {bucket}")

    # Prepare data
    print("\nPreparing training data...")

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

    # Create data.yaml for SageMaker
    data_yaml = create_data_yaml_for_sagemaker()

    # Copy the SageMaker data.yaml to overwrite the local one
    import shutil
    shutil.copy('data.yaml', str(yolo_data_dir / 'data.yaml'))
    print("[OK] Overwrote local data.yaml with SageMaker version")

    # Upload data to S3
    s3_data_path = upload_data_to_s3(
        bucket,
        'room-detection-training/data',
        str(yolo_data_dir)
    )

    # Upload YOLO weights (if not using pretrained)
    # For now, we'll download in the container

    # Create PyTorch estimator
    print("\nCreating SageMaker estimator...")

    estimator = PyTorch(
        entry_point='train_yolo_sagemaker.py',  # We'll create this
        source_dir=None,  # Don't upload source directory - script is standalone
        role=role,
        instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
        framework_version=FRAMEWORK_VERSION,
        py_version=PYTHON_VERSION,
        hyperparameters=HYPERPARAMETERS,
        # Spot training for cost savings (70% discount)
        use_spot_instances=True,
        max_wait=172800,  # 48 hours (for long 200 epoch training)
        max_run=43200,    # 12 hours training time (for 200 epochs on large model)
        # Enable multi-GPU training on single instance with multiple GPUs
        distribution={
            'pytorch': {
                'enabled': True,
                'processes_per_host': 2  # 2 GPUs on g5.2xlarge
            }
        },
    )

    # Launch training
    print("\nStarting training job...")

    job_name = f'room-detection-yolo-{int(time.time())}'

    estimator.fit(
        inputs={'training': s3_data_path},
        job_name=job_name,
        wait=False  # Don't wait for completion
    )

    print(f"[OK] Training job launched: {job_name}")
    print(f"Monitor progress at: https://{sess.boto_region_name}.console.aws.amazon.com/sagemaker/home?region={sess.boto_region_name}#/jobs/{job_name}")

    # Estimate costs and time for LARGE MODEL + 200 EPOCHS
    hourly_rate = get_instance_price(INSTANCE_TYPE)
    spot_rate = hourly_rate * 0.3  # ~70% spot discount
    estimated_hours = 8.0  # A10G x2 GPUs with large model: ~8 hours for 200 epochs
    estimated_cost = spot_rate * estimated_hours

    print(f"\nCost Estimate ({INSTANCE_TYPE}) - YOLO Large 200 Epochs:")
    print(f"- Instance: {INSTANCE_TYPE} (${hourly_rate}/hour -> ${spot_rate:.2f}/hour spot)")
    print("- Multi-GPU training: 2x A10G GPUs for faster convergence")
    print("- Model: YOLOv8 Large (better accuracy than Small model)")
    print("- Spot discount: ~70% savings")
    print(f"- Estimated training time: {estimated_hours} hours")
    print(f"- Estimated cost: ${estimated_cost:.2f} (with spot pricing)")
    print(f"- Expected improvement: 15-25% better mAP vs Small model")

    return job_name, estimator


def get_instance_price(instance_type):
    """Get approximate hourly price for instance type"""
    prices = {
        'ml.g4dn.xlarge': 0.736,   # T4 GPU (what we used for small model)
        'ml.g5.xlarge': 1.206,     # A10G single GPU
        'ml.g5.2xlarge': 2.412,    # A10G x2 GPUs (current choice for large model!)
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
    print("\\nLoading YOLO detection model...")
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
    print("\\nStarting training...")
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
    print(f"\\n[OK] Model saved to {model_path}")

    # Export for inference
    onnx_path = '/opt/ml/model/model.onnx'
    model.export(format='onnx', imgsz=args.imgsz)
    print(f"[OK] ONNX model exported to {onnx_path}")

    # Run validation
    print("\\nRunning final validation...")
    metrics = model.val()
    print(f"Validation mAP: {metrics.box.map:.4f}")

    print("\\nSageMaker training completed!")

if __name__ == "__main__":
    main()
'''

    with open('train_yolo_sagemaker.py', 'w') as f:
        f.write(script_content)

    print("Created SageMaker training script")


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
        print(f"[OK] AWS Account: {account_id}")

        # Test required services
        sm = boto3.client('sagemaker')
        s3 = boto3.client('s3')
        iam = boto3.client('iam')
        print("[OK] All required AWS services accessible")

    except Exception as e:
        print(f"[ERROR] AWS access issue: {e}")
        print("\n[SETUP] Setup Options:")
        print("1. AWS CLI: Run 'aws configure' with your access keys")
        print("2. Environment variables:")
        print("   export AWS_ACCESS_KEY_ID=your_key")
        print("   export AWS_SECRET_ACCESS_KEY=your_secret")
        print("   export AWS_DEFAULT_REGION=us-east-1")
        print("3. IAM role (if on EC2/SageMaker)")
        print("4. ~/.aws/credentials file")
        print("\n[TIP] Since Gauntlet AI is paying, ask your team for:")
        print("   - AWS console access")
        print("   - IAM user with SageMaker permissions")
        print("   - Budget approval confirmation")
        sys.exit(1)

    # Create training script
    create_sagemaker_training_script()

    # Launch training
    job_name, estimator = train_on_sagemaker()

    print("\\n Next steps:")
    print("1. Monitor training at the SageMaker console URL above")
    print("2. Training will take 2-4 hours with cloud GPU")
    print("3. Download trained model from S3 when complete")
    print("4. Deploy model for inference testing")
