#!/usr/bin/env python3
"""
Simple SageMaker YOLO training launcher
"""

import boto3
import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker import get_execution_role
import json
from pathlib import Path
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main():
    print("Starting YOLO Training on AWS SageMaker")
    print("=" * 50)
    print("DEBUG: Script started")

    # Check AWS access
    print("DEBUG: Testing AWS access...")
    try:
        print("DEBUG: Creating STS client...")
        sts = boto3.client('sts')
        print("DEBUG: Calling get_caller_identity...")
        identity = sts.get_caller_identity()
        account_id = identity['Account']
        print(f"AWS Account: {account_id}")

        # Test services
        print("DEBUG: Creating SageMaker client...")
        sagemaker_client = boto3.client('sagemaker')
        print("DEBUG: Creating S3 client...")
        s3_client = boto3.client('s3')
        print("AWS services accessible")
        print("DEBUG: AWS access test completed")

    except Exception as e:
        print(f"AWS access failed: {e}")
        print("Check your .env file credentials")
        return

    # Get SageMaker session
    print("DEBUG: Creating SageMaker session...")
    sess = sagemaker.Session()
    bucket = sess.default_bucket()
    print(f"Using S3 bucket: {bucket}")
    print("DEBUG: SageMaker session created")

    # Get or create IAM role
    print("DEBUG: Setting up IAM role...")
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

        try:
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

        except Exception as e:
            print(f"Failed to create IAM role: {e}")
            print("You may need to create the role manually in AWS IAM console")
            print("Required policies: AmazonSageMakerFullAccess, AmazonS3FullAccess, CloudWatchLogsFullAccess")
            return
    except Exception as e:
        print(f"Failed to get IAM role: {e}")
        return

    # Training configuration - matching what worked locally
    hyperparameters = {
        'epochs': 50,  # Full training run
        'batch_size': 16,  # Larger batch for cloud GPU
        'imgsz': 640,  # Standard YOLO size (local used 320 for testing)
        'data': 'data.yaml',
        'weights': 'yolov8s.pt',
        # Additional parameters that worked locally
        'workers': 8,  # More workers for cloud instance
        'patience': 10,  # Early stopping patience
        'save_period': 10,  # Save checkpoints every 10 epochs
        'optimizer': 'AdamW',  # Optimizer from local run
        'lr0': 0.01,  # Learning rate from local run
        'lrf': 0.01,  # Final learning rate
        'momentum': 0.937,  # Momentum
        'weight_decay': 0.0005,  # Weight decay
        'warmup_epochs': 3.0,  # Warmup epochs
        'box': 7.5,  # Box loss gain
        'cls': 0.5,  # Classification loss gain
        'dfl': 1.5,  # DFL loss gain
    }

    # Create training job
    # For SageMaker, we'll create a minimal script that just installs ultralytics
    # and runs the training - no need for complex dependencies
    estimator = PyTorch(
        entry_point='train_yolo_sagemaker.py',
        source_dir=None,  # Don't upload local directory
        role=role,
        instance_count=1,
        instance_type='ml.g5.xlarge',  # A10G GPU - better performance
        framework_version='2.0.0',
        py_version='py310',
        hyperparameters=hyperparameters,
        use_spot_instances=True,
        max_wait=86400,  # 24 hours
        max_run=14400,   # 4 hours training time (for 50 epochs)
    )

    # Check for pre-processed YOLO data
    print("Checking for YOLO dataset...")
    yolo_data_dir = Path("./yolo_data")

    if not yolo_data_dir.exists() or not (yolo_data_dir / "data.yaml").exists():
        print("❌ ERROR: YOLO dataset not found!")
        print("Please run the preprocessing script first:")
        print("  cd room_detection_training")
        print("  python prepare_yolo_data.py --coco-train ../room_detection_dataset_coco/train/annotations.json --coco-val ../room_detection_dataset_coco/val/annotations.json --output-dir ./yolo_data")
        return

    # Verify dataset integrity
    train_images = len(list(yolo_data_dir.glob("train/images/*.png")))
    val_images = len(list(yolo_data_dir.glob("val/images/*.png")))
    test_images = len(list(yolo_data_dir.glob("test/images/*.png")))
    train_labels = len(list(yolo_data_dir.glob("train/labels/*.txt")))
    val_labels = len(list(yolo_data_dir.glob("val/labels/*.txt")))
    test_labels = len(list(yolo_data_dir.glob("test/labels/*.txt")))

    print(f"Dataset stats: Train={train_images} images/{train_labels} labels, Val={val_images} images/{val_labels} labels, Test={test_images} images/{test_labels} labels")

    if train_images == 0 or val_images == 0:
        print("❌ ERROR: No images found in dataset!")
        return

    # Create SageMaker-compatible data structure and upload
    print("Uploading YOLO dataset to S3...")

    # Create SageMaker data.yaml
    sagemaker_data_yaml = """# YOLO Data Configuration for SageMaker
path: /opt/ml/input/data/training  # SageMaker data dir
train: train/images  # train images (relative to 'path')
val: val/images  # val images (relative to 'path')
test: test/images  # test images (relative to 'path')

# Classes
nc: 1  # number of classes
names: ['room']  # class names

# Segmentation task
task: segment
"""

    # Upload data to S3
    s3_prefix = 'room-detection-training/data'
    s3_client = boto3.client('s3')

    # Upload data.yaml
    s3_client.put_object(
        Bucket=bucket,
        Key=f"{s3_prefix}/data.yaml",
        Body=sagemaker_data_yaml,
        ContentType='text/yaml'
    )

    # Upload all YOLO data files recursively
    uploaded_count = 0
    for root, dirs, files in os.walk(str(yolo_data_dir)):
        for file in files:
            if file.endswith(('.png', '.txt', '.yaml')):
                local_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_path, str(yolo_data_dir))

                # Skip the local data.yaml, use SageMaker version
                if file == 'data.yaml' and 'yolo_data' in relative_path:
                    continue

                s3_key = f"{s3_prefix}/{relative_path}"
                s3_client.upload_file(local_path, bucket, s3_key)
                uploaded_count += 1

                if uploaded_count % 50 == 0:
                    print(f"Uploaded {uploaded_count} files...")

    print(f"[SUCCESS] Uploaded {uploaded_count} files to S3")

    # Download and upload YOLO weights
    weights_path = yolo_data_dir / "yolov8s.pt"
    if not weights_path.exists():
        print("Downloading YOLO weights...")
        try:
            import urllib.request
            weights_url = "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8s.pt"
            urllib.request.urlretrieve(weights_url, str(weights_path))
            print("Weights downloaded")
        except Exception as e:
            print(f"Failed to download weights: {e}")
            return

    # Upload weights
    s3_key = f"{s3_prefix}/yolov8s.pt"
    s3_client.upload_file(str(weights_path), bucket, s3_key)
    print("[SUCCESS] Weights uploaded")

    training_data_path = f's3://{bucket}/{s3_prefix}'
    print(f"[SUCCESS] Data uploaded to {training_data_path}")

    print("Launching training job...")
    print("This will take 3-5 hours (50 epochs)")
    print("Check AWS SageMaker console for progress")

    # Start training with unique name
    import time
    timestamp = int(time.time())
    job_name = f'room-detection-yolo-{timestamp}'

    estimator.fit(
        inputs={'training': training_data_path},
        job_name=job_name,
        wait=False  # Don't wait for completion
    )

    print(f"Training job started: {job_name}")
    print("Monitor at: https://us-east-1.console.aws.amazon.com/sagemaker/home")
    print("Estimated cost: $3-5 (with spot pricing - full training run)")
    print("Estimated time: 3-5 hours (50 epochs on G5 A10G GPU)")

if __name__ == "__main__":
    main()
