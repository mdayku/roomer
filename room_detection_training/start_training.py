#!/usr/bin/env python3
"""
Simple SageMaker YOLO training launcher
"""

import boto3
import sagemaker
from sagemaker.pytorch import PyTorch
import json
from pathlib import Path
import os
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Disable SageMaker config file loading to avoid hangs
os.environ['SAGEMAKER_DISABLE_CONFIG_FILE'] = 'true'

def main():
    print("Starting YOLO Training on AWS SageMaker")
    print("=" * 50)

    # Check AWS access
    print("Testing AWS access...")
    sts = boto3.client('sts')
    identity = sts.get_caller_identity()
    account_id = identity['Account']
    print(f"AWS Account: {account_id}")

    # Get SageMaker session
    print("Creating SageMaker session...")
    sess = sagemaker.Session()
    bucket = sess.default_bucket()
    print(f"Using S3 bucket: {bucket}")

    # Get or create IAM role
    print("Setting up IAM role...")
    iam = boto3.client('iam')
    role_name = 'room-detection-sagemaker-role'

    print("Checking for existing IAM role...")
    try:
        # Try to get existing role
        response = iam.get_role(RoleName=role_name)
        role = response['Role']['Arn']
        print(f"Using existing IAM role: {role}")
    except iam.exceptions.NoSuchEntityException:
        print("Role not found, creating new IAM role...")
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

        print("Creating IAM role...")
        iam.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(trust_policy),
            Description='IAM role for Room Detection ML service'
        )
        print("IAM role created, attaching policies...")

        # Attach policies
        policies = [
            'arn:aws:iam::aws:policy/AmazonSageMakerFullAccess',
            'arn:aws:iam::aws:policy/AmazonS3FullAccess',
            'arn:aws:iam::aws:policy/CloudWatchLogsFullAccess'
        ]

        for i, policy in enumerate(policies):
            print(f"Attaching policy {i+1}/3: {policy}")
            iam.attach_role_policy(RoleName=role_name, PolicyArn=policy)

        print("Policies attached, getting role ARN...")
        # Get the role ARN
        response = iam.get_role(RoleName=role_name)
        role = response['Role']['Arn']
        print(f"Created IAM role: {role}")

    # Training configuration - EXACTLY matching what worked locally
    hyperparameters = {
        'epochs': 20,  # Same as local training that was learning
        'batch_size': 8,  # Same as local training
        'imgsz': 640,  # Same as local training
        'data': 'data.yaml',
        'weights': 'yolov8s.pt',
        # Exact same parameters as local training
        'workers': 2,  # Same as local training
        'patience': 5,  # Same as local training
        'optimizer': 'AdamW',  # Same as local training
        'lr0': 0.01,  # Same as local training
        'save_period': 10,  # Save checkpoints every 10 epochs
        'lrf': 0.01,  # Final learning rate
        'momentum': 0.937,  # Momentum
        'weight_decay': 0.0005,  # Weight decay
        'warmup_epochs': 3.0,  # Warmup epochs
        'box': 7.5,  # Box loss gain
        'cls': 0.5,  # Classification loss gain
        'dfl': 1.5,  # DFL loss gain
    }

    # Create training job
    print("Creating training job...")
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
        max_run=7200,   # 2 hours training time (for 20 epochs)
    )

    # Check for YOLO dataset
    print("Checking for YOLO dataset...")
    yolo_data_dir = Path("./yolo_data")

    if not yolo_data_dir.exists() or not (yolo_data_dir / "data.yaml").exists():
        print("[ERROR] YOLO dataset not found!")
        print("Please run the preprocessing script first:")
        print("  cd room_detection_training")
        print("  python prepare_yolo_data.py --coco-train ../room_detection_dataset_coco/train/annotations.json --coco-val ../room_detection_dataset_coco/val/annotations.json --output-dir ./yolo_data")
        return

    # Verify dataset integrity
    train_images = len(list(yolo_data_dir.glob("train/images/*.png")))
    val_images = len(list(yolo_data_dir.glob("val/images/*.png")))
    print(f"Dataset stats: Train={train_images} images, Val={val_images} images")

    if train_images == 0 or val_images == 0:
        print("[ERROR] No images found in dataset!")
        return

    # Upload data to S3
    print("Uploading YOLO dataset to S3...")
    s3_prefix = 'room-detection-training/data'
    s3_client = boto3.client('s3')

    print("Creating and uploading data.yaml...")
    # Create SageMaker data.yaml
    sagemaker_data_yaml = """# YOLO Data Configuration for SageMaker
path: /opt/ml/input/data/training  # SageMaker data dir
train: train/images  # train images (relative to 'path')
val: val/images  # val images (relative to 'path')

# Classes
nc: 1  # number of classes
names: ['room']  # class names

# Detection task (using bounding boxes)
task: detect
"""

    # Upload data.yaml
    s3_client.put_object(
        Bucket=bucket,
        Key=f"{s3_prefix}/data.yaml",
        Body=sagemaker_data_yaml
    )
    print("Uploaded data.yaml to S3")

    print("Uploading training data files...")
    # Upload all data files
    uploaded_count = 0
    for split in ['train', 'val']:
        print(f"Uploading {split} data...")
        for data_type in ['images', 'labels']:
            data_path = yolo_data_dir / split / data_type
            if data_path.exists():
                files_in_dir = list(data_path.glob('*'))
                print(f"Uploading {len(files_in_dir)} {data_type} files for {split}...")
                for file_path in files_in_dir:
                    if file_path.is_file():
                        s3_key = f"{s3_prefix}/{split}/{data_type}/{file_path.name}"
                        s3_client.upload_file(str(file_path), bucket, s3_key)
                        uploaded_count += 1
                        if uploaded_count % 500 == 0:
                            print(f"Uploaded {uploaded_count} files...")

    print(f"Total uploaded: {uploaded_count} files to S3")

    print("Uploading YOLO weights...")
    # Upload YOLO weights
    weights_path = yolo_data_dir / "yolov8s.pt"
    if weights_path.exists():
        s3_key = f"{s3_prefix}/yolov8s.pt"
        s3_client.upload_file(str(weights_path), bucket, s3_key)
        print("Uploaded weights to S3")
    else:
        print("Warning: yolov8s.pt not found locally")

    training_data_path = f's3://{bucket}/{s3_prefix}'
    print(f"Data uploaded to {training_data_path}")

    print("Creating SageMaker estimator...")
    # Create training job
    print("Creating training job...")
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
        max_run=7200,   # 2 hours training time (for 20 epochs)
    )
    print("Estimator created")

    print("Launching training job...")
    print("This will take ~2 hours (20 epochs)")
    print("Check AWS SageMaker console for progress")

    # Start training with unique name
    timestamp = int(time.time())
    job_name = f'room-detection-yolo-{timestamp}'
    print(f"Job name: {job_name}")

    print("Calling estimator.fit()...")
    estimator.fit(
        inputs={'training': training_data_path},
        job_name=job_name,
        wait=False  # Don't wait for completion
    )
    print("estimator.fit() completed")

    print(f"[SUCCESS] Training job launched: {job_name}")
    print("Monitor at: https://us-east-1.console.aws.amazon.com/sagemaker/home")
    print("Estimated cost: $2-3 (with spot pricing)")
    print("Estimated time: ~2 hours (20 epochs on G5 A10G GPU)")

if __name__ == "__main__":
    main()