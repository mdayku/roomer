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

    # Check AWS access
    try:
        sts = boto3.client('sts')
        identity = sts.get_caller_identity()
        account_id = identity['Account']
        print(f"AWS Account: {account_id}")

        # Test services
        sagemaker_client = boto3.client('sagemaker')
        s3_client = boto3.client('s3')
        print("AWS services accessible")

    except Exception as e:
        print(f"AWS access failed: {e}")
        print("Check your .env file credentials")
        return

    # Get SageMaker session
    sess = sagemaker.Session()
    bucket = sess.default_bucket()
    print(f"Using S3 bucket: {bucket}")

    # Get or create IAM role
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

    # Training configuration - simplified for testing
    hyperparameters = {
        'epochs': 10,  # Start small for testing
        'batch_size': 4,  # Smaller batch size
        'imgsz': 416,  # Smaller images for testing
        'data': 'data.yaml',
        'weights': 'yolov8s-seg.pt',
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
        max_run=3600,    # 1 hour training time
    )

    # Prepare and upload training data
    print("Preparing training data...")
    yolo_data_dir = Path("./yolo_data")

    # Convert COCO to YOLO format if not already done
    if not yolo_data_dir.exists():
        print("Converting COCO data to YOLO format...")
        # Import here to avoid issues if not needed
        sys.path.append('.')
        try:
            from train_yolo import convert_coco_to_yolo_format, create_data_yaml
            convert_coco_to_yolo_format(
                "../room_detection_dataset_coco/train/annotations.json",
                yolo_data_dir / "train"
            )
            convert_coco_to_yolo_format(
                "../room_detection_dataset_coco/val/annotations.json",
                yolo_data_dir / "val"
            )
            create_data_yaml(yolo_data_dir / "train", yolo_data_dir / "val", yolo_data_dir / "data.yaml")
            print("Data conversion completed")
        except Exception as e:
            print(f"Data conversion failed: {e}")
            return

    # Create SageMaker-compatible data structure and upload
    print("Creating SageMaker data structure...")

    # Create SageMaker data.yaml
    sagemaker_data_yaml = """# YOLO Data Configuration for SageMaker
path: /opt/ml/input/data/training  # SageMaker data dir
train: images/train  # train images (relative to 'path')
val: images/val  # val images (relative to 'path')

# Classes
nc: 1  # number of classes
names: ['room']  # class names

# Segmentation task
task: segment
"""

    # Upload data to S3
    print("Uploading data to S3...")
    s3_prefix = 'room-detection-training/data'
    s3_client = boto3.client('s3')

    # Upload data.yaml
    s3_client.put_object(
        Bucket=bucket,
        Key=f"{s3_prefix}/data.yaml",
        Body=sagemaker_data_yaml,
        ContentType='text/yaml'
    )

    # Upload training data in correct SageMaker structure
    for split in ['train', 'val']:
        split_dir = yolo_data_dir / split

        if split_dir.exists():
            # Upload images
            images_dir = split_dir / "images"
            if images_dir.exists():
                for img_file in images_dir.glob("*.png"):
                    s3_key = f"{s3_prefix}/images/{split}/{img_file.name}"
                    print(f"Uploading {img_file.name}...")
                    s3_client.upload_file(str(img_file), bucket, s3_key)

            # Upload labels
            labels_dir = split_dir / "labels"
            if labels_dir.exists():
                for lbl_file in labels_dir.glob("*.txt"):
                    s3_key = f"{s3_prefix}/labels/{split}/{lbl_file.name}"
                    s3_client.upload_file(str(lbl_file), bucket, s3_key)

    # Download YOLO weights if not present
    weights_path = yolo_data_dir / "yolov8s-seg.pt"
    if not weights_path.exists():
        print("Downloading YOLO weights...")
        try:
            import urllib.request
            weights_url = "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8s-seg.pt"
            urllib.request.urlretrieve(weights_url, str(weights_path))
        except Exception as e:
            print(f"Failed to download weights: {e}")
            print("Training may fail without weights")

    # Upload weights
    if weights_path.exists():
        s3_key = f"{s3_prefix}/yolov8s-seg.pt"
        s3_client.upload_file(str(weights_path), bucket, s3_key)
        print("Weights uploaded")

    training_data_path = f's3://{bucket}/{s3_prefix}'
    print(f"Data uploaded to {training_data_path}")

    print("Launching training job...")
    print("This will take 2-4 hours")
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
    print("Estimated cost: $1-3 (with spot pricing - G5 upgrade!)")
    print("Estimated time: 1-2 hours (G5 is faster!)")

if __name__ == "__main__":
    main()
