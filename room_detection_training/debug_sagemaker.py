#!/usr/bin/env python3
"""
Debug SageMaker hanging issue
"""

import boto3
import sagemaker
from sagemaker import get_execution_role
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main():
    print("Debug SageMaker Script")
    print("=" * 30)

    print("1. Testing AWS access...")
    try:
        sts = boto3.client('sts')
        identity = sts.get_caller_identity()
        print(f"   AWS Account: {identity['Account']}")
        print("   AWS access OK")
    except Exception as e:
        print(f"   AWS access failed: {e}")
        return

    print("2. Testing SageMaker session...")
    try:
        sess = sagemaker.Session()
        bucket = sess.default_bucket()
        print(f"   S3 bucket: {bucket}")
        print("   SageMaker session OK")
    except Exception as e:
        print(f"   SageMaker session failed: {e}")
        return

    print("3. Testing IAM role...")
    try:
        iam = boto3.client('iam')
        role_name = 'room-detection-sagemaker-role'

        # Try to get existing role
        response = iam.get_role(RoleName=role_name)
        role = response['Role']['Arn']
        print(f"   Using existing role: {role}")
    except iam.exceptions.NoSuchEntityException:
        print(f"   Role {role_name} does not exist")
        # Try to create it
        try:
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
                AssumeRolePolicyDocument=str(trust_policy).replace("'", '"'),
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
            print(f"   Created role: {role}")

        except Exception as e:
            print(f"   Failed to create role: {e}")
            return
    except Exception as e:
        print(f"   IAM error: {e}")
        return

    print("4. Testing data upload preparation...")
    try:
        from pathlib import Path
        yolo_data_dir = Path("./yolo_data")
        if yolo_data_dir.exists():
            train_images = len(list(yolo_data_dir.glob("train/images/*.png")))
            print(f"   Found {train_images} training images")
            print("   Data check OK")
        else:
            print("   YOLO data directory not found")
            return
    except Exception as e:
        print(f"   Data check error: {e}")
        return

    print("5. Testing S3 data upload...")
    try:
        import os
        s3_client = boto3.client('s3')
        bucket = sess.default_bucket()

        # Try uploading just a few test files
        yolo_data_dir = Path("./yolo_data")
        uploaded_count = 0

        for root, dirs, files in os.walk(str(yolo_data_dir)):
            for file in files[:5]:  # Just upload first 5 files as test
                if file.endswith(('.png', '.txt', '.yaml')):
                    local_path = os.path.join(root, file)
                    relative_path = os.path.relpath(local_path, str(yolo_data_dir))
                    s3_key = f"test-upload/{relative_path}"

                    print(f"   Uploading test file: {relative_path}")
                    s3_client.upload_file(local_path, bucket, s3_key)
                    uploaded_count += 1

                    if uploaded_count >= 3:  # Upload just 3 files for testing
                        break
            if uploaded_count >= 3:
                break

        print(f"   Successfully uploaded {uploaded_count} test files")
        print("   S3 upload test OK")

    except Exception as e:
        print(f"   S3 upload error: {e}")
        return

    print("SUCCESS: All checks passed!")
    print("The full data upload or job creation might be the issue...")
    print("Try running start_training.py now - it should work!")

if __name__ == "__main__":
    main()
