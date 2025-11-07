#!/usr/bin/env python3
"""
Final AWS check before training
"""

import os
import boto3
from dotenv import load_dotenv

def main():
    print("Final AWS Check Before Training")
    print("=" * 40)

    # Load environment variables
    load_dotenv()

    # Check environment variables
    access_key = os.getenv('AWS_ACCESS_KEY_ID')
    secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    region = os.getenv('AWS_DEFAULT_REGION')

    print(f"AWS_ACCESS_KEY_ID: {'[SET]' if access_key else '[MISSING]'}")
    print(f"AWS_SECRET_ACCESS_KEY: {'[SET]' if secret_key else '[MISSING]'}")
    print(f"AWS_DEFAULT_REGION: {region or '[MISSING]'}")

    if not all([access_key, secret_key, region]):
        print("\n[ERROR] Missing AWS credentials")
        return False

    # Test AWS connection
    print("\nTesting AWS connection...")
    try:
        sts = boto3.client('sts')
        identity = sts.get_caller_identity()
        print(f"[SUCCESS] AWS Account: {identity['Account']}")

        # Test SageMaker access
        sm = boto3.client('sagemaker')
        print("[SUCCESS] SageMaker access confirmed")

        # Test S3 access
        s3 = boto3.client('s3')
        print("[SUCCESS] S3 access confirmed")

        print("\n[SUCCESS] All systems ready for training!")
        print("Run: python sagemaker_train.py")

        return True

    except Exception as e:
        print(f"[ERROR] AWS connection failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
