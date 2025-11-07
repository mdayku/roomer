#!/usr/bin/env python3
"""
Simple AWS credentials check for Room Detection training
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file if it exists
load_dotenv()

def check_credentials():
    """Check AWS credentials from all sources"""
    print("Checking AWS credentials...")

    # Check .env file
    env_file = Path('.env')
    if env_file.exists():
        print(f"[OK] .env file found: {env_file}")
        # Check if .env has AWS vars
        with open(env_file, 'r') as f:
            content = f.read()
            if 'AWS_ACCESS_KEY_ID' in content and 'AWS_SECRET_ACCESS_KEY' in content:
                print("[OK] AWS credentials found in .env file")
                return True

    # Check environment variables (includes .env loaded ones)
    env_vars = ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'AWS_DEFAULT_REGION']
    env_set = all(os.getenv(var) for var in env_vars)

    if env_set:
        source = ".env file" if env_file.exists() else "environment variables"
        print(f"[OK] AWS credentials set via {source}")
        return True

    # Check AWS credentials file
    creds_file = Path.home() / '.aws' / 'credentials'
    if creds_file.exists():
        print(f"[OK] AWS credentials file found: {creds_file}")
        return True

    print("[WARN] No AWS credentials found")
    return False

def main():
    print("AWS Setup Checker for Room Detection Training")
    print("=" * 60)

    # Check if boto3 is available
    try:
        import boto3
    except ImportError:
        print("[ERROR] boto3 not installed. Run: pip install boto3")
        return

    # Check credentials
    creds_ok = check_credentials()

    if not creds_ok:
        print("\n[ERROR] No AWS credentials found")
        print("\nSetup Options:")
        print("1. Create .env file with AWS credentials")
        print("2. Set environment variables")
        print("3. Use AWS CLI: aws configure")
        print("\nContact Gauntlet AI team for AWS access keys")
        return

    print("\n[OK] Credentials found! Testing AWS access...")

    # Test AWS access
    try:
        sts = boto3.client('sts')
        identity = sts.get_caller_identity()
        account_id = identity['Account']
        print(f"[OK] AWS Account: {account_id}")

        # Test required services
        sagemaker = boto3.client('sagemaker')
        s3 = boto3.client('s3')
        iam = boto3.client('iam')
        print("[OK] All required AWS services accessible")

        print("\nSUCCESS: AWS setup complete! Ready for training.")
        print("Next: Run 'python sagemaker_train.py'")

    except Exception as e:
        print(f"[ERROR] AWS access failed: {e}")
        print("Check your credentials and permissions")

if __name__ == "__main__":
    main()
