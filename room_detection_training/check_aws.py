#!/usr/bin/env python3
"""
Check AWS credentials and permissions for Room Detection training
Run this before starting SageMaker training
"""

import boto3
import sys
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

def test_aws_access():
    """Test access to required AWS services"""
    print("Testing AWS service access...")

    try:
        # Test STS (basic AWS access)
        sts = boto3.client('sts')
        identity = sts.get_caller_identity()
        account_id = identity['Account']
        user_arn = identity['Arn']
        print(f"[OK] AWS Account: {account_id}")
        print(f"[OK] User/Role: {user_arn.split('/')[-1]}")

        # Test SageMaker
        sm = boto3.client('sagemaker')
        print("[OK] SageMaker access confirmed")

        # Test S3
        s3 = boto3.client('s3')
        print("[OK] S3 access confirmed")

        # Test IAM (for role creation)
        iam = boto3.client('iam')
        print("[OK] IAM access confirmed")

        # Test CloudWatch (for logging)
        cw = boto3.client('logs')
        print("[OK] CloudWatch access confirmed")

        return True

    except Exception as e:
        print(f"[ERROR] AWS access failed: {e}")
        return False

def check_permissions():
    """Check specific permissions needed"""
    print("Checking specific permissions...")

    issues = []

    try:
        # Check SageMaker permissions
        sm = boto3.client('sagemaker')
        sm.list_training_jobs(MaxResults=1)
        print("[OK] SageMaker training permissions")
    except:
        issues.append("SageMaker training permissions")

    try:
        # Check S3 bucket creation
        s3 = boto3.client('s3')
        # Try to list buckets (basic permission)
        s3.list_buckets()
        print("[OK] S3 permissions")
    except:
        issues.append("S3 permissions")

    try:
        # Check IAM permissions
        iam = boto3.client('iam')
        iam.list_roles(MaxItems=1)
        print("[OK] IAM permissions")
    except:
        issues.append("IAM permissions")

    if issues:
        print(f"[WARN] Permission issues: {', '.join(issues)}")
        return False

    print("[OK] All required permissions confirmed")
    return True

def provide_setup_instructions():
    """Provide setup instructions"""
    print("\n🛠️ AWS Setup Instructions:")
    print("=" * 50)

    print("\n📋 Option 1 - .env File (Easiest for Development):")
    print("Create a .env file in this directory:")
    print("")
    print("# AWS Credentials for Room Detection Training")
    print("AWS_ACCESS_KEY_ID=your_access_key_here")
    print("AWS_SECRET_ACCESS_KEY=your_secret_key_here")
    print("AWS_DEFAULT_REGION=us-east-1")
    print("")
    print("Replace with your actual keys from Gauntlet AI team!")

    print("\n📋 Option 2 - AWS CLI Setup (Recommended for Production):")
    print("1. Install AWS CLI: https://aws.amazon.com/cli/")
    print("2. Run: aws configure")
    print("3. Enter your AWS Access Key ID")
    print("4. Enter your AWS Secret Access Key")
    print("5. Enter default region (us-east-1)")
    print("6. Enter default output format (json)")

    print("\n📋 Option 3 - Environment Variables:")
    print("export AWS_ACCESS_KEY_ID=your_access_key_here")
    print("export AWS_SECRET_ACCESS_KEY=your_secret_key_here")
    print("export AWS_DEFAULT_REGION=us-east-1")

    print("\n📋 Option 4 - AWS Credentials File:")
    print("Create ~/.aws/credentials:")
    print("[default]")
    print("aws_access_key_id = your_access_key_here")
    print("aws_secret_access_key = your_secret_key_here")
    print("")
    print("Create ~/.aws/config:")
    print("[default]")
    print("region = us-east-1")

    print("\n💡 Since Gauntlet AI is paying:")
    print("- Ask your team for AWS console access")
    print("- Request IAM user with SageMaker permissions")
    print("- Confirm budget allocation")
    print("- Get your access keys from AWS console -> IAM -> Users -> Security credentials")

    print("\n🔗 Useful Links:")
    print("- AWS Console: https://console.aws.amazon.com/")
    print("- IAM Users: https://console.aws.amazon.com/iam/home#/users")
    print("- SageMaker: https://console.aws.amazon.com/sagemaker/home")

def main():
    """Main AWS check function"""
    print("AWS Setup Checker for Room Detection Training")
    print("=" * 60)

    # Check if boto3 is available
    try:
        import boto3
    except ImportError:
        print("[ERROR] boto3 not installed. Run: pip install boto3")
        sys.exit(1)

    # Check credentials from all sources
    creds_ok = check_credentials()

    if not creds_ok:
        print("\n[ERROR] No AWS credentials found")
        provide_setup_instructions()
        sys.exit(1)

    # Test AWS access
    access_ok = test_aws_access()

    if not access_ok:
        print("\n[ERROR] AWS access test failed")
        provide_setup_instructions()
        sys.exit(1)

    # Check permissions
    perms_ok = check_permissions()

    print("\n" + "=" * 60)
    if perms_ok:
        print("SUCCESS: AWS setup complete! Ready for SageMaker training.")
        print("\nNext step: Run 'python sagemaker_train.py'")
    else:
        print("WARNING: Some permissions may be missing.")
        print("The script will attempt to create required resources,")
        print("but you may need to ask your AWS admin for additional permissions.")
        print("\nYou can try running: python sagemaker_train.py")

if __name__ == "__main__":
    main()
