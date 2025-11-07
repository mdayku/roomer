#!/usr/bin/env python3
"""
Test if .env file is being loaded correctly
"""

import os
from pathlib import Path
from dotenv import load_dotenv

def main():
    print("Testing .env file loading")
    print("=" * 30)

    # Check if .env file exists
    env_file = Path('.env')
    if env_file.exists():
        print(f"[OK] .env file exists at: {env_file.absolute()}")

        # Try to read it (without printing contents for security)
        try:
            with open(env_file, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.strip().split('\n')
                print(f"[OK] .env file has {len(lines)} lines")

                # Check for required variables (without showing values)
                has_access_key = 'AWS_ACCESS_KEY_ID=' in content
                has_secret_key = 'AWS_SECRET_ACCESS_KEY=' in content
                has_region = 'AWS_DEFAULT_REGION=' in content

                print(f"[OK] AWS_ACCESS_KEY_ID found: {has_access_key}")
                print(f"[OK] AWS_SECRET_ACCESS_KEY found: {has_secret_key}")
                print(f"[OK] AWS_DEFAULT_REGION found: {has_region}")

        except Exception as e:
            print(f"[ERROR] Cannot read .env file: {e}")
            return

    else:
        print(f"[ERROR] .env file not found at: {env_file.absolute()}")
        return

    # Test loading with dotenv
    print("\nTesting dotenv loading...")
    load_dotenv()  # This should load the .env file

    # Check if variables are now in environment
    access_key = os.getenv('AWS_ACCESS_KEY_ID')
    secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    region = os.getenv('AWS_DEFAULT_REGION')

    print(f"AWS_ACCESS_KEY_ID loaded: {access_key is not None}")
    print(f"AWS_SECRET_ACCESS_KEY loaded: {secret_key is not None}")
    print(f"AWS_DEFAULT_REGION loaded: {region is not None}")

    if region:
        print(f"Region value: {region}")

    # Test boto3 import and basic connection
    if access_key and secret_key:
        print("\nTesting AWS connection...")
        try:
            import boto3
            sts = boto3.client('sts')
            identity = sts.get_caller_identity()
            print(f"[SUCCESS] AWS Account: {identity['Account']}")
        except Exception as e:
            print(f"[ERROR] AWS connection failed: {e}")
    else:
        print("\n[ERROR] Environment variables not loaded by dotenv")

if __name__ == "__main__":
    main()
