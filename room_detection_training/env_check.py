#!/usr/bin/env python3
"""
Check environment variables for AWS credentials
"""

import os

def main():
    print("Checking AWS Environment Variables")
    print("=" * 40)

    vars_to_check = [
        'AWS_ACCESS_KEY_ID',
        'AWS_SECRET_ACCESS_KEY',
        'AWS_DEFAULT_REGION'
    ]

    all_set = True
    for var in vars_to_check:
        value = os.getenv(var)
        if value:
            # Show first few and last few characters for security
            if len(value) > 10:
                display = value[:4] + "..." + value[-4:]
            else:
                display = value
            print(f"[OK] {var} = {display}")
        else:
            print(f"[MISSING] {var} = Not set")
            all_set = False

    print("\n" + "=" * 40)
    if all_set:
        print("SUCCESS: All AWS environment variables are set!")
        print("You can now run: python sagemaker_train.py")
    else:
        print("ERROR: Missing AWS environment variables")
        print("\nTo set them, run these commands:")
        print("set AWS_ACCESS_KEY_ID=your_key_here")
        print("set AWS_SECRET_ACCESS_KEY=your_secret_here")
        print("set AWS_DEFAULT_REGION=us-east-1")
        print("\nOr create a .env file with:")
        print("AWS_ACCESS_KEY_ID=your_key_here")
        print("AWS_SECRET_ACCESS_KEY=your_secret_here")
        print("AWS_DEFAULT_REGION=us-east-1")

if __name__ == "__main__":
    main()
