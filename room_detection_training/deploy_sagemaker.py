#!/usr/bin/env python3
"""
Deploy trained YOLO model to SageMaker for inference
"""

import boto3
import sagemaker
from sagemaker.pytorch import PyTorchModel
from sagemaker import get_execution_role
import json
from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main():
    print("Deploying YOLO Room Detection Model to SageMaker")
    print("=" * 60)

    # Check AWS access
    try:
        sts = boto3.client('sts')
        identity = sts.get_caller_identity()
        account_id = identity['Account']
        print(f"AWS Account: {account_id}")

        # Test services
        sagemaker_client = boto3.client('sagemaker')
        print("AWS services accessible")

    except Exception as e:
        print(f"AWS access failed: {e}")
        print("Check your .env file credentials")
        return

    # Get SageMaker session
    sess = sagemaker.Session()
    bucket = sess.default_bucket()
    print(f"Using S3 bucket: {bucket}")

    # Get IAM role
    iam = boto3.client('iam')
    role_name = 'room-detection-sagemaker-role'

    try:
        response = iam.get_role(RoleName=role_name)
        role = response['Role']['Arn']
        print(f"Using IAM role: {role}")
    except Exception as e:
        print(f"Failed to get IAM role: {e}")
        print("Make sure the role exists and has proper permissions")
        return

    # Check for trained model
    model_dir = Path("./local_training_output")
    model_files = list(model_dir.glob("**/weights/best.pt"))

    if not model_files:
        print("❌ No trained model found!")
        print("Please train a model first using train_yolo_local.py")
        return

    model_path = model_files[0]
    print(f"Found trained model: {model_path}")

    # Upload model to S3
    print("Uploading model to S3...")
    s3_prefix = 'room-detection-models'
    model_s3_key = f"{s3_prefix}/model.tar.gz"

    # Create model.tar.gz
    import tarfile
    import tempfile

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Copy model file
        import shutil
        model_copy = Path(tmp_dir) / "best.pt"
        shutil.copy2(model_path, model_copy)

        # Copy inference script
        inference_script = Path("./inference_sagemaker.py")
        if inference_script.exists():
            shutil.copy2(inference_script, Path(tmp_dir) / "inference.py")

        # Create tar.gz
        tar_path = Path(tmp_dir) / "model.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(model_copy, arcname="best.pt")
            if inference_script.exists():
                tar.add(Path(tmp_dir) / "inference.py", arcname="inference.py")

        # Upload to S3
        s3_client = boto3.client('s3')
        s3_client.upload_file(str(tar_path), bucket, model_s3_key)

    model_s3_uri = f"s3://{bucket}/{model_s3_key}"
    print(f"Model uploaded to: {model_s3_uri}")

    # Create SageMaker model
    print("Creating SageMaker model...")
    model_name = f'room-detection-yolo-{int(__import__("time").time())}'

    pytorch_model = PyTorchModel(
        model_data=model_s3_uri,
        role=role,
        entry_point='inference_sagemaker.py',
        source_dir='.',
        framework_version='2.0.0',
        py_version='py310',
        name=model_name
    )

    # Deploy to endpoint
    print("Deploying to endpoint...")
    endpoint_name = f'room-detection-endpoint-{int(__import__("time").time())}'

    predictor = pytorch_model.deploy(
        initial_instance_count=1,
        instance_type='ml.t2.medium',  # Cost-effective for inference
        endpoint_name=endpoint_name
    )

    print("🎉 Deployment successful!")
    print(f"Endpoint name: {endpoint_name}")
    print(f"Endpoint URL: https://runtime.sagemaker.{sess.boto_region_name}.amazonaws.com/endpoints/{endpoint_name}/invocations")

    # Save endpoint info
    with open('endpoint_info.json', 'w') as f:
        json.dump({
            'endpoint_name': endpoint_name,
            'model_name': model_name,
            'region': sess.boto_region_name
        }, f, indent=2)

    print("Endpoint info saved to endpoint_info.json")

    # Test the endpoint
    print("Testing endpoint...")
    try:
        # Create a simple test image (just for testing connectivity)
        import base64
        from PIL import Image
        import numpy as np

        # Create a small test image
        test_img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        img_buffer = __import__("io").BytesIO()
        test_img.save(img_buffer, format='PNG')
        img_bytes = img_buffer.getvalue()

        test_data = {
            'image': base64.b64encode(img_bytes).decode('utf-8')
        }

        result = predictor.predict(test_data)
        print("✅ Endpoint test successful!")
        print(f"Response: {result}")

    except Exception as e:
        print(f"⚠️ Endpoint test failed: {e}")
        print("The endpoint was created but may need debugging")

    print("\n📋 Summary:")
    print(f"Model: {model_name}")
    print(f"Endpoint: {endpoint_name}")
    print(f"Region: {sess.boto_region_name}")
    print("\nTo use this endpoint in your React app, update the API calls to:")
    print(f"https://runtime.sagemaker.{sess.boto_region_name}.amazonaws.com/endpoints/{endpoint_name}/invocations")

if __name__ == "__main__":
    main()
