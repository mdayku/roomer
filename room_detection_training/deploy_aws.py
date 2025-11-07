"""
AWS Deployment Infrastructure for Room Detection Service
Creates SageMaker Endpoint, API Gateway, Lambda, and S3 storage
"""

import boto3
import json
import zipfile
import os
from pathlib import Path
import base64
from botocore.exceptions import ClientError

# AWS Configuration
REGION = 'us-east-1'  # Change to your preferred region
S3_BUCKET = None  # Will be created
STACK_NAME = 'room-detection-service'
MODEL_NAME = 'room-detection-yolo'
ENDPOINT_NAME = f'{MODEL_NAME}-endpoint'
API_NAME = f'{MODEL_NAME}-api'
LAMBDA_NAME = f'{MODEL_NAME}-inference'

# IAM Role ARN (needs to be created with proper permissions)
IAM_ROLE_ARN = None  # Will be created or you can specify existing

def create_s3_bucket(bucket_name):
    """Create S3 bucket for model artifacts and results"""
    s3 = boto3.client('s3', region_name=REGION)

    try:
        if REGION == 'us-east-1':
            s3.create_bucket(Bucket=bucket_name)
        else:
            s3.create_bucket(
                Bucket=bucket_name,
                CreateBucketConfiguration={'LocationConstraint': REGION}
            )
        print(f"✅ Created S3 bucket: {bucket_name}")
        return bucket_name
    except ClientError as e:
        if e.response['Error']['Code'] == 'BucketAlreadyExists':
            print(f"⚠️ Bucket {bucket_name} already exists")
            return bucket_name
        raise

def create_iam_role():
    """Create IAM role for SageMaker"""
    iam = boto3.client('iam', region_name=REGION)

    trust_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Service": "sagemaker.amazonaws.com"},
                "Action": "sts:AssumeRole"
            },
            {
                "Effect": "Allow",
                "Principal": {"Service": "lambda.amazonaws.com"},
                "Action": "sts:AssumeRole"
            }
        ]
    }

    try:
        response = iam.create_role(
            RoleName=f'{MODEL_NAME}-role',
            AssumeRolePolicyDocument=json.dumps(trust_policy),
            Description='IAM role for Room Detection ML service'
        )

        role_arn = response['Role']['Arn']
        print(f"✅ Created IAM role: {role_arn}")

        # Attach necessary policies
        policies = [
            'arn:aws:iam::aws:policy/AmazonSageMakerFullAccess',
            'arn:aws:iam::aws:policy/AmazonS3FullAccess',
            'arn:aws:iam::aws:policy/CloudWatchLogsFullAccess',
            'arn:aws:iam::aws:policy/AWSLambda_FullAccess'
        ]

        for policy in policies:
            iam.attach_role_policy(RoleName=f'{MODEL_NAME}-role', PolicyArn=policy)

        print("✅ Attached required policies")
        return role_arn

    except ClientError as e:
        if e.response['Error']['Code'] == 'EntityAlreadyExists':
            # Get existing role ARN
            response = iam.get_role(RoleName=f'{MODEL_NAME}-role')
            role_arn = response['Role']['Arn']
            print(f"✅ Using existing IAM role: {role_arn}")
            return role_arn
        raise

def create_model_package(model_artifact_s3_uri, role_arn):
    """Create SageMaker model package"""
    sm = boto3.client('sagemaker', region_name=REGION)

    model_data_url = model_artifact_s3_uri

    # For YOLO, we'll use a PyTorch container
    image_uri = f'763104351884.dkr.ecr.{REGION}.amazonaws.com/pytorch-inference:2.0.0-gpu-py310'

    try:
        response = sm.create_model(
            ModelName=MODEL_NAME,
            PrimaryContainer={
                'Image': image_uri,
                'ModelDataUrl': model_data_url,
                'Environment': {
                    'SAGEMAKER_PROGRAM': 'inference.py',  # Custom inference script
                    'SAGEMAKER_MODEL_SERVER_TIMEOUT': '300',  # 5 minutes
                    'SAGEMAKER_MODEL_SERVER_WORKERS': '1'
                }
            },
            ExecutionRoleArn=role_arn
        )

        print(f"✅ Created SageMaker model: {MODEL_NAME}")
        return response['ModelArn']

    except ClientError as e:
        print(f"❌ Failed to create model: {e}")
        raise

def create_endpoint_config():
    """Create SageMaker endpoint configuration"""
    sm = boto3.client('sagemaker', region_name=REGION)

    endpoint_config_name = f'{MODEL_NAME}-config'

    try:
        response = sm.create_endpoint_config(
            EndpointConfigName=endpoint_config_name,
            ProductionVariants=[
                {
                    'VariantName': 'primary',
                    'ModelName': MODEL_NAME,
                    'InstanceType': 'ml.g4dn.xlarge',  # T4 GPU for inference
                    'InitialInstanceCount': 1,
                    'InitialVariantWeight': 1.0,
                    # Enable autoscaling (optional)
                    # 'ManagedInstanceScaling': {
                    #     'Status': 'ENABLED',
                    #     'MinInstanceCount': 1,
                    #     'MaxInstanceCount': 4
                    # }
                }
            ]
        )

        print(f"✅ Created endpoint config: {endpoint_config_name}")
        return endpoint_config_name

    except ClientError as e:
        print(f"❌ Failed to create endpoint config: {e}")
        raise

def create_endpoint(endpoint_config_name):
    """Create SageMaker endpoint"""
    sm = boto3.client('sagemaker', region_name=REGION)

    try:
        response = sm.create_endpoint(
            EndpointName=ENDPOINT_NAME,
            EndpointConfigName=endpoint_config_name
        )

        print(f"✅ Creating endpoint: {ENDPOINT_NAME}")
        print("This may take 5-10 minutes...")

        # Wait for endpoint to be ready
        waiter = sm.get_waiter('endpoint_in_service')
        waiter.wait(EndpointName=ENDPOINT_NAME)

        print(f"✅ Endpoint ready: {ENDPOINT_NAME}")
        return ENDPOINT_NAME

    except ClientError as e:
        print(f"❌ Failed to create endpoint: {e}")
        raise

def create_lambda_function(role_arn, s3_bucket):
    """Create Lambda function for inference coordination"""
    lambda_client = boto3.client('lambda', region_name=REGION)

    # Create Lambda deployment package
    lambda_code = f'''
import json
import boto3
import base64
import uuid
from datetime import datetime

def lambda_handler(event, context):
    """
    Lambda function to coordinate room detection inference
    Receives blueprint image, calls SageMaker endpoint, stores results in S3
    """

    try:
        # Parse input
        if 'body' in event:
            if event.get('isBase64Encoded'):
                image_data = base64.b64decode(event['body'])
            else:
                image_data = event['body'].encode('utf-8')
        else:
            return {{
                'statusCode': 400,
                'body': json.dumps({{'error': 'No image data provided'}})
            }}

        # Generate unique ID for this inference
        inference_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()

        # Save input image to S3
        s3 = boto3.client('s3')
        input_key = f"inputs/{{inference_id}}.png"
        s3.put_object(
            Bucket='{s3_bucket}',
            Key=input_key,
            Body=image_data,
            ContentType='image/png'
        )

        # Call SageMaker endpoint
        sm_runtime = boto3.client('sagemaker-runtime')

        # Prepare input for SageMaker (base64 encoded image)
        payload = {{
            'image': base64.b64encode(image_data).decode('utf-8'),
            'inference_id': inference_id
        }}

        response = sm_runtime.invoke_endpoint(
            EndpointName='{ENDPOINT_NAME}',
            ContentType='application/json',
            Body=json.dumps(payload)
        )

        # Parse results
        result = json.loads(response['Body'].read().decode())

        # Save results to S3
        result_key = f"results/{{inference_id}}.json"
        result_data = {{
            'inference_id': inference_id,
            'timestamp': timestamp,
            'input_image': input_key,
            'result': result,
            'status': 'completed'
        }}

        s3.put_object(
            Bucket='{s3_bucket}',
            Key=result_key,
            Body=json.dumps(result_data, indent=2),
            ContentType='application/json'
        )

        # Return result with S3 URLs
        return {{
            'statusCode': 200,
            'headers': {{
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type',
                'Access-Control-Allow-Methods': 'OPTIONS,POST,GET'
            }},
            'body': json.dumps({{
                'inference_id': inference_id,
                'timestamp': timestamp,
                'rooms': result.get('rooms', []),
                'input_url': f"https://{s3_bucket}.s3.{REGION}.amazonaws.com/{{input_key}}",
                'result_url': f"https://{s3_bucket}.s3.{REGION}.amazonaws.com/{{result_key}}",
                'status': 'completed'
            }})
        }}

    except Exception as e:
        print(f"Error: {{str(e)}}")
        return {{
            'statusCode': 500,
            'body': json.dumps({{'error': str(e)}})
        }}
'''

    # Create ZIP file
    with zipfile.ZipFile('lambda_function.zip', 'w') as zip_file:
        zip_file.writestr('lambda_function.py', lambda_code)

    # Read ZIP content
    with open('lambda_function.zip', 'rb') as f:
        zip_content = f.read()

    try:
        response = lambda_client.create_function(
            FunctionName=LAMBDA_NAME,
            Runtime='python3.9',
            Role=role_arn,
            Handler='lambda_function.lambda_handler',
            Code={'ZipFile': zip_content},
            Description='Room Detection Inference Coordinator',
            Timeout=300,  # 5 minutes (matches SageMaker timeout)
            MemorySize=1024,  # 1GB RAM
            Environment={
                'Variables': {
                    'S3_BUCKET': s3_bucket,
                    'SAGEMAKER_ENDPOINT': ENDPOINT_NAME,
                    'REGION': REGION
                }
            }
        )

        print(f"✅ Created Lambda function: {LAMBDA_NAME}")
        return response['FunctionArn']

    except ClientError as e:
        print(f"❌ Failed to create Lambda function: {e}")
        raise

def create_api_gateway(lambda_arn):
    """Create API Gateway for the Lambda function"""
    apigateway = boto3.client('apigateway', region_name=REGION)

    # Create REST API
    try:
        api_response = apigateway.create_rest_api(
            name=API_NAME,
            description='Room Detection Service API',
            endpointConfiguration={'types': ['REGIONAL']}
        )

        api_id = api_response['id']
        print(f"✅ Created API Gateway: {API_NAME} (ID: {api_id})")

        # Get root resource
        resources = apigateway.get_resources(restApiId=api_id)
        root_id = next(r['id'] for r in resources['items'] if r['path'] == '/')

        # Create /detect resource
        resource_response = apigateway.create_resource(
            restApiId=api_id,
            parentId=root_id,
            pathPart='detect'
        )
        resource_id = resource_response['id']

        # Add POST method
        apigateway.put_method(
            restApiId=api_id,
            resourceId=resource_id,
            httpMethod='POST',
            authorizationType='NONE',
            apiKeyRequired=False
        )

        # Add Lambda integration
        apigateway.put_integration(
            restApiId=api_id,
            resourceId=resource_id,
            httpMethod='POST',
            type='AWS_PROXY',
            integrationHttpMethod='POST',
            uri=f'arn:aws:apigateway:{REGION}:lambda:path/2015-03-31/functions/{lambda_arn}/invocations'
        )

        # Add method response
        apigateway.put_method_response(
            restApiId=api_id,
            resourceId=resource_id,
            httpMethod='POST',
            statusCode='200'
        )

        # Add integration response
        apigateway.put_integration_response(
            restApiId=api_id,
            resourceId=resource_id,
            httpMethod='POST',
            statusCode='200'
        )

        # Add OPTIONS method for CORS
        apigateway.put_method(
            restApiId=api_id,
            resourceId=resource_id,
            httpMethod='OPTIONS',
            authorizationType='NONE'
        )

        apigateway.put_integration(
            restApiId=api_id,
            resourceId=resource_id,
            httpMethod='OPTIONS',
            type='MOCK',
            requestTemplates={'application/json': '{"statusCode": 200}'}
        )

        apigateway.put_method_response(
            restApiId=api_id,
            resourceId=resource_id,
            httpMethod='OPTIONS',
            statusCode='200',
            responseHeaders={
                'Access-Control-Allow-Origin': "'*'",
                'Access-Control-Allow-Headers': "'Content-Type'",
                'Access-Control-Allow-Methods': "'OPTIONS,POST'"
            }
        )

        apigateway.put_integration_response(
            restApiId=api_id,
            resourceId=resource_id,
            httpMethod='OPTIONS',
            statusCode='200',
            responseTemplates={'application/json': ''}
        )

        # Deploy API
        deployment = apigateway.create_deployment(
            restApiId=api_id,
            stageName='prod',
            description='Production deployment'
        )

        api_url = f'https://{api_id}.execute-api.{REGION}.amazonaws.com/prod'
        print(f"✅ API deployed at: {api_url}")

        # Add Lambda permission for API Gateway
        lambda_client = boto3.client('lambda', region_name=REGION)
        lambda_client.add_permission(
            FunctionName=LAMBDA_NAME,
            StatementId='AllowAPIGatewayInvoke',
            Action='lambda:InvokeFunction',
            Principal='apigateway.amazonaws.com',
            SourceArn=f'arn:aws:apigateway:{REGION}::/restapis/{api_id}/*'
        )

        return api_url

    except ClientError as e:
        print(f"❌ Failed to create API Gateway: {e}")
        raise

def main():
    """Deploy complete AWS infrastructure for Room Detection service"""
    print("🚀 Deploying AWS Room Detection Service")
    print("=" * 50)

    global S3_BUCKET, IAM_ROLE_ARN

    try:
        # 1. Create S3 bucket
        print("\n📦 Step 1: Creating S3 bucket...")
        account_id = boto3.client('sts').get_caller_identity()['Account']
        S3_BUCKET = f'room-detection-{account_id}'
        s3_bucket = create_s3_bucket(S3_BUCKET)

        # 2. Create IAM role
        print("\n🔐 Step 2: Creating IAM role...")
        role_arn = create_iam_role()
        IAM_ROLE_ARN = role_arn

        # 3. Upload model to S3 (you'll need to upload your trained model first)
        print("\n📤 Step 3: Upload model artifact...")
        model_artifact_key = 'models/room_detection_yolo/model.tar.gz'
        print(f"⚠️ Please upload your trained model to: s3://{s3_bucket}/{model_artifact_key}")
        print("Then update MODEL_ARTIFACT_S3_URI below")

        # For now, we'll use a placeholder - you'll need to upload your actual model
        model_artifact_s3_uri = f's3://{s3_bucket}/{model_artifact_key}'

        # 4. Create SageMaker model
        print("\n🤖 Step 4: Creating SageMaker model...")
        model_arn = create_model_package(model_artifact_s3_uri, role_arn)

        # 5. Create endpoint configuration
        print("\n⚙️ Step 5: Creating endpoint configuration...")
        endpoint_config_name = create_endpoint_config()

        # 6. Create SageMaker endpoint
        print("\n🌐 Step 6: Creating SageMaker endpoint...")
        endpoint_name = create_endpoint(endpoint_config_name)

        # 7. Create Lambda function
        print("\n⚡ Step 7: Creating Lambda function...")
        lambda_arn = create_lambda_function(role_arn, s3_bucket)

        # 8. Create API Gateway
        print("\n🚪 Step 8: Creating API Gateway...")
        api_url = create_api_gateway(lambda_arn)

        # Save configuration
        config = {
            's3_bucket': s3_bucket,
            'iam_role_arn': role_arn,
            'model_name': MODEL_NAME,
            'endpoint_name': endpoint_name,
            'lambda_name': LAMBDA_NAME,
            'api_url': api_url,
            'region': REGION,
            'deployed_at': str(datetime.now())
        }

        with open('aws_config.json', 'w') as f:
            json.dump(config, f, indent=2)

        print("\n🎉 AWS Deployment Complete!")
        print("=" * 50)
        print(f"API Endpoint: {api_url}/detect")
        print(f"S3 Bucket: {s3_bucket}")
        print(f"SageMaker Endpoint: {endpoint_name}")
        print(f"Configuration saved: aws_config.json")

        print("\n📋 Integration Instructions:")
        print("1. Upload your trained model to S3")
        print("2. Update your React frontend to POST to the API endpoint")
        print("3. Test with a sample blueprint image")

        print("\n💰 Estimated Monthly Cost:")
        print("- SageMaker Endpoint: $50-100/month (with autoscaling)")
        print("- S3 Storage: $1-5/month")
        print("- API Gateway: $1-3/month")
        print("- Lambda: $0.20 per 1K requests")

    except Exception as e:
        print(f"\n❌ Deployment failed: {e}")
        print("Check AWS credentials and permissions")
        raise

if __name__ == "__main__":
    main()
