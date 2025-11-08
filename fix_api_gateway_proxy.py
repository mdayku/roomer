#!/usr/bin/env python3
"""
Fix API Gateway to handle /api/detect by creating a proxy resource
"""
import boto3
from botocore.exceptions import ClientError

API_ID = 'qhhfso7u0h'
REGION = 'us-east-1'
STAGE = 'prod'
ALLOWED_ORIGIN = 'https://master.d7ra9ayxxa84o.amplifyapp.com'

def fix_api_gateway_proxy():
    """Create proxy resource to handle all paths including /api/detect"""
    apigateway = boto3.client('apigateway', region_name=REGION)
    
    try:
        # Get all resources
        resources = apigateway.get_resources(restApiId=API_ID)
        
        # Find root resource
        root_resource = next((r for r in resources['items'] if r['path'] == '/'), None)
        if not root_resource:
            print("[ERROR] Could not find root resource")
            return False
        
        root_id = root_resource['id']
        print(f"[OK] Found root resource: / (ID: {root_id})")
        
        # Get Lambda ARN from existing POST integration
        try:
            integration = apigateway.get_integration(
                restApiId=API_ID,
                resourceId=root_id,
                httpMethod='POST'
            )
            # Extract Lambda ARN from URI: arn:aws:apigateway:.../functions/{lambda_arn}/invocations
            uri = integration['uri']
            # Find the functions/ part and extract everything after it until /invocations
            if '/functions/' in uri:
                lambda_arn = uri.split('/functions/')[1].split('/invocations')[0]
                print(f"[OK] Found Lambda ARN: {lambda_arn}")
            else:
                print(f"[ERROR] Could not parse Lambda ARN from URI: {uri}")
                return False
        except Exception as e:
            print(f"[ERROR] Could not get Lambda ARN: {e}")
            return False
        
        # Check if proxy resource already exists
        proxy_resource = next((r for r in resources['items'] if r['path'] == '/{proxy+}'), None)
        
        if proxy_resource:
            proxy_id = proxy_resource['id']
            print(f"[OK] Proxy resource already exists: /{{proxy+}} (ID: {proxy_id})")
        else:
            # Create proxy resource
            print("[INFO] Creating proxy resource /{proxy+}...")
            proxy_resource = apigateway.create_resource(
                restApiId=API_ID,
                parentId=root_id,
                pathPart='{proxy+}'
            )
            proxy_id = proxy_resource['id']
            print(f"[OK] Created proxy resource: /{{proxy+}} (ID: {proxy_id})")
        
        # Configure POST method on proxy
        print("[INFO] Configuring POST method on proxy resource...")
        try:
            apigateway.get_method(restApiId=API_ID, resourceId=proxy_id, httpMethod='POST')
            print("[OK] POST method already exists on proxy")
        except ClientError:
            # Create POST method
            apigateway.put_method(
                restApiId=API_ID,
                resourceId=proxy_id,
                httpMethod='POST',
                authorizationType='NONE'
            )
            print("[OK] Created POST method on proxy")
        
        # Set up AWS_PROXY integration for POST
        apigateway.put_integration(
            restApiId=API_ID,
            resourceId=proxy_id,
            httpMethod='POST',
            type='AWS_PROXY',
            integrationHttpMethod='POST',
            uri=f'arn:aws:apigateway:{REGION}:lambda:path/2015-03-31/functions/{lambda_arn}/invocations'
        )
        print("[OK] Configured POST integration on proxy")
        
        # Configure OPTIONS method on proxy for CORS
        print("[INFO] Configuring OPTIONS method on proxy resource...")
        try:
            apigateway.get_method(restApiId=API_ID, resourceId=proxy_id, httpMethod='OPTIONS')
            print("[OK] OPTIONS method already exists on proxy, updating...")
            # Delete and recreate to ensure clean config
            try:
                apigateway.delete_method(restApiId=API_ID, resourceId=proxy_id, httpMethod='OPTIONS')
            except:
                pass
        except ClientError:
            pass
        
        # Create OPTIONS method
        apigateway.put_method(
            restApiId=API_ID,
            resourceId=proxy_id,
            httpMethod='OPTIONS',
            authorizationType='NONE'
        )
        print("[OK] Created OPTIONS method on proxy")
        
        # Create MOCK integration for OPTIONS
        apigateway.put_integration(
            restApiId=API_ID,
            resourceId=proxy_id,
            httpMethod='OPTIONS',
            type='MOCK',
            requestTemplates={'application/json': '{"statusCode": 200}'}
        )
        print("[OK] Created MOCK integration for OPTIONS")
        
        # Create method response for OPTIONS
        apigateway.put_method_response(
            restApiId=API_ID,
            resourceId=proxy_id,
            httpMethod='OPTIONS',
            statusCode='200',
            responseParameters={
                'method.response.header.Access-Control-Allow-Origin': True,
                'method.response.header.Access-Control-Allow-Headers': True,
                'method.response.header.Access-Control-Allow-Methods': True,
                'method.response.header.Access-Control-Allow-Credentials': True,
            },
            responseModels={'application/json': 'Empty'}
        )
        print("[OK] Created method response for OPTIONS")
        
        # Create integration response for OPTIONS
        apigateway.put_integration_response(
            restApiId=API_ID,
            resourceId=proxy_id,
            httpMethod='OPTIONS',
            statusCode='200',
            responseParameters={
                'method.response.header.Access-Control-Allow-Origin': f"'{ALLOWED_ORIGIN}'",
                'method.response.header.Access-Control-Allow-Headers': "'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token'",
                'method.response.header.Access-Control-Allow-Methods': "'OPTIONS,POST,GET'",
                'method.response.header.Access-Control-Allow-Credentials': "'true'",
            },
            responseTemplates={'application/json': ''}
        )
        print(f"[OK] Created integration response with origin: {ALLOWED_ORIGIN}")
        
        # Also configure GET method (for health checks, etc.)
        print("[INFO] Configuring GET method on proxy resource...")
        try:
            apigateway.get_method(restApiId=API_ID, resourceId=proxy_id, httpMethod='GET')
            print("[OK] GET method already exists on proxy")
        except ClientError:
            apigateway.put_method(
                restApiId=API_ID,
                resourceId=proxy_id,
                httpMethod='GET',
                authorizationType='NONE'
            )
            apigateway.put_integration(
                restApiId=API_ID,
                resourceId=proxy_id,
                httpMethod='GET',
                type='AWS_PROXY',
                integrationHttpMethod='POST',
                uri=f'arn:aws:apigateway:{REGION}:lambda:path/2015-03-31/functions/{lambda_arn}/invocations'
            )
            print("[OK] Created GET method on proxy")
        
        # Deploy API
        print("\n[INFO] Deploying API...")
        deployment = apigateway.create_deployment(
            restApiId=API_ID,
            stageName=STAGE,
            description='Add proxy resource for /api/detect routing'
        )
        print(f"[OK] API deployed successfully!")
        print(f"[INFO] API URL: https://{API_ID}.execute-api.{REGION}.amazonaws.com/{STAGE}")
        print(f"[INFO] Test endpoint: https://{API_ID}.execute-api.{REGION}.amazonaws.com/{STAGE}/api/detect")
        
        return True
        
    except Exception as e:
        print(f"\n[ERROR] Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    print("=" * 60)
    print("Fixing API Gateway Proxy Resource")
    print("=" * 60)
    print(f"API ID: {API_ID}")
    print(f"Region: {REGION}")
    print(f"Stage: {STAGE}\n")
    
    success = fix_api_gateway_proxy()
    
    if success:
        print("\n" + "=" * 60)
        print("[OK] Proxy resource configured!")
        print("[INFO] The /api/detect endpoint should now work.")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("[ERROR] Failed to configure proxy resource")
        print("=" * 60)

