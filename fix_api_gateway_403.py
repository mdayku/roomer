#!/usr/bin/env python3
"""
Fix API Gateway 403 error by ensuring proper CORS and resource configuration
This script checks all resources and ensures OPTIONS is configured correctly
"""
import boto3
from botocore.exceptions import ClientError

API_ID = 'qhhfso7u0h'
REGION = 'us-east-1'
STAGE = 'prod'
ALLOWED_ORIGIN = 'https://master.d7ra9ayxxa84o.amplifyapp.com'

def fix_api_gateway_403():
    """Fix 403 error by ensuring proper API Gateway configuration"""
    apigateway = boto3.client('apigateway', region_name=REGION)
    
    try:
        # Get all resources
        print("[INFO] Fetching API Gateway resources...")
        resources = apigateway.get_resources(restApiId=API_ID)
        
        print("\n=== Available Resources ===")
        for resource in resources['items']:
            print(f"  {resource['path']} (ID: {resource['id']})")
        
        # Find resources that might handle /api/detect
        # Could be root /, /{proxy+}, /api, or /api/{proxy+}
        target_resources = []
        for resource in resources['items']:
            path = resource['path']
            if path == '/' or path == '/{proxy+}' or path == '/api' or path == '/api/{proxy+}':
                target_resources.append(resource)
        
        if not target_resources:
            print("\n[ERROR] Could not find any suitable resource")
            return False
        
        # Use root resource if available, otherwise use proxy
        root_resource = next((r for r in target_resources if r['path'] == '/'), None)
        proxy_resource = next((r for r in target_resources if r['path'] == '/{proxy+}'), None)
        target_resource = root_resource or proxy_resource or target_resources[0]
        
        resource_id = target_resource['id']
        resource_path = target_resource['path']
        print(f"\n[OK] Using resource: {resource_path} (ID: {resource_id})")
        
        # Check POST method
        print("\n=== Checking POST Method ===")
        post_exists = False
        try:
            post_method = apigateway.get_method(restApiId=API_ID, resourceId=resource_id, httpMethod='POST')
            post_exists = True
            print("[OK] POST method exists")
            
            # Check integration
            try:
                integration = apigateway.get_integration(
                    restApiId=API_ID,
                    resourceId=resource_id,
                    httpMethod='POST'
                )
                print(f"  Integration Type: {integration.get('type')}")
                print(f"  Integration URI: {integration.get('uri', 'N/A')}")
            except Exception as e:
                print(f"  [WARN] No integration found: {e}")
        except ClientError as e:
            if e.response['Error']['Code'] == 'NotFoundException':
                print("[WARN] POST method does not exist - this might be the problem!")
            else:
                print(f"[ERROR] Error checking POST: {e}")
        
        # Ensure OPTIONS method exists and is configured correctly
        print("\n=== Configuring OPTIONS Method ===")
        try:
            # Try to get existing OPTIONS
            apigateway.get_method(restApiId=API_ID, resourceId=resource_id, httpMethod='OPTIONS')
            print("[OK] OPTIONS method exists, updating...")
            
            # Delete and recreate to ensure clean config
            try:
                apigateway.delete_method(restApiId=API_ID, resourceId=resource_id, httpMethod='OPTIONS')
            except:
                pass
        except ClientError:
            print("[INFO] OPTIONS method does not exist, creating...")
        
        # Create OPTIONS method
        apigateway.put_method(
            restApiId=API_ID,
            resourceId=resource_id,
            httpMethod='OPTIONS',
            authorizationType='NONE'
        )
        print("[OK] Created OPTIONS method")
        
        # Create MOCK integration for OPTIONS
        apigateway.put_integration(
            restApiId=API_ID,
            resourceId=resource_id,
            httpMethod='OPTIONS',
            type='MOCK',
            requestTemplates={'application/json': '{"statusCode": 200}'}
        )
        print("[OK] Created MOCK integration for OPTIONS")
        
        # Create method response
        apigateway.put_method_response(
            restApiId=API_ID,
            resourceId=resource_id,
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
        
        # Create integration response with specific origin
        apigateway.put_integration_response(
            restApiId=API_ID,
            resourceId=resource_id,
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
        
        # Ensure POST method response allows CORS headers
        if post_exists:
            print("\n=== Updating POST Method Response ===")
            try:
                # Get existing method response
                try:
                    method_response = apigateway.get_method_response(
                        restApiId=API_ID,
                        resourceId=resource_id,
                        httpMethod='POST',
                        statusCode='200'
                    )
                    response_parameters = method_response.get('responseParameters', {})
                except:
                    response_parameters = {}
                
                # Add CORS headers to method response
                response_parameters['method.response.header.Access-Control-Allow-Origin'] = True
                response_parameters['method.response.header.Access-Control-Allow-Headers'] = True
                response_parameters['method.response.header.Access-Control-Allow-Methods'] = True
                
                apigateway.put_method_response(
                    restApiId=API_ID,
                    resourceId=resource_id,
                    httpMethod='POST',
                    statusCode='200',
                    responseParameters=response_parameters,
                    responseModels={'application/json': 'Empty'}
                )
                print("[OK] Updated POST method response to allow CORS headers")
            except Exception as e:
                print(f"[WARN] Could not update POST method response: {e}")
        
        # Deploy API
        print("\n=== Deploying API ===")
        deployment = apigateway.create_deployment(
            restApiId=API_ID,
            stageName=STAGE,
            description='Fix 403 error - CORS configuration'
        )
        print(f"[OK] API deployed successfully!")
        print(f"[INFO] API URL: https://{API_ID}.execute-api.{REGION}.amazonaws.com/{STAGE}")
        
        return True
        
    except Exception as e:
        print(f"\n[ERROR] Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    print("=" * 60)
    print("Fixing API Gateway 403 Error")
    print("=" * 60)
    print(f"API ID: {API_ID}")
    print(f"Region: {REGION}")
    print(f"Stage: {STAGE}")
    print(f"Allowed Origin: {ALLOWED_ORIGIN}\n")
    
    success = fix_api_gateway_403()
    
    if success:
        print("\n" + "=" * 60)
        print("[OK] API Gateway configuration updated!")
        print("[INFO] The 403 error should be resolved.")
        print("[INFO] Try your API call again.")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("[ERROR] Failed to fix API Gateway configuration")
        print("=" * 60)

