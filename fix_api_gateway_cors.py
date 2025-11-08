#!/usr/bin/env python3
"""
Fix CORS configuration for Amplify-deployed API Gateway
"""
import boto3
from botocore.exceptions import ClientError

API_ID = 'qhhfso7u0h'
REGION = 'us-east-1'
STAGE = 'prod'
ALLOWED_ORIGIN = 'https://master.d7ra9ayxxa84o.amplifyapp.com'

def fix_api_gateway_cors():
    """Configure CORS for the Amplify API Gateway"""
    apigateway = boto3.client('apigateway', region_name=REGION)
    
    try:
        # Get all resources
        resources = apigateway.get_resources(restApiId=API_ID)
        
        # Find the proxy resource (usually {proxy+}) or /api resource
        proxy_resource = None
        api_resource = None
        root_resource = None
        
        for resource in resources['items']:
            if resource['path'] == '/{proxy+}':
                proxy_resource = resource
            elif resource['path'] == '/api':
                api_resource = resource
            elif resource['path'] == '/api/{proxy+}':
                proxy_resource = resource
            elif resource['path'] == '/':
                root_resource = resource
        
        # Use proxy resource if found, otherwise fall back to root
        target_resource = proxy_resource or api_resource or root_resource
        
        if not target_resource:
            print("[ERROR] Could not find any resource. Available resources:")
            for r in resources['items']:
                print(f"  - {r['path']} (ID: {r['id']})")
            return False
        
        resource_id = target_resource['id']
        resource_path = target_resource['path']
        print(f"[OK] Found resource: {resource_path} (ID: {resource_id})")
        
        # Check if POST method exists and has integration
        post_method_exists = False
        integration_exists = False
        lambda_arn = None
        
        try:
            method = apigateway.get_method(restApiId=API_ID, resourceId=resource_id, httpMethod='POST')
            post_method_exists = True
            print("[OK] POST method already exists")
            
            # Check if integration exists
            try:
                integration = apigateway.get_integration(restApiId=API_ID, resourceId=resource_id, httpMethod='POST')
                integration_exists = True
                if 'uri' in integration:
                    lambda_arn = integration['uri'].split('/')[-1] if 'lambda' in integration['uri'] else None
                    print(f"[OK] Integration exists: {integration.get('type', 'unknown')}")
            except ClientError:
                print("[WARN] POST method exists but no integration found")
        except ClientError as e:
            if e.response['Error']['Code'] == 'NotFoundException':
                print("[WARN] POST method not found. Creating...")
                # Create POST method
                apigateway.put_method(
                    restApiId=API_ID,
                    resourceId=resource_id,
                    httpMethod='POST',
                    authorizationType='NONE'
                )
                print("[OK] Created POST method")
                post_method_exists = True
            else:
                raise
        
        # If no integration exists, try to find Amplify Lambda function
        if not integration_exists:
            print("[INFO] Attempting to find Amplify Lambda function...")
            lambda_client = boto3.client('lambda', region_name=REGION)
            try:
                # List Lambda functions that might be Amplify-related
                functions = lambda_client.list_functions()['Functions']
                amplify_functions = [f for f in functions if 'amplify' in f['FunctionName'].lower() or 'roomer' in f['FunctionName'].lower()]
                
                if amplify_functions:
                    # Use the first one found (you might need to adjust this)
                    lambda_func = amplify_functions[0]
                    lambda_arn = lambda_func['FunctionArn']
                    print(f"[OK] Found Lambda function: {lambda_func['FunctionName']}")
                    
                    # Create AWS_PROXY integration
                    apigateway.put_integration(
                        restApiId=API_ID,
                        resourceId=resource_id,
                        httpMethod='POST',
                        type='AWS_PROXY',
                        integrationHttpMethod='POST',
                        uri=f'arn:aws:apigateway:{REGION}:lambda:path/2015-03-31/functions/{lambda_arn}/invocations'
                    )
                    print("[OK] Created Lambda integration for POST method")
                    integration_exists = True
                else:
                    print("[WARN] No Amplify Lambda functions found. You may need to configure integration manually.")
            except Exception as e:
                print(f"[WARN] Could not find Lambda function: {e}")
        
        # Ensure OPTIONS method exists for CORS preflight
        options_exists = False
        try:
            apigateway.get_method(restApiId=API_ID, resourceId=resource_id, httpMethod='OPTIONS')
            options_exists = True
            print("[OK] OPTIONS method already exists")
        except ClientError as e:
            if e.response['Error']['Code'] == 'NotFoundException':
                print("[WARN] OPTIONS method not found. Creating...")
                # Create OPTIONS method
                apigateway.put_method(
                    restApiId=API_ID,
                    resourceId=resource_id,
                    httpMethod='OPTIONS',
                    authorizationType='NONE'
                )
                
                # Create MOCK integration for OPTIONS
                apigateway.put_integration(
                    restApiId=API_ID,
                    resourceId=resource_id,
                    httpMethod='OPTIONS',
                    type='MOCK',
                    requestTemplates={'application/json': '{"statusCode": 200}'}
                )
                options_exists = True
            else:
                raise
        
        # Always update OPTIONS integration response to ensure correct origin (remove any wildcard '*')
        if options_exists:
            print("[INFO] Updating OPTIONS integration response CORS headers to use specific origin...")
            try:
                # Get existing integration response to preserve other settings
                try:
                    existing = apigateway.get_integration_response(
                        restApiId=API_ID,
                        resourceId=resource_id,
                        httpMethod='OPTIONS',
                        statusCode='200'
                    )
                    response_params = existing.get('responseParameters', {})
                except:
                    response_params = {}
                
                # CRITICAL: Set specific origin, NOT '*'
                response_params['method.response.header.Access-Control-Allow-Origin'] = f"'{ALLOWED_ORIGIN}'"
                response_params['method.response.header.Access-Control-Allow-Headers'] = "'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token'"
                response_params['method.response.header.Access-Control-Allow-Methods'] = "'OPTIONS,POST,GET'"
                
                # Update integration response
                apigateway.put_integration_response(
                    restApiId=API_ID,
                    resourceId=resource_id,
                    httpMethod='OPTIONS',
                    statusCode='200',
                    responseParameters=response_params,
                    responseTemplates={'application/json': ''}
                )
                print(f"[OK] Updated OPTIONS integration response to use origin: {ALLOWED_ORIGIN}")
            except Exception as e:
                print(f"[WARN] Could not update OPTIONS integration response: {e}")
                import traceback
                traceback.print_exc()
        
        # Update POST method response to include CORS headers
        try:
            # Try to get existing method response (might not exist)
            try:
                method_response = apigateway.get_method_response(
                    restApiId=API_ID,
                    resourceId=resource_id,
                    httpMethod='POST',
                    statusCode='200'
                )
                response_parameters = method_response.get('responseParameters', {})
                response_models = method_response.get('responseModels', {})
            except ClientError as e:
                if e.response['Error']['Code'] == 'NotFoundException':
                    # Method response doesn't exist, create it
                    response_parameters = {}
                    response_models = {}
                else:
                    raise
            
            # Update method response to include CORS headers
            response_parameters['method.response.header.Access-Control-Allow-Origin'] = True
            response_parameters['method.response.header.Access-Control-Allow-Headers'] = True
            response_parameters['method.response.header.Access-Control-Allow-Methods'] = True
            
            apigateway.put_method_response(
                restApiId=API_ID,
                resourceId=resource_id,
                httpMethod='POST',
                statusCode='200',
                responseParameters=response_parameters,
                responseModels=response_models
            )
            print("[OK] Updated POST method response with CORS headers")
            
            # Update integration response for POST (only if integration exists)
            if integration_exists:
                try:
                    integration_response = apigateway.get_integration_response(
                        restApiId=API_ID,
                        resourceId=resource_id,
                        httpMethod='POST',
                        statusCode='200'
                    )
                    
                    response_params = integration_response.get('responseParameters', {})
                    response_params['method.response.header.Access-Control-Allow-Origin'] = f"'{ALLOWED_ORIGIN}'"
                    response_params['method.response.header.Access-Control-Allow-Headers'] = "'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token'"
                    response_params['method.response.header.Access-Control-Allow-Methods'] = "'POST,OPTIONS'"
                    
                    apigateway.put_integration_response(
                        restApiId=API_ID,
                        resourceId=resource_id,
                        httpMethod='POST',
                        statusCode='200',
                        responseParameters=response_params,
                        responseTemplates=integration_response.get('responseTemplates', {})
                    )
                    print("[OK] Updated POST integration response with CORS headers")
                except ClientError as e:
                    if e.response['Error']['Code'] == 'NotFoundException':
                        # For AWS_PROXY, integration responses are handled by Lambda, so we skip
                        print("[INFO] Skipping integration response (AWS_PROXY handles responses)")
                    else:
                        print(f"[WARN] Could not update integration response: {e}")
            else:
                print("[INFO] Skipping integration response update (no integration configured)")
        except ClientError as e:
            print(f"[WARN] Could not update POST method response: {e}")
        
        # Deploy the API
        print("\n[INFO] Deploying API...")
        deployment = apigateway.create_deployment(
            restApiId=API_ID,
            stageName=STAGE,
            description='CORS configuration update'
        )
        print(f"[OK] API deployed successfully!")
        print(f"[INFO] API URL: https://{API_ID}.execute-api.{REGION}.amazonaws.com/{STAGE}")
        
        return True
        
    except ClientError as e:
        print(f"[ERROR] Error: {e}")
        return False

if __name__ == '__main__':
    print("Fixing API Gateway CORS configuration...")
    print(f"API ID: {API_ID}")
    print(f"Region: {REGION}")
    print(f"Allowed Origin: {ALLOWED_ORIGIN}\n")
    
    success = fix_api_gateway_cors()
    
    if success:
        print("\n[OK] CORS configuration complete!")
        print("[INFO] Your API should now accept requests from the Amplify frontend.")
    else:
        print("\n[ERROR] Failed to configure CORS. Please check the errors above.")

