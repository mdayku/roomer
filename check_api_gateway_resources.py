#!/usr/bin/env python3
"""Check API Gateway resource structure and methods"""
import boto3
from botocore.exceptions import ClientError

API_ID = 'qhhfso7u0h'
REGION = 'us-east-1'

apigateway = boto3.client('apigateway', region_name=REGION)

print("=== API Gateway Resources ===\n")

try:
    resources = apigateway.get_resources(restApiId=API_ID)
    
    for resource in resources['items']:
        path = resource['path']
        resource_id = resource['id']
        print(f"Resource: {path} (ID: {resource_id})")
        
        # Check what methods exist on this resource
        try:
            methods = apigateway.get_resource(restApiId=API_ID, resourceId=resource_id)
            if 'resourceMethods' in methods:
                for method_name in methods['resourceMethods'].keys():
                    print(f"  - {method_name}")
                    
                    # Get method details
                    try:
                        method = apigateway.get_method(
                            restApiId=API_ID,
                            resourceId=resource_id,
                            httpMethod=method_name
                        )
                        
                        # Check integration
                        try:
                            integration = apigateway.get_integration(
                                restApiId=API_ID,
                                resourceId=resource_id,
                                httpMethod=method_name
                            )
                            print(f"    Integration: {integration.get('type')}")
                            if integration.get('type') == 'AWS_PROXY':
                                print(f"    URI: {integration.get('uri', 'N/A')}")
                        except:
                            print(f"    Integration: None")
                    except Exception as e:
                        print(f"    Error getting method: {e}")
            else:
                print("  (no methods)")
        except Exception as e:
            print(f"  Error: {e}")
        
        print()
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

