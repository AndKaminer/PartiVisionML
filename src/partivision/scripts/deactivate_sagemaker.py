import argparse
import os

import boto3

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--endpoint', required=True)
    parsed = parser.parse_args()
    endpoint_name = parsed.endpoint

    sm_client = boto3.client(service_name="sagemaker",
                             region_name="us-east-1",
                             aws_access_key_id=os.environ["ACCESS_KEY"],
                             aws_secret_access_key=os.environ["SECRET_KEY"])

    response = sm_client.describe_endpoint_config(EndpointConfigName=endpoint_name)
    print(response)
    endpoint_config_name = response['EndpointConfigName']

# Delete Endpoint
    sm_client.delete_endpoint(EndpointName=endpoint_name)

# Delete Endpoint Configuration
    sm_client.delete_endpoint_config(EndpointConfigName=endpoint_config_name)

# Delete Model
    for prod_var in response['ProductionVariants']:
        model_name = prod_var['ModelName']
        sm_client.delete_model(ModelName=model_name)
