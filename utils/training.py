import os
import sys
import boto3
import sagemaker
from datetime import datetime

from sagemaker.sklearn.estimator import SKLearn

from modules import config

social_network = sys.argv[1]

try:
    test = sys.argv[2]
except:
    test = False

# get role with sagemaker, s3, permissions
iam = boto3.client('iam')
role = iam.get_role(RoleName='datascience-sagemaker-s3-redshift')['Role']['Arn']
s3_client = boto3.client('s3')

# start session 
sagemaker_session = sagemaker.Session()

# bucket folder name (prefix)
bucket = sagemaker_session.default_bucket() # retrieves bucket based on your region and account ID

if test == 'test':
    prefix = f'social-media-cx-monitor/dafiti_br/comentarios_{social_network}/testes'
    base_job_name = f'test-{social_network}'
else:
    prefix = f'social-media-cx-monitor/dafiti_br/comentarios_{social_network}'
    base_job_name = f'cx-social-comments-dafiti-br-{social_network}'
s3_data_uri = f's3://{bucket}/{prefix}'

env = {'SAGEMAKER_REQUIREMENTS': 'requirements.txt'}

sklearn_model = SKLearn(
    entry_point='processing.py',
    framework_version='0.23-1',
    instance_type='ml.c5.4xlarge', # 36 CPUs ml.c5.4xlarge
    role=role,
    sagemaker_session=sagemaker_session,
    base_job_name=base_job_name,
    source_dir='.',
    env=env,
    # hyperparameters={'social-network': social_network},
    code_location=f's3://{bucket}/{prefix}/training-jobs',
    output_path=f's3://{bucket}/{prefix}/training-jobs'
)

data_path = os.path.join(s3_data_uri, 'data.csv')

sklearn_model.fit({'data': data_path})


# create model from training job

iam = boto3.client('iam')
role = iam.get_role(RoleName='datascience-sagemaker-s3-redshift')['Role']['Arn']
client = boto3.client('sagemaker')

client.create_model(
    ModelName=f'/'.join(sklearn_model.model_data.split('/')[-3:-2]),
    PrimaryContainer={
        'Image': '296025910508.dkr.ecr.us-east-1.amazonaws.com/cx_social_comments:latest', # genérico, pois é necessário informar
        'Mode': 'SingleModel',
        'ModelDataUrl': sklearn_model.model_data
    },
    ExecutionRoleArn=role
)