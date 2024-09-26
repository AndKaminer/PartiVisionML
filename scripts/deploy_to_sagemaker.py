import os
import argparse
import tarfile
import tempfile
import shutil
import datetime

import huggingface_hub
import boto3
import sagemaker
from sagemaker.pytorch import PyTorchModel
from sagemaker.deserializers import JSONDeserializer


if __name__ == '__main__':
    code_folder_path = './code'

    parser = argparse.ArgumentParser()
    parser.add_argument('--code_path', required=False)
    parser.add_argument('--model_name', required=True)
    parser.add_argument('--model_repo', required=True)
    parser.add_argument('--bucket', required=True)
    parser.add_argument('--hfkey', required=False)

    parsed = parser.parse_args()
    if parsed.code_path:
        code_folder_path = parsed.code_path

    model_name = parsed.model_name
    model_repo = parsed.model_repo
    bucket = parsed.bucket
    hfkey = parsed.hfkey

    if not os.path.isdir(code_folder_path):
        raise Exception("Code directory not found")

    if hfkey:
        huggingface_hub.login(token=hfkey)
    else:
        huggingface_hub.login()

    session = boto3.Session(
        aws_access_key_id=os.environ["ACCESS_KEY"],
        aws_secret_access_key=os.environ["SECRET_KEY"],
        region_name="us-east-1")
    sagemaker_session = sagemaker.session.Session(session)
    iam_client = session.client('iam')
    response = iam_client.get_role(RoleName='sagemaker-role')

    
    with tempfile.TemporaryDirectory() as tmpdirname:
        huggingface_hub.hf_hub_download(repo_id=model_repo, local_dir=tmpdirname, filename=model_name)
        tar_filename = os.path.join(tmpdirname, "model.tar.gz")

        with tarfile.open(tar_filename, "w:gz") as tar:
            for file in os.listdir(code_folder_path):
                tar.add(os.path.join(code_folder_path, file))
            tar.add(os.path.join(tmpdirname, model_name))

        model_data = sagemaker.s3.S3Uploader.upload(tar_filename, bucket + "/yolov8/endpoint", sagemaker_session=sagemaker_session)
    print(model_data)

    model = PyTorchModel(
            entry_point='inference.py',
            framework_version='1.12',
            py_version='py38',
            model_data=model_data,
            env={'TS_MAX_RESPONSE_SIZE':'20000000', 'YOLOV8_MODEL': model_name, 'SAGEMAKER_REQUIREMENTS': 'requirements.txt'},
            sagemaker_session=sagemaker_session,
            role=response['Role']['Arn'])

    # INSTANCE_TYPE = 'ml.g4dn.xlarge'
    INSTANCE_TYPE = 'ml.g5.xlarge'
    ENDPOINT_NAME = 'yolov8-pytorch-' + str(datetime.datetime.utcnow().strftime('%Y-%m-%d-%H-%M-%S-%f'))

    predictor = model.deploy(initial_instance_count=1,
                             instance_type=INSTANCE_TYPE,
                             deserializer=JSONDeserializer(),
                             endpoint_name=ENDPOINT_NAME)
