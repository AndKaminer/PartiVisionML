import os
import logging
import shutil

from .config import Configs

import huggingface_hub
from ultralytics import YOLO

class Inference:

    available_models = os.listdir(Configs.weights_path)
    model = None
   

    def get_model():
        if not Inference.model:
            raise Exception("No model found")

        return Inference.model


    def select_current_model(model_name):
        if model_name in Inference.available_models:
            Inference.model = YOLO(os.path.join(Configs.weights_path, model_name))
        else:
            raise Exception("Model not found")
        

    def download_model_weights(huggingface_repo_id, model_name):
        try:
            huggingface_hub.hf_hub_download(repo_id=Configs.huggingface_repo_id,
                                            local_dir=Configs.weights_path,
                                            filename=model_name)

            if os.path.exists(os.path.join(Configs.weights_path, ".huggingface")):
                shutil.rmtree(os.path.join(Configs.weights_path, ".huggingface"))

        except Exception as e:
            logging.error(e)
            return False
        
        Inference.available_models = os.listdir(Configs.weights_path)
        return True


    def list_weights():
        return Inference.available_models


    def delete_weights(model_name):
        if model_name in Inference.available_models:
            os.remove(os.path.join(Configs.weights_path, model_name))
            Inference.available_models = os.listdir(Configs.weights_path)

        else:
            raise Exception("Model not found")
