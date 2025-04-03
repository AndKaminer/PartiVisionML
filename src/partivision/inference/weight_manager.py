import os
import logging
import shutil

from .config import Configs
from .util import Util

import huggingface_hub
from ultralytics import YOLO

class WeightManager:

    available_models = Util.list_pt(Configs.weights_path)
    model = None

    def get_model():
        if not WeightManager.model:
            raise Exception("No model found")

        return WeightManager.model


    def select_current_model(model_name):
        if model_name in WeightManager.available_models:
            WeightManager.model = YOLO(os.path.join(Configs.weights_path, model_name))
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
        
        WeightManager.available_models = Util.list_pt(Configs.weights_path)
        return True


    def list_weights():
        return WeightManager.available_models


    def delete_weights(model_name):
        if model_name in WeightManager.available_models:
            os.remove(os.path.join(Configs.weights_path, model_name))
            WeightManager.available_models = Util.list_pt(Configs.weights_path)

        else:
            raise Exception("Model not found")


    def check_weights(model_name):
        return model_name in WeightManager.available_models

    def set_model_default():
        WeightManager.model = YOLO('yolov8m-seg.pt')
