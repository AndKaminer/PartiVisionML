import argparse
import os

import cv2

from .auto_annotation import ImageAutoAnnotater
from .image_annotation import ImageAnnotater
from .model import LocalModel

from ..training import Configs
from ..inference import WeightManager

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("annotation_type")
    parser.add_argument("image_path")
    parser.add_argument("-m", "--model", required=False)
    parser.add_argument("-k", "--api_key", required=False)

    args = parser.parse_args()
    
    VALID_TYPES = ["auto", "manual"]
    if args.annotation_type not in VALID_TYPES:
        raise Exception(f"Please choose from: {VALID_TYPES}")

    try:
        img = cv2.imread(args.image_path)
    except Exception:
        raise Exception("Invalid image path")

    if args.annotation_type == "auto":

        if "model" not in args:
            raise Exception("Must specify model")

        if args.model not in WeightManager.list_weights():
            raise Exception("Model not found")

        WeightManager.select_current_model(args.model)

        m = LocalModel(WeightManager.get_model())
        ia = ImageAutoAnnotater(img, Configs.resize_constant, m)

    else:
        ia = ImageAnnotater(img, Configs.resize_constant)
        
    ann_img = ia.annotate()
    
    ann_img.show()
    
    if "api_key" in args:
        ann_img.roboflow_upload(workspace=Configs.workspace_name,
                                project=Configs.project_name,
                                api_key=args.api_key)
    else:
        ann_img.roboflow_upload(workspace=Configs.workspace_name,
                                project=Configs.project_name)
