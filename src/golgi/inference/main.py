from .config import Configs
from .inference_pipeline import InferencePipeline
from .weight_manager import WeightManager

import argparse
import os

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("output_folder")
    parser.add_argument("video_path")
    parser.add_argument("weight_name")

    args = parser.parse_args()

    if not os.path.isdir(args.output_folder):
        raise Exception("Invalid output folder")

    if not os.path.isfile(args.video_path):
        raise Exception("Invalid video path")

    if args.weight_name not in WeightManager.list_weights():
        raise Exception("Invalid model specified")
    
    WeightManager.select_current_model(args.weight_name)

    ip = InferencePipeline(
            model=WeightManager.get_model(),
            framerate=Configs.framerate,
            window_width=Configs.window_width,
            scaling_factor=Configs.scaling_factor,
            um_per_pixel=Configs.um_per_pixel,
            output_folder=args.output_folder)

    ip.process_video(
            video_path=args.video_path,
            scatter=Configs.scatter,
            verbose=Configs.verbose)

    print("Tracking complete!")

def download_weights():
    parser = argparse.ArgumentParser()
    parser.add_argument("repo_id")
    parser.add_argument("model_name")
    parser.add_argument("huggingface_token")

    args = parser.parse_args()

    Configs.huggingface_login(args.huggingface_token)

    if WeightManager.download_model_weights(args.repo_id, args.model_name):
        print("Downloaded successfully")
    else:
        print("Download failed")

def list_weights():
    print(WeightManager.list_weights())
