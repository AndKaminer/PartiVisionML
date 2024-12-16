from .config import Configs
from .inference_pipeline import InferencePipeline
from .weight_manager import WeightManager
import os


def main(output_folder, video_path, weight_name):
    """
    Run the tracking process with the specified parameters.

    Args:
        output_folder (str): Path to the output folder.
        video_path (str): Path to the video file.
        weight_name (str): Name of the model weights to use.

    Raises:
        Exception: If any validation checks fail.
    """
    # Validate inputs
    if not os.path.isdir(output_folder):
        raise Exception("Invalid output folder")
    if not os.path.isfile(video_path):
        raise Exception("Invalid video path")
    if weight_name not in WeightManager.list_weights():
        raise Exception("Invalid model specified")

    # Configure the weight manager
    WeightManager.select_current_model(weight_name)

    # Initialize and configure the inference pipeline
    ip = InferencePipeline(
        model=WeightManager.get_model(),
        framerate=Configs.framerate,
        window_width=Configs.window_width,
        scaling_factor=Configs.scaling_factor,
        um_per_pixel=Configs.um_per_pixel,
        output_folder=output_folder
    )

    # Run video processing
    ip.process_video(
        video_path=video_path,
        scatter=Configs.scatter,
        verbose=Configs.verbose
    )

    return "Tracking complete!"


def get_available_weights():
    """
    Retrieve the list of available weights.

    Returns:
        list: List of available weight names.
    """
    return WeightManager.list_weights()


def download_model(repo_id, model_name, huggingface_token):
    """
    Download model weights from the specified repository.

    Args:
        repo_id (str): Huggingface repository ID.
        model_name (str): Name of the model.
        huggingface_token (str): Authentication token for Huggingface.

    Returns:
        bool: True if download succeeds, False otherwise.
    """
    Configs.huggingface_login(huggingface_token)
    return WeightManager.download_model_weights(repo_id, model_name)