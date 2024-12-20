import cv2
from .auto_annotation import ImageAutoAnnotater
from .image_annotation import ImageAnnotater
from .model import LocalModel

from ..training import Configs # causing error due to Roboflow API

from .config import Configs as AnnConfigs
from ..inference import WeightManager


def annotate_image(annotation_type, image_path, model=None, api_key=None):
    """
    Annotates an image based on the specified annotation type.

    Args:
        annotation_type (str): Type of annotation ("auto" or "manual").
        image_path (str): Path to the image file.
        model (str, optional): Model name for auto-annotation. Required for "auto".
        api_key (str, optional): API key for uploading the annotated image.

    Returns:
        str: Success or error message.
    """

    VALID_TYPES = ["auto", "manual"]
    if annotation_type not in VALID_TYPES:
        raise ValueError(f"Invalid annotation type. Please choose from: {VALID_TYPES}")

    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Invalid image path or unable to read image.")
    except Exception as e:
        raise ValueError(f"Error reading image: {e}")

    if annotation_type == "auto":
        if not model:
            raise ValueError("Must specify model for auto annotation.")

        if model not in WeightManager.list_weights():
            raise ValueError("Model not found.")

        # Load the model and perform auto annotation
        WeightManager.select_current_model(model)
        m = LocalModel(WeightManager.get_model())
        ia = ImageAutoAnnotater(img, AnnConfigs.resize_constant, m)
    else:
        # Manual annotation
        ia = ImageAnnotater(img, AnnConfigs.resize_constant)

    try:
        if not api_key:
            return "Error: API key is required to upload annotations to Roboflow."
        # Annotate the image
        ann_img = ia.annotate()
        # Show the annotated image (if needed for debugging or visualization)
        # Comment out in production environments
        # ann_img.show()
        ann_img.roboflow_upload(
            workspace=Configs.workspace_name,
            project=Configs.project_name,
            api_key=api_key
        )
        return "Annotation completed and uploaded successfully."

    except Exception as e:
        raise RuntimeError(f"Error during annotation: {e}")
