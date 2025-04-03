from .trainer import Trainer
from .configs import Configs as TrainConfigs
from ..inference import WeightManager


def train_model(epochs, batch, patience, weight_destination, api_key):
    """
    Train the model with the given parameters.

    Args:
        epochs (int): Number of epochs to train.
        batch (int): Batch size.
        patience (int): Patience for early stopping.
        weight_destination (str): Destination folder for the weights.

    Returns:
        str: Success message or error message.
    """
    try:
        # Initialize default model
        WeightManager.set_model_default()

        # Create Trainer instance
        t = Trainer(
            WeightManager.get_model(),
            TrainConfigs.workspace_name,
            TrainConfigs.project_name,
            TrainConfigs.version_number,
            api_key
        )

        # Start training
        t.train(int(epochs), int(batch), int(patience), weight_destination)

        return f"Training completed successfully. Weights saved to {weight_destination}."
    except Exception as e:
        return f"Error during training: {str(e)}"
