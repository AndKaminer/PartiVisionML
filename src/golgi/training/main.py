from .trainer import Trainer
from .configs import Configs as TrainConfigs
from ..inference import WeightManager

import argparse

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("epochs")
    parser.add_argument("batch")
    parser.add_argument("patience")
    parser.add_argument("weight_destination")

    args = parser.parse_args()

    WeightManager.set_model_default()
    t = Trainer(WeightManager.get_model(), TrainConfigs.workspace_name, TrainConfigs.project_name, TrainConfigs.version_number)
    t.train(int(args.epochs), int(args.batch), int(args.patience), args.weight_destination)
