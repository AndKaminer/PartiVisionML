from partivision.training import Trainer
from partivision.training import Configs as TrainConfigs
from partivision.inference import WeightsManager

import argparse

parser = argparser.ArgumentParser()
parser.add_argument("epochs")
parser.add_argument("batch")
parser.add_argument("patience")
parser.add_argument("weight_destination")

args = parser.parse_args()

WeightsManager.set_default_model()
t = Trainer(WeightsManager.get_model(), TrainConfigs.workspace_name, TrainConfigs.project_name, TrainConfigs.version_number)
t.train(int(args.epochs), int(args.batch), int(args.patience), args.weight_destination)
