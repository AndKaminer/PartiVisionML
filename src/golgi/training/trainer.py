import roboflow

import tempfile
import os

roboflow.login()

class Trainer:
    def __init__(self, starting_model, workspace_name, project_name, version_number):
        self.starting_model = starting_model
        self.tempdir = tempfile.TemporaryDirectory()
        self.dataset_dir = os.path.join(self.tempdir.name, "dataset")

        rf = roboflow.Roboflow()
        project = rf.workspace(workspace_name).project(project_name)
        dataset = project.version(version_number).download("yolov8", location=self.dataset_dir)

        with open(os.path.join(self.dataset_dir, "data.yaml"), "r") as f:
            lines = f.readlines()
        
        out = ""
        for l in lines:
            if l.startswith("test:"):
                out += f"test: {os.path.join(self.dataset_dir, 'test/images')}\n"
            else:
                out += l

        with open(os.path.join(self.dataset_dir, "data.yaml"), "w") as f:
            f.write(out)

    def train(self, epochs, batch, patience, weight_destination):
        out = self.starting_model.train(
                data=os.path.join(self.dataset_dir, "data.yaml"),
                name="Training",
                epochs=epochs,
                batch=batch,
                patience=patience,
                project=weight_destination)

        return out
