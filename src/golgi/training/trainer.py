import roboflow
import tempfile
import os


class Trainer:
    def __init__(self, starting_model, workspace_name, project_name, version_number, api_key=None):
        self.starting_model = starting_model
        self.tempdir = tempfile.TemporaryDirectory()
        self.dataset_dir = os.path.join(self.tempdir.name, "dataset")

        # Login explicitly with the provided API key
        if api_key:
            rf = roboflow.Roboflow(api_key=api_key)
        else:
            raise ValueError("API key is required to initialize Trainer.")


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