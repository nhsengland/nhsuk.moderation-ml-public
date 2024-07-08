import pickle
import uuid

import azureml.core
from azureml.core import Model, Workspace


def make_endpoint_name(name_of_rule: str) -> str:

    return name_of_rule + str(uuid.uuid4())[:8]


def download_model(model_name, version, target_dir="./"):
    ws = Workspace.from_config()

    model = Model(workspace=ws, name=model_name, version=version)
    model_path = model.download(target_dir=target_dir, exist_ok=True)

    print("model_downloaded")
