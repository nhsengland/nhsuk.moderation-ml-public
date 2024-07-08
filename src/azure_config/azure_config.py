import os
import shutil
import subprocess
from dataclasses import dataclass

from azure.ai.ml import MLClient, UserIdentityConfiguration
from azure.ai.ml.dsl import pipeline
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
from azureml.core import Dataset, Experiment, Run, Workspace

RANDOMSEED = 42
TRAIN_FRAC = 0.5
TEST_FRAC = 0.25
VAL_FRAC = 0.25
ENVIRONMENT_NAME = "embeddings_main_A"
ENVIRONMENT_VERSION = 3


def clear_all_spark_files():
    [
        shutil.rmtree(os.path.join("/mnt/", d))
        for d in os.listdir("/mnt/")
        if d.startswith("spark-")
    ]
    print("all spark stuff deleted from /mnt/")


def setup_mnt_permissions_and_check_mount():
    # Check if /mnt is a mount point
    result = subprocess.run(["mountpoint", "-q", "/mnt"])
    if result.returncode != 0:
        raise Exception("/mnt is not a mounted directory.")

    # Check permissions of /mnt
    permissions = subprocess.check_output(["stat", "-c", "%a", "/mnt"]).decode().strip()
    if permissions != "777":
        # Try to change permissions
        result = subprocess.run(
            ["sudo", "chmod", "-R", "777", "/mnt"], stderr=subprocess.PIPE
        )

        # Check for errors
        if result.returncode != 0:
            # Decode error message
            error_message = result.stderr.decode()

            # Check if the error specifically concerns DATALOSS_WARNING_README.txt
            if "DATALOSS_WARNING_README.txt" not in error_message:
                # If it's about some other file, raise an exception or handle appropriately
                raise Exception(
                    f"Error encountered while changing permissions: {error_message}"
                )

    print("mnt drive ready now")


def start_run(expeiment_name: str):
    ws = get_workspace()

    experiment = Experiment(workspace=ws, name=expeiment_name)
    run = experiment.start_logging(snapshot_directory=None)
    return run


def get_ml_client():
    ml_client_ws = MLClient.from_config(credential=get_credential())
    return ml_client_ws


def get_credential():
    try:
        credential = DefaultAzureCredential()
        # Check if given credential can get token successfully.
        credential.get_token("https://management.azure.com/.default")
    except Exception as ex:
        # Fall back to InteractiveBrowserCredential in case DefaultAzureCredential not work
        credential = InteractiveBrowserCredential()
    return credential


def get_workspace():
    subscription_id = "REDACTED"
    resource_group = "REDACTED"
    workspace_name = "REDACTED"

    return Workspace(subscription_id, resource_group, workspace_name)


# def get_api_keys(key_name: str):
#     return {
#         "complaints4-janaks": "REDACTED",
#         "falcon-7b-instruct": "REDACTED",
#         }.get(key_name)


@dataclass
class EndpointAccessDetails:
    name: str
    api_key: str
    url: str
    json_name_for_content: str
    query_must_be_in_list: bool = False


complaints4_janaks_details = EndpointAccessDetails(
    name="REDACTED",
    api_key="REDACTED",
    url="REDACTED",
    json_name_for_content="data",
    query_must_be_in_list=True,
)

complaints_redeploymeny_multimodel_details = EndpointAccessDetails(
    name="REDACTED",
    api_key="REDACTED",
    url="REDACTED",
    json_name_for_content="data",
    query_must_be_in_list=True,
)

falcon_7b_instruct_details = EndpointAccessDetails(
    name="REDACTED",
    api_key="REDACTED",
    url="REDACTED",
    json_name_for_content="inputs",
)

llama_7b_details = EndpointAccessDetails(
    name="REDACTED",
    api_key="REDACTED",
    url="REDACTED",
    json_name_for_content="input_data",
)

llama_7b_chat_details = EndpointAccessDetails(
    name="REDACTED",
    api_key="REDACTED",
    url="REDACTED",
    json_name_for_content="input_data",
)


setup_mnt_permissions_and_check_mount()

TEMP_MODELS_FOLDER = "/mnt/temporary_models/"
os.makedirs(TEMP_MODELS_FOLDER, exist_ok=True)

# def get_endpoint_url(endpoint_name: str):
#     return {
#         "complaints4-janaks": "REDACTED",
#         "falcon-7b-instruct":"REDACTED"
#     }.get(endpoint_name)
