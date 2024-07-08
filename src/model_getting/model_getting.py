import json
import os
import ssl
import subprocess
import sys
import urllib.request
from typing import Callable, Dict

import azureml.core
import joblib
import langchain
import torch
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Model
from azureml.core import Dataset, Model, Run, Workspace
from azureml.core.webservice import Webservice

from src.azure_config import azure_config

azure_config.setup_mnt_permissions_and_check_mount()


class ModelRetriever:
    """Interface for model retrievers."""

    def get_model(self, model_name: str):
        raise NotImplementedError


class ModelGetter:
    """A class to unify model retrieval from different sources."""

    def __init__(self, retriever: ModelRetriever, model_name: str):
        self.retriever = retriever
        self.model = self.retriever.get_model(model_name=model_name)


class HuggingFaceAPIRetriever(ModelRetriever):
    """Model retriever for Hugging Face API."""

    def get_model(self, model_name: str):
        # Implement model retrieval from Hugging Face API here
        pass


class HuggingFaceLocalRetriever(ModelRetriever):
    """Model retriever for local Hugging Face models."""

    def get_model(self, model_name: str):
        # Implement model retrieval from local Hugging Face models here
        pass


def get_experiment_by_name(experiment_name: str):
    ws = azure_config.get_workspace()
    if experiment_name in ws.experiments:
        experiment = ws.experiments[experiment_name]
        return experiment
    else:
        print("experiment not found")
        return None


def get_run_by_experiment_and_run_names(experiment_name: str, run_name: str):
    #! Looks like this one won't work because the display name isn't actually stored with the run
    experiment = get_experiment_by_name(experiment_name)
    if experiment is not None:
        for run in experiment.get_runs():
            if run.name == run_name:
                return run
    print("Run not found")
    return None


def get_run_by_run_id(run_id: str):
    ws = azure_config.get_workspace()
    return Run.get(workspace=ws, run_id=run_id)


def get_model_from_run(run: Run):
    ws = azure_config.get_workspace()
    metrics = run.get_metrics()
    model_name = metrics["model name"]
    try:
        model = get_model_by_name(model_name=model_name)

        return model
    except Exception as e:
        print(f"Error: {str(e)}")


class AzureRegistryModelRetriever(ModelRetriever):
    def get_model(self, model_name: str):
        return super().get_model(model_name)


class OpenAIRetriever(ModelRetriever):
    """Model retriever for OpenAI."""

    def get_model(self, model_name: str):
        from dotenv import load_dotenv

        load_dotenv()
        from langchain.llms import OpenAI

        llm = OpenAI(model_name=model_name)
        return llm


class EndpointsRetriever(ModelRetriever):
    """Model retriever for specific endpoints."""

    def __init__(self, model_name) -> None:
        self.model_name = model_name

    def get_model(self, model_name: str):
        print("Endpoint Retriever does not return a model")
        pass

    def get_prediction(self, input_text: str):
        # TODO: Currently, this takes single entries at a time. Must find way to batch query
        endpoint_url = azure_config.get_endpoint_url(self.model_name)
        response = get_endpoint_response(
            endpoint_url=endpoint_url,
            endpoint_name=self.model_name,
            query_content=input_text,
        )
        return response


class AMLDeploymentsRetriever(ModelRetriever):
    """Model retriever for Azure Machine Learning deployments."""

    def get_model(self, model_name: str):
        # Implement model retrieval from Azure Machine Learning deployments here
        pass


def list_all_models_in_workspace():
    ws = azure_config.get_workspace()

    for model in Model.list(ws):
        print(model.name)


def list_all_models_in_workspace_by_date_created():
    ws = azure_config.get_workspace()
    models = Model.list(ws)

    sorted_models = sorted(models, key=lambda model: model.created_time, reverse=True)
    for model in sorted_models:
        print(model.name)


def get_model_by_name(model_name):
    ws = azure_config.get_workspace()
    model = Model(ws, model_name)
    model_path = model.download(
        target_dir=azure_config.TEMP_MODELS_FOLDER, exist_ok=True
    )
    if not ("pkl" in model_path):
        model_path += ".pkl"
    model = joblib.load(model_path)
    return model


def list_all_endpoints():
    # TODO: This is quitte an important funciton but currently not working, cant tell why
    ws = azure_config.get_workspace()

    try:
        for webservice in Webservice.list(ws):
            print(webservice.name)
    except Exception as e:
        print(f"An error occurred: {e}")


def get_endpoint_response(
    query_content: str, access_details: azure_config.EndpointAccessDetails
):
    """
    An example would be
    endpoint_url = 'REDACTED'
    """

    def allowSelfSignedHttps(allowed):
        # bypass the server certificate verification on client side
        if (
            allowed
            and not os.environ.get("PYTHONHTTPSVERIFY", "")
            and getattr(ssl, "_create_unverified_context", None)
        ):
            ssl._create_default_https_context = ssl._create_unverified_context

    allowSelfSignedHttps(
        True
    )  # this line is needed if you use self-signed certificate in your scoring service.

    if access_details.query_must_be_in_list:
        query_content = (
            [query_content] if isinstance(query_content, str) else query_content
        )

    data = {access_details.json_name_for_content: query_content}
    # Some endpoints want 'data':, some want 'input': , etc

    body = str.encode(json.dumps(data))

    # Replace this with the primary/secondary key or AMLToken for the endpoint
    api_key = access_details.api_key
    if not api_key:
        raise Exception("A key should be provided to invoke the endpoint")

    headers = {
        "Content-Type": "application/json",
        "Authorization": ("Bearer " + api_key),
    }

    req = urllib.request.Request(access_details.url, body, headers)

    try:
        response = urllib.request.urlopen(req)

        result = response.read()
        result_str = result.decode(
            "utf-8"
        )  # Decode the byte string to a regular string
        result_str = result_str.strip('"')  # Remove the quotes
        return result_str
        # return result

    except urllib.error.HTTPError as error:
        print("The request failed with status code: " + str(error.code))

        # Print the headers - they include the requert ID and the timestamp, which are useful for debugging the failure
        print(error.info())
        print(error.read().decode("utf8", "ignore"))


def register_model(model, model_name, description=""):

    ws = azure_config.get_workspace()
    base_path = "embeddings_models/"

    path_for_model = os.path.join(base_path, model_name + ".pkl")
    os.makedirs(base_path, exist_ok=True)

    joblib.dump(value=model, filename=path_for_model)
    Model.register(
        workspace=ws,
        model_path=path_for_model,
        model_name=model_name,
        description=description,
    )

    # file_model = Model(
    #     workspace=ws,
    #     path=path_for_model,
    #     type=AssetTypes.CUSTOM_MODEL,
    #     name=model_name,
    #     description=description,
    #     )
    # ml_client = azure_config.get_ml_client
    # ml_client.models.create_or_update(file_model)

    print(f"Model registered under the name {model_name}")
