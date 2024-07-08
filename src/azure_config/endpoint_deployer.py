from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    CodeConfiguration,
    Environment,
    ManagedOnlineDeployment,
    ManagedOnlineEndpoint,
    Model,
)
from azure.identity import DefaultAzureCredential

from ..azure_config import azure_config

ws = azure_config.get_workspace()

endpoing_name = "testing_endpoint"


endpoint = ManagedOnlineEndpoint(
    name=endpoing_name, description="example endpoint", auth_mode="key"
)
