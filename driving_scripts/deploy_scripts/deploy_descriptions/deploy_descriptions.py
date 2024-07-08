"""
The purpose of this script is to deploy the relevant model to an endpoint. It
also contains the necessities for setting that endpoint up in the first place.
We have significant documentation about this process on our (internal)
Confluence. 

This script basically: - Gets all model files from the Azure registry - Creates
an online inference endpoint (unless that code is commented out - see below) -
Gets an Environment object from Azure (python environment and linux distro) -
Creates a deployment on the endpoint. This deployment takes:
    - The environment
    - The model files
    - The score.py script which is in the same folder as this file
- The 'poller' object then checks that this deployment is created fine. 


If you're using or updating this script, the common mistakes are around allowing
the model name here and the model name in the score.py script to drift from one
another. Beware that the name of the model object in the Azure registry doesn't
necessarily match the name of the .pkl file which will be downloaded. 
"""
import os
import sys
import sklearn
from azure.ai.ml.entities import CodeConfiguration
from azure.ai.ml.entities import ManagedOnlineDeployment 
from azure.ai.ml.entities import Model as Model_entity
from azure.ai.ml.exceptions import DeploymentException

from src.azure_config import azure_config
from src.deployment import deployment
from src.environment_management import environment_management

# This block exist to allow the driving script(s) to see the /src folder and us
# the modules within, without having to put all scripts in the root. 
CURRENT_DIR = os.path.dirname(__file__)
root_path = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", ".."))
sys.path.insert(0, root_path)


# environment_management.check_environment_is_correct()

ws = azure_config.get_workspace()
ml_client = azure_config.get_ml_client()

REGISTERED_MODEL_NAME = "spacy_descriptionsdg_redo.pkl" 
MODEL_DIR = os.path.join(CURRENT_DIR, "models")
deployment.download_model(
    model_name=REGISTERED_MODEL_NAME, version=1, target_dir=MODEL_DIR
)


# endpoint_name = REGISTERED_MODEL_NAME.replace('_','-')
# endpoint_name = endpoint_name[0:30] 
# endpoint = ManagedOnlineEndpoint(name=endpoint_name)
# poller = ml_client.begin_create_or_update(endpoint)

# poller.wait()
# status = poller.status()
# if status != "Succeeded":
#     raise DeploymentException(status)
# else:
#     print("Endpoint creation succeeded")
#     endpoint = poller.result()
#     print(endpoint)

ENVIRONMENT_NAME = "dg_descriptions_environment_redo"
ENVIRONMENT_VERSION = 8
env = ml_client.environments.get(
    name=ENVIRONMENT_NAME, version=str(ENVIRONMENT_VERSION)
)
print(env)


deployment = ManagedOnlineDeployment(
    name="REDACTED", 
    endpoint_name="REDACTED", 
    model=Model_entity(name="spacy_descriptionsdg_redo", path=MODEL_DIR),
    code_configuration=CodeConfiguration(code=CURRENT_DIR, scoring_script="score.py"),
    environment=env,
    instance_type="Standard_DS3_v2",
    instance_count=1,
)

# Next we create a 'poller' object. This handles the communication from azure as
# our deployment is created. Allows us to see a progress bar and outcomes in the
# terminal. 

poller = ml_client.online_deployments.begin_create_or_update(deployment)
poller.wait()

status = poller.status()
if status != "Succeeded":
    raise DeploymentException(status)
else:
    print("Deployment creation succeeded")
    deployment = poller.result()
    print(deployment)

poller.wait()
