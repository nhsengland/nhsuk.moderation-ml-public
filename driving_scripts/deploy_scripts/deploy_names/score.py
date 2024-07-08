"""
This is the score script. This script is what gets sent to the deployed
endpoint, and indeed is the only thing which runs there. This consists of two
main parts; an 'init' function and a 'run' function. Both of these function
names are references by the Azure endpoint - you need an init and a run function
somewhere in here or you'll get an error in deployment. 

The init function is run once. This loads in the model files and does whatever
else. The Azure recommendation is to attach these to global vars, which is what
we've done. 

The `run` function expects a serialised json with a 'data' field. 

Note that Azure actually serialises the return for you - so if you return
json.dumps(output) then queries will actually get something double-serialised.
This mistake was made in this script, but we're keeping it here for reference. 
"""
import json
import os

import joblib

REGISTERED_MODEL_NAME = "Jorgeutd-bert-large-uncased-finetuned-ner"


def init():
    """
    Initializes and loads the NER model into a global variable `NER_model`.
    """
    global NER_model
    #Using global vars in this way is the suggested methodology from azure
    #documentation. 

    # Construct the model file path
    base_model_path = os.getenv("AZUREML_MODEL_DIR")
    model_file_path = os.path.join(
        base_model_path, "models", REGISTERED_MODEL_NAME + ".pkl"
    )

    # Load the model
    NER_model = joblib.load(model_file_path)


# Called when a request is received
def run(raw_data):
    """
    Processes input data and returns named entity recognition predictions.
    
    Parameters:
    - raw_data (str): JSON string containing the data for prediction.
    
    Returns:
    - str: JSON string of predictions indexed by their occurrence in the input.
    """
    # Get the input data as a numpy array
    data = json.loads(raw_data)["data"]
    # Get a prediction from the model
    predictions = NER_model(data)
    predictions = [
        dict(zip(a.keys(), [str(a) for a in list(a.values())])) for a in predictions
    ]
    # This line takes a list of dictionaries (predictions), and for each
    # dictionary, it converts all the values to strings, preserving the keys,
    # and then reconstructs a list of these new dictionaries where all values
    # are ensured to be in string format. If you are re-deploying, please
    # re-write this to make it more legible. 
    predictions_dict = dict(enumerate(predictions))
    # Return the predictions as JSON
    return json.dumps(predictions_dict)
