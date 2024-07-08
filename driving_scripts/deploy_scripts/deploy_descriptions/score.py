"""
This is the score script. This script is what gets sent to the deployed endpoint, and indeed is the only thing which runs there. 
This consists of two main parts; an 'init' function and a 'run' function.
Both of these function names are references by the Azure endpoint - you need an init and a run function somewhere in here or you'll get an error in deployment. 

The init function is run once. This loads in the model files and does whatever else. The Azure recommendation is to attach these to global vars, which is what we've done. 

The `run` function expects a serialised json with a 'data' field. 

Note that Azure actually serialises the return for you - so if you return json.dumps(output) then queries will actually get something double-serialised. This mistake was made in this script, but we're keeping it here for reference. 
"""
import json
import os
import joblib

REGISTERED_MODEL_NAME = "spacy_descriptionsdg_redo.pkl"


def init():
    """
    Loads the model into a global variable `desc_model` from Azure's model
    directory.
    """
    global desc_model 
    #Using global vars in this way is the suggested methodology from azure
    #documentation. 

    # Construct the model file path
    base_model_path = os.getenv("AZUREML_MODEL_DIR")
    model_file_path = os.path.join(base_model_path, "models", REGISTERED_MODEL_NAME)

    # Load the model
    desc_model = joblib.load(model_file_path)


# Called when a request is received
def run(raw_data):
    """
    Processes input data to generate and return adjective-noun pairs from model
    predictions.
    
    Parameters: - raw_data (str): JSON string with key 'data' containing text
    for analysis.
    
    Returns: - str: JSON string mapping indices to adjective-noun pairs derived
    from the text.
    """
    # Get the input data:
    data = json.loads(raw_data)["data"]
    # Get a prediction from the model:
    predictions = desc_model(data)
    # filter on dependency labels:
    predictions_tokens = []
    for token in predictions:
        word = token.text
        syntactic_head = token.head.text
        # if the word is an adjective:
        if token.dep_ in ["amod", "advmod", "compound"]:
            # record the adjective and it's noun as a pair
            predictions_tokens.append([word, syntactic_head])
            # Returned as [adjective, noun], like [tall, man]
    # Convert to dict
    predictions_dict = dict(enumerate(predictions_tokens))
    # Return the predictions as JSON
    return json.dumps(predictions_dict)
