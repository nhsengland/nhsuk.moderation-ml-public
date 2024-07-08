"""
This is the score script. This script is what gets sent to the deployed endpoint, and indeed is the only thing which runs there. 
This consists of two main parts; an 'init' function and a 'run' function.
Both of these function names are references by the Azure endpoint - you need an init and a run function somewhere in here or you'll get an error in deployment. 

The init function is run once. This loads in the model files and does whatever else. The Azure recommendation is to attach these to global vars, which is what we've done. 

The `run` function expects a serialised json with a 'data' field. 

Note that Azure actually serialises the return for you - so if you return json.dumps(output) then queries will actually get something double-serialised. This mistake was made in this script, but we're keeping it here for reference. 
"""
import json
import logging
import os
import re

import joblib
import numpy as np
import pandas as pd
import sentence_transformers

#This score script has been sent to deployment without logging configured. I'm
#leaving it that way for posterity; but if you're editing this for a new
#deployment, *definitely!!!* configure logging

def init():
    """
    Run once, when the deployment is first set up. This loads in the models and
    assigns them to global variables, as per the Azure documentation. 
    """
    global classification_model
    global sentence_transformer_model

    model_dir = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "models")
    classifier_path = os.path.join(model_dir, "complaints_svm_4.pkl")
    sentence_transformer_path = os.path.join(
        model_dir, "model_sentence_transformer_cpu_for_complaints_.pkl"
    )
    sentence_transformer_model = joblib.load(sentence_transformer_path)
    classification_model = joblib.load(classifier_path)


def clean_string_nospell(s: str)-> str:
    """
    Simple function to remove double space, newlines, and to lower the case.
    """
    s = s.strip()
    s = re.sub("  ", " ", s)
    s = re.sub("   ", " ", s)
    s = re.sub("\n", "", s)
    s = s.lower()

    return s


def encode_comment_text_for_row(text: str):
    """
    Gets embeddings for the comment on this row. 
    """
    return np.array(sentence_transformer_model.encode(text))


def run(raw_data):
    """
    This is the main function of the score script. This is what gets run each
    time some data is sent in. This 
    - Takes in some data, 'cleans' the string
    - Gets the embeddings for that string
    - Passes this embedding to the Complaints model to be classified. 
    - Returns a serialised version of that outcome.
    """
    df = pd.DataFrame(json.loads(raw_data))
    df["string_cleaned"] = df["data"].apply(clean_string_nospell)
    df["embeddings"] = df["string_cleaned"].apply(encode_comment_text_for_row)

    Y_pred = classification_model.predict(np.array(list(df["embeddings"])))

    return json.dumps(int(Y_pred[0]))
#! Note! If you're editing or re-deploying, I recommend returning just a dict,
#instead of this. Beware though that if you're doing this, you should edit the
#querying code in tandem