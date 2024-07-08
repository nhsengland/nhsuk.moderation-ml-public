"""
This is the score script. This script is what gets sent to the deployed endpoint, and indeed is the only thing which runs there. 
This consists of two main parts; an 'init' function and a 'run' function.
Both of these function names are references by the Azure endpoint - you need an init and a run function somewhere in here or you'll get an error in deployment. 

The init function is run once. This loads in the model files and does whatever else. The Azure recommendation is to attach these to global vars, which is what we've done. 

The `run` function expects a serialised json with a 'data' field. 

Note that Azure actually serialises the return for you - so if you return json.dumps(output) then queries will actually get something double-serialised. This mistake was made in this script, but we're keeping it here for reference. 
"""
# Sam
import json
import logging
import os
import re

import joblib
import numpy as np
import pandas as pd
import sentence_transformers
from azureml.core.model import Model

logging.basicConfig(level=logging.DEBUG)


def init():
    global classification_model
    global vectorizer_model

    model_dir = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "models")
    classifier_path = os.path.join(model_dir, "nae_bow_final_data_LR_test.pkl")
    vectorizer_path = os.path.join(model_dir, "nae_bow_vectorizer_final_data_test.pkl")
    vectorizer_model = joblib.load(vectorizer_path)
    classification_model = joblib.load(classifier_path)


def encode_comment_text_for_row(text):
    X_val_bow = vectorizer_model.transform([text])
    return X_val_bow


def run(raw_data):
    df = pd.DataFrame(json.loads(raw_data))
    df["bow_rep"] = df["data"].apply(encode_comment_text_for_row)

    # Y_pred = classification_model.predict(np.array(list(df["embeddings"])))
    # Y_pred = classification_model.predict(df.loc[0, "bow_rep"])
    Y_probs = classification_model.predict_proba(
        df.loc[0, "bow_rep"]
    )  # get probabilities
    max_index = np.argmax(Y_probs)  # extract classification
    max_value = Y_probs[0, max_index]  # extract probability

    keys = ["0", "1"]  # following safeguarding return structure
    values = [str(max_index), str(max_value)]
    return_dict = dict(zip(keys, values))

    return json.dumps(return_dict)
