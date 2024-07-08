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
import logging
import os

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import BertForSequenceClassification, BertTokenizer


def init():
    logging.basicConfig(
        filename="score_log.txt", level=logging.INFO, format="%(asctime)s - %(message)s"
    )

    global model
    global device
    global tokenizer
    device = "cpu" 

    base_model_path = os.getenv("AZUREML_MODEL_DIR")
    model_file_path = os.path.join(base_model_path, "safeguarding3cat.pkl")

    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=3,
        output_attentions=False,
        output_hidden_states=False,
    )

    model.load_state_dict(torch.load(model_file_path, map_location=torch.device(device)))
    print(model)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)


def run(raw_data):
    """
    Processes input text for safeguarding classification, returning prediction and probability.
    
    Parameters:
    - raw_data (str): JSON string with 'data' key containing text for classification.
    
    Returns:
    - str: JSON string mapping an index to a tuple of prediction and its probability.
    """
    data = json.loads(raw_data)["data"]

    def preprocessing(input_text, tokenizer):
        """
        Prepares text for model input using specified tokenizer.
        """
        return tokenizer.encode_plus(
            input_text,
            add_special_tokens=True,
            max_length=512,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors="pt",  # Return pytorch tensors.
        )

    def safeguarding_prediction(text_to_classify):
        """
        Classifies text into the predefined safeguarding categories and calculates probability. 
        """
        # Apply the tokenizer
        encoding = preprocessing(text_to_classify, tokenizer)

        # Extract IDs and Attention Mask
        test_ids = torch.cat(test_ids, dim=0)
        test_attention_mask = torch.cat(test_attention_mask, dim=0)

        # Forward pass, calculate logit predictions
        with torch.no_grad():  # not to construct the compute graph during this forward pass (since we won’t be running backprop here)
            output = model(
                test_ids.to(device),
                token_type_ids=None,
                attention_mask=test_attention_mask.to(device),
            )

        prediction = (
            "Possibly Concerning"
            if np.argmax(output.logits.cpu().numpy()).flatten().item() == 1
            else (
                "Strongly Concerning"
                if np.argmax(output.logits.cpu().numpy()).flatten().item() == 2
                else "No safeguarding"
            )
        )
        probs = (
            F.softmax(output.logits.cpu(), dim=1).cpu().numpy()
        )  # Apply softmax to calculate probabilities
        probs_max = np.max(probs)  # Apply softmax to calculate probabilities
        return prediction, str(probs_max)

    predictions = safeguarding_prediction(data)

    return json.dumps(dict(enumerate(predictions)))
