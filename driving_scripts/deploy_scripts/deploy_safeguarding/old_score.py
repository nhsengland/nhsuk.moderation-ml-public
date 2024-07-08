import json
import os
import pickle

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import BertForSequenceClassification, BertTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Called when the service is loaded
def init():
    global model
    # Get the path to the deployed model file and load it

    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=3,
        output_attentions=False,
        output_hidden_states=False,
    )

    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "safeguarding3cat.pkl")
    print(model_path)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    print(model)

    global tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)


# Called when a request is received
def run(raw_data):
    # Get the input data as a numpy array
    data = json.loads(raw_data)["data"]

    def preprocessing(input_text, tokenizer):
        return tokenizer.encode_plus(
            input_text,
            add_special_tokens=True,
            max_length=512,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors="pt",  # Return pytorch tensors.
        )

    def safeguarding_prediction(text_to_classify):
        # We need Token IDs and Attention Mask for inference on the new sentence
        test_ids = []
        test_attention_mask = []
        # Apply the tokenizer
        encoding = preprocessing(text_to_classify, tokenizer)

        # Extract IDs and Attention Mask
        test_ids.append(encoding["input_ids"])
        test_attention_mask.append(encoding["attention_mask"])
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

    # Call prediction
    predictions = safeguarding_prediction(data)

    # Return the predictions as JSON
    return json.dumps(dict(enumerate(predictions)))
