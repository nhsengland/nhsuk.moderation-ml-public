import os
import subprocess
import sys

import torch
from azureml.core import Dataset, Model, Workspace
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
)

from azure_config import azure_config

# NOTE: In this submodule, we run the 'prep_for_huggingface

ws = azure_config.get_workspace()
MODELS_PATH = os.path.join("/mnt", "models")


def prep_for_huggingface():
    subprocess.run(["sudo", "chmod", "777", "/mnt"])
    os.environ["HF_HOME"] = "/mnt/huggingface_cache"
    os.environ["HUGGINGFACE_HUB_CACHE"] = "/mnt/huggingface_cache"
    os.environ["TRANSFORMERS_CACHE"] = "/mnt/huggingface_cache"
    if not os.path.exists(MODELS_PATH):
        os.mkdir(MODELS_PATH)


prep_for_huggingface()


def is_model_registered(workspace: Workspace, model_name: str) -> bool:
    # List all models in the workspace
    models = Model.list(workspace)

    # Check if model_name is among the names of the models in the workspace
    for model in models:
        if model.name == model_name:
            return True

    return False


def get_hf_model_DEPRECATED(model_name: str):
    """
    This function is deprecated because it relies on pre-registeding models, which we don't need to do.
    """
    registered = is_model_registered(workspace=ws, model_name=model_name)
    if not registered:
        print("You're about to download the model.")
        print(
            f"This will use the cache at {os.getenv('TRANSFORMERS_CACHE')}. If this isnt big enough for the model, change the cache location first. See README for this. "
        )

        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            force_download=True,
            resume_download=False,
        )

        # Step 2: Save the model locally
        model_name = model_name.replace("/", "_")

        model_path = MODELS_PATH + "/" + model_name
        if os.path.isdir(model_path):
            print("Model folder already exists, though model not registered. ")
            delete_choice = input("type 'yes' to delete, anything else to quit")
            if delete_choice == "yes":
                os.rmdir(model_path)
            else:
                print("Exiting")
                sys.exit()
        os.mkdir(model_path)
        model.save_pretrained(model_path)

        # Step 3: Register the model in Azure
        azure_model = Model.register(
            model_path=model_path,  # This points to a local file
            model_name=model_name,
            description="",
            workspace=ws,  # This is the workspace object we have from the code cells above
        )


def Question_answering_with_context(
    question, context, model_name="distilbert-base-uncased-distilled-squad"
):
    """
    question = "What is the capital of France?"
    context = "Paris is the capital and most populous city of France."
    Tested Models:
        distilbert-base-uncased-distilled-squad
    """

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        #   allow_remote_code=True,
    )
    model = AutoModelForQuestionAnswering.from_pretrained(
        model_name,
        #   allow_remote_code=True
    )

    inputs = tokenizer.encode_plus(question, context, return_tensors="pt")
    outputs = model(**inputs)
    answer_start = torch.argmax(outputs.start_logits)
    answer_end = torch.argmax(outputs.end_logits) + 1

    answer = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end])
    )
    return answer


def chat_microsoft_DialoGPT_medium(prompt, model_name="microsoft/DialoGPT-medium"):
    """
    What is the capital of France?
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(input_ids, max_length=100, num_return_sequences=1)
    response = tokenizer.decode(output[0], skip_special_tokens=True)

    return response
