import json
import os
from typing import List


def get_test_parameters():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    target_dir = os.path.join(current_dir, "test_parameters.json")

    with open(target_dir, "r") as file:
        data = json.load(file)
    return data


def update_test_parameters(field_to_update: str, new_value):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    target_dir = os.path.join(current_dir, "test_parameters.json")

    with open(target_dir, "r") as file:
        data = json.load(file)

    data[field_to_update] = new_value

    with open(target_dir, "w") as file:
        json.dump(data, file, indent=4)


def define_column_to_embed_for_tests(column_to_embed_for_tests: str = "Comment Text"):
    update_test_parameters(
        field_to_update="name_of_column_to_embed", new_value=column_to_embed_for_tests
    )


def define_list_of_datasets_for_tests(list_of_dataset_names: List[str]):
    update_test_parameters(
        field_to_update="dataset_names", new_value=list_of_dataset_names
    )


def define_list_of_models_for_tests(list_of_model_names: List[str]):
    update_test_parameters(
        field_to_update="embeddings_model_names", new_value=list_of_model_names
    )
