#! WARNING
# The data_augmentation module has undergone significant refactoring since this script was used. We keep it for posterity. If using, it will need some adaptation.
import concurrent.futures
import json
import os
import pathlib
import random
import sys

from tqdm import tqdm

from src.data_augmentation import data_generation

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, root_path)


nae_params = data_generation.generate_parameter_dictionaries_combinatorically(
    dictionary_of_parameter_names_to_value_lists={
        "location": [
            "GP Practice",
            "dental practice",
            "hospital",
            "sexual health clinic",
            "optician",
            "pharmacy",
            "hospital department",
        ],
        "sentiment": [
            "negative",
            "slightly negative",
            "neutral",
            "neutral",
            "neutral",
            "slightly positive" "slightly positive",
            "positive",
        ],
        "tense": ["present", "present", "present", "past"],
        "word_count": ["30", "30", "40", "50", "100", "150"],
    }
)
nae_params = list(nae_params)


def process_param_set(param_set):
    reviews_data = []
    context = data_generation.generate_parameterised_context(
        base_context_filename="nae_generation_context_1", prompt_parameters=param_set
    )
    try:
        reviews = data_generation.generate_openai_review(
            context=context, n=1, model="gpt-4"
        )
        for review in reviews:
            json_str = json.dumps(review)
            reviews_data.append(json_str + "\n")
    except Exception as e:
        print("This context failed!")
        print(context)
        print(e)
    return reviews_data


BASE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "src", "data_augmentation", "outputs"
)
print(BASE_PATH)

filename = os.path.join(BASE_PATH, "nae_gen_gp4_finalsample.json")
nae_params = random.sample(nae_params, 100)

with open(filename, "a") as file:
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Use list(tqdm()) to ensure tqdm displays progress correctly with concurrent futures
        for review_data in list(
            tqdm(executor.map(process_param_set, nae_params), total=len(nae_params))
        ):
            for line in review_data:
                file.write(line)


with open(filename, "r") as file:
    reviews_loaded = [json.loads(line) for line in file]
