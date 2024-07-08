#! WARNING
# The data_augmentation module has undergone significant refactoring since this script was used. We keep it for posterity. If using, it will need some adaptation.
import concurrent.futures
import json
import os
import pathlib
import random

from tqdm import tqdm

from src.data_augmentation import data_augmentation

complaints_params = data_augmentation.generate_parameter_dictionaries_combinatorically(
    dictionary_of_parameter_names_to_value_lists={
        "location": [
            "GP Practice",
            "dental practice",
            "hospital",
            "sexual health clinic",
            "optician",
            "pharmacy",
            "hopsital department",
        ],
        "topic": [
            "case of criminality by a member of staff",
            "formal complaint being raised, or intent to do so",
            "racist, homophobic, mysoginist or ableist behaviour by a member of staff",
            "medical negligence by a member of staff",
            "violent behaviour by a member of staff",
            "overcharging patients, or not being transparent about (and or charging) excessive fees",
        ],
        "sentiment": ["very negative", "slightly negative", "negative", "neutral"],
        "word_count": ["100", "200", "300", "200", "250"],
    }
)
complaints_params = list(complaints_params)

def process_param_set(param_set):
    reviews_data = []
    context = data_augmentation.generate_parameterised_context(
        base_context_filename="comment_generation_context_1",
        prompt_parameters=param_set,
    )
    try:
        reviews = data_augmentation.generate_openai_review(
            context=context, n=4, model="gpt-4"
        )
        for review in reviews:
            json_str = json.dumps(review)
            reviews_data.append(json_str + "\n")
    except:
        print("This context failed!")
        print(context)
    return reviews_data


BASE_PATH = os.path.join(
    os.path.dirname(__file__), "src", "data_augmentation", "outputs"
)

filename = os.path.join(BASE_PATH, "complaints_gen_gpt4_v3.json")
complaints_params = random.sample(complaints_params,50)

with open(filename, "a") as file:
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Use list(tqdm()) to ensure tqdm displays progress correctly with concurrent futures
        for review_data in list(
            tqdm(
                executor.map(process_param_set, complaints_params),
                total=len(complaints_params),
            )
        ):
            for line in review_data:
                file.write(line)


with open(filename, "r") as file:
    reviews_loaded = [json.loads(line) for line in file]
