"""
Module for generating and processing text data using GPT and other language
models, focusing on review generation. This is mostly built around making some
kind of 'base' prompt, and then making parameterised variations on that prompt
using parameter dicts which we're generating also in this module. 
"""

from typing import List
import concurrent.futures
import itertools
import json
import os
from difflib import get_close_matches

import dotenv
import openai
import pandas
from tenacity import retry, stop_after_attempt, wait_fixed
from tqdm import tqdm

from ..data_ingestion import data_ingestion

dotenv.load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")


BASE_PATH = os.path.join(os.path.dirname(__file__), "outputs")
READING_AGE_VALUE_LIST = ["8", "12", "15", "adult"]
SENTIMENT_VALUE_LIST = [
    "very negative",
    "slightly negative",
    "neutral",
    "slightly positive",
    "very positive",
]
LOCATION_VALUE_LIST = ["GP Practice", "hospital", "dentist", "care centre"]
TOPIC_VALUE_LIST = [""]
WORD_COUNT_VALUE_LIST = ["50", "200", "400"]


def process_param_set_into_n_reviews(param_set: dict, generation_params: dict):
    """
    Generates reviews based on parameter sets and adds them to a list.
    
    Parameters:
    - param_set (dict): Parameters for review generation.
    - generation_params (dict): Context and model parameters for generating reviews.
    
    Returns:
    - List[str]: Generated reviews in JSON string format.
    """
    reviews_data = []
    context = generate_parameterised_context(
        base_context_filename=generation_params["base_context_filename"],
        prompt_parameters=param_set,
    )
    try:
        reviews = generate_openai_review(
            context=context, n=generation_params["n"], model=generation_params["model"]
        )
        for review in reviews:
            json_str = json.dumps(review)
            reviews_data.append(json_str + "\n")
    except:
        print("This context failed!")
        print(context)
    return reviews_data


def load_json_reviews(json_name: str):
    """
    Loads reviews from a JSON file into a list.
    
    Parameters:
    - json_name (str): Name of the JSON file without extension.
    
    Returns:
    - List[dict]: Reviews loaded as list of dictionaries.
    """
    filepath = os.path.join(os.path.dirname(__file__), "outputs", json_name + ".json")
    with open(filepath, "r") as file:
        reviews_loaded = [json.loads(line) for line in file]
    return reviews_loaded


def register_generated_json(json_name: str, positive_label_column: str = None):
    """
    Registers a JSON file as a dataset. Optionally adds a positive label column.
    
    Parameters:
    - json_name (str): Name of the JSON file without extension to register.
    - positive_label_column (str, optional): Column name for a positive label, defaults to None.
    
    If the dataset name is already registered, the function aborts without action.
    """
    if json_name in set(data_ingestion.get_all_dataset_names_in_registry()):
        print("This one is already in registry. Aborting")
        return None
    reviews_loaded = load_json_reviews(json_name=json_name)
    df = pandas.DataFrame()
    df["Comment Text"] = reviews_loaded
    if positive_label_column:
        df[positive_label_column] = 1
    data_ingestion.register_dataframe(df=df, dataset_name=json_name)
    print("registered")


def get_gpt_completion(user_msg: str, n=1, model="gpt-3.5-turbo")-> List:
    """
    Fetches completions from a GPT model based on the user's message.
    
    Parameters:
    - user_msg (str): The message to get completions for.
    - n (int, optional): Number of completions to generate. Default is 1.
    - model (str, optional): The GPT model to use. Default is "gpt-3.5-turbo".
    
    Returns:
    - The model's completion(s) for the given message.
    """ 
    completions = openai.ChatCompletion.create(
        # model="gpt-4",
        model=model,
        n=n,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_msg},
        ],
    )
    return completions


def get_anthropic_completion(pre_prompt: str, anthropic_model="claude-1")-> str:
    """
    Generates a completion using an Anthropic model based on the given pre-prompt.
    
    Parameters:
    - pre_prompt (str): The initial text to generate completions for.
    - anthropic_model (str, optional): The Anthropic model to use. Default is "claude-1".
    
    Returns:
    - The model's generated completion for the pre-prompt.
    """
    anthropic = Anthropic()
    completion = anthropic.completions.create(
        model=anthropic_model,
        max_tokens_to_sample=1000,
        prompt=f"{HUMAN_PROMPT} {pre_prompt} {AI_PROMPT}",
    )
    return completion.completion


def filter_close_matches(main_list: List[str], filter_list: List[str], cutoff: float = 0.8) -> List[str]:
    """
    Filters out strings in main_list that are close matches to any string in filter_list.

    Parameters:
    - main_list (List[str]): The main list of strings.
    - filter_list (List[str]): The list of strings to check for close matches.
    - cutoff (float): A ratio for string similarity. Strings with similarity above this ratio will be considered a match.

    Returns:
    - List[str]: Filtered main_list without close matches.
    """

    return [s for s in main_list if not get_close_matches(s, filter_list, n=1, cutoff=cutoff)]



def modify_string(
    base_context_filename: str, string_to_lower: str, n=1, model="gpt-3.5-turbo"
):
    """
    Generates modified versions of a string using a GPT model and a specified context.

    Parameters:
    - base_context_filename (str): Filename for the base context to prepend to the string.
    - string_to_lower (str): The string to be modified.
    - n (int): Number of modifications to generate.
    - model (str): GPT model to use for generation.

    Returns:
    - List[str]: A list of modified string versions.
    """
    completions = openai.ChatCompletion.create(
        model=model,
        n=n,
        messages=[
            {
                "role": "system",
                "content": get_base_prompt(base_context_filename=base_context_filename),
            },
            {"role": "user", "content": string_to_lower},
        ],
    )
    completions = completions["choices"]
    generations = [message["message"]["content"] for message in completions]
    return generations


def generate_openai_review(context: str, model: str = "gpt-3.5-turbo", n=1)-> List:
    """
    Generates `n` reviews for a given context, returns as a list of strings
    """
    completions = openai.ChatCompletion.create(
        model=model,
        n=n,
        messages=[
            {"role": "system", "content": context},
            {"role": "user", "content": "Please generate a comment"},
        ],
    )
    completions = completions["choices"]
    generations = [message["message"]["content"] for message in completions]
    return generations


def get_base_prompt(base_context_filename: str)-> str:
    """Opens a base prompt based on filepath"""
    file_path = os.path.join(
        os.path.dirname(__file__), "prompts_and_text", base_context_filename + ".txt"
    )
    with open(file_path, "r") as file:
        text = file.read()

    return text


def generate_parameterised_context(base_context_filename: str, prompt_parameters: dict)-> str:
    """
    Makes a full prompt by injecting the parameters into it a base prompt appropriately. 
    """
    text = get_base_prompt(base_context_filename=base_context_filename)

    text = text.format(**prompt_parameters)

    return text


def generate_parameter_dictionaries_combinatorically(
    dictionary_of_parameter_names_to_value_lists,
):
    """
    This is to create the combinatoric set of possible parameter dictionaries.
    It's nice! You supply all your keys and all your values, and it will give
    you all the possible distinct dictionaries comprised of these. 
    Example: 
        {'foo': [1,2], 'bar': ['a','b']} # As an input, goes to
        -> [
        {'foo':1, bar:'a'},
        {'foo':1, bar:'b'},
        {'foo':2, bar:'a'},
        {'foo':2, bar:'b'},
        ]
    """
    keys = list(dictionary_of_parameter_names_to_value_lists.keys())
    value_lists = list(dictionary_of_parameter_names_to_value_lists.values())

    for values in itertools.product(*value_lists):
        yield dict(zip(keys, values))


def generate_lowered_version_of_data(dataset_name, positive_value_column):
    """
    This is for creating a version of a dataset with a lower literacy level.
    Parameters:
        - dataset name: name of source dataset
        - positive_value_column: name of the labelled column, like
          'safeguarding' or 'complaint'. This is so that the lowered version of
          the dataset can have the same functional label. 
    
    Note that this is basically about making 100's of very similar queries to
    OpenAi. If you do this in series it takes a very long time. Hence the
    paralllel processing, which is approriate for I/O bound stuff like this. 
    """
    dataset_name_lowered = dataset_name + "_lowered"
    df = data_ingestion.DataRetrieverDatastore(dataset_name=dataset_name).dataset

    def process_string(sstring):
        return modify_string(
            base_context_filename="literacy_level_lowering_1", string_to_lower=sstring
        )

    # Using this 'retry' decorator from tenacity to gracefully handle waiting for API.
    @retry(wait=wait_fixed(1), stop=stop_after_attempt(5))
    def parallel_process(review_to_lower):
        try:
            review = process_string(review_to_lower)[0]
            json_str = json.dumps(review)
            return json_str + "\n"
        except Exception as e:
            print(f"Error processing review '{review_to_lower}': {e}")
            return None

    def save_result_to_file(result, file):
        if result:
            file.write(result)

    num_workers = os.cpu_count()
    filepath = os.path.join(os.path.dirname(__file__), dataset_name_lowered + ".json")

    with open(filepath, "a") as file:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            for result in tqdm(
                executor.map(parallel_process, df["Comment Text"]), total=len(df)
            ):
                save_result_to_file(result, file)
            #todo Take the file saving component out of the parallel process. 
    print("lower literacy version saved")
    register_generated_json(
        dataset_name_lowered, positive_label_column=positive_value_column
    )



def make_json_of_generated_reviews(
    dictionary_of_parameter_names_to_value_lists: dict,
    base_context_filename: str,
    model: str = "gpt-3.5-turbo",
    n: int = 1,
    version: int = 1,
)-> None:
    """
    Creates a JSON file of generated reviews based on parameterized contexts and
    writes it to the outputs directory.

    Parameters: - dictionary_of_parameter_names_to_value_lists (dict):
    Parameters for generating varied contexts. - base_context_filename (str):
    Base filename for the context template. - model (str, optional): GPT model
    identifier. Defaults to "gpt-3.5-turbo". - n (int, optional): Number of
    reviews to generate per context. Defaults to 1. - version (int, optional):
    Version number for naming the output file. Defaults to 1.

    Returns: Nothing, but it will have written something to disk.
    """
    def process_param_set(param_set):
        reviews_data = []
        context = generate_parameterised_context(
            base_context_filename=base_context_filename, prompt_parameters=param_set
        )
        try:
            reviews = generate_openai_review(context=context, n=n, model=model)
            for review in reviews:
                json_str = json.dumps(review)
                reviews_data.append(json_str + "\n")
        except Exception as e:
            print(f"This context failed due to error: {e}")
        return reviews_data

    params = generate_parameter_dictionaries_combinatorically(
        dictionary_of_parameter_names_to_value_lists
    )
    params = list(params)

    filename_string = base_context_filename + "_" + model + "_" + str(version) + ".json"
    filename = os.path.join(BASE_PATH, filename_string)

    all_review_data = []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        for review_data in list(
            tqdm(executor.map(process_param_set, params), total=len(params))
        ):
            all_review_data.extend(review_data)

    with open(filename, "w") as file:  
        for line in all_review_data:
            file.write(line)

    print(f"All written to {filename}")
