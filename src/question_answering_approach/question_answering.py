import json
import os
from dataclasses import dataclass

import azureml.core
import dotenv
import pandas
from anthropic import AI_PROMPT, HUMAN_PROMPT, Anthropic
from src.azure_config import azure_config
from src.model_getting import model_getting

from ..data_ingestion import data_ingestion
from ..model_assessment import model_assessment

dotenv.load_dotenv()


# model_getting.EndpointsRetriever()


class LlamaClassifier:
    """Instantiates a Llama classifier. The classifier will use either the general model (llama-7b) by default, or can use the chat model (llama-7b-chat), and will use a prompt file as set. The classifier can classify single reviews, or classify entire datasets by iterating the single review classification function.

    Parameters:
        pre_prompt_name: a string, the name of the txt file that contains the model prompt
        classifier_type: a string, either 'general' to use the general llama model or 'chat' to use the chat llama model
        model_name: string, the name of the model for logging purposes
        temperature: decimal, the temperature parameter by Llama when generating the response
        top_p: decimal, the top_p parameter used by Llama when generating the response

    Returns:
        A Llama classifier that can then be used to classify either single reviews or entire datasets
        '"""

    def __init__(
        self,
        pre_prompt_name: str,
        classifier_type="general",
        model_name="llama-2-7b",
        temperature=0,
        top_p=0.1,
    ) -> None:
        self.classifier_type = classifier_type
        self.pre_prompt_name = pre_prompt_name
        self.pre_prompt = self.get_pre_prompt()
        self.model_name = model_name + "-" + classifier_type
        self.temperature = temperature
        self.top_p = top_p

    def get_pre_prompt(self):
        filepath = os.path.join(
            os.path.dirname(__file__), "prompts", self.pre_prompt_name + ".txt"
        )
        with open(filepath, "r") as file:
            text = file.read()
        return text

    def classify_single_review(self, review: str):
        if self.classifier_type == "general":
            query = {
                "input_string": [self.pre_prompt + "Input: '" + review + "'. Output: "],
                "parameters": {
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "max_new_tokens": 1,
                    "do_sample": True,
                },
            }

            response = model_getting.get_endpoint_response(
                query_content=query, access_details=azure_config.llama_7b_details
            )

            return response[-4]

        elif self.classifier_type == "chat":
            query = {
                "input_string": [{"role": "user", "content": self.pre_prompt + review}],
                "parameters": {
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "do_sample": True,
                    "max_new_tokens": 150,
                },
            }

            response = model_getting.get_endpoint_response(
                query_content=query, access_details=azure_config.llama_7b_chat_details
            )

            # We need to map the output to the labels. The model will give a 150 token text that generally includes the labels as 'Label 0' etc. We check for these, allowing for an 'unknown' classification of '3' for the responses that don't include those labels.
            if response.rfind("Label 2") != -1:
                return "2"
            elif response.rfind("Label 1") != -1:
                return "1"
            elif response.rfind("Label 0") != -1:
                return "0"
            else:
                return "3"

        else:
            print("Please enter a valid classifier type")
            return None

    def classify_datasets(
        self,
        positive_label_dataset_name_list,
        negative_label_dataset_name_list,
        y_column_name: str,
        name_of_column_to_classify: str,
        train_test_val_label: str = "test",
        n_to_sample: int = 0,
        balance_data: bool = False,
    ):
        self.y_column_name = y_column_name
        self.positive_label_dataset_name_list = positive_label_dataset_name_list
        self.negative_label_dataset_name_list = negative_label_dataset_name_list
        self.n_to_sample = n_to_sample

        dfs = [
            data_ingestion.DataRetrieverDatastore(name).dataset
            for name in positive_label_dataset_name_list
            + negative_label_dataset_name_list
        ]
        df = pandas.concat(dfs)

        if train_test_val_label != "all":
            df = df[df[train_test_val_label] == 1]

        if balance_data:
            min_count = df[y_column_name].value_counts().min()

            balanced_df = pandas.DataFrame()
            for value in df[y_column_name].unique():
                sampled_rows = df[df[y_column_name] == value].sample(
                    min_count, random_state=azure_config.RANDOMSEED
                )
                balanced_df = pandas.concat([balanced_df, sampled_rows])
            df = balanced_df.sample(frac=1, random_state=azure_config.RANDOMSEED)

        if n_to_sample > 0:
            df = df.sample(n=n_to_sample, random_state=azure_config.RANDOMSEED)

        self.data_to_classify = df

        preds = []
        for i, row in df.iterrows():
            try:
                pred = self.classify_single_review(
                    review=row[name_of_column_to_classify]
                )
                pred = int(pred)
            except:
                pred = None

            preds += [pred]
            print(".", end="", flush=True)
        print("\n")

        self.preds = preds
        self.data_to_classify["predictions"] = preds
        print("classifications gotten")

    def log_all_attributes(self, run: azureml.core.Run):
        attribute_dict = {
            "classifier_type": self.classifier_type,
            "pre_prompt_name": self.pre_prompt_name,
            "model_name": self.model_name,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "positive_label_dataset_name_list": self.positive_label_dataset_name_list,
            "negative_label_dataset_name_list": self.negative_label_dataset_name_list,
            "n_to_sample": self.n_to_sample,
        }
        for key in attribute_dict:
            if isinstance(attribute_dict[key], list):
                if len(attribute_dict[key]) > 0:
                    run.log(name=key, value="; ".join(attribute_dict[key]))
            else:
                run.log(name=key, value=attribute_dict[key])
        print("All attributes logged")

    def get_assessor(self):
        self.assessor = model_assessment.PredictionsAssessor(
            predictions_column_name="predictions",
            test_data=self.data_to_classify,
            y_column_name=self.y_column_name,
        )
        print("Assessor gotten")


class AnthropicClassifier:
    def __init__(self, pre_prompt_name: str, model="claude-1") -> None:
        self.pre_prompt_name = pre_prompt_name
        self.pre_prompt = self.get_pre_prompt()
        self.model = model

    def get_pre_prompt(self):
        filepath = os.path.join(
            os.path.dirname(__file__), "prompts", self.pre_prompt_name + ".txt"
        )
        with open(filepath, "r") as file:
            text = file.read()
        return text

    def classify_single_review(self, review: str):
        anthropic = Anthropic()
        completion = anthropic.completions.create(
            model=self.model,
            max_tokens_to_sample=1000,
            prompt=f"{HUMAN_PROMPT} {self.pre_prompt} \n {review}{AI_PROMPT}",
        )
        return completion.completion

    def classify_datasets(
        self,
        positive_label_dataset_name_list,
        negative_label_dataset_name_list,
        y_column_name: str,
        name_of_column_to_classify: str,
        train_test_val_label: str = "test",
        n_to_sample: int = 0,
        balance_data: bool = False,
    ):
        self.y_column_name = y_column_name

        dfs = [
            data_ingestion.DataRetrieverDatastore(name).dataset
            for name in positive_label_dataset_name_list
            + negative_label_dataset_name_list
        ]
        df = pandas.concat(dfs)

        df = df[df[train_test_val_label] == 1]

        if balance_data:
            min_count = df[y_column_name].value_counts().min()

            balanced_df = pandas.DataFrame()
            for value in df[y_column_name].unique():
                sampled_rows = df[df[y_column_name] == value].sample(
                    min_count, random_state=azure_config.RANDOMSEED
                )
                balanced_df = balanced_df.append(sampled_rows)
            df = balanced_df.sample(frac=1, random_state=azure_config.RANDOMSEED)

        if n_to_sample > 0:
            df = df.sample(n=n_to_sample, random_state=azure_config.RANDOMSEED)

        self.data_to_classify = df

        preds = []
        for i, row in df.iterrows():
            try:
                pred = self.classify_single_review(
                    review=row[name_of_column_to_classify]
                )
                pred = int(pred)
            except:
                pred = None

            preds += [pred]
            print(".", end="", flush=True)
        print("\n")

        self.preds = preds
        self.data_to_classify["predictions"] = preds
        print("classifications gotten")

    def log_all_attributes(self, run: azureml.core.Run):
        attribute_dict = {
            "pre_prompt_name": self.pre_prompt_name,
            "model": self.model,
            "positive_label_dataset_name_list": self.positive_label_dataset_name_list,
            "negative_label_dataset_name_list": self.negative_label_dataset_name_list,
            "n_to_sample": self.n_to_sample,
        }
        for key in attribute_dict:
            if isinstance(attribute_dict[key], list):
                if len(attribute_dict[key]) > 0:
                    run.log(name=key, value="; ".join(attribute_dict[key]))
            else:
                run.log(name=key, value=attribute_dict[key])
        print("All attributes logged")

    def get_assessor(self):
        self.assessor = model_assessment.PredictionsAssessor(
            predictions_column_name="predictions",
            test_data=self.data_to_classify,
            y_column_name=self.y_column_name,
        )
        print("Assessor gotten")


# anth = AnthropicClassifier(
#     pre_prompt_name='Complaints_detector_1'
# )
# review_to_classify = "I had a horrific experience at the Greenfield Medical Centre recently. I went in for a routine check-up and left feeling violated and unsafe. During my appointment, the receptionist stole my purse from my bag. I only realized it when I got home and found my bag open and my money missing. I couldn't believe that a member of the clinic's staff would stoop so low as to steal from patients. It was a clear case of criminality, and I felt utterly disgusted and violated. This incident has shattered my trust in this practice, and I strongly advise everyone to stay away from Greenfield Medical Centre."
# ans = anth.classify_single_review(review=review_to_classify)

# # print(anth.pre_prompt)

# print(ans)
