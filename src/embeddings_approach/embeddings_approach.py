import gc
import itertools
import math
import multiprocessing
import os
import time
import typing
import zipfile
from dataclasses import dataclass
from typing import List, Optional

import azureml.core
import hyperopt
import joblib
import numpy as np
import optuna
import pandas
import petname
import sentence_transformers
import sklearn
import sklearn.ensemble
import sklearn.linear_model
import sklearn.metrics
import sklearn.svm
import torch
import xgboost
from azureml.core import Model, Workspace
from hyperopt import STATUS_OK, SparkTrials, Trials, fmin, hp, mongoexp, tpe
from optuna.samplers import TPESampler
from pyspark import SparkConf
from pyspark.sql import SparkSession

from src.azure_config import azure_config
from src.data_ingestion import data_ingestion, data_utils
from src.environment_management import environment_management
from src.model_assessment import model_assessment

azure_config.setup_mnt_permissions_and_check_mount()


#! This is imporant! If we don't put this here, the spark parallelisation won't work - it will default to using the root folder as cache, and will fill up and crash.
os.environ["TMPDIR"] = "/mnt"
#! This is imporant! If we don't put this here, the spark parallelisation won't work - it will default to using the root folder as cache, and will fill up and crash.


def get_embeddings_model(model_for_embeddings_name: str):
    model_for_embeddings_name = model_for_embeddings_name.replace("/", "-")
    return model_assessment.model_getting.get_model_by_name(model_for_embeddings_name)


def make_embedding_column_name(
    name_of_column_to_embed: str, model_for_embeddings_name: str
):
    return "embeddings_" + name_of_column_to_embed + model_for_embeddings_name


@dataclass
class EmbeddingsApproach:
    """
    A class for managing and optimizing text classification models using embeddings.

    Attributes:
        positive_label_dataset_name_list (List[str]): List of dataset names with positive labels.
        negative_label_dataset_name_list (List[str]): List of dataset names with negative labels.
        augmented_dataset_name_list (List[str]): List of augmented dataset names.
        model_for_embeddings_name (str): Name of the model used for generating embeddings.
        name_of_column_to_embed (str): Name of the column containing text to be embedded.
        name_of_y_column (str): Name of the column containing target labels.
        classifier_class_name (str): Name of the classifier class.
        scorer (Optional[sklearn.metrics.make_scorer]): Scorer function for model evaluation. Default is None.
        balance_test (bool): Whether to balance the test dataset. Default is True.
        balance_val (bool): Whether to balance the validation dataset. Default is True.
        max_evals (int): Maximum number of evaluations for hyperparameter optimization. Default is 20.
        multiclass (bool): Indicates if the classification task is multiclass. Default is False.
        prefix (str): A prefix string for internal usage. Default is an empty string.
        override_with_new_embeddings (bool): Flag to override existing embeddings. Default is False.
        parallelise (bool): Enable parallel execution. Default is False.
        parallel_limit (int): Limit for parallel threads. Default is 0, meaning no limit.

    Methods:
        __post_init__: Initializes additional attributes and prepares the embedding model.
        get_and_clean_df: Retrieves and cleans a dataframe based on its name.
        refresh_all_dfs: Refreshes all dataframes, cleaning and concatenating them as needed.
        get_test_and_val_dfs: Splits the data into test and validation datasets, balancing them if required.
        find_optimised_classifier: Optimizes the classifier using hyperparameter tuning.
        get_assessor_for_optimised_model: Creates an assessor object for the optimized model.
        log_all_dataset_version_numbers: Logs version numbers of all datasets involved in the run.
        generate_description_for_model: Generates a structured description of the model.
        log_all_attributes: Logs all attributes of the model and the run.
        make_model_name: Generates a name for the model.
        register_optimal_model: Registers the optimized model in an external system (e.g., Azure ML).
    """

    positive_label_dataset_name_list: List[str]
    negative_label_dataset_name_list: List[str]
    augmented_dataset_name_list: List[str]
    model_for_embeddings_name: str
    name_of_column_to_embed: str  # Probably Comment_Text
    name_of_y_column: str  # like 'Complaint', or 'safeguarding'
    classifier_class_name: str
    scorer: Optional[sklearn.metrics.make_scorer] = None
    balance_test: bool = True
    balance_val: bool = True
    max_evals: int = 20
    multiclass: bool = False
    prefix: str = ""
    override_with_new_embeddings: bool = False
    parallelise: bool = False
    parallel_limit: int = 0  # Set this to a number above 0 to limit thread count
    SVM_probability: bool = False

    def __post_init__(self):
        self.__initial_attributes = self.__dict__.copy()
        self.model_for_embeddings = get_embeddings_model(self.model_for_embeddings_name)
        self.all_embeddings_gotten = False
        # self.embeddings_column_name = (
        #     "embeddings_"
        #     + self.name_of_column_to_embed
        #     + self.model_for_embeddings_name
        # )
        self.embeddings_column_name = make_embedding_column_name(
            name_of_column_to_embed=self.name_of_column_to_embed,
            model_for_embeddings_name=self.model_for_embeddings_name,
        )
        self.argmin = None
        self.refresh_all_dfs()
        self.get_test_and_val_dfs()
        self.assessor = None
        self.make_model_name()
        self.embedding_dimension = None

    def get_and_clean_df(self, name):
        df = data_ingestion.DataRetrieverDatastore(dataset_name=name).dataset
        return df.dropna(subset=[self.name_of_column_to_embed])

    def refresh_all_dfs(self):
        self.positive_label_dfs_dict = {
            name: self.get_and_clean_df(name)
            for name in self.positive_label_dataset_name_list
        }
        self.negative_label_dfs_dict = {
            name: self.get_and_clean_df(name)
            for name in self.negative_label_dataset_name_list
        }
        self.augmented_datasets_dict = {
            name: self.get_and_clean_df(name)
            for name in self.augmented_dataset_name_list
        }
        self.dfs_to_sample_dict = {
            **self.augmented_datasets_dict,
            **self.negative_label_dfs_dict,
        }
        self.positive_label_df = pandas.concat(self.positive_label_dfs_dict.values())
        self.negative_label_df = pandas.concat(self.negative_label_dfs_dict.values())
        if len(self.augmented_dataset_name_list) > 0:
            self.augmented_data_df = pandas.concat(
                self.augmented_datasets_dict.values()
            )
        else:
            self.augmented_data_df = None
        self.get_test_and_val_dfs()

    def get_test_and_val_dfs(self):
        test_df = pandas.concat(
            [
                df[df["test"] == 1]
                for df in (
                    self.positive_label_df,
                    self.negative_label_df,
                    self.augmented_data_df,
                )
                if df is not None
            ]
        )

        val_df = pandas.concat(
            [
                df[df["val"] == 1]
                for df in (
                    self.positive_label_df,
                    self.negative_label_df,
                    self.augmented_data_df,
                )
                if df is not None
            ]
        )

        if self.balance_test:
            test_df = data_ingestion.balance_dataframe(
                df=test_df, column_to_balance=self.name_of_y_column
            )
        if self.balance_val:
            val_df = data_ingestion.balance_dataframe(
                df=val_df, column_to_balance=self.name_of_y_column
            )

        self.val_df = val_df
        self.test_df = test_df

    def find_optimised_classifier(self):
        """
        Finds the optimised classifier based on the provided configurations.

        This method uses Optuna to perform hyperparameter optimization on the
        specified classifier using the training data. The scoring mechanism can
        be customized; by default, it uses recall score for multiclass tasks and
        F1 score for binary classification tasks.

        After optimization, the method updates the `optimised_classifier` attribute
        with the best model and `best_value` attribute with its corresponding score.
        Additionally, it prints the winning hyperparameters and the best score achieved.

        Notes:
        - The method constructs the training dataset based on the positive label data
          and the sampling of other datasets in `dfs_to_sample_dict`.
        - The number of samples drawn from each dataset in `dfs_to_sample_dict` is
          also optimized in the process.
        - If parallelism is enabled, Optuna will perform optimization in parallel using
          all available CPUs.
        - Once done, the model and winning values are logged
        - There's a possible race condition in the optimisation. Each trial will save it's model and best value to the object if it beats the current one which it can see - depending on how optuna handles the object instance (I'm not sure), this *could* create a race condition. This is why we print the best value from two sources. Testing indicates that there's no race condition, but this isn't watertight. If there is a race condition, it would only mean that the optimisation is slightly sub-optimal.

        Raises:
            ValueError: If the necessary attributes (e.g., `classifier_class_name`) are not set.
        """
        print("Creating Optimised Classifier")

        if self.scorer is None:
            if self.multiclass:
                self.scorer = sklearn.metrics.make_scorer(
                    sklearn.metrics.recall_score, average="macro"
                )
            else:
                self.scorer = sklearn.metrics.make_scorer(sklearn.metrics.f1_score)

        self.optimised_classifier = None
        self.best_value = 0

        def objectives(trial: optuna.Trial):
            params_for_classifier = get_optuna_params(
                self.classifier_class_name, trial=trial
            )
            if (self.classifier_class_name == "SVM") & (self.SVM_probability == True):
                params_for_classifier["probability"] = True

            model = get_classifier_by_class_name(self.classifier_class_name)(
                **params_for_classifier
            )
            dfs_for_training_df = [
                self.positive_label_df[self.positive_label_df["train"] == 1]
            ]
            for name, df_to_sample in self.dfs_to_sample_dict.items():
                df_to_sample = df_to_sample[df_to_sample["train"] == 1]
                n_rows = trial.suggest_int(name, 1, len(df_to_sample))
                dfs_for_training_df.append(
                    df_to_sample.sample(n=n_rows, random_state=azure_config.RANDOMSEED)
                )

            training_df = pandas.concat(dfs_for_training_df)

            model.fit(
                X=np.array(list(training_df[self.embeddings_column_name])),
                y=list(training_df[self.name_of_y_column]),
            )
            test_X = np.array(list(self.test_df[self.embeddings_column_name]))
            test_y = list(self.test_df[self.name_of_y_column])
            test_score = self.scorer(model, test_X, test_y)
            # gc.collect()

            if test_score > self.best_value:
                self.best_value = test_score
                self.optimised_classifier = model

            del training_df
            del model

            return test_score

        study = optuna.create_study(
            sampler=TPESampler(seed=azure_config.RANDOMSEED), direction="maximize"
        )
        if self.parallelise:
            if self.parallel_limit > 0:
                n_jobs = self.parallel_limit
            else:
                n_jobs = -1
        else:
            n_jobs = 1
        study.optimize(objectives, n_trials=self.max_evals, n_jobs=n_jobs)

        self.argmin = study.best_params

        print(f"Winning values were:{study.best_params}")
        print(f"Best score was {self.best_value}")
        print(f"Best score was {study.best_value}")

    def get_assessor_for_optimised_model(self):
        df_for_assessor = self.val_df.copy()
        start_time = time.time()
        df_for_assessor["predictions"] = self.optimised_classifier.predict(
            np.array(list(df_for_assessor[self.embeddings_column_name]))
        )
        end_time = time.time()
        self.time_take_to_predict = end_time - start_time
        if self.scorer is not None:
            assessor = model_assessment.PredictionsAssessor(
                test_data=df_for_assessor,
                y_column_name=self.name_of_y_column,
                predictions_column_name="predictions",
                scorer=self.scorer,
            )
        else:
            assessor = model_assessment.PredictionsAssessor(
                test_data=df_for_assessor,
                y_column_name=self.name_of_y_column,
                predictions_column_name="predictions",
            )
        self.assessor = assessor
        print("assessor gotten")

    def log_all_dataset_version_numbers(self, run: azureml.core.Run):
        for dataset_name in (
            self.positive_label_dataset_name_list
            + self.negative_label_dataset_name_list
            + self.augmented_dataset_name_list
        ):
            version_number = data_utils.get_latest_dataset_version(
                dataset_name=dataset_name
            )
            run.log(name=dataset_name + "_version", value=str(version_number))

    def generate_description_for_model(self):
        structured_string = ""
        for attribute, value in self.__initial_attributes.items():
            structured_string += (
                f"Attribute Name: {attribute}, Attribute Value: {value}\n"
            )
        return structured_string

    def log_all_attributes(self, run: azureml.core.Run):

        environment_management.check_environment_is_correct()

        run.log(
            name="Classifier Class", value=self.optimised_classifier.__class__.__name__
        )
        for attribute, value in self.__initial_attributes.items():
            if attribute != "scorer":
                if isinstance(value, list):
                    run.log(name=attribute, value="; ".join(value))
                else:
                    run.log(name=attribute, value=value)
        for attribute, value in self.argmin.items():
            run.log(name=attribute, value=value)
        run.log(name="Environment name", value=azure_config.ENVIRONMENT_NAME)
        run.log(name="Environment version", value=azure_config.ENVIRONMENT_VERSION)
        run.log(name="model name", value=self.model_name)
        run.log(name="time to predict val set", value=self.time_take_to_predict)
        self.log_all_dataset_version_numbers(run=run)
        print("All attributes logged")

    def make_model_name(self):
        self.model_name = petname.Generate(2)

    def register_optimal_model(self):
        from azure.ai.ml.constants import AssetTypes
        from azure.ai.ml.entities import Model

        if not hasattr(self, "model_name"):
            self.make_model_name()

        base_path = "embeddings_models/"

        path_for_model = os.path.join(base_path, self.model_name + ".pkl")
        os.makedirs(base_path, exist_ok=True)

        joblib.dump(value=self.optimised_classifier, filename=path_for_model)

        file_model = Model(
            path=path_for_model,
            type=AssetTypes.CUSTOM_MODEL,
            name=self.model_name,
            description=self.generate_description_for_model(),
        )

        ml_client = azure_config.get_ml_client()
        ml_client.models.create_or_update(file_model)
        print(f"Model registered under the name {self.model_name}")


def get_optuna_params(classifier_class_name: str, trial: optuna.Trial):
    args_getter = {
        "SVM": get_svm_optuna_args(trial),
        "random_forest": get_random_forest_optuna_args(trial),
        "logistic_regression": get_lr_optuna_args(trial),
        "xgboost": get_xgboost_optuna_args(trial),
    }
    return args_getter.get(classifier_class_name)


def get_classifier_by_class_name(classifier_class_name: str):
    return {
        "SVM": sklearn.svm.SVC,
        "random_forest": sklearn.ensemble.RandomForestClassifier,
        "logistic_regression": sklearn.linear_model.LogisticRegression,
        "xgboost": xgboost.XGBClassifier,
    }.get(classifier_class_name)


def get_svm_optuna_args(trial: optuna.Trial):
    C = trial.suggest_float("C", 0.0001, 1, log=True)
    # C = trial.suggest_loguniform('C', 1e-4, 1,)
    gamma = trial.suggest_float("gamma", 1e-5, 1e-1, log=True)
    return {"C": C, "gamma": gamma}


def get_random_forest_optuna_args(trial):
    n_estimators = trial.suggest_int("n_estimators", 100, 1000)
    max_depth = trial.suggest_int("max_depth", 1, 100)
    min_samples_split = trial.suggest_float("min_samples_split", 0.1, 1.0)
    min_samples_leaf = trial.suggest_float("min_samples_leaf", 0.1, 0.5)

    return {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf,
    }


def get_lr_optuna_args(trial: optuna.Trial):
    C = trial.suggest_float("C", 0.0001, 1e3, log=True)
    # penalty = trial.suggest_categorical('penalty', ['l2', 'none'])
    solver = trial.suggest_categorical(
        "solver", ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]
    )
    max_iter = trial.suggest_int("max_iter", 1e2, 1e3, log=True)

    return {"C": C, "penalty": "l2", "solver": solver, "max_iter": max_iter}


def get_xgboost_optuna_args(trial):
    # Number of boosting rounds
    n_estimators = trial.suggest_int("n_estimators", 100, 1000)

    # Maximum depth of a tree
    max_depth = trial.suggest_int("max_depth", 1, 20)

    # Boosting learning rate
    learning_rate = trial.suggest_float("learning_rate", 0.001, 0.3)

    # Minimum sum of instance weight (hessian) needed in a child
    min_child_weight = trial.suggest_int("min_child_weight", 1, 10)

    # L1 regularization term on weights
    reg_alpha = trial.suggest_float("reg_alpha", 0.1, 1.0)

    # L2 regularization term on weights
    reg_lambda = trial.suggest_float("reg_lambda", 0.1, 1.0)

    # Proportion of training data to randomly sample in each round
    subsample = trial.suggest_float("subsample", 0.5, 1.0)

    # Proportion of columns to randomly sample for each tree
    colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0)

    return {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "min_child_weight": min_child_weight,
        "reg_alpha": reg_alpha,
        "reg_lambda": reg_lambda,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
    }


def make_experiment_run_list_on_disk(
    filepath: str,
    classifier_name_list: List[str],
    embedding_model_name_list: List[str],
    betas: Optional[List[float]] = None,
):
    if not os.path.exists(filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        beta_values = betas if betas is not None else [None]
        with open(filepath, "w") as file:
            for embeddings_model, classifier_name, beta_value in itertools.product(
                embedding_model_name_list, classifier_name_list, beta_values
            ):
                line = f"{embeddings_model},{classifier_name}"
                if beta_value is not None:
                    line += f",{beta_value}"
                file.write(line + "\n")

        print("experiment run list created")


def check_if_combination_in_file(
    embeddings_model, classifier_name, filepath, beta=None
):
    search_string = f"{embeddings_model},{classifier_name}"
    if beta is not None:
        search_string += f",{beta}"

    with open(filepath, "r") as file:
        if search_string in file.read():
            return True
    return False


def remove_combination_from_file(
    embeddings_model, classifier_name, filepath, beta=None
):
    search_string = f"{embeddings_model},{classifier_name}"
    if beta is not None:
        search_string += f",{beta}"

    with open(filepath, "r") as file:
        lines = file.readlines()

    with open(filepath, "w") as file:
        for line in lines:
            if line.strip("\n") != search_string:
                file.write(line)
    print(f"line '{search_string}' removed")
