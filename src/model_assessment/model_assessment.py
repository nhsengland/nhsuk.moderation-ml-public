import json
import os
import ssl
import urllib.request
from abc import ABC, abstractmethod
from typing import Callable, Optional

import azureml
import joblib
import matplotlib.pyplot as plt
import numpy
import pandas
import sklearn.metrics
from azureml.core import Run

from src.data_ingestion import data_ingestion
from src.embeddings_approach import embeddings_approach
from src.model_getting import model_getting

from ..azure_config import azure_config


class PredictionsAssessor:
    """
    This is a class for handling all the evaluation of our model outputs. Pass it a panads dataframe containing your ground truths and the predictions of the model, and you can easily see the performance of the model in various ways.
    """

    def __init__(
        self,
        test_data: pandas.DataFrame,
        y_column_name: str,  # This is the label column; like 'Is Complaint' etc. Must be 0 or 1
        predictions_column_name: str,  # What ever you've called the predictions column, labelling the predictions from your model
        scorer: Optional[sklearn.metrics.make_scorer] = None,
    ):
        self.test_data = test_data
        self.y_column_name = y_column_name
        if predictions_column_name in test_data.columns:
            self.predictions = test_data[predictions_column_name]
        else:
            print("No Predictions")
            self.predictions = None
        if scorer:
            self.scorer = scorer

    def get_confusion_matrix(self):
        return sklearn.metrics.confusion_matrix(
            self.test_data[self.y_column_name], self.predictions
        )

    def get_f1_score(self):
        if (
            self.predictions is None
            or self.predictions.empty
            or self.predictions.isnull().all()
        ):
            print("Predictions are not available. Cannot calculate F1 score.")
            return None
        return sklearn.metrics.f1_score(
            self.test_data[self.y_column_name], self.predictions
        )

    def get_fbeta_score(self, beta):
        return sklearn.metrics.fbeta_score(
            y_pred=self.predictions,
            y_true=self.test_data[self.y_column_name],
            beta=beta,
        )

    def get_precision(self):
        return sklearn.metrics.precision_score(
            y_true=self.test_data[self.y_column_name], y_pred=self.predictions
        )

    def get_custom_score(self):
        if self.scorer is None:
            return None
        else:
            return self.scorer._score_func(
                self.test_data[self.y_column_name],
                self.predictions,
                **self.scorer._kwargs,
            )

    def get_recall(self):
        return sklearn.metrics.recall_score(
            y_true=self.test_data[self.y_column_name], y_pred=self.predictions
        )

    def get_multiclass_f1_score(self, average="macro"):
        if (
            self.predictions is None
            or self.predictions.empty
            or self.predictions.isnull().all()
        ):
            print("Predictions are not available. Cannot calculate F1 score.")
            return None
        return sklearn.metrics.f1_score(
            self.test_data[self.y_column_name], self.predictions, average=average
        )

    def get_multiclass_recall_score(self, average="macro"):
        if (
            self.predictions is None
            or self.predictions.empty
            or self.predictions.isnull().all()
        ):
            print("Predictions are not available. Cannot calculate recall score.")
            return None
        return sklearn.metrics.recall_score(
            self.test_data[self.y_column_name], self.predictions, average=average
        )

    def get_and_display_confusion_matrix(self, display_labels=["No Flag", "Flag"]):
        mat = self.get_confusion_matrix()
        disp = sklearn.metrics.ConfusionMatrixDisplay(
            mat, display_labels=display_labels
        )
        disp.plot()
        plt.show()

    def log_all_metrics(self, display_labels, run: Run):
        confusion_matrix = self.get_confusion_matrix()
        TN, FP, FN, TP = confusion_matrix.ravel()

        to_log = [
            ("True Positives", float(TP)),
            ("False Positives", float(FP)),
            ("True Negatives", float(TN)),
            ("False Negatives", float(FN)),
            ("precision", self.get_precision()),
            ("f1", self.get_f1_score()),
            ("recall", self.get_recall()),
        ]
        if self.scorer is not None:
            score_name = self.scorer._score_func.__name__
            score = self.get_custom_score()
            to_log += [(score_name, score)]
            to_log += list(self.scorer._kwargs.items())

        for attribute, value in to_log:
            if isinstance(
                value, (int, float)
            ):  # These values are expected to be numbers
                run.log(name=attribute, value=value)
            else:
                print(
                    f"Warning: Unknown type for attribute {attribute}. Value not logged."
                )

        disp = sklearn.metrics.ConfusionMatrixDisplay(
            confusion_matrix, display_labels=display_labels
        )
        disp.plot(values_format="")
        run.log_image("confusion_matrix_plot", plot=disp.figure_)

    def log_all_multiclass_metrics(self, display_labels, run: Run):
        confusion_matrix = self.get_confusion_matrix()

        to_log = [
            ("f1_macro", self.get_multiclass_f1_score(average="macro")),
            ("f1_micro", self.get_multiclass_f1_score(average="micro")),
            ("recall_macro", self.get_multiclass_recall_score(average="macro")),
        ]

        for attribute, value in to_log:
            if isinstance(
                value, (int, float)
            ):  # These values are expected to be numbers
                run.log(name=attribute, value=float(value))
            else:
                print(
                    f"Warning: Unknown type for attribute {attribute}. Value not logged."
                )

        disp = sklearn.metrics.ConfusionMatrixDisplay(
            confusion_matrix, display_labels=display_labels
        )
        disp.plot()
        run.log_image("confusion_matrix_plot", plot=disp.figure_)


def get_model_prediction_from_text_function_from_run_id(run_id: str) -> Callable:
    """
    This function returns a function. Supply it with the ID of a run, and you'll get a function which supplies the predictions for text, as per the model from that run.
    Example:
    predictor_A = spin_up_model_from_run(1231231231232)
    comment_1 = "I went to the GP"
    prediction_1 = predictor_A(comment_1)
    #... = [0], or [1]
    """
    run = model_getting.get_run_by_run_id(run_id=run_id)
    model = model_getting.get_model_from_run(run=run)
    run_metrics = run.get_metrics()
    embedder = embeddings_approach.get_embeddings_model(
        model_for_embeddings_name=run_metrics["model_for_embeddings_name"]
    )

    def get_pred_from_this_model(input_text: str):
        embedding = embedder.encode(input_text)
        return model.predict(numpy.array([embedding]))

    return get_pred_from_this_model


def assess_model_from_run(run_id: str):
    run = model_getting.get_run_by_run_id(run_id=run_id)
    model = model_getting.get_model_from_run(run=run)
    run_metrics = run.get_metrics()
    positive_label_data_names = run_metrics["positive_label_dataset_name_list"].split(
        ";"
    )
    negative_label_data_names = run_metrics["negative_label_dataset_name_list"].split(
        ";"
    )

    augmented_datasets_names = run_metrics["augmented_dataset_name_list"].split(";")
    setnames = [
        s.strip()
        for s in positive_label_data_names
        + negative_label_data_names
        + augmented_datasets_names
    ]
    datasets = []
    for setname in setnames:
        datasets.append(
            data_ingestion.DataRetrieverDatastore(dataset_name=setname).dataset
        )

    data = pandas.concat(datasets)
    X_column_name = embeddings_approach.make_embedding_column_name(
        name_of_column_to_embed=run_metrics["name_of_column_to_embed"],
        model_for_embeddings_name=run_metrics["model_for_embeddings_name"],
    )
    y_column_name = run_metrics["name_of_y_column"]
    val_set = data[data["val"] == 1]
    test_set = data[data["test"] == 1]
    val_set["preds"] = model.predict(numpy.array(list(val_set[X_column_name])))

    if run_metrics["balance_val"]:
        val_set = data_ingestion.balance_dataframe(
            df=val_set, column_to_balance=run_metrics["name_of_y_column"]
        )

    assessor = PredictionsAssessor(
        test_data=val_set, y_column_name=y_column_name, predictions_column_name="preds"
    )
    return assessor


def get_predictions_for_run_against_new_df(
    run_id: str, new_dataset_name: str, name_of_column_to_inference_from: str
):

    df = data_ingestion.DataRetrieverDatastore(dataset_name=new_dataset_name).dataset
    predictor = get_model_prediction_from_text_function_from_run_id(run_id=run_id)
    df["predictions"] = df[name_of_column_to_inference_from].apply(predictor)
    return df


def get_predictions_for_endpoint_against_new_df(
    df: pandas.DataFrame,
    access_details: azure_config.EndpointAccessDetails,
    name_of_column_to_inference_from,
):
    def get_comment_response(comment):
        return model_getting.get_endpoint_response(
            access_details=access_details, query_content=comment
        )

    df["predictions"] = df[name_of_column_to_inference_from].apply(get_comment_response)

    return df
