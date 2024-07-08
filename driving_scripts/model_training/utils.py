#! NOTE: THIS CODE IS DEFUNCT AND KEPT FOR ARCHIVING / REFERENCE PURPOSES ONLY.

print(
    "WARNING: YOU HAVE IMPORTED THE WRONG UTILS. THIS CODE IS DEFUNCT AND KEPT FOR ARCHIVING / REFERENCE PURPOSES ONLY. "
)


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import sentence_transformers
import sklearn
import tqdm
from azureml.core import Dataset, Workspace
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)

subscription_id = "REDACTED" 
resource_group = "REDACTED" 
workspace_name = "REDACTED" 

workspace = Workspace(subscription_id, resource_group, workspace_name)


gibson_model_sentence_transformer = sentence_transformers.SentenceTransformer(
    "all-MiniLM-L6-v2"
)  # FASTER
model_sentence_transformer = sentence_transformers.SentenceTransformer(
    "all-mpnet-base-v2"
)  # BETTER


CONFIDENCE = 0.95

z_value_for_ci = scipy.stats.norm.ppf((1 + CONFIDENCE) / 2.0)


def get_confidence_for_accuracy(model, df_validate: pd.DataFrame):
    X = np.array(list(df_validate["combined_embedding"]))
    y = df_validate["Is_Complaint"]
    accuracy = model.score(X, y)
    confidence_interval_length = z_value_for_ci * np.sqrt(
        (accuracy * (1 - accuracy) / len(df_validate))
    )
    ci = (
        max(accuracy - confidence_interval_length, 0),
        min(accuracy + confidence_interval_length, 1),
    )
    return ci


def fit_and_score_model(model, X_train, X_test, Y_train, Y_test):
    model = model.fit(X_train, Y_train)
    y_predicted = model.predict(list(X_test))
    print(model)
    print(sklearn.metrics.classification_report(Y_test, y_predicted))


def validate_model(model, df_validate: pd.DataFrame):
    y_pred = model.predict(np.array(list(df_validate["combined_embedding"])))
    # print(y_pred)
    conf_mat = confusion_matrix(y_true=df_validate["Is_Complaint"], y_pred=y_pred)
    conf_disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat)
    conf_disp.plot()
    plt.show()
    print(classification_report(y_true=df_validate["Is_Complaint"], y_pred=y_pred))
    confidence_interval = get_confidence_for_accuracy(
        model=model, df_validate=df_validate
    )
    print(
        f"The {CONFIDENCE} confidence interval for model accuracy is {confidence_interval}"
    )


def fit_and_validate(
    df_train: pd.DataFrame, df_validate: pd.DataFrame, clf=sklearn.svm.SVC()
):
    clf = clf.fit(
        np.array(list(df_train["combined_embedding"])), df_train["Is_Complaint"]
    )
    validate_model(model=clf, df_validate=df_validate)
    return clf


def encode_comment_text_for_row(row):
    return np.array(model_sentence_transformer.encode(row["Comment_Text"]))


def encode_comment_title_for_row(row):
    return np.array(model_sentence_transformer.encode(row["Comment_Title"]))


def create_embeddings(df: pd.DataFrame):
    df["embeddings_comment"] = df.apply(encode_comment_text_for_row, axis=1)
    df["embeddings_title"] = df.apply(encode_comment_title_for_row, axis=1)

    def get_combined_embedding(row):
        return np.concatenate((row["embeddings_comment"], row["embeddings_title"]))

    df["combined_embedding"] = df.apply(get_combined_embedding, axis=1)
    return df


def make_training_and_validation_sets(
    published: pd.DataFrame,
    complaints: pd.DataFrame,
    balance: bool,
    validation_size: int,
):
    """
    We want a validation (or holdout) set to test against in the end. This is not a duplication of the functionality of the test set; since we will use the test set to optimise the hyperparameters.
    """

    validation_complaints = complaints.sample(n=validation_size // 2)
    validation_published = published.sample(n=validation_size // 2)
    validation = pd.concat([validation_complaints, validation_published]).sample(frac=1)
    test_and_train_complaints = complaints.drop(validation_complaints.index)
    test_and_train_published = published.drop(validation_published.index)

    if balance:
        test_and_train = pd.concat(
            [
                test_and_train_complaints,
                test_and_train_published.sample(len(test_and_train_complaints)),
            ]
        ).sample(frac=1)
    else:
        test_and_train = pd.concat(
            [test_and_train_complaints, test_and_train_published]
        ).sample(frac=1)

    return test_and_train, validation


def get_and_clean_complaints_v1(workspace=workspace):
    dataset = Dataset.get_by_name(
        workspace=workspace, name="complaints_complete_cleansed_v1"
    )
    df = dataset.to_pandas_dataframe()
    df.rename(
        columns={
            "Org Type (Org Name) (Organisation)": "Org_Type",
            "Overall Rating": "Rating",
            "Comment Title": "Comment_Title",
            "Comment Text": "Comment_Text",
            "Rejection Reason": "Rejection_Reason",
        },
        inplace=True,
    )
    # df = df[['Comment_Title', 'Rejection_Reason', 'Org_Type', 'Rating', 'Comment_Text']]
    df = df[["Comment_Title", "Comment_Text"]]
    df["Is_Complaint"] = 1
    df.dropna(inplace=True)
    return df


def get_original_complaints_data(workspace=workspace):
    dataset = Dataset.get_by_name(workspace=workspace, name="complaints_data_original")
    df = dataset.to_pandas_dataframe()
    df = df[["Feature", "Complaints"]]
    df.rename(
        columns={"Feature": "Comment_Text", "Complaints": "Is_Complaint"}, inplace=True
    )
    return df


def get_simple_combined_data(workspace=workspace):
    df_1 = get_original_complaints_data()[["Is_Complaint", "Comment_Text"]]
    df_2 = get_and_clean_complaints_v1()[["Is_Complaint", "Comment_Text"]]
    df_comb = pd.concat([df_1, df_2], join="inner")
    return df_comb.sample(frac=1)


def get_and_clean_published_4000_data(workspace=workspace):
    dataset = Dataset.get_by_name(workspace=workspace, name="published_reviews")
    df = dataset.to_pandas_dataframe()
    df.rename(
        columns={"Comment Title": "Comment_Title", "Comment Text": "Comment_Text"},
        inplace=True,
    )
    df = df[["Comment_Title", "Comment_Text"]]
    df.dropna(inplace=True)
    df["Is_Complaint"] = 0
    return df.sample(frac=1)
