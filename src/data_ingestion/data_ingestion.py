"""
This module is to handle all the data I/O from azure. Much of this was written
with a level of abstraction in mind which never materialised - we'd hoped to use
different data sources, and wanted to be robust against that. In the end we used
only the Azure datastore. This means that the module could benefit from some
de-abstracting at some point. 
Name is also a bit of a misnomer because it also has data writing functions.
"""
#todo De-abstract this module to make it reflect the fact we only use Azure datastore. 
import os
import pickle
import pandas
import re
import numpy
from typing import List

from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import (
    Data,
)
from azureml.core import Dataset

from ..azure_config import azure_config


class DatasetGetter:
    """
    This class serves as something like an abstract base class. We have several
    methods for loading in data - from the azure datastore, from the SQL server,
    or from CSV files. The idea of this class is that we ought to be able to
    unify these into a single 'data getter', and that if we change from one of
    these methods to another we simple changer the retriever class which this
    calls on.
    """

    def __init__(self, retriever, dataset_name: str):
        self.retriever = retriever(dataset_name)
        self.dataset = self.retriever.dataset
        self.dataset_object = self.retriever._dataset_object


class DataRetrieverSQLServer:
    """
    This class masks some complexity. The original data being brought out here
    is stored at the following URL REDACTED

    The Azure Machine Learning Studio has a DataStore object which queries the
    SQL server version and ingests it. The function called below loads this
    DataStore object, and preforms quite a lot of processing on it. !NOTE 1:
    Calling this class only loads what's already in DataStore. To refresh from
    the SQL Server, go to the DataStore interface and tell it to update. URL
    REDACTED 
    !NOTE 2: There is a *significant* problem with versioning between
    the SQL Server data and the Dynamics data. Until this is resolved it is
    advised *NOT TO USE THIS CLASS FOR MODEL BUILDING* Once this problem is
    resolved - the idea is that this class ought to be the gold-standard and
    most commonly used form of data ingestion.
    """

    def __init__(self, dataset_name):
        self._dataset_object = self._get_dataset(dataset_name)
        self.dataset = self._dataset_object.to_pandas_dataframe()



def deserialize_numpy_columns(df: pandas.DataFrame)-> pandas.DataFrame:
    """
    This function, and its twin below, are very important. When we're working
    with NLP models and using embeddings, there are always generated as numpy
    arrays. Unfortunately, the Azure datastore doesn't allow you to store this
    data type. So in order to store (and load) this data to (from) the
    datastore, we need to serialise it first. 
    """
    for col in df.columns:
        if (
            df[col].apply(lambda x: isinstance(x, bytes)).any()
        ):  # Check if any cell in column is bytes (serialized data)
            df[col] = df[col].apply(
                lambda x: pickle.loads(x) if isinstance(x, bytes) else x
            )
    return df


def serialize_numpy_columns(df: pandas.DataFrame)-> pandas.DataFrame:
    """
    This function, and its twin above, are very important. When we're working
    with NLP models and using embeddings, there are always generated as numpy
    arrays. Unfortunately, the Azure datastore doesn't allow you to store this
    data type. So in order to store (and load) this data to (from) the
    datastore, we need to serialise it first. 
    """
    for col in df.columns:
        if (
            df[col].apply(lambda x: isinstance(x, numpy.ndarray)).any()
        ):  # Check if any cell in column is a numpy array
            df[col] = df[col].apply(
                lambda x: pickle.dumps(x) if isinstance(x, numpy.ndarray) else x
            )
    return df


class DataRetrieverDatastore:
    """
    This is a wrapper around the azureml Dataset functionality. Just loads the
    datastore but with our credentials. Handles the deserialising of data too. 
    """

    def __init__(self, dataset_name: str):
        self.workspace = azure_config.get_workspace()
        self._dataset_object = Dataset.get_by_name(
            workspace=self.workspace, name=dataset_name
        )
        df = self._dataset_object.to_pandas_dataframe()
        self.dataset = deserialize_numpy_columns(df)

    def print_description(self):
        print(self._dataset_object["description"])


def get_all_dataset_names_in_registry()-> List:
    ws = azure_config.get_workspace()
    names = ws.datasets
    return names


def add_train_test_val_labels_to_df(dataset_name: str, split_proportions=None)-> None:
    """
    This function modifies a dataset by adding 'train', 'test', and 'validation' labels to it based on
    provided or default split proportions. It then re-registers the modified dataframe.

    The split proportions are either provided via the 'split_proportions' parameter,
    or default values are fetched from 'azure_config'. The proportions are used to assign
    labels randomly to each row of the dataframe, creating new columns 'train', 'test',
    and 'val' which are binary indicators of the respective splits.

    If the split proportions do not sum up to 1, a ValueError is raised. After adding the labels,
    the modified dataframe is re-registered with the 'dataset_name'.

    """
    df = DataRetrieverDatastore(dataset_name=dataset_name).dataset
    if split_proportions is None:
        train_frac = azure_config.TRAIN_FRAC
        test_frac = azure_config.TEST_FRAC
        val_frac = azure_config.VAL_FRAC
    else:
        train_frac = split_proportions["train_frac"]
        test_frac = split_proportions["test_frac"]
        val_frac = split_proportions["val_frac"]

    total_frac = train_frac + test_frac + val_frac
    if not numpy.isclose(total_frac, 1):
        raise ValueError(
            f"The fractions do not add up to 1. The total was {total_frac}."
        )

    splits = ["train", "test", "val"]
    fracs = [train_frac, test_frac, val_frac]

    df["split"] = numpy.random.choice(splits, size=len(df), p=fracs)

    for split in splits:
        df[split] = (df["split"] == split).astype(int)

    # Drop the 'split' column
    df = df.drop(columns="split")

    register_dataframe(df=df, dataset_name=dataset_name)
    print("'train' 'test', 'val' columns created, and dataframe re-registered. ")


def balance_dataframe(df: pandas.DataFrame, column_to_balance: str):
    """
    Balance the DataFrame based on the provided column.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame that needs to be balanced.

    column : str
        The column name based on which the DataFrame should be balanced.

    Returns:
    --------
    balanced_df : pd.DataFrame
        Balanced DataFrame.
    """

    unique_values = df[column_to_balance].unique()
    balanced_dfs = []
    min_group_size = min(df[column_to_balance].value_counts())

    for value in unique_values:
        sub_df = df[df[column_to_balance] == value]
        sampled_df = sub_df.sample(min_group_size)
        balanced_dfs.append(sampled_df)

    # Concatenate all the balanced dataframes
    balanced_df = pandas.concat(balanced_dfs, ignore_index=True)

    return balanced_df.sample(frac=1)


def add_categorical_column_to_dataset(dataset_name: str, column: str, value: int):
    """
    Use this to add a label column to an entire dataset in the registry. For
    example, adding a column 'Complaint' with a value of 1. 
    """
    df = DataRetrieverDatastore(dataset_name=dataset_name).dataset
    df[column] = value
    register_dataframe(df=df, dataset_name=dataset_name)


def register_dataframe(df: pandas.DataFrame, dataset_name: str)-> None:
    """
    Puts a dataframe on the azure datastore. Overwrites if it's already there.
    Handles the serialisation of the embeddings columns. 
    """
    df = serialize_numpy_columns(df)
    local_file = f"{dataset_name}.pkl"
    df.to_pickle(local_file)
    ws = azure_config.get_workspace()
    datastore = ws.get_default_datastore()

    target_path = f"{dataset_name}/"

    Dataset.Tabular.register_pandas_dataframe(
        name=dataset_name, dataframe=df, target=(datastore, target_path)
    )
    os.remove(local_file)


