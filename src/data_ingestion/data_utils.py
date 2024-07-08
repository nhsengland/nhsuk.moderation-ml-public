"""
This module contains various odds and sods for manipulating data. Most of these
utils are used by the data integrity test suite. 
"""
import pandas
from azureml.core.dataset import Dataset

from src.azure_config import azure_config
from src.data_ingestion import data_ingestion



def clean_up_registered_dataset(dataset_name: str)-> None:
    """
    This performs some basic cleaning tasks on your registered dataset and
    re-registers it. This is not a substitute for using the data quality test
    suite in './tests'. This removes duplicates and empties. 
    """
    df = data_ingestion.DataRetrieverDatastore(dataset_name).dataset
    df = df.drop_duplicates(subset=["Comment ID"])
    df = df.dropna(subset=["Comment Text"])
    data_ingestion.register_dataframe(df=df, dataset_name=dataset_name)
    print("done")

def get_latest_dataset_version(dataset_name: str) -> int:
    """
    Get the latest version number of a dataset in Azure ML workspace.
    """
    ws = azure_config.get_workspace()
    dataset = Dataset.get_by_name(ws, dataset_name, version="latest")

    # Return the version number of the dataset
    return dataset.version


def check_duplicates(dataframe: pandas.DataFrame, column_name: str)-> pandas.DataFrame:
    """
    Gets a df of all the duplicates in a registered dataframe. 
    """
    duplicates = dataframe[dataframe.duplicated(subset=[column_name], keep=False)]
    return duplicates


def remove_duplicates_from_df(dataset_name: str, column_name: str)-> None:
    """
    Removes all the duplicates from a dataset and re-registers it
    """
    df = data_ingestion.DataRetrieverDatastore(dataset_name=dataset_name).dataset
    dups = check_duplicates(dataframe=df, column_name=column_name)
    if len(dups) > 0:
        print(f"Found {len(dups)} duplicates in {dataset_name}")
        df = df.drop_duplicates(subset=column_name, keep="first")
    data_ingestion.register_dataframe(df=df, name=dataset_name)


def check_train_test_val_splits(dataframe: pandas.DataFrame)-> int:
    """
    Check that each row has exactly one '1' across 'train', 'test', and 'val' columns.

    Parameters:
        dataframe (pd.DataFrame): The input DataFrame.

    Returns:
        int: Number of rows where the sum across 'train', 'test', and 'val' columns is not 1.
    """
    # List of columns to check
    split_columns = ["train", "test", "val"]

    # Calculate the sum across the specified columns for each row
    split_sums = dataframe[split_columns].sum(axis=1)

    # Count and return the number of rows where the sum is not 1
    invalid_rows = sum(split_sums != 1)
    return invalid_rows


def get_all_dataset_objects()-> dict:
    # Get all datasets
    ws = azure_config.get_workspace()
    all_datasets = Dataset.get_all(ws)
    return all_datasets