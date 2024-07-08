"""
This is a bunch of tests to check that your data passes some sanity checks.
We've had a lot of problems with, in particular, embeddings drifting: you create
some embeddings for a dataset, and then you can't recreate those embeddings.
This is ofc a huge problem if there's drift between the way you built embeddings
to train the classifier, and th way you're getting embeddings to do inference
on. 

There are tests in here to check that your embeddings are functioning as you
would hope. There are also tests in here for more basic stuff like making sure
you don't have duplicates in your data, that your train test and val sets don't
overlap, etc. 

!!! To use this test suite, you *have* to first set things up in the
utils_for_tests/test_parameters.json
"""
import itertools
import os
import sys

import numpy
import pandas
import pytest

from src.data_ingestion import data_ingestion, data_utils
from src.embeddings_approach import embeddings_approach, embeddings_getting
from src.utils_for_tests import utils_for_tests

# This block exist to allow the driving script(s) to see the /src folder and us
# the modules within, without having to put all scripts in the root. 
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root_path)


#todo: Need testing to assure that sentence transformer model versions are logged / pegged


# Read in the test parameters (essentially which datasets, and which embeddings models, to include in the tests). 
# The workflow of these tests requires these test parameters to be changed before running these tests.
TEST_PARAMS = utils_for_tests.get_test_parameters()



# Test data creation. Just a usefully dummy df for the tests. 
@pytest.fixture
def create_test_data():
    data = {"A": [1, 2, 1, 2, 1, 2, 1, 2], "B": [2, 3, 4, 3, 5, 6, 7, 8]}
    df = pandas.DataFrame(data)
    return df


def test_balance_dataframe(create_test_data):
    """
    Tests the functionality of the `balance_dataframe` function with respect to uniform distribution of class values.

    Parameters:
        create_test_data (pd.DataFrame): A DataFrame used for testing, generated through a fixture or setup function.

    Checks:
        1. The balanced dataframe has an equal number of each class in column "A".
        2. The count of each class in the balanced dataframe equals the count of the least frequent class in the original dataframe.
    """
    df = create_test_data
    balanced_df = data_ingestion.balance_dataframe(df, "A")

    assert len(set(balanced_df["A"].value_counts())) == 1

    assert balanced_df["A"].value_counts().iloc[0] == min(df["A"].value_counts())


def test_balance_dataframe_retain_unique_values(create_test_data):
    """
    Tests the functionality of the `balance_dataframe` function with respect to retaining unique class values.

    Parameters:
        create_test_data (pd.DataFrame): A DataFrame used for testing, generated through a fixture or setup function.

    Check:
        The balanced dataframe retains all unique class values present in the original dataframe for column "A".
    """
    df = create_test_data
    balanced_df = data_ingestion.balance_dataframe(df, "A")

    assert set(balanced_df["A"].unique()) == set(df["A"].unique())


# Check iteratively that for the test/train/val split has been performed correctly on each dataset
@pytest.mark.parametrize("dataset_name", TEST_PARAMS["dataset_names"])
def test_train_test_val_splits(dataset_name: str):
    df = data_ingestion.DataRetrieverDatastore(dataset_name=dataset_name).dataset
    assert data_utils.check_train_test_val_splits(dataframe=df) == 0

@pytest.mark.parametrize(
    "dataset_name, embedding_model_name",
    itertools.product(
        TEST_PARAMS["dataset_names"], TEST_PARAMS["embeddings_model_names"]
    ),
)
def test_embedding_values_match_new_ones(embedding_model_name: str, dataset_name: str):
    """
    Tests whether the stored embeddings align with freshly produced embeddings.

    Parameters:
        embedding_model_name (str): The specific embedding model's name being tested.
        dataset_name (str): The dataset's name being evaluated.

    For each combination of dataset and embedding model, a sample of rows (specified by TEST_PARAMS["number_of_rows_to_sample_when_comparing_embeddings"]) is taken.
    For each sample row, the function checks if the previously stored embedding is close to the one produced now.

    Raises:
        AssertionError: If any of the stored embeddings do not closely match their freshly produced counterparts, highlighting a possible inconsistency between old and new embeddings.
    """
    sub_sample_df_old = data_ingestion.DataRetrieverDatastore(
        dataset_name=dataset_name
    ).dataset.sample(
        n=TEST_PARAMS["number_of_rows_to_sample_when_comparing_embeddings"]
    )
    print("sub_sample_df_old generated")
    sub_sample_df_new = embeddings_getting.get_embeddings_for_model_and_dataframe(
        df=sub_sample_df_old.copy(),
        name_of_column_to_embed=TEST_PARAMS["name_of_column_to_embed"],
        model_for_embeddings_name=embedding_model_name,
    )
    print("sub_sample_df_new generated")
    column_name = embeddings_approach.make_embedding_column_name(
        name_of_column_to_embed=TEST_PARAMS["name_of_column_to_embed"],
        model_for_embeddings_name=embedding_model_name,
    )
    print("column generated generated")
    matching = []

    for index, row_old in sub_sample_df_old.iterrows():
        old_embedding = row_old[column_name]

        # It's best to do it like this because we can't guarantuee the dfs are like, in the same order
        row_new = sub_sample_df_new.loc[index]
        new_embedding = row_new[column_name]

        # Check if the two embeddings are close to each other
        matching.append(numpy.allclose(old_embedding, new_embedding))

    assert all(
        matching
    ), f"Stored embeddings don't match the ones being produced for {dataset_name} with {embedding_model_name}"


@pytest.mark.parametrize("dataset_name", TEST_PARAMS["dataset_names"])
def test_duplicates(dataset_name: str):
    """
    Tests for the presence of duplicates in a specific dataset based on a specified column.

    Retrieves the dataset specified by the given dataset name and checks for duplicates
    based on the column specified in TEST_PARAMS["name_of_column_to_embed"].

    Parameters:
        dataset_name (str): The name of the dataset being tested for duplicates.

    Raises:
        AssertionError: If duplicates are found in the dataset for the specified column,
                        indicating potential redundancy or data inconsistencies.
    """
    df = data_ingestion.DataRetrieverDatastore(dataset_name=dataset_name).dataset

    duplicates_text = data_utils.check_duplicates(
        df, TEST_PARAMS["name_of_column_to_embed"]
    )

    assert (
        duplicates_text.empty
    ), f"Duplicates found in {TEST_PARAMS['name_of_column_to_embed']} for {dataset_name}!"


def test_duplication_for_combined_set_of_datasets():
    """
    Tests for the presence of duplicates when combining multiple datasets based on a specific column.

    Retrieves each dataset specified in TEST_PARAMS["dataset_names"], concatenates them into a single dataframe,
    and then checks for duplicates based on the column specified in TEST_PARAMS["name_of_column_to_embed"].

    Raises:
        AssertionError: If any duplicates are found in the combined dataframe for the specified column, indicating
                        potential redundancy or overlap among datasets.
    """
    dfs = [
        data_ingestion.DataRetrieverDatastore(name).dataset
        for name in TEST_PARAMS["dataset_names"]
    ]
    total_df = pandas.concat(dfs)

    duplicates_text = data_utils.check_duplicates(
        total_df, TEST_PARAMS["name_of_column_to_embed"]
    )
    assert (
        duplicates_text.empty
    ), f"Duplicates found in {TEST_PARAMS['name_of_column_to_embed']} for combined dataframes"
