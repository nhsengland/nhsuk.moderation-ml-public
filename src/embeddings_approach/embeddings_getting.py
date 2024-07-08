import pandas
from src.azure_config import azure_config
from src.data_ingestion import data_ingestion
from src.embeddings_approach import embeddings_approach
from src.model_assessment import model_assessment

list_of_all_models = [
    "all-mpnet-base-v2",
    "BAAI/bge-small-en",
    "BAAI/bge-base-en",
    "thenlper/gte-base",
    "thenlper/gte-large",
    "BAAI/bge-large-en",
    "intfloat/e5-large-v2",
]


def get_embeddings_for_model_and_dataframe(
    df: pandas.DataFrame, name_of_column_to_embed: str, model_for_embeddings_name: str
) -> pandas.DataFrame:
    """Generate embeddings for a given dataset and model."""
    embeddings_column_name = embeddings_approach.make_embedding_column_name(
        name_of_column_to_embed=name_of_column_to_embed,
        model_for_embeddings_name=model_for_embeddings_name,
    )

    # This logic is just to take care of the weird prefix thing which e5 wants
    prefix = "query: " if model_for_embeddings_name == "intfloat/e5-large-v2" else ""

    model = embeddings_approach.get_embeddings_model(
        model_for_embeddings_name=model_for_embeddings_name
    )
    temp_series = prefix + df[name_of_column_to_embed]
    df[embeddings_column_name] = temp_series.apply(model.encode)

    return df


def register_embeddings_for_dataset(df, dataset_name: str):
    """Register the given dataset after embeddings generation."""
    embeddings_column_name = df.columns[
        -1
    ]  # assuming the last column is the generated embeddings
    if df[embeddings_column_name].isna().sum() > 0:
        print("Some embeddings are problems")
        breakpoint()

    data_ingestion.register_dataframe(df=df, name=dataset_name)
    print(f"Embeddings made and registered for {dataset_name}")


def get_and_register_embeddings_for_dataset(
    dataset_name: str,
    name_of_column_to_embed: str,
    model_for_embeddings_name: str,
    override: bool = False,
):
    df = data_ingestion.DataRetrieverDatastore(dataset_name=dataset_name).dataset
    embeddings_column_name = embeddings_approach.make_embedding_column_name(
        name_of_column_to_embed=name_of_column_to_embed,
        model_for_embeddings_name=model_for_embeddings_name,
    )

    if (embeddings_column_name in df.columns) and (not override):
        print("Embeddings already exist")
        return

    print(f"Getting embeddings for {dataset_name}")

    df = get_embeddings_for_model_and_dataframe(
        df, name_of_column_to_embed, model_for_embeddings_name
    )
    register_embeddings_for_dataset(df, dataset_name)


def _get_all_models_embeddings_for_dataset(
    dataset_name: str, name_of_column_to_embed: str, override: bool
):
    for model_name in list_of_all_models:
        get_and_register_embeddings_for_dataset(
            dataset_name=dataset_name,
            name_of_column_to_embed=name_of_column_to_embed,
            override=override,
            model_for_embeddings_name=model_name,
        )
    print(f"All embeddings gotten for {dataset_name}")


def get_or_check_all_embeddings(dataset_name: str, name_of_column_to_embed: str):
    _get_all_models_embeddings_for_dataset(
        dataset_name=dataset_name,
        name_of_column_to_embed=name_of_column_to_embed,
        override=False,
    )


def override_all_embeddings_with_new_ones(
    dataset_name: str, name_of_column_to_embed: str
):
    _get_all_models_embeddings_for_dataset(
        dataset_name=dataset_name,
        name_of_column_to_embed=name_of_column_to_embed,
        override=True,
    )
