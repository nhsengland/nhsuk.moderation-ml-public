# %%
import os
import sys

import azureml
import sklearn.metrics

from src.data_augmentation import data_augmentation
from src.data_ingestion import data_ingestion

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, root_path)

i = 0
for dataset_name in ["published_1897_17Oct23_dynamics_copied"]:
    for genmethod in [
        data_augmentation.generate_word_embedded_comments,
        data_augmentation.generate_sentence_shuffled_comments,
    ]:
        print(f"On run {i}")

        df = data_ingestion.DataRetrieverDatastore(dataset_name=dataset_name).dataset

        df = df[df["train"] == 1]

        new_df = data_augmentation.make_and_augmented_dataset(
            df=df, number_of_new_comments_per_comment=2, generation_function=genmethod
        )
        new_df["Is Complaint"] = 0
        new_df["train"] = 1
        new_df["val"] = 0
        new_df["test"] = 0
        new_name = genmethod.__name__ + dataset_name
        data_ingestion.register_dataframe(df=new_df, dataset_name=new_name)
        print(new_df.head())
        i += 1
