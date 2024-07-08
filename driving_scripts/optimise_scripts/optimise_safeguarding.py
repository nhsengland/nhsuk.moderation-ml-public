## This is the most recent iteration of the optimise safeguarding script, following a more elegant refactoring of code to functionalise more parts. Simply change
## the 'multiclass' argument (and change log_all_multiclass_metrics to log_all_metrics towards the end of the script) to switch between binary and multiclass safeguarding models.

import os
import sys

import numpy
import sklearn
import sklearn.metrics
import subprocess
import pytest

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, root_path)

from src.azure_config import azure_config
from src.embeddings_approach import embeddings_approach
from src.utils_for_tests import utils_for_tests

embeddings_models = [
    # "all-mpnet-base-v2",
    # "BAAI/bge-small-en",
    # "BAAI/bge-base-en",
    # "thenlper/gte-base",
    # "thenlper/gte-large",
    "BAAI/bge-large-en",
    "intfloat/e5-large-v2",
]

classifier_name_list = [
    # 'SVM',
    "logistic_regression",
    # 'random_forest'
]

utils_for_tests.define_list_of_datasets_for_tests([
    "safeguarding_472_Sept22_DanFinola",
    "safeguarding_184_Nov22_DanFinola",
    "safeguarding_108_Nov22_copycatNHS",
    "safeguarding_113_Nov22_copycat2",
    "safeguarding_200_Nov22_copycatTwitter",
    "safeguarding_low_gen_gpt4_500_24Aug23_DanG",
    "safeguarding_gen_gpt4_600_24Aug23_DanG",
    "published_10k_DanFinola_subset",
])

utils_for_tests.define_list_of_models_for_tests(embeddings_models)


def run_tests():
    test_data_file_path = os.path.join(root_path, 'tests/test_data.py')
    test_exit_code = pytest.main([test_data_file_path])

    if test_exit_code == 0:
        print("All tests passed, continuing...")
    else:
        print("Some tests failed, exiting...")
        sys.exit(test_exit_code) 

run_tests()


EXPERIMENT_NAME = "dg_safeguarding_multi_400"
EXPERIMENT_NAME_LIST_PATH = os.path.join("experiment_lists/", EXPERIMENT_NAME + ".txt")

embeddings_approach.make_experiment_run_list_on_disk(
    filepath=EXPERIMENT_NAME_LIST_PATH,
    classifier_name_list=classifier_name_list,
    embedding_model_name_list=embeddings_models,
)

NUMBER_OF_RUNS = len(embeddings_models) * len(classifier_name_list)
i = 0
for embeddings_model in embeddings_models:
    for classifier_name in classifier_name_list:
        i += 1
        print("-" * 50)
        print(f"\n\n On Run {i} out of {NUMBER_OF_RUNS}")

        if not embeddings_approach.check_if_combination_in_file(
            embeddings_model=embeddings_model,
            classifier_name=classifier_name,
            filepath=EXPERIMENT_NAME_LIST_PATH,
        ):
            continue

        if embeddings_model == "intfloat/e5-large-v2":
            pre_prompt = "query: "
        else:
            pre_prompt = ""

        run = azure_config.start_run(expeiment_name=EXPERIMENT_NAME)

        this_one = embeddings_approach.EmbeddingsApproach(
            classifier_class_name=classifier_name,
            model_for_embeddings_name=embeddings_model,
            positive_label_dataset_name_list=[
                "safeguarding_472_Sept22_DanFinola",
                "safeguarding_184_Nov22_DanFinola",
                "safeguarding_108_Nov22_copycatNHS",
                "safeguarding_113_Nov22_copycat2",
                "safeguarding_200_Nov22_copycatTwitter",
            ],
            augmented_dataset_name_list=[
                "safeguarding_low_gen_gpt4_500_24Aug23_DanG",
                "safeguarding_gen_gpt4_600_24Aug23_DanG",
            ],
            name_of_column_to_embed="Comment Text",
            name_of_y_column="label_multi",
            max_evals=400,
            negative_label_dataset_name_list=[
                # "published_80k_DanFinola",
                "published_10k_DanFinola_subset",
                # "published_3k_dg_devset"
            ],
            prefix=pre_prompt,
            balance_test=False,
            balance_val=False,
            parallelise=True,
            parallel_limit=30,
            multiclass=True,
        )
        this_one.find_optimised_classifier()
        this_one.get_assessor_for_optimised_model()
        this_one.register_optimal_model()
        this_one.log_all_attributes(run=run)
        this_one.assessor.log_all_multiclass_metrics(
            run=run, display_labels=["no risk", "low risk", "high risk"]
        )
        run.complete()

        embeddings_approach.remove_combination_from_file(
            embeddings_model=embeddings_model,
            classifier_name=classifier_name,
            filepath=EXPERIMENT_NAME_LIST_PATH,
        )
        azure_config.clear_all_spark_files()
        print("-" * 50)
