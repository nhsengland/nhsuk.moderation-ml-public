import os
import subprocess
import sys

import numpy
import pytest
import sklearn
import sklearn.metrics
import subprocess
import pytest

from src.azure_config import azure_config
from src.embeddings_approach import embeddings_approach
from src.utils_for_tests import utils_for_tests


root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, root_path)


embeddings_models = [
    "all-mpnet-base-v2",
    # "BAAI/bge-small-en",
    "BAAI/bge-base-en",
    # "thenlper/gte-base",
    # "thenlper/gte-large",
    # "BAAI/bge-large-en",
    # "intfloat/e5-large-v2",
]

utils_for_tests.define_list_of_datasets_for_tests(
    [
        "complaints_complete_cleansed_v1_reboot",
        "complaints_dg_DanFinola_dec_reboot",
        "published_1897_17Oct23_dynamics_copied",
        "augmented_combined_methods_published_1897_17Oct23_dynamics_copied",
        "dg_aug_shuffled_and_embed_complaints_dec_and_v1_reboots",
        "published_10k_DanFinola_subset",
        # "published_80k_DanFinola"
    ]
)

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


EXPERIMENT_NAME = "dg_complaints_with10k_cleaned"
EXPERIMENT_NAME_LIST_PATH = os.path.join("experiment_lists/", EXPERIMENT_NAME + ".txt")

classifier_name_list = [
    "SVM",
    # "logistic_regression",
    # "xgboost"
    # 'random_forest'
]

embeddings_approach.make_experiment_run_list_on_disk(
    filepath=EXPERIMENT_NAME_LIST_PATH,
    classifier_name_list=classifier_name_list,
    embedding_model_name_list=embeddings_models,
)
embeddings_approach.make_experiment_run_list_on_disk(
    filepath=EXPERIMENT_NAME_LIST_PATH,
    classifier_name_list=classifier_name_list,
    embedding_model_name_list=embeddings_models,
)

# Calculating the total number of runs for progress tracking
NUMBER_OF_RUNS = len(embeddings_models) * len(classifier_name_list)
i = 0

scorer = sklearn.metrics.make_scorer(sklearn.metrics.fbeta_score, beta=0.75)

parallel_values = [0, 60, 50, 40, 30, 20]
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
        retries = len(parallel_values)
        parallelise_values_index = 0

        success = False
        this_one = embeddings_approach.EmbeddingsApproach(
            classifier_class_name=classifier_name,
            model_for_embeddings_name=embeddings_model,
            positive_label_dataset_name_list=[
                "complaints_complete_cleansed_v1_reboot",
                "complaints_dg_DanFinola_dec_reboot",
            ],
            augmented_dataset_name_list=[
                "dg_aug_shuffled_and_embed_complaints_dec_and_v1_reboots",
                "augmented_combined_methods_published_1897_17Oct23_dynamics_copied",
                # "complaints_gen_35turbo_context1_v6_3600_lowered",
                # "complaints_gen_35turbo_context1_v6_3600",
                "complaints_gen_gpt4_v3",
                # "complaints_gen_gpt4_v3_lowered",
                # "dg_aug_shuffled_complaints_dec_and_complete_cleansed_v1_reboot",
                # "dg_aug_word_embed_complaints_dec_and_complete_cleansed_v1_reboot",
            ],
            name_of_column_to_embed="Comment Text_cleaned",
            name_of_y_column="Is Complaint",
            max_evals=800,
            negative_label_dataset_name_list=[
                "published_1897_17Oct23_dynamics_copied",
                "published_10k_DanFinola_subset",
            ],
            scorer=scorer,
            prefix=pre_prompt,
            balance_test=False,
            balance_val=True,
            # parallelise=False,
            parallelise=True,
            parallel_limit=0,
        )
        this_one.find_optimised_classifier()
        this_one.get_assessor_for_optimised_model()
        this_one.register_optimal_model()
        this_one.log_all_attributes(run=run)
        this_one.assessor.log_all_metrics(
            run=run, display_labels=["no complaint", "complaint"]
        )

        run.complete()

        embeddings_approach.remove_combination_from_file(
            embeddings_model=embeddings_model,
            classifier_name=classifier_name,
            filepath=EXPERIMENT_NAME_LIST_PATH,
        )
        azure_config.clear_all_spark_files()
        print("-" * 50)
