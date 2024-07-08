import os
import sys

import pytest
import sklearn
import sklearn.metrics

from src.azure_config import azure_config
from src.embeddings_approach import embeddings_approach
from src.utils_for_tests import utils_for_tests

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, root_path)


embeddings_models = [
    "all-mpnet-base-v2",
    "BAAI/bge-small-en",
    "BAAI/bge-base-en",
    "thenlper/gte-base",
    "thenlper/gte-large",
    "BAAI/bge-large-en",
    "intfloat/e5-large-v2",
]

classifier_name_list = [
    "SVM",
    "logistic_regression",
    # "xgboost",
]


utils_for_tests.define_list_of_datasets_for_tests(
    [
        "published_1897_17Oct23_dynamics",
        "published_1000_24Nov23_DanFinola",
        "experience_pub_3248_6dec23_from10k",
        "generate_word_embedded_commentsnae_220_24Nov23_DanFinola",
        "generateSentenceShuffledCommentsnae_220_24Nov23_DanFinola",
        "generate_word_embedded_commentsnot_an_experience_1000_Oct23_Finola",
        "generateSentenceShuffledCommentsnot_an_experience_1000_Oct23_Finola",
        "not_an_experience_1000_Oct23_Finola",
        "nae_220_24Nov23_DanFinola",
        "nae_1500_08Dec23_DanFinola",
        "nae_100_15Dec23_gpt4_GPsDentists",
        "nae_90_15Dec23_gpt4_sampled",
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


EXPERIMENT_NAME = "at_not_experience_final_data_genaug"
EXPERIMENT_NAME_LIST_PATH = os.path.join("experiment_lists/", EXPERIMENT_NAME + ".txt")


embeddings_approach.make_experiment_run_list_on_disk(
    filepath=EXPERIMENT_NAME_LIST_PATH,
    classifier_name_list=classifier_name_list,
    embedding_model_name_list=embeddings_models,
)

NUMBER_OF_RUNS = len(embeddings_models) * len(classifier_name_list)
i = 0


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
        while retries > 0 and not success:
            parallelise_value = parallel_values[parallelise_values_index]
            try:
                this_one = embeddings_approach.EmbeddingsApproach(
                    classifier_class_name=classifier_name,
                    model_for_embeddings_name=embeddings_model,
                    positive_label_dataset_name_list=[
                        "not_an_experience_1000_Oct23_Finola",
                        "nae_220_24Nov23_DanFinola",
                        "nae_1500_08Dec23_DanFinola",
                    ],
                    augmented_dataset_name_list=[
                        "nae_100_15Dec23_gpt4_GPsDentists",
                        "nae_90_15Dec23_gpt4_sampled",
                        "generate_word_embedded_commentsnae_220_24Nov23_DanFinola",
                        "generateSentenceShuffledCommentsnae_220_24Nov23_DanFinola",
                        "generate_word_embedded_commentsnot_an_experience_1000_Oct23_Finola",
                        "generateSentenceShuffledCommentsnot_an_experience_1000_Oct23_Finola",
                    ],
                    name_of_column_to_embed="Comment Text",
                    name_of_y_column="not_an_experience",
                    max_evals=300,
                    negative_label_dataset_name_list=[
                        "published_1897_17Oct23_dynamics",
                        "published_1000_24Nov23_DanFinola",
                        "experience_pub_3248_6dec23_from10k",
                    ],
                    prefix=pre_prompt,
                    balance_test=True,
                    balance_val=False,
                    # parallelise=False,
                    parallelise=True,
                    parallel_limit=parallelise_value,
                    SVM_probability=True,
                )
                this_one.find_optimised_classifier()
                this_one.get_assessor_for_optimised_model()
                this_one.register_optimal_model()
                this_one.log_all_attributes(run=run)
                this_one.assessor.log_all_metrics(
                    run=run, display_labels=["publishable", "not an experience"]
                )
                success = True
            except Exception as e:
                print(
                    f"Error occurred: {e}. Retrying with different parallelise value."
                )
                parallelise_values_index += 1
                retries -= 1

        run.complete()

        embeddings_approach.remove_combination_from_file(
            embeddings_model=embeddings_model,
            classifier_name=classifier_name,
            filepath=EXPERIMENT_NAME_LIST_PATH,
        )
        azure_config.clear_all_spark_files()
        print("-" * 50)
