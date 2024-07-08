## This is an older iteration of the optimise script, before a more elegant refactoring of code to functionalise more parts, and before tests were introduced. Readers should look to the optimise_safeguarding.py file instead, and 
## simply change the 'multiclass' argument (and change log_all_metrics to log_all_multiclass_metrics towards the end of the script) to switch between binary and multiclass safeguarding models.


import itertools
import os
import sys

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from src.azure_config import azure_config
from src.embeddings_approach import embeddings_approach

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, root_path)


EXPERIMENT_NAME = "safeguarding_binary_broad_50"
EXPERIMENT_NAME_LIST_PATH = os.path.join("experiment_lists/", EXPERIMENT_NAME + ".txt")

embeddings_models = [
    "all-mpnet-base-v2",
    "BAAI/bge-small-en",
    "BAAI/bge-base-en",
    "thenlper/gte-base",
    "thenlper/gte-large",
    "BAAI/bge-large-en",
    "intfloat/e5-large-v2",
]

classifier_mapping = {
    # "Gradient Boosting": (GradientBoostingClassifier, embeddings_approach.gradient_boosting_classifier_space_arguments_default),
    "SVM": (SVC, embeddings_approach.svm_space_arguments_default),
    "Random Forest": (
        RandomForestClassifier,
        embeddings_approach.random_forest_classifier_space_arguments_default,
    ),
    "Logistic Regression": (
        LogisticRegression,
        embeddings_approach.logistic_regression_space_arguments_default,
    ),
}


classifier_names = list(classifier_mapping.keys())

if not os.path.exists(EXPERIMENT_NAME_LIST_PATH):
    os.makedirs(os.path.dirname(EXPERIMENT_NAME_LIST_PATH), exist_ok=True)
    with open(EXPERIMENT_NAME_LIST_PATH, "w") as file:
        for embeddings_model, classifier_name in itertools.product(
            embeddings_models, classifier_names
        ):
            file.write(f"{embeddings_model},{classifier_name}\n")


def check_if_combination_in_file(embeddings_model, classifier_name, filename):
    with open(filename, "r") as file:
        if f"{embeddings_model},{classifier_name}" in file.read():
            return True
    return False


def remove_combination_from_file(embeddings_model, classifier_name, filename):
    with open(filename, "r") as file:
        lines = file.readlines()

    with open(filename, "w") as file:
        for line in lines:
            if line.strip("\n") != f"{embeddings_model},{classifier_name}":
                file.write(line)
    print("line removed")


NUMBER_OF_RUNS = len(embeddings_models) * len(classifier_names)
i = 0
for embeddings_model in embeddings_models:
    for classifier_name in classifier_names:
        i += 1
        print("-" * 50)
        print(f"\n\n On Run {i} out of {NUMBER_OF_RUNS}")

        if not check_if_combination_in_file(
            embeddings_model=embeddings_model,
            classifier_name=classifier_name,
            filename=EXPERIMENT_NAME_LIST_PATH,
        ):
            continue

        base_classifier, default_classifier_space = classifier_mapping[classifier_name]

        if embeddings_model == "intfloat/e5-large-v2":
            pre_prompt = "query: "
        else:
            pre_prompt = ""

        run = azure_config.start_run(expeiment_name=EXPERIMENT_NAME)

        this_one = embeddings_approach.EmbeddingsApproach(
            classifier_class=base_classifier,
            model_for_embeddings_name=embeddings_model,
            default_classifier_arguments=default_classifier_space,
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
            name_of_y_column="Safeguarding",
            max_evals=50,
            negative_label_dataset_name_list=[
                # 'published_80k_DanFinola',
                # "published_10k_DanFinola_subset"
                "published_3k_dg_devset"
            ],
            prefix=pre_prompt,
            balance_test=False,
            balance_val=False,
            multiclass=False,
            parallelise=True,
        )
        this_one.find_optimised_classifier()
        this_one.make_and_fit_optimal_classifier()
        this_one.assessor.log_all_metrics(
            run=run, display_labels=["publishable", "safeguarding"]
        )
        this_one.register_optimal_model()
        this_one.log_all_attributes(run=run)
        run.complete()
        remove_combination_from_file(
            embeddings_model=embeddings_model,
            classifier_name=classifier_name,
            filename=EXPERIMENT_NAME_LIST_PATH,
        )
        print("-" * 50)
