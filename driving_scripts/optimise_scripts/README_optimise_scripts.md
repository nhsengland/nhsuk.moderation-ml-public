# Optimise scripts

## Overview

This folder contains a collection of optimisation scripts designed to fine-tune and optimise models for certain automoderation rules, by making use of the pipeline modules found in `src/`. The models tested and optimised were all classifier models, and used an embeddings approach to vectorize the review text. A range of classifiers and embeddings models were tested as part of the optimisation process.

These scripts were run as part of the model development process, but ultimately:
- for complaints, an old model outperformed the models resulting from this new optimisation script
- for safeguarding, an old model also outperformed the newer models
- for not an experience, a model using bag-of-words representation outperformed the embeddings approach used here

Despite the fact that none of the live models were actually developed using these scripts, we have included them in the public repo to keep the model selection process as transparent as possible.

## Combinations explored

For each optimise script, the following embeddings models were tested:
- all-mpnet-base-v2
- BAAI/bge-small-en
- BAAI/bge-base-en
- BAAI/bge-large-en
- thenlper/gte-base
- thenlper/gte-large
- intfloat/e5-large-v2

And the following classifiers were tested:
- logistic regression
- svm
- xgboost

## Script structure

The optimise scripts all follow the same structure:

1. Define embeddings models and classifier to use in the script.

2. Define datasets to use. These will be more specifically defined according to their label in step 4.

3. Run tests. These tests will confirm that there are no duplicates in the datasets, either within the datasets or across datasets. The tests will also confirm that the embeddings saved in the datafiles are reproducible, which helps to confirm that the script is being run in the correct environment.

4. Start Azure experiment, and set up a .txt file which can keep track of the embeddings+classifier combinations that have been completed. This allows us to cancel the script mid-run, and restart without repeating combinations that have already been completed.

5. Ensure the experiment is running on the right part of the compute disc space, to ensure the embeddings models can all be downloaded and run if needed.

6. Code loops through every combination, performing hyperparameter optimisation to find the optimum model for each combination. Within this loop we define several lists and parameters, the most notable of which are:
- the datasets that have 'positive' labels, i.e. datasets where the reviews break the rule (`positive_label_dataset_name_list`)
- the datasets that have 'negative' labels, i.e. datasets where the reviews are publishable (`negative_label_dataset_name_list`)
- any datasets that have generated or augmented, where the data breaks the rule, which will be used as training data (`augmented_dataset_name_list`)
- whether or not we want to balance the labels in the test dataset (`balance_test`), and the validation dataset (`balance_val`)
- whether or not the data is multiclass (`multiclass`), which will also affect the evaluation metrics

7. Find the best-performing combination from those that have been tested. Important model information is logged (such as the embeddings model used and values for the evaluation metrics), and the classifier model is registered as a .pkl file.
