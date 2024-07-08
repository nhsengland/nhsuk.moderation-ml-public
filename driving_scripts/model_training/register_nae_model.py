# Sam
# %%

# This is a notebook that trains and registers a non-semantic model, using a bag-of-words vectorizer and a logistic regression classifier

import os
import sys
import azureml

from sklearn import svm, datasets
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, ConfusionMatrixDisplay
import pandas as pd
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Model
import joblib

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, root_path)
from src.azure_config import azure_config
from src.data_ingestion import data_ingestion
from src.data_ingestion import data_utils
from src.model_getting import model_getting
from src.environment_management import environment_management

environment_management.check_environment_is_correct()

EXPERIMENT_NAME = "at_nae_non_semantic_new_data"
run = azure_config.start_run(expeiment_name=EXPERIMENT_NAME)

run.log(name="Environment name", value=azure_config.ENVIRONMENT_NAME)
run.log(name="Environment version", value=azure_config.ENVIRONMENT_VERSION)

# Define the data used - both the dataset names and the amount of data to sample from that dataset
published_ds_name_dict = {'published_1897_17Oct23_dynamics': 'all',
                          'published_1000_24Nov23_DanFinola': 'all',
                          'experience_pub_3248_6dec23_from10k': 'all'}
nae_ds_name_dict = {'not_an_experience_1000_Oct23_Finola': 'all',
                    'nae_220_24Nov23_DanFinola': 'all',
                    'nae_1500_08Dec23_DanFinola': 'all',
                    'generate_word_embedded_commentsnae_220_24Nov23_DanFinola': 80,
                    'generateSentenceShuffledCommentsnae_220_24Nov23_DanFinola': 80,
                    'generate_word_embedded_commentsnot_an_experience_1000_Oct23_Finola': 400,
                    'generateSentenceShuffledCommentsnot_an_experience_1000_Oct23_Finola': 400}
# Define whether or not the training data and validation data should be balanced
balance_train = False
balance_val = False


# Construct published dataset and nae dataset using above lists
published_dataframes = []
for ds_name, sample_size in published_ds_name_dict.items():
    current_df = data_ingestion.DataRetrieverDatastore(ds_name).dataset
    if sample_size=='all':
        published_dataframes.append(current_df)
    else:
        current_df_sample = current_df.sample(n=sample_size)
        published_dataframes.append(current_df_sample)

published_df = pd.concat(published_dataframes)

nae_dataframes = []
for ds_name, sample_size in nae_ds_name_dict.items():
    current_df = data_ingestion.DataRetrieverDatastore(ds_name).dataset
    if sample_size=='all':
        nae_dataframes.append(current_df)
    else:
        current_df_sample = current_df.sample(n=sample_size)
        nae_dataframes.append(current_df_sample)

nae_df = pd.concat(nae_dataframes)

# Merge both dataframes into a single one
df = pd.concat([nae_df, published_df], axis=0, ignore_index=True)
df = df[["Comment ID", "Comment Text", "not_an_experience", "train", "test", "val"]] # keep relevant columns


# Train/test/val split based on stored labels. In this case we are using cross-validation on the train+test data, and validating using the validation data
trained_on = 'train & test'
df_train = df[(df['train']==1) | (df['test']==1)]
df_val = df[df['val']==1]

if balance_train:
    min_samples_train = min(df_train['not_an_experience'].value_counts())
    train_data = pd.concat([
        df_train[df_train['not_an_experience']==0].sample(n=min_samples_train, random_state=42),
        df_train[df_train['not_an_experience']==1].sample(n=min_samples_train, random_state=42)
    ])
else:
    train_data = df_train

if balance_val:
    min_samples_val = min(df_val['not_an_experience'].value_counts())
    val_data = pd.concat([
        df_val[df_val['not_an_experience']==0].sample(n=min_samples_val, random_state=42),
        df_val[df_val['not_an_experience']==1].sample(n=min_samples_val, random_state=42)
    ])
else:
    val_data = df_val

X_train = train_data['Comment Text']
y_train = train_data['not_an_experience']

X_val = val_data['Comment Text']
y_val = val_data['not_an_experience']


# Convert to BoW representation
representation = "bow, include stopwords, 10000 features"
vectorizer = CountVectorizer(max_features=10000)  # Limiting to 10000 features for simplicity, including stop words
X_train_bow = vectorizer.fit_transform(X_train)
X_val_bow = vectorizer.transform(X_val)


# Setting the parameter grid for logistic regression
param_grid = {
    'penalty': ['l1', 'l2'],           # Regularization term (L1 or L2)
    'C': [0.001, 0.01, 0.1, 1, 10, 100],  # Inverse of regularization strength
    'solver': ['liblinear', 'saga'],  # Algorithm to use for optimization problem
}

# Performing crossvalidation
grid = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, refit=True, verbose=2, cv=5)
classifier = "LR"
grid.fit(X_train_bow, y_train)

print("Best parameters found: ", grid.best_params_)
run.log(name="classifier", value=classifier)
run.log(name="best_parameters", value=str(grid.best_params_))

# Evaluate on val set
best_estimator = grid.best_estimator_

y_pred = best_estimator.predict(X_val_bow)

# Log appropriate evaluation metrics
confusion_mat = confusion_matrix(y_val, y_pred)
f1_score_metric = f1_score(y_val, y_pred)
run.log(name="f1_score", value=f1_score_metric)
disp = ConfusionMatrixDisplay(confusion_mat, display_labels=["publishable","not_an_experience"])
disp.plot(values_format="")
run.log_image("confusion_matrix_plot", plot=disp.figure_)

# Log any other relevant details
run.log(name="balance_train", value=balance_train)
run.log(name="balance_val", value=balance_val)
run.log(name="representation", value=representation)
run.log(name="env_name", value=azure_config.ENVIRONMENT_NAME)
run.log(name="env_version", value=azure_config.ENVIRONMENT_VERSION)


# Register the models:

environment_management.check_environment_is_correct()

base_path = "non_semantic_models/"
classifier_model_name = "nae_bow_final_data_LR_aug2"
vectorizer_model_name = "nae_bow_vectorizer_final_data_aug2"
run.log(name="classifier_model_name", value=classifier_model_name)
run.log(name="vectorizer_model_name", value=vectorizer_model_name)


# First register the vectorizer
path_for_vectorizer_model = os.path.join(base_path, vectorizer_model_name + ".pkl")
os.makedirs(base_path, exist_ok=True)

joblib.dump(value=vectorizer, filename=path_for_vectorizer_model)

vectorizer_file_model = Model(
    path=path_for_vectorizer_model,
    type=AssetTypes.CUSTOM_MODEL,
    name=vectorizer_model_name,
    description=f"neg label data = {published_ds_name_dict}; pos label data = {nae_ds_name_dict}; trained on: {trained_on}; representation: {representation}; environment name = {azure_config.ENVIRONMENT_NAME}; environment version = {azure_config.ENVIRONMENT_VERSION}",
)

ml_client = azure_config.get_ml_client()
ml_client.models.create_or_update(vectorizer_file_model)
print(f"Model registered under the name {vectorizer_model_name}")

# And then register the classifier
path_for_classifier_model = os.path.join(base_path, classifier_model_name + ".pkl")
os.makedirs(base_path, exist_ok=True)

joblib.dump(value=best_estimator, filename=path_for_classifier_model)

classifier_file_model = Model(
    path=path_for_classifier_model,
    type=AssetTypes.CUSTOM_MODEL,
    name=classifier_model_name,
    description=f"neg label data = {published_ds_name_dict}; pos label data = {nae_ds_name_dict}; trained on: {trained_on}; validated on: val; representation: {representation}; classifier: {classifier}; optimised: gridsearchCV; best parameters: {grid.best_params_}; environment name = {azure_config.ENVIRONMENT_NAME}; environment version = {azure_config.ENVIRONMENT_VERSION}"
)

ml_client = azure_config.get_ml_client()
ml_client.models.create_or_update(classifier_file_model)
print(f"Model registered under the name {classifier_model_name}")


run.complete()

