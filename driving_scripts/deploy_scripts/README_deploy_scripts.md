# Deploy scripts

## Overview

This folder contains a collection of deployment scripts, each of which:
- sets up an endpoint for one of the automoderation rules
- deploys the appropriate model to the endpoint

These deployment scripts are stored together with their corresponding score scripts, in a folder per rule. The score script provides the code which is run in order to fetch model predictions for an input.

This deployment process is done for:
- complaints
- safeguarding
- not an experience
- names
- descriptions

In the case of names and descriptions, this equates to simply deploying the out-of-the-box BERT and spacy models respectively to the endpoints, and any post-processing that is required to filter the results is done on the Flask app.

## Script structure - deployment scripts

The deployment scripts all follow the same general structure:

1. Create a model directory folder in the current folder.

2. Download the appropriate pre-registered models from the Azure model directory. For example, for the not an experience model, these are the bag-of-words vectorizer and the trained logistic regression model.

3. Define the endpoint that we want to deploy to, creating it if needed.

4. Define and create the new deployment, including the deployment name, the directory path pointing to the previously downloaded models, the score script, the deployment environment, the instance type and count.

Some deployment scripts have an extra step, assigning 100% endpoint traffic to the new deployment. This will automatically direct endpoint requests solely to the new deployment, unless otherwise specified.

## Script structure - scoring scripts

Different models for different rules will process the inputs and fetch model predictions differently. However, each scoring script generally has:

1. An init() step, where the appropriate models are loaded and set as global variables.

2. An encode() step, where the input text is translated to a numerical
vector representation.

3. A run() step, which runs the encode() step above on the input text, and then proceeds to perform the classification on the resulting vector. Results are returned in json format.
