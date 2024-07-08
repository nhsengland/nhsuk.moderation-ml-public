- [Automoderation and more for NHS UK](#automoderation-and-more-for-nhs-uk)
  - [Installation and environment management](#installation-and-environment-management)
    - [`requirements.txt`](#requirementstxt)
    - [What to do](#what-to-do)
  - [Driving Scripts](#driving-scripts)
    - [Data gen scripts](#data-gen-scripts)
    - [Deploy scripts](#deploy-scripts)
    - [Model training](#model-training)
    - [Optimise scripts](#optimise-scripts)
    - [Get embeddings](#get-embeddings)
  - [src/ : Modules and purpuse](#src--modules-and-purpuse)
    - [Azure config](#azure-config)
    - [Data augmentation](#data-augmentation)
    - [Data ingestion](#data-ingestion)
    - [Embeddings approach](#embeddings-approach)
    - [Embeddings models](#embeddings-models)
    - [Environment management](#environment-management)
    - [Model assessment](#model-assessment)
    - [Question answering approach](#question-answering-approach)
    - [Utils for tests](#utils-for-tests)
  - [tests](#tests)
    - [Test Data](#test-data)
    - [Test Models](#test-models)
    - [archive](#archive)
  - [Model getting:](#model-getting)
    - [Running out of disk space](#running-out-of-disk-space)

# Automoderation and more for NHS UK
This repo exists to manage the resources on AMLS. This includes, but is not limited to, the creation of new ML models.


## Installation and environment management
The environment management is important for this project, because we want to ensure that the *environment in which our models are built is identical to the environment in which they are deployed*.

To do this, we use azure's own environmnent feature.

We pick an environment, load it locally, and build models in that environment. It's very important that any models which we're deploying have been created in one of the existing registered environments. It's also very important that those existing registered environments have fully specified versions for every package, and that those environments have been built and tested. There are various situations in which these environments might need to rebuild (for example, a cluster reboots) -- and it's imperative that they can rebuild safely and reliably.

We have some functionality to handle all of this built into the `environment_management` module. Some headlines on using this:

- It's important that scripts include the `environment_management.check_environment_is_correct` method.
- Do not modify the environment direcly in your dev workspace. If the environment needs updating, go to the corresponding environment object in AMLS and update the conda yaml file. From there, use the `environment_management.load_environment` function.
- This whole approach might seem cumbersome and counter-intuitive. What's wrong with a simple requirements file? The reason we've chosen this approach has to do with how python environments are passed to deployments in AMLS. Since deployment environments must be taken from ALMS environment objects anyway, it made sense for us to take our dev environment(s) from the same source. This allows us to guaruantuee consistency. The cumbersome aspecs are hopefully taken care of by using the module we've built to handle this stuff.

### `requirements.txt`
If we're doing all our environment management via this `environment_management` module, and using AMLS objects - then why is there a `requirements.txt` file in the project root?

The answer is that, in order to run all the functions in `environment_management`, and to interface with AMLS, we first need some Azure packages and such. So the `requirements.txt` file sets you up to be able to run the functions in `environment_management`.

### What to do
Install the pre-environment using
```bash
pip install -r requirements.txt
```
Then, in a scratch script, run
```python
import environment_management

environment_management.load_correct_environment()
```

At the top of every script which is going to create any assets (models, embeddings, etc) - make sure to run

```python
import environment_management

environment_management.check_environment_is_correct()
```

If you want to do dev work in notebooks, you might have some difficulty getting the notebook kernel to pick up the conda environment installed by the above method. If there is a simple way to achieve this, please add it to this repo.

## Driving Scripts

In the root there's a folder called `driving_scripts/`. These use the functionality provided in `src/` to generate, register, and use assets on AMLS. We'll go through these in turn. N.B. There are specific READMEs in each sub-folder, to outline the structure of each type of script in more detail.

```
driving_scripts/
├── data_gen_scripts/
├── deploy_scripts/
├── model_training/
├── optimise_scripts/
└── get_embeddings.py
```

### Data gen scripts

This is a set of scripts which use the functionality in `data_augmentation.data_generation`. This functionality calls an LLM via API to generate reviews for the specific problems we've worked on.

### Deploy scripts

These are the scripts used to deploy the models we've used to their respective endpoints. [See significant documentation on the topic here](https://nhsd-confluence.digital.nhs.uk/display/DAT/DS_233%3A+Model+Deployment+via+the+SDK).

### Model training

Models created not using the functionality or tooling from this repo were created in standalone notebooks. Those notebooks are kept in this folder.

### Optimise scripts

A significant amount of the tooling in `src/` is geared towards building hyper-parameter optimised classifiers for text classification. Most of that is in the `embeddings_approach` module. Each script in the `optimise_scripts` folder calls this functionality with some particular set of arguments. For example, it will specify some datasets to use, some model embeddings, a maximum number of trials, and so forth.


### Get embeddings

The main computationally intensive thing about the `embeddings_approach` to building classifiers is generating the text embeddings. Rather than do this repeatedly, we do this once and register those embeddings with the main data asset on AMLS.

The `get_embeddings.py` file is a super simple script which calls some functionality from the relevant module. Basically you'd just adjust and use this script when you're introducing a new dataset, or if you want to override some embeddings with new values. You could also use it if you're introducing a new embeddings generating model.

## src/ : Modules and purpuse

### Azure config

Most things which we do in this repo need to interact with the assets stored on AMLS. This involves some minor config. For both ease and consistency, this has been abstracted out to this module.

We also set some effectively global parameters here, such as which environment we're using [see the section on this.](#installation-and-environment-management)

### Data augmentation

This has two sub modules:
- `data_augmentation`: This is for re-mixing text data so as to bootstrap a training set.
- `data_generation`: This is for creating entirely new, fictional data via LLMs for training purposes.

### Data ingestion

This is just to deal with the ways in which we fetch data from AMLS, and also how we register data assets there.

### Embeddings approach

This is a fairly hefty module. In short, it has the tooling for:
- Fetching all embeddings for some datasets and embeddings models
- Creating classifiers which look at those embeddings and segment them according to label
- Doing a broad hyperparameter search over:
  - Which embeddings
  - Which classifier
  - Which classifier hyperparameters
  - Which augmented and generated datasets, and in which proportion
  Lead to the best solution to the classification problem.

For more detail, see the README at `src/embeddings_approach/README_embeddings.md`, and also the docstrings within the module itself.

### Embeddings models

This folder should be empty. It's used to store model files which are generated in the embeddings approach optimisation process.

### Environment management

This contains the tooling for interacting with the environment objects on AMLS. See the (installation instructions)[#installation-and-environment-management] for more detail.

### Model assessment

In the early days of this repo, it wasn't clear which approach to clasifying text would be the best. We wanted to try out a variety of different approaches. In order to compare these in a simple and consistent way, it made sense to abstract their assessment out one layer so a seperate module which could treat these different approaches on the same plane.

This is the module we made for this purpose. In the end, we only really used embeddings_aproach type classifiers, so some of this abstraction wasn't that useful.

### Question answering approach

One of the approaches we explored to the classification problem was using LLMs to classify text directly, by asking them. This is where that functionality lives. We abandoned this approach due to poor performance and other issues, so this module isn't very developed.

### Utils for tests

As described. See (Tests)[#tests] for more.

## tests
This contains two sets of test, one for data and one for models.

### Test Data

This contains **very** useful tests. Most of what we've done here relies on the embeddings which we can generate for text using open source models.

Due to reasons which remain somewhat mysterious to us, these generated embeddings seem to be very sensitive to hardware environment.
To illustrate:

- John and Jen both have identical versions of an embeddings model, both work in identical python environments, and are both operating on the same OS.
- John and Jen both generate a set of embeddings for some piece of text.
- If John and Jen both use the same compute hardware, the embeddings will match.
- If John uses a GPU and Jen uses a CPU, the embeddings **might not match**.

This can be very difficult to debug and to track down. These test exist to make sure that the embeddings which you have registered with the data assets you're working with mathc the ones you would get if you generated them all fresh, today.


> [!WARNING] WARNING: We very strongly recommend calling the `test_data` suite at the top of any driving script which is going to register any models or data.

### Test Models

This was the beginnings of some assurance testing for registered models. Incomplete.

### archive

This folder contains code that is no longer relevant, e.g. it contais unused code or code pertaining to the development of retired models (e.g. the uninformative model). The code will be left in this archive in the private repo in case the relevant development ever gets picked up again, but it is not neccesary to review this code and it will not be included in the final published open repo.

## Model getting:
There are a number of problems which we can encounter when we try to download and register a model.

### Running out of disk space
The compute clusters typically have small disk space, and can get filled up. This leads to a crash and, so far, I've needed to delete the cluster when this happens, because it's somewhat finicky to chase down and delete the unwanted files.
If we create a compute with a large disk space, this doesn't necessarily resolve the issue. This is because the large drive is an attached drive, rather than the main drive. We need to a fair bit of configuring to actually make use of this.

Most of the configuring is taken care of in the huggingface_model_getting submodule. There are three important things which happen in the submodule:
- The permissions are granted to write to the `/mnt` drive (which is where AMLS seems to put the large storage)
- The write location for the files themselves are set to the large storage
- The cache for HF is set to the large storage. This one's important.
