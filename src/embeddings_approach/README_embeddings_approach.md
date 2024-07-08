## Overview

This module provides a class `EmbeddingsApproach` for training classifiers on text data using sentence embeddings.

The key steps are:

1. Several datasets containing text samples are provided, some with positive labels and some with negative. The positive labels are the data which _have the thing_; i.e., which are complaints.

2. The `EmbeddingsApproach` generates embeddings for the text data using a specified sentence embeddings model like BERT. These embeddings are registered as new versions of the datasets. This only happens once so as to save on compute.

3. The class splits the data into train, validation and test sets. It can balance the classes in these sets. By default the `balance` variable is set to True. If you don't want this, it's possible to input custom splits. The balancing is achieved by downsampling. Remember that the `train, test, val` flags are fixed *at the level of the registered datasets*.

4. A search space is defined for the classifier hyperparameters and the number of datasets which you might be sampling from.

5. Hyperopt is used to find the optimal hyperparameters and training set by maximising the chosen scorer.

6. Within each hyperopt `trial`, the classifier is fit to the `training data`. Outside this, in the optimization routine, the hyperparameters are fit to the `test data`. The score which you see in the end is against the `val` data -- look at the `assessor` module for more info on this.

7. The best classifier found is fitted on the training data and registered as a model.

8. The results of the `assessor` get logged as well

## Wide and Shallow: Optimising via a script

The `optimise_complaints.py` exists to try all combinations of embeddings models with all combinations of classifiers.
An experiment list is generated, consisting of various combinations of embeddings models and classifier names.
If the list file for the experiment does not exist, it's created. If it already exists, the script will use it to keep track of pending combinations.

For each combination of embedding model and classifier:
The script first checks the list to see if this combination exists in the experiment `txt` file.
If it does, the script proceeds with the run. Otherwise, the combination is skipped.
Upon the successful completion of an run for a given combination, that combination is removed from the list to ensure it won't be rerun in the future.



## Parallelising
The tricky thing with all of this - whether doing a broad and shallow approach or a deep dive on a couple of models, is managing the compute resources and parallelisation. The 'wide and shallow' version driven by the `optimise_complaints` script leverages `Spark` to parallelise. Here, we're using spark to create workers, each of whom will run a series of hyperopt trials. There are some difficulties we can run into here, both with memory and with drive space. Steps we've taken to alleviate these:
- We limit to 16 workers. More workers is better, but incurs more of a memory overhead on the driver node. 16 has gotten successes, so that's good enough.
- Big ram (48GB) for the driver
- 2GB ram per worker seems to work fine
- Assigning the entirety of python to use `/mnt/` as a caching space. This happens at the top of the `embeddigngs_approach` module. *this part is important!* If your python kernel isn't using `mnt` as a cache, then spark won't either, even if you tell it to. So we tell python to use `mnt`, and then we tell spark to - in that order. You can see all the spark config inside the `find_optimal_classifier` method inside `EmbeddingsApproach` class.
- After each run, we run `azure_config.clear_all_spark_files()`. This is a scorched earth approach - we delete anything related to spark. This seems like total overkill - but we've somehow filled up >1TB drives whilst using the recommended garbage collection tools after each run - so scorched earth it is.
