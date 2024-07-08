# Data generation scripts

## Overview

This folder contains 2 types of data creation scripts: some that perform data generation (entirely new data is created by querying external LLM endpoints), and some that perform data augmentation (real data is changed slightly to introduce variety).

The generated and augmented datasets created here can be fed into the optimise scripts as an optional argument `augmented_dataset_name_list`, where they will be used as training data **only**.

N.B. Although datasets created using these methods were used as part of the model selection process, none of the final models selected actually used any of this data for model development/training. This code is left here for archive and interest purposes.

## Script structure - generation scripts

In these scripts, we generally query the OpenAI gpt-3.5-turbo and gpt-4 endpoints with a prompt that is a combination of a base prompt (stored in `src/data_augmentation/prompts_and_text`) and various input parameters. These queries prompt the endpoint to return a generated review, and the queries were developed in collaboration with product owner and expert moderator Lee Morgan, to check that the resulting generated reviews were similar in style and content to real reviews. 

In the case of `safeguarding_generation.py`, the prompts here do not take any parameter inputs, and there is some post-processing applied to filter out any undesired endpoint returns (particularly common with safeguarding, and generally the LLMs are resistant to generate this kind of content).

## Script structure - augmentation scripts

In these scripts, we import a dataset and then we run the `make_and_augmented_dataset` function from the `data_augmentation` module, to create augmented data using only the imported reviews that have been **labelled as training**. This function takes an argument `generation_function`, which defines the augmentation method and which can take two values: `data_augmentation.generate_word_embedded_comments` and `data_augmentation.generateSentenceShuffledComments`. The first of these methods swaps words at random with other words that have a similar contextual embedding. The second of these methods simply shuffles sentences at random. These methods both help to add variety to the dataset, but in both cases it is unlikely that the review is changed so much that the new review has a different label to the old review. In this way we can create large amounts of augmented data without having to repeat the labelling exercise.

The `make_and_augmented_dataset` function also takes an argument `number_of_new_comments_per_comment`. This sets the number of augmented comments that are created per seed review.
