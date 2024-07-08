"""
This module provides functionalities for cleaning and augmenting textual data.
- Cleaning strings and adding cleaned text as a new column to a dataset. 
- Generating augmented textual data by shuffling sentences within comments or through word embedding techniques using NLPAug and NLTK
libraries.
"""
import re
from typing import List

import nlpaug
import nlpaug.augmenter.sentence as nas
import nlpaug.augmenter.word as naw
import nltk
import pandas
from nltk.corpus import stopwords
from tqdm import tqdm

from src.data_ingestion import data_ingestion

nltk.download("stopwords")
nltk.download("punkt")


def clean_string(s: str)-> str:
    """
    Removes double spaces and newlines from strings, lowers the case.
    """
    s = s.strip()
    s = re.sub("  ", " ", s)
    s = re.sub("   ", " ", s)
    s = re.sub("\n", "", s)
    s = s.lower()
    return s


def clean_text_and_add_column(dataset_name: str, column_name: str):
    """
    Cleans text in a specified column and adds a new column with cleaned text to the dataset.
    
    Parameters:
    - dataset_name (str): Name of the dataset.
    - column_name (str): Name of the column to clean.
    """

    new_column_name = column_name + "_cleaned"
    df = data_ingestion.DataRetrieverDatastore(dataset_name=dataset_name).dataset
    if new_column_name in df.columns:
        print(f"I already have the column {new_column_name}, so Im aborting")
        return
    df[new_column_name] = df[column_name].apply(clean_string)
    data_ingestion.register_dataframe(dataset_name=dataset_name, df=df)


def generate_sentence_shuffled_comments(
    comment_in, number_of_new_comments_per_comment: int
)-> List:
    """
    Generates shuffled sentence comments based on the input comment.
    
    Parameters:
    - comment_in (str): The input comment.
    - number_of_new_comments_per_comment (int): Number of new shuffled sentence comments to generate.
    
    Returns:
    - List[str]: A list of shuffled sentence comments.
    """
    aug = nas.RandomSentAug(mode="random", aug_p=1)

    generatedComments = []

    sentences = comment_in.split(".")
    cleaned_sentences = [
        s for s in sentences if s.strip()
    ]  # Remove any empty or whitespace-only strings

    # Check if the comment has at least two sentences. If not, return the comment itself.
    if len(cleaned_sentences) < 2:
        return [comment_in] * number_of_new_comments_per_comment

    # Here we're generating N comments for a single input comment. The 'x' below
    # is intentionally not called, it's just there for us to loop with. The [0]
    # index on the augment object gives us the actual string we're after. 
    for x in range(number_of_new_comments_per_comment):
        augmentedComment = aug.augment(comment_in)[0]
        generatedComments.append(augmentedComment)

    return generatedComments


def generate_word_embedded_comments(
    comment_in: str,
    number_of_new_comments_per_comment: int,
    aug_in: naw.ContextualWordEmbsAug = None,
) -> List[str]:
    """
    Generates word embedded comments based on the input comment.

    Parameters:
    - comment_in (str): The input comment.
    - number_of_new_comments_per_comment (int): The number of new comments to generate per input comment.
    - aug_in (naw.ContextualWordEmbsAug, optional): The word embeddings augmentation. If None, will use BERT-base-cased by default.

    Returns:
    - List[str]: A list of augmented comments.
    """

    # Check if aug_in is None, if so instantiate default
    if aug_in is None:
        aug_in = naw.ContextualWordEmbsAug(
            model_path="bert-base-cased",
            aug_p=0.05,
            stopwords=stopwords.words("english"),
        )

    new_comments = []

    for _ in range(number_of_new_comments_per_comment):
        sentences = comment_in.split(".")
        augmented_comment = []
        for sentence in sentences:
            augmented_sentence = aug_in.augment(sentence, n=1)
            if augmented_sentence:
                if "[UNK]" not in augmented_sentence[0]:
                    augmented_comment.append(augmented_sentence[0])
                else:
                    augmented_comment.append(sentence)
        joined_comment = ". ".join(augmented_comment)
        new_comments.append(joined_comment)

    return new_comments


def make_and_augmented_dataset(
    df: pandas.DataFrame,
    number_of_new_comments_per_comment: int,
    generation_function: callable,
)-> pandas.DataFrame:
    """
    Augments a dataset by generating new comments for each row using a specified function.
    
    Parameters:
    - df (pandas.DataFrame): Original dataset.
    - number_of_new_comments_per_comment (int): Number of new comments to generate per original comment.
    - generation_function (callable): Function to generate new comments.
    
    Returns:
    - pandas.DataFrame: Augmented dataset with new comments.
    """
    aug_df = []
    successful_augments = 0
    for i, row in tqdm(df.iterrows(), total=df.shape[0]):
        try:
            new_comments = generation_function(
                comment_in=row["Comment Text"],
                number_of_new_comments_per_comment=number_of_new_comments_per_comment,
            )
            comment_id = row["Comment ID"]
            for comment in new_comments:
                aug_df.append({"Comment ID": comment_id, "Comment Text": comment})
            successful_augments += 1
        except:
            pass
    print(f"Successfully augmented {successful_augments} out of {len(df)} rows")
    aug_df = pandas.DataFrame(aug_df)
    return aug_df
