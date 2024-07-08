#! WARNING
# The data_augmentation module has undergone significant refactoring since this script was used. We keep it for posterity. If using, it will need some adaptation. 

import pandas
from src.data_augmentation import data_augmentation
from src.data_ingestion import data_ingestion


# create augmented data:
context = data_augmentation.generate_parameterised_context(base_context_filename='safeguarding_low',
                                                 prompt_parameters={})

revs = data_augmentation.generate_openai_review(context=context,model='gpt-4',n=3)


# alternatively, read in previously augmented data:
revs = []
for name in [
    "safeguarding_low_working_gpt-4_1",
    "safeguarding_low_working_gpt-4_2",
    "safeguarding_low_working_gpt-4_3",
    "safeguarding_low_working_gpt-4_4",
    "safeguarding_low_2_gpt-4_1",
    "safeguarding_low_2_gpt-4_2",
]:
    revs += data_augmentation.load_json_reviews(name)


def save_list_to_txt(strings, filename):
    with open(filename, "w") as file:
        for s in strings:
            file.write(s + "\n")


filter_list = [
    "I'm sorry but I can't help with that",
    "I'm really sorry but I can't help with that",
]

revs = data_augmentation.filter_close_matches(revs, filter_list, cutoff=0.5)


df = pandas.DataFrame()
df["Comment Text"] = revs
df["Safeguarding"] = 1
df["train"] = 1
data_ingestion.register_dataframe(
    df=df, dataset_name="safeguarding_low_gen_gpt4_500_24Aug23_DanG"
)

