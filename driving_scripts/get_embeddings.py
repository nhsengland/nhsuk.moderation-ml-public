import os
import sys

from src.embeddings_approach import embeddings_getting

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root_path)

list_of_dataset_names = [
    # "complaints_complete_cleansed_v1_reboot",
    # "complaints_dg_DanFinola_dec_reboot",
    # "published_1897_17Oct23_dynamics_copied",
    # "augmented_combined_methods_published_1897_17Oct23_dynamics_copied",
    # "dg_aug_shuffled_and_embed_complaints_dec_and_v1_reboots",
    # "published_10k_DanFinola_subset"
    "complaints_gen_gpt4_v3"
]


for dataset_name in list_of_dataset_names:
    embeddings_getting.get_or_check_all_embeddings(
        dataset_name=dataset_name, name_of_column_to_embed="Comment Text_cleaned"
    )
