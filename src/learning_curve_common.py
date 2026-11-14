import pandas as pd

import config

# Learning curve (Phase 7): the same nested train subsets for TF-IDF and BERTurk.
TRAIN_SIZES = [1000, 5000, 20000]  # the full train split is added as the last point


def get_train_subset(n_examples):
    train_df = pd.read_csv(config.PROCESSED_DIR / "train.csv")
    # One fixed shuffle: the first 1000 rows are inside the first 5000 rows, and so on.
    shuffled_df = train_df.sample(frac=1, random_state=config.SEED)
    return shuffled_df.iloc[:n_examples]
