import pandas as pd
from transformers import AutoTokenizer

import config

# Only the tokenizer is downloaded here (a few hundred KB), not the model.
tokenizer = AutoTokenizer.from_pretrained(config.BERTURK_MODEL_NAME)

train_df = pd.read_csv(config.PROCESSED_DIR / "train.csv")
validation_df = pd.read_csv(config.PROCESSED_DIR / "validation.csv")

for split_name, df in [("train", train_df), ("validation", validation_df)]:
    # No padding or truncation: we want the real length of every review.
    encoded = tokenizer(df["text"].tolist(), truncation=False, padding=False)
    token_counts = pd.Series([len(ids) for ids in encoded["input_ids"]])
    print(f"--- {split_name}: token counts (including [CLS] and [SEP])")
    print("mean:", round(token_counts.mean(), 1), "| median:", token_counts.median())
    for percentile in [0.90, 0.95, 0.99]:
        print(f"{int(percentile * 100)}th percentile:", int(token_counts.quantile(percentile)))
    for max_length in [64, 128, 192, 256, 512]:
        share_cut = (token_counts > max_length).mean()
        print(f"max_length {max_length}: {share_cut:.1%} of reviews would be cut")
