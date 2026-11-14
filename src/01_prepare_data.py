import json

import pandas as pd
from datasets import load_dataset

import config

SPLIT_NAMES = ["train", "validation", "test"]


def load_splits():
    dataset = load_dataset(config.DATASET_NAME, config.DATASET_CONFIG)
    splits = {}
    for split_name in SPLIT_NAMES:
        df = dataset[split_name].to_pandas()
        # example_id = row position in the official split. It stays the same
        # even after we drop rows, so ids in json files always point to the right row.
        df["example_id"] = df.index
        splits[split_name] = df
    return splits


def add_match_key(df):
    # Only used to find duplicates, never saved. Lowercase and collapsed spaces
    # make "Çok güzel  film" and "çok güzel film" the same. Python lower() is not
    # perfect for Turkish I/İ, but it treats both sides the same way, which is enough here.
    lowered = df["text"].str.lower()
    collapsed = lowered.str.replace(r"\s+", " ", regex=True)
    df["match_key"] = collapsed.str.strip()
    return df


def drop_empty_rows(splits, removed_rows):
    # Step 1: same rule for all splits.
    for split_name in SPLIT_NAMES:
        df = splits[split_name]
        is_empty = df["match_key"] == ""
        print(f"step 1 empty text, {split_name}: {len(df)} rows -> {len(df) - is_empty.sum()} rows")
        removed_rows.append(make_log(df[is_empty], split_name, "empty text"))
        splits[split_name] = df[~is_empty]
    return splits


def drop_train_overlap(splits, removed_rows):
    # Step 2: train rows whose text also appears in validation or test.
    # Validation and test stay as the official splits, so the fix is done on train.
    train_df = splits["train"]
    other_keys = set(splits["validation"]["match_key"]) | set(splits["test"]["match_key"])
    in_other_split = train_df["match_key"].isin(other_keys)
    print(f"step 2 overlap with validation/test, train: {len(train_df)} rows -> {len(train_df) - in_other_split.sum()} rows")
    removed_rows.append(make_log(train_df[in_other_split], "train", "same text in validation or test"))
    splits["train"] = train_df[~in_other_split]
    return splits


def drop_train_duplicates(splits, removed_rows):
    # Step 3: repeated texts inside train. We keep the first copy.
    train_df = splits["train"]
    is_copy = train_df.duplicated("match_key", keep="first")
    print(f"step 3 duplicates inside train: {len(train_df)} rows -> {len(train_df) - is_copy.sum()} rows")
    removed_rows.append(make_log(train_df[is_copy], "train", "duplicate inside train"))
    splits["train"] = train_df[~is_copy]
    return splits


def make_log(df, split_name, reason):
    log_df = df[["example_id", "text", "label"]].copy()
    log_df["split"] = split_name
    log_df["reason"] = reason
    return log_df


def log_remaining_val_test_overlap(splits):
    # Not removed (official splits), only counted so that Phase 5 can report with/without them.
    val_keys = set(splits["validation"]["match_key"])
    test_keys = set(splits["test"]["match_key"])
    overlap_in_test = splits["test"]["match_key"].isin(val_keys)
    overlap_in_val = splits["validation"]["match_key"].isin(test_keys)
    print(f"kept: {overlap_in_test.sum()} test rows have the same text as a validation row")
    print(f"kept: {overlap_in_val.sum()} validation rows have the same text as a test row")
    return make_log(splits["test"][overlap_in_test], "test", "KEPT: same text in validation")


def draw_llm_subset(test_df):
    subset_df = test_df.sample(n=config.LLM_SUBSET_SIZE, random_state=config.SEED)
    print("llm subset class counts:", subset_df["label"].value_counts().to_dict())
    return sorted(subset_df["example_id"].tolist())


def draw_fewshot_pool(train_df):
    word_counts = train_df["text"].str.split().str.len()
    right_length = (word_counts >= config.FEWSHOT_MIN_WORDS) & (word_counts <= config.FEWSHOT_MAX_WORDS)
    candidates_df = train_df[right_length]
    picked = []
    for label_id in [config.NEGATIVE_LABEL_ID, config.POSITIVE_LABEL_ID]:
        label_df = candidates_df[candidates_df["label"] == label_id]
        picked.append(label_df.sample(n=config.FEWSHOT_PER_CLASS, random_state=config.SEED))
    pool_df = pd.concat(picked)
    # One fixed shuffled order, so the classes are mixed and the order never changes.
    pool_df = pool_df.sample(frac=1, random_state=config.SEED)
    return pool_df["example_id"].tolist()


def main():
    splits = load_splits()
    for split_name in SPLIT_NAMES:
        splits[split_name] = add_match_key(splits[split_name])

    removed_rows = []
    splits = drop_empty_rows(splits, removed_rows)
    splits = drop_train_overlap(splits, removed_rows)
    splits = drop_train_duplicates(splits, removed_rows)
    removed_rows.append(log_remaining_val_test_overlap(splits))

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for split_name in SPLIT_NAMES:
        columns = ["example_id", "text", "label"]
        splits[split_name][columns].to_csv(config.PROCESSED_DIR / f"{split_name}.csv", index=False)
        print(f"saved {split_name}: {len(splits[split_name])} rows")
    pd.concat(removed_rows).to_csv(config.PROCESSED_DIR / "cleaning_log.csv", index=False)

    llm_subset_ids = draw_llm_subset(splits["test"])
    fewshot_ids = draw_fewshot_pool(splits["train"])
    # tolist() above gives plain Python ints, which json can write.
    config.LLM_SUBSET_IDS_FILE.write_text(json.dumps(llm_subset_ids))
    config.FEWSHOT_IDS_FILE.write_text(json.dumps(fewshot_ids))
    print("few-shot ids (train):", fewshot_ids)


main()
