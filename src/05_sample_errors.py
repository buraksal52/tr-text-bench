import pandas as pd

import config

# Draws about 100 examples for the error analysis (docs/ERROR_TAXONOMY.md, procedure step 2).
# The three main models: tfidf_lr, berturk, gemini_0shot. Gemini 5-shot is left out, it is almost the same as 0-shot.
MODELS = ["tfidf_lr", "berturk", "gemini_0shot"]
SAMPLE_SIZES = {3: 35, 1: 40, 2: 25}  # key = how many of the 3 models are wrong

predictions_df = pd.read_csv(config.RESULTS_DIR / "test_subset1000_all_predictions.csv")
test_df = pd.read_csv(config.PROCESSED_DIR / "test.csv")[["example_id", "text"]]

wrong_count = 0
for model_name in MODELS:
    is_wrong = (predictions_df[model_name] != predictions_df["true_label"]).astype(int)
    predictions_df["wrong_" + model_name] = is_wrong
    wrong_count = wrong_count + is_wrong
predictions_df["n_models_wrong"] = wrong_count

parts = []
for n_wrong, sample_size in SAMPLE_SIZES.items():
    group_df = predictions_df[predictions_df["n_models_wrong"] == n_wrong]
    parts.append(group_df.sample(n=sample_size, random_state=config.SEED))
sample_df = pd.concat(parts)
# Shuffle, so the reading order does not reveal how many models were wrong.
sample_df = sample_df.sample(frac=1, random_state=config.SEED).reset_index(drop=True)
sample_df = sample_df.merge(test_df, on="example_id")

# The key file knows which model was wrong. It is NOT looked at while labeling.
key_columns = ["example_id", "n_models_wrong"] + ["wrong_" + name for name in MODELS] + ["gemini_5shot"]
sample_df[key_columns].to_csv(config.RESULTS_DIR / "error_sample_key.csv", index=False)
# The blind file has only what a labeler may see: text and the true label.
sample_df[["example_id", "text", "true_label"]].to_csv(config.RESULTS_DIR / "error_sample_blind.csv", index=False)
print("sample size:", len(sample_df))
print(sample_df["n_models_wrong"].value_counts().to_dict())
