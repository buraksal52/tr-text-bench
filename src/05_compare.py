import json
from itertools import combinations
from math import comb

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

import config

# Phase 5: compares all models on the SAME 1000 test examples. Reads saved predictions only.
# No model is run here and no API is called.
N_BOOTSTRAP = 1000
RESULTS = config.RESULTS_DIR
PREDICTION_FILES = {
    "tfidf_lr": RESULTS / "tfidf_lr_test_subset1000_predictions.csv",
    "berturk": RESULTS / "berturk_test_subset1000_seed42_predictions.csv",
    "gemini_0shot": RESULTS / "llm_gemini-3.6-flash_zeroshot_test_n1000_predictions.csv",
    "gemini_5shot": RESULTS / "llm_gemini-3.6-flash_fiveshot_test_n1000_predictions.csv",
}
# Speed and cost come from the result files written in the earlier phases.
INFO_FILES = {
    "tfidf_lr": RESULTS / "tfidf_lr_test_subset1000.json",
    "berturk": RESULTS / "berturk_test_subset1000_seed42.json",
    "gemini_0shot": RESULTS / "llm_gemini-3.6-flash_zeroshot_test_n1000.json",
    "gemini_5shot": RESULTS / "llm_gemini-3.6-flash_fiveshot_test_n1000.json",
}


def load_predictions():
    merged_df = None
    for model_name, path in PREDICTION_FILES.items():
        model_df = pd.read_csv(path).sort_values("example_id")
        model_df = model_df.rename(columns={"predicted_label": model_name})
        model_df = model_df[["example_id", "true_label", model_name]]
        if merged_df is None:
            merged_df = model_df
        else:
            merged_df = merged_df.merge(model_df, on=["example_id", "true_label"])
    return merged_df


def bootstrap_interval(true_labels, predictions):
    # Resample the 1000 examples with replacement, recompute macro-F1, take the 2.5% and 97.5% values.
    random_generator = np.random.default_rng(config.SEED)
    scores = []
    for _ in range(N_BOOTSTRAP):
        picked = random_generator.integers(0, len(true_labels), size=len(true_labels))
        scores.append(f1_score(true_labels[picked], predictions[picked], average="macro"))
    return np.percentile(scores, 2.5), np.percentile(scores, 97.5)


def mcnemar_exact_p(correct_a, correct_b):
    # Only examples where exactly one of the two models is right carry information.
    only_a = int(((correct_a == 1) & (correct_b == 0)).sum())
    only_b = int(((correct_a == 0) & (correct_b == 1)).sum())
    n_different = only_a + only_b
    if n_different == 0:
        return only_a, only_b, 1.0
    smaller = min(only_a, only_b)
    tail = sum(comb(n_different, k) for k in range(smaller + 1)) / 2**n_different
    return only_a, only_b, min(1.0, 2 * tail)


def main():
    merged_df = load_predictions()
    print("examples compared:", len(merged_df))
    true_labels = merged_df["true_label"].to_numpy()

    rows = []
    for model_name in PREDICTION_FILES:
        predictions = merged_df[model_name].to_numpy()
        low, high = bootstrap_interval(true_labels, predictions)
        info = json.loads(INFO_FILES[model_name].read_text())
        rows.append({
            "model": model_name,
            "macro_f1": f1_score(true_labels, predictions, average="macro"),
            "macro_f1_ci_low": low, "macro_f1_ci_high": high,
            "accuracy": accuracy_score(true_labels, predictions),
            "ms_per_example": info.get("time_per_example_ms", info.get("seconds_per_example", 0) * 1000),
            "cost_usd": info["cost_usd"],
        })
    table_df = pd.DataFrame(rows)
    table_df.to_csv(RESULTS / "comparison_table.csv", index=False)
    print(table_df.round(4).to_string(index=False))

    pair_rows = []
    for model_a, model_b in combinations(PREDICTION_FILES, 2):
        correct_a = (merged_df[model_a] == merged_df["true_label"]).astype(int)
        correct_b = (merged_df[model_b] == merged_df["true_label"]).astype(int)
        only_a, only_b, p_value = mcnemar_exact_p(correct_a, correct_b)
        pair_rows.append({"model_a": model_a, "model_b": model_b, "only_a_right": only_a, "only_b_right": only_b, "p_value": p_value})
    pair_df = pd.DataFrame(pair_rows)
    pair_df.to_csv(RESULTS / "pairwise_mcnemar.csv", index=False)
    print(pair_df.round(4).to_string(index=False))

    merged_df.to_csv(RESULTS / "test_subset1000_all_predictions.csv", index=False)
    compare_full_test()


def compare_full_test():
    # TF-IDF and BERTurk were also run on the whole test split (8,934 examples).
    full_files = {
        "tfidf_lr": RESULTS / "tfidf_lr_test_predictions.csv",
        "berturk": RESULTS / "berturk_test_seed42_predictions.csv",
    }
    rows = []
    for model_name, path in full_files.items():
        predictions_df = pd.read_csv(path)
        true_labels = predictions_df["true_label"].to_numpy()
        predictions = predictions_df["predicted_label"].to_numpy()
        low, high = bootstrap_interval(true_labels, predictions)
        rows.append({"model": model_name, "n_examples": len(true_labels),
                     "macro_f1": f1_score(true_labels, predictions, average="macro"),
                     "macro_f1_ci_low": low, "macro_f1_ci_high": high,
                     "accuracy": accuracy_score(true_labels, predictions)})
    full_df = pd.DataFrame(rows)
    full_df.to_csv(RESULTS / "comparison_table_full_test.csv", index=False)
    print(full_df.round(4).to_string(index=False))


main()
