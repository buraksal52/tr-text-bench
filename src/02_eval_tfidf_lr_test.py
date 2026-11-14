import json
import platform
import time
from datetime import date

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

import config
from tfidf_pipeline import build_pipeline

# FINAL evaluation on the test split. Run once, only after the user said yes.
# The model settings come from the validation grid (02_tune_tfidf_lr.py).


def save_results(split_name, test_df, predictions, ms_per_example, train_seconds, best_params):
    true_labels = test_df["label"]
    accuracy = accuracy_score(true_labels, predictions)
    macro_f1 = f1_score(true_labels, predictions, average="macro")
    precision, recall, _, _ = precision_recall_fscore_support(true_labels, predictions, labels=[0, 1])
    matrix = confusion_matrix(true_labels, predictions, labels=[0, 1])

    print(f"--- {split_name}: {len(test_df)} examples")
    print("accuracy:", round(accuracy, 4), "| macro-F1:", round(macro_f1, 4))
    print("predicted label counts:", pd.Series(predictions).value_counts().to_dict())
    print("confusion matrix (rows = true, columns = predicted):")
    print(matrix)

    result = {
        "model": "tfidf_lr",
        "split": split_name,
        "n_examples": len(test_df),
        "seed": config.SEED,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "precision_negative": precision[0],
        "precision_positive": precision[1],
        "recall_negative": recall[0],
        "recall_positive": recall[1],
        "time_per_example_ms": ms_per_example,
        "train_seconds": train_seconds,
        "hardware": platform.processor() or platform.machine(),
        "params": best_params,
        "cost_usd": 0.0,
        "date": date.today().isoformat(),
    }
    (config.RESULTS_DIR / f"tfidf_lr_{split_name}.json").write_text(json.dumps(result, indent=2))
    predictions_df = pd.DataFrame(
        {
            "example_id": test_df["example_id"],
            "true_label": true_labels,
            "predicted_label": predictions,
            "model": "tfidf_lr",
        }
    )
    predictions_df.to_csv(config.RESULTS_DIR / f"tfidf_lr_{split_name}_predictions.csv", index=False)


def main():
    best_params = json.loads((config.RESULTS_DIR / "tfidf_lr_best_params.json").read_text())
    train_df = pd.read_csv(config.PROCESSED_DIR / "train.csv")
    test_df = pd.read_csv(config.PROCESSED_DIR / "test.csv")
    subset_ids = json.loads(config.LLM_SUBSET_IDS_FILE.read_text())

    pipeline = build_pipeline(best_params["c_value"], best_params["word_ngram_max"], best_params["class_weight"])
    start_time = time.time()
    pipeline.fit(train_df["text"], train_df["label"])
    train_seconds = time.time() - start_time

    # The first call is slower (warm-up), so it is not timed.
    pipeline.predict(test_df["text"].head(100))
    start_time = time.time()
    predictions = pipeline.predict(test_df["text"])
    ms_per_example = (time.time() - start_time) * 1000 / len(test_df)
    print("train seconds:", round(train_seconds, 1), "| ms per example:", round(ms_per_example, 3))

    save_results("test", test_df, predictions, ms_per_example, train_seconds, best_params)

    # Same predictions, only the 1000 examples the LLM will see, so the models are comparable.
    in_subset = test_df["example_id"].isin(subset_ids)
    save_results("test_subset1000", test_df[in_subset], predictions[in_subset.to_numpy()],
                 ms_per_example, train_seconds, best_params)


main()
