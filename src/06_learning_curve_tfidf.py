import json
import time

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

import config
from learning_curve_common import TRAIN_SIZES, get_train_subset
from tfidf_pipeline import build_pipeline

# TF-IDF + LogReg with the best parameters from D-016, trained on 1K / 5K / 20K / full train.
# Measured on VALIDATION only. Test is never loaded.
best_params = json.loads((config.RESULTS_DIR / "tfidf_lr_best_params.json").read_text())
validation_df = pd.read_csv(config.PROCESSED_DIR / "validation.csv")
full_size = len(pd.read_csv(config.PROCESSED_DIR / "train.csv"))

rows = []
for n_examples in TRAIN_SIZES + [full_size]:
    train_subset_df = get_train_subset(n_examples)
    start_time = time.time()
    pipeline = build_pipeline(best_params["c_value"], best_params["word_ngram_max"], best_params["class_weight"])
    pipeline.fit(train_subset_df["text"], train_subset_df["label"])
    train_seconds = time.time() - start_time
    predictions = pipeline.predict(validation_df["text"])
    rows.append({
        "model": "tfidf_lr", "n_train": n_examples,
        "val_macro_f1": f1_score(validation_df["label"], predictions, average="macro"),
        "val_accuracy": accuracy_score(validation_df["label"], predictions),
        "train_seconds": train_seconds,
    })
    print(rows[-1])

pd.DataFrame(rows).to_csv(config.RESULTS_DIR / "learning_curve_tfidf.csv", index=False)
