import json
import time

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

import config
from tfidf_pipeline import build_pipeline

# The grid is small on purpose. Everything is tuned on validation only. Test is never loaded here.
C_VALUES = [0.1, 1, 10]
WORD_NGRAM_MAX_VALUES = [1, 2]
CLASS_WEIGHT_VALUES = [None, "balanced"]


def main():
    train_df = pd.read_csv(config.PROCESSED_DIR / "train.csv")
    validation_df = pd.read_csv(config.PROCESSED_DIR / "validation.csv")

    rows = []
    for c_value in C_VALUES:
        for word_ngram_max in WORD_NGRAM_MAX_VALUES:
            for class_weight in CLASS_WEIGHT_VALUES:
                start_time = time.time()
                pipeline = build_pipeline(c_value, word_ngram_max, class_weight)
                # fit() learns the vocabulary and the weights from train only.
                pipeline.fit(train_df["text"], train_df["label"])
                # predict() only applies what was learned, validation is never fitted on.
                predictions = pipeline.predict(validation_df["text"])

                accuracy = accuracy_score(validation_df["label"], predictions)
                macro_f1 = f1_score(validation_df["label"], predictions, average="macro")
                seconds = time.time() - start_time
                rows.append(
                    {
                        "C": c_value,
                        "word_ngram_max": word_ngram_max,
                        "class_weight": class_weight,
                        "val_accuracy": round(accuracy, 4),
                        "val_macro_f1": round(macro_f1, 4),
                        "seconds": round(seconds, 1),
                    }
                )
                print(rows[-1], flush=True)

    grid_df = pd.DataFrame(rows).sort_values("val_macro_f1", ascending=False)
    print()
    print(grid_df.to_string(index=False))

    config.RESULTS_DIR.mkdir(exist_ok=True)
    grid_df.to_csv(config.RESULTS_DIR / "tfidf_lr_grid_validation.csv", index=False)

    best_row = grid_df.iloc[0]
    # pandas turns None into NaN in a column, so we translate it back for json.
    class_weight = best_row["class_weight"]
    if pd.isna(class_weight):
        class_weight = None
    best_params = {
        "c_value": float(best_row["C"]),
        "word_ngram_max": int(best_row["word_ngram_max"]),
        "class_weight": class_weight,
    }
    (config.RESULTS_DIR / "tfidf_lr_best_params.json").write_text(json.dumps(best_params, indent=2))
    print("best params:", best_params)


main()
