import json

import matplotlib.pyplot as plt
import pandas as pd

import config

# Reads the saved learning curve results (validation only) and draws one plot and one table.
RESULTS = config.RESULTS_DIR
tfidf_df = pd.read_csv(RESULTS / "learning_curve_tfidf.csv")

berturk_rows = []
for n_train in [1000, 5000, 20000]:
    result = json.loads((RESULTS / f"learning_curve_berturk_n{n_train}.json").read_text())
    berturk_rows.append({"n_train": n_train, "val_macro_f1": result["val_macro_f1"], "n_epochs": 3, "train_minutes": result["train_seconds"] / 60})
# The full train run (seed 42) has its own log from Phase 3.
full_log = json.loads((RESULTS / "berturk_training_log_seed42.json").read_text())
full_scores = []
for entry in full_log["history"]:
    if "eval_macro_f1" in entry:
        full_scores.append(entry["eval_macro_f1"])
berturk_rows.append({"n_train": int(tfidf_df["n_train"].max()), "val_macro_f1": max(full_scores), "n_epochs": 3, "train_minutes": full_log["train_seconds"] / 60})
berturk_df = pd.DataFrame(berturk_rows)

# 1K with 15 epochs: a separate point, because 3 epochs gave too few steps.
long_run = json.loads((RESULTS / "learning_curve_berturk_n1000_epochs15.json").read_text())
long_df = pd.DataFrame([{"n_train": 1000, "val_macro_f1": long_run["val_macro_f1"], "n_epochs": 15, "train_minutes": long_run["train_seconds"] / 60}])

table_df = pd.concat([berturk_df, long_df]).assign(model="berturk_3_epochs_or_15")
tfidf_table_df = tfidf_df[["n_train", "val_macro_f1", "train_seconds"]].assign(model="tfidf_lr", n_epochs=0, train_minutes=tfidf_df["train_seconds"] / 60)
all_df = pd.concat([tfidf_table_df[["model", "n_train", "val_macro_f1", "n_epochs", "train_minutes"]], table_df[["model", "n_train", "val_macro_f1", "n_epochs", "train_minutes"]]])
all_df.to_csv(RESULTS / "learning_curve.csv", index=False)
print(all_df.round(4).to_string(index=False))

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(tfidf_df["n_train"], tfidf_df["val_macro_f1"], marker="o", label="TF-IDF + LogReg")
ax.plot(berturk_df["n_train"], berturk_df["val_macro_f1"], marker="o", label="BERTurk (3 epochs)")
ax.plot(long_df["n_train"], long_df["val_macro_f1"], marker="s", linestyle="none", label="BERTurk, 1K with 15 epochs")
ax.set_xscale("log")
ax.set_xlabel("training examples (log scale)")
ax.set_ylabel("validation macro-F1")
ax.legend()
plt.tight_layout()
plt.savefig(RESULTS / "plot_learning_curve.png", dpi=150)
