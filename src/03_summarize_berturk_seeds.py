import json

import pandas as pd

import config

# Mean and standard deviation of the BERTurk test results over the seeds.
# Only reads result files that already exist. It never touches the model or the data.
SEEDS = [42, 43, 44]
SPLITS = ["test", "test_subset1000"]

summary = {}
for split_name in SPLITS:
    rows = []
    for seed in SEEDS:
        result = json.loads((config.RESULTS_DIR / f"berturk_{split_name}_seed{seed}.json").read_text())
        rows.append({"seed": seed, "accuracy": result["accuracy"], "macro_f1": result["macro_f1"]})
    seed_df = pd.DataFrame(rows)
    print(f"--- {split_name}")
    print(seed_df.round(4).to_string(index=False))

    # Standard deviation of only 3 values is a rough number, pandas uses n-1 (sample std).
    summary[split_name] = {
        "n_seeds": len(SEEDS),
        "accuracy_mean": seed_df["accuracy"].mean(),
        "accuracy_std": seed_df["accuracy"].std(),
        "macro_f1_mean": seed_df["macro_f1"].mean(),
        "macro_f1_std": seed_df["macro_f1"].std(),
    }
    print("mean macro-F1:", round(summary[split_name]["macro_f1_mean"], 4),
          "+/-", round(summary[split_name]["macro_f1_std"], 4))

(config.RESULTS_DIR / "berturk_test_summary.json").write_text(json.dumps(summary, indent=2))
