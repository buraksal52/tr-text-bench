import json
import platform
import sys
import time
from datetime import date

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments

import config
from berturk_common import load_tokenized_split

# FINAL evaluation on the test split. Run once per seed, only after the user said yes.
# Usage: python src/03_eval_berturk_test.py [seed]
EVAL_BATCH_SIZE = 64


def save_results(split_name, seed, example_ids, true_labels, predictions, timing, train_seconds):
    accuracy = accuracy_score(true_labels, predictions)
    macro_f1 = f1_score(true_labels, predictions, average="macro")
    precision, recall, _, _ = precision_recall_fscore_support(true_labels, predictions, labels=[0, 1])
    matrix = confusion_matrix(true_labels, predictions, labels=[0, 1])

    print(f"--- {split_name}: {len(true_labels)} examples")
    print("accuracy:", round(accuracy, 4), "| macro-F1:", round(macro_f1, 4))
    print("predicted label counts:", pd.Series(predictions).value_counts().to_dict())
    print("confusion matrix (rows = true, columns = predicted):")
    print(matrix)

    result = {
        "model": "berturk",
        "split": split_name,
        "n_examples": len(true_labels),
        "seed": seed,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "precision_negative": precision[0],
        "precision_positive": precision[1],
        "recall_negative": recall[0],
        "recall_positive": recall[1],
        "time_per_example_ms": timing["total_ms"],
        "tokenize_ms_per_example": timing["tokenize_ms"],
        "model_ms_per_example": timing["model_ms"],
        "eval_batch_size": EVAL_BATCH_SIZE,
        "train_seconds": train_seconds,
        "hardware": f"{platform.machine()}, device mps",
        "max_length": config.BERT_MAX_LENGTH,
        "cost_usd": 0.0,
        "date": date.today().isoformat(),
    }
    (config.RESULTS_DIR / f"berturk_{split_name}_seed{seed}.json").write_text(json.dumps(result, indent=2))
    predictions_df = pd.DataFrame(
        {
            "example_id": example_ids,
            "true_label": true_labels,
            "predicted_label": predictions,
            "model": "berturk",
        }
    )
    predictions_df.to_csv(config.RESULTS_DIR / f"berturk_{split_name}_seed{seed}_predictions.csv", index=False)


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else config.SEED
    best_model_dir = config.CHECKPOINT_DIR / f"berturk_seed{seed}" / "best"
    training_log = json.loads((config.RESULTS_DIR / f"berturk_training_log_seed{seed}.json").read_text())
    subset_ids = json.loads(config.LLM_SUBSET_IDS_FILE.read_text())

    tokenizer = AutoTokenizer.from_pretrained(str(best_model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(best_model_dir))
    args = TrainingArguments(output_dir=str(config.CHECKPOINT_DIR / "eval_tmp"), per_device_eval_batch_size=EVAL_BATCH_SIZE, report_to="none")
    trainer = Trainer(model=model, args=args, data_collator=DataCollatorWithPadding(tokenizer))
    print("device used by the Trainer:", trainer.args.device)

    start_time = time.time()
    test_data = load_tokenized_split("test", tokenizer)
    tokenize_seconds = time.time() - start_time

    # The first call is slower (warm-up), so it is not timed.
    trainer.predict(test_data.select(range(100)))
    start_time = time.time()
    output = trainer.predict(test_data)
    model_seconds = time.time() - start_time
    predictions = output.predictions.argmax(axis=-1)

    n_examples = len(test_data)
    timing = {
        "tokenize_ms": tokenize_seconds * 1000 / n_examples,
        "model_ms": model_seconds * 1000 / n_examples,
    }
    timing["total_ms"] = timing["tokenize_ms"] + timing["model_ms"]
    print("ms per example:", {key: round(value, 2) for key, value in timing.items()})

    example_ids = pd.Series(test_data["example_id"])
    true_labels = pd.Series(test_data["label"])
    train_seconds = training_log["train_seconds"]
    save_results("test", seed, example_ids, true_labels, predictions, timing, train_seconds)

    # Same predictions, only the 1000 examples the LLM will see, so the models are comparable.
    in_subset = example_ids.isin(subset_ids).to_numpy()
    save_results("test_subset1000", seed, example_ids[in_subset], true_labels[in_subset],
                 predictions[in_subset], timing, train_seconds)


main()
