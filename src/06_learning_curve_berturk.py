import json
import sys
import time

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

import config
from berturk_common import compute_metrics, load_tokenized_split
from learning_curve_common import get_train_subset

# Usage: python src/06_learning_curve_berturk.py <number of train examples> [number of epochs]
# Same settings as 03_finetune_berturk.py (3 epochs, seed 42), but trained on a smaller train subset.
# Validation only. No checkpoint is saved. The best epoch is the one with the highest validation macro-F1,
# like in the full run.
n_examples = int(sys.argv[1])
# A small subset has few steps per epoch, so more epochs can be needed to give the model enough steps.
n_epochs = int(sys.argv[2]) if len(sys.argv) > 2 else config.BERT_EPOCHS
run_name = f"n{n_examples}" if n_epochs == config.BERT_EPOCHS else f"n{n_examples}_epochs{n_epochs}"
subset_ids = get_train_subset(n_examples)["example_id"]

tokenizer = AutoTokenizer.from_pretrained(config.BERTURK_MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(config.BERTURK_MODEL_NAME, num_labels=2)
train_data = load_tokenized_split("train", tokenizer)
# Keep only the rows of the subset. example_id identifies a row, the row position is not the same number.
subset_id_set = set(subset_ids)
positions = []
for position, example_id in enumerate(train_data["example_id"]):
    if example_id in subset_id_set:
        positions.append(position)
train_data = train_data.select(positions)
validation_data = load_tokenized_split("validation", tokenizer)
print("train rows:", len(train_data), "| validation rows:", len(validation_data))

training_args = TrainingArguments(
    output_dir=str(config.CHECKPOINT_DIR / f"learning_curve_{run_name}"),
    num_train_epochs=n_epochs,
    learning_rate=config.BERT_LEARNING_RATE,
    per_device_train_batch_size=config.BERT_BATCH_SIZE,
    per_device_eval_batch_size=64,
    weight_decay=0.01,
    warmup_steps=0.1,
    eval_strategy="epoch",
    save_strategy="no",
    train_sampling_strategy="group_by_length",
    logging_steps=100,
    report_to="none",
    seed=config.SEED,
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=validation_data,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics,
)

start_time = time.time()
trainer.train()
train_seconds = time.time() - start_time

epoch_rows = []
for entry in trainer.state.log_history:
    if "eval_macro_f1" in entry:
        epoch_rows.append({"epoch": entry["epoch"], "val_macro_f1": entry["eval_macro_f1"], "val_accuracy": entry["eval_accuracy"]})
best_row = max(epoch_rows, key=lambda row: row["val_macro_f1"])
result = {"model": "berturk", "n_train": n_examples, "seed": config.SEED, "n_epochs": n_epochs, "train_seconds": train_seconds,
          "val_macro_f1": best_row["val_macro_f1"], "val_accuracy": best_row["val_accuracy"],
          "best_epoch": best_row["epoch"], "epochs": epoch_rows}
(config.RESULTS_DIR / f"learning_curve_berturk_{run_name}.json").write_text(json.dumps(result, indent=2))
print(result)
