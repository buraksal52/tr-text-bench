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

# Usage: python src/03_finetune_berturk.py [seed] [checkpoint folder to resume from]
# Trains on train, picks the best epoch by VALIDATION macro-F1. Test is never loaded here.
seed = int(sys.argv[1]) if len(sys.argv) > 1 else config.SEED
output_dir = config.CHECKPOINT_DIR / f"berturk_seed{seed}"
# Optional: continue a run that crashed, from the last complete checkpoint.
resume_checkpoint = sys.argv[2] if len(sys.argv) > 2 else None

tokenizer = AutoTokenizer.from_pretrained(config.BERTURK_MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(config.BERTURK_MODEL_NAME, num_labels=2)
train_data = load_tokenized_split("train", tokenizer)
validation_data = load_tokenized_split("validation", tokenizer)
print("train rows:", len(train_data), "| validation rows:", len(validation_data))

training_args = TrainingArguments(
    output_dir=str(output_dir),
    num_train_epochs=config.BERT_EPOCHS,
    learning_rate=config.BERT_LEARNING_RATE,
    per_device_train_batch_size=config.BERT_BATCH_SIZE,
    per_device_eval_batch_size=64,
    weight_decay=0.01,
    # First 10% of the steps raise the learning rate slowly, which keeps the start stable.
    warmup_steps=0.1,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    greater_is_better=True,
    # Puts reviews of similar length in the same batch. Less padding, about 2x faster on MPS.
    train_sampling_strategy="group_by_length",
    logging_steps=100,
    report_to="none",
    seed=seed,
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=validation_data,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics,
)
print("device used by the Trainer:", trainer.args.device)

start_time = time.time()
trainer.train(resume_from_checkpoint=resume_checkpoint)
train_seconds = time.time() - start_time

# The best epoch is already loaded (load_best_model_at_end). Save it as the final model.
best_model_dir = output_dir / "best"
trainer.save_model(str(best_model_dir))
tokenizer.save_pretrained(str(best_model_dir))

validation_metrics = trainer.evaluate()
print("validation metrics of the best epoch:", validation_metrics)

config.RESULTS_DIR.mkdir(exist_ok=True)
log = {
    "seed": seed,
    "train_seconds": train_seconds,
    "device": str(trainer.args.device),
    "best_checkpoint": trainer.state.best_model_checkpoint,
    "validation_metrics": validation_metrics,
    "history": trainer.state.log_history,
}
(config.RESULTS_DIR / f"berturk_training_log_seed{seed}.json").write_text(json.dumps(log, indent=2))
print("train minutes:", round(train_seconds / 60, 1))
