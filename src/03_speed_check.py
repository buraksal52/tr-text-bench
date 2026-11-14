import time

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

import config
from berturk_common import load_tokenized_split

# Short speed test before the real training: 30 training steps on 1000 train reviews.
# It only measures time per step. The model is thrown away afterwards.
STEPS = 30

tokenizer = AutoTokenizer.from_pretrained(config.BERTURK_MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(config.BERTURK_MODEL_NAME, num_labels=2)
small_train = load_tokenized_split("train", tokenizer, max_rows=1000)

training_args = TrainingArguments(
    output_dir=str(config.CHECKPOINT_DIR / "speed_check"),
    per_device_train_batch_size=config.BERT_BATCH_SIZE,
    learning_rate=config.BERT_LEARNING_RATE,
    max_steps=STEPS,
    save_strategy="no",
    report_to="none",
    train_sampling_strategy="group_by_length",
    seed=config.SEED,
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train,
    data_collator=DataCollatorWithPadding(tokenizer),
)
print("device used by the Trainer:", trainer.args.device)

start_time = time.time()
trainer.train()
seconds = time.time() - start_time

seconds_per_step = seconds / STEPS
steps_per_epoch = 60294 / config.BERT_BATCH_SIZE
print(f"{seconds_per_step:.2f} s per step (includes start-up, so a little pessimistic)")
print(f"estimated full train, 1 epoch: {seconds_per_step * steps_per_epoch / 60:.0f} min")
print(f"estimated full train, {config.BERT_EPOCHS} epochs: {seconds_per_step * steps_per_epoch * config.BERT_EPOCHS / 60:.0f} min")
