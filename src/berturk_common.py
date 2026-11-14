import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score

import config


def load_tokenized_split(split_name, tokenizer, max_rows=None):
    # Shared by the training and the test scripts, so both prepare text in the same way.
    split_df = pd.read_csv(config.PROCESSED_DIR / f"{split_name}.csv")
    if max_rows is not None:
        split_df = split_df.sample(n=max_rows, random_state=config.SEED)
    dataset = Dataset.from_pandas(split_df[["example_id", "text", "label"]], preserve_index=False)

    def tokenize(batch):
        # No padding here: the data collator pads every batch only to its own longest text.
        return tokenizer(batch["text"], truncation=True, max_length=config.BERT_MAX_LENGTH)

    return dataset.map(tokenize, batched=True)


def compute_metrics(eval_prediction):
    # The Trainer calls this after every epoch on the validation split.
    logits, labels = eval_prediction
    predictions = logits.argmax(axis=-1)
    return {
        "macro_f1": f1_score(labels, predictions, average="macro"),
        "accuracy": accuracy_score(labels, predictions),
    }
