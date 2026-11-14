import json
from datetime import date

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

import config

# Majority-class baseline: always predicts the most common train label.
# Every real model has to beat this, otherwise it learned nothing.
train_df = pd.read_csv(config.PROCESSED_DIR / "train.csv")
validation_df = pd.read_csv(config.PROCESSED_DIR / "validation.csv")

model = DummyClassifier(strategy="most_frequent")
# The dummy model ignores the text, but fit() still needs some input.
model.fit(train_df[["text"]], train_df["label"])
predictions = model.predict(validation_df[["text"]])

true_labels = validation_df["label"]
accuracy = accuracy_score(true_labels, predictions)
macro_f1 = f1_score(true_labels, predictions, average="macro", zero_division=0)
precision, recall, _, _ = precision_recall_fscore_support(
    true_labels, predictions, labels=[0, 1], zero_division=0
)
matrix = confusion_matrix(true_labels, predictions, labels=[0, 1])

print("predicted label:", sorted(set(predictions.tolist())))
print("accuracy:", round(accuracy, 4))
print("macro-F1:", round(macro_f1, 4))
print("precision per class [negative, positive]:", precision.round(4).tolist())
print("recall per class [negative, positive]:", recall.round(4).tolist())
print("confusion matrix (rows = true, columns = predicted):")
print(matrix)

config.RESULTS_DIR.mkdir(exist_ok=True)
result = {
    "model": "majority",
    "split": "validation",
    "n_examples": len(validation_df),
    "seed": config.SEED,
    "accuracy": accuracy,
    "macro_f1": macro_f1,
    "precision_negative": precision[0],
    "precision_positive": precision[1],
    "recall_negative": recall[0],
    "recall_positive": recall[1],
    "time_per_example_ms": None,
    "cost_usd": 0.0,
    "date": date.today().isoformat(),
}
result_path = config.RESULTS_DIR / "majority_validation.json"
result_path.write_text(json.dumps(result, indent=2))

predictions_df = pd.DataFrame(
    {
        "example_id": validation_df["example_id"],
        "true_label": true_labels,
        "predicted_label": predictions,
        "model": "majority",
    }
)
predictions_df.to_csv(config.RESULTS_DIR / "majority_validation_predictions.csv", index=False)
print("saved:", result_path)
