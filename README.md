# trsent-bench

> Status: work in progress. This file is a skeleton and will be completed in Phase 6.

**Question:** For Turkish sentiment classification, how do TF-IDF + Logistic Regression, a fine-tuned BERTurk model and a prompted LLM compare in accuracy, macro-F1, speed, cost and the kinds of errors they make?

## Results

_To be filled in Phase 5._

| Model | Accuracy | Macro-F1 | Time / example | Cost / example |
|---|---|---|---|---|
| Majority class | | | | |
| TF-IDF + LogReg | | | | |
| BERTurk (fine-tuned) | | | | |
| LLM 0-shot | | | | |
| LLM 5-shot | | | | |

## Error analysis

_To be filled in Phase 5._

## Setup and reproduction

_To be filled in Phase 6 (tested from a fresh virtual environment)._

## Project structure

_To be filled in Phase 6._

## Limitations

_To be filled in Phase 6._

## Data and model attribution

- Data: SentiTurca (`movies` config), turkish-nlp-suite, https://huggingface.co/datasets/turkish-nlp-suite/SentiTurca (the BüyükSinema dataset it is based on is CC BY-SA 4.0; check the SentiTurca card for its own license).
- Model: BERTurk, `dbmdz/bert-base-turkish-cased`, MIT license, https://huggingface.co/dbmdz/bert-base-turkish-cased
