# Plan

Every phase starts and ends with user approval (see `CLAUDE.md`).
Order matters, dates are flexible. Each session is about 2-3 hours.

| Session | Planned date | Phase |
|---|---|---|
| S0 | Sat 14 Nov 2026 | Phase 0, setup |
| S1 | Sun 15 Nov | Phase 1, data |
| S2 | Mon-Tue 16-17 Nov | Phase 2, baseline |
| S3 | Wed-Thu 18-19 Nov | Phase 3, BERTurk (most likely to slip) |
| S4 | Fri 20 Nov | Phase 4, LLM |
| S5 | Sat 21 Nov | Phase 5, comparison and error analysis |
| S6 | Sun 22 Nov | Phase 6, publish |

## Phase 0: Setup
- [x] Create the folder structure: `src/`, `data/`, `notebooks/`, `results/`, `prompts/`, `docs/`
- [x] Python virtual environment (Python 3.11+), pinned `requirements.txt`
- [x] `src/config.py` with seed, paths, model names
- [x] `.env.example` with an API key placeholder; `.env` stays out of git
- [x] Check which device PyTorch can use (CPU / MPS / CUDA)
- [x] First commit
Done when: environment runs, `config.py` imports, repo is committed.

## Phase 1: Data
- [x] Load SentiTurca `movies` config; print columns, label names, class counts per split
- [x] Verify label meaning from the dataset card (how many classes, which id is positive). The cards looked inconsistent, so check by reading examples
- [x] EDA notebook: text length, empty or very short texts, class balance, duplicates
- [x] Cleaning only if justified. One step at a time, row counts printed, same cleaning for all splits
- [x] Duplicate check inside and across splits; decide and log it
- [x] Save processed splits to `data/processed/`
- [x] Draw the LLM subset (1000 from test) to `data/llm_subset_ids.json`
- [x] Draw the few-shot pool from train to `data/fewshot_ids.json`
- [x] Majority-class baseline on validation
Done when: label meaning is confirmed and logged, splits are saved, id files exist.

## Phase 2: Baseline (TF-IDF + Logistic Regression)
- [x] Word n-gram and character n-gram TF-IDF, Logistic Regression, in one sklearn pipeline
- [x] Small grid search on validation only (`C`, n-gram range)
- [x] Final test evaluation (ask first): accuracy, macro-F1, per-class precision/recall, confusion matrix, time per example
- [x] Save `results/` JSON and predictions CSV
Done when: results are saved and the leakage checklist in `EVAL_PROTOCOL.md` passes.

## Phase 3: BERTurk fine-tuning
- [x] Check token length distribution and choose `max_length`
- [x] Fine-tune `dbmdz/bert-base-turkish-cased` with Hugging Face `Trainer`
- [x] Starting values: learning rate 2e-5 to 5e-5, 2-3 epochs, batch size 16 or 32
- [x] Use the full train split if time allows, otherwise a subset (record which)
- [x] Pick the checkpoint by validation macro-F1
- [x] Test evaluation (ask first) and timing
- [ ] If time: 3 seeds (not done, 1 seed only, see D-017)
- [x] If the local device is too slow: ask before moving to Colab
Done when: checkpoint chosen, test results saved, training time recorded.

## Phase 4: LLM few-shot
- [x] Choose the API model together with the user and log it in `DECISIONS.md`
- [x] Finalize `prompts/fewshot_v1.md`; 0-shot and 5-shot
- [x] Dry run on 20 validation examples to check parsing and cost
- [x] Ask before the 1000-example run (state number of calls and estimated cost)
- [x] Cache all responses; count parse errors
- [x] Log tokens and cost
- [x] Mini analysis: token counts of BERTurk tokenizer vs the LLM tokenizer on the same sentences
Done when: results on the 1000-example subset are saved with cost and parse error count.

## Phase 5: Comparison and error analysis
- [x] One table: all models on the same 1000-example subset; classical and BERTurk also on the full test split
- [x] Bootstrap confidence intervals and McNemar test
- [x] Error analysis notebook following `docs/ERROR_TAXONOMY.md` (about 100 examples)
- [x] 2-3 plots
Done when: table, plots and categorized errors are saved in `results/`.

## Phase 6: Publish
- [ ] Complete README (question, results, errors, limitations, attribution)
- [ ] Re-run the instructions from a fresh virtual environment
- [ ] Tag `v1.0`
Done when: a stranger can reproduce the main table from the README.

## Phase 7: Bonus (optional, only after Phase 6)
- [ ] Prompt sensitivity: 3-4 prompt variants on validation, then the best on the subset
- [ ] Calibration of LLM confidence (only if the API gives token probabilities)
- [ ] A small local open model on the same subset
- [ ] Learning curve: classical vs BERTurk with 1K / 5K / 20K / full train
- [ ] 10-class version (BüyükSinema)
