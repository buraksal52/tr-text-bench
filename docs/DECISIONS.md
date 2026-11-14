# Decisions

Short log. Format: what, why, alternatives, status.
The walkthrough files in `learn/` explain the reasoning in more detail and only link here.

## D-001: Dataset
- Decision: SentiTurca, `movies` config (binary sentiment).
- Why: official splits, binary task, part of a published Turkish benchmark, film reviews give good error analysis material.
- Alternatives: BüyükSinema 10-class (kept as a bonus), Beyazperde-only sets, other Turkish sentiment sets.
- Status: verified in Phase 1 (config `movies` exists; 0 = negative, 1 = positive). See D-013.

## D-002: Splits
- Decision: use the official train / validation / test splits.
- Why: comparable with the benchmark, no extra choices to defend.
- Status: fixed.

## D-003: Fine-tuned model
- Decision: BERTurk, `dbmdz/bert-base-turkish-cased`.
- Why: well-known Turkish BERT, MIT license, ~111M parameters, fine-tunable on a laptop with a subset or on Colab.
- Alternatives: multilingual BERT, other Turkish encoders.
- Status: fixed.

## D-004: LLM
- Decision: Gemini 3.6 Flash (`gemini-3.6-flash`), chosen with the user in Phase 4 (see D-018).
- Why: fast and cheap enough for 1000 examples, no local setup.
- Alternatives: a local open model (bonus in Phase 7).
- Status: fixed (D-018).

## D-005: Primary metric
- Decision: macro-F1, with accuracy reported as well.
- Why: the star-based labels are probably imbalanced.
- Status: fixed.

## D-006: Environment
- Decision: Python virtual environment (`.venv`) with a pinned `requirements.txt`.
- Why: simplest option, no extra tools to learn.
- Status: fixed.

## D-007: Code style
- Decision: simple, boring, beginner-readable code (rules in `CLAUDE.md`).
- Why: the user must understand every step; clever code hides mistakes.
- Status: fixed.

## D-008: Language policy
- Decision: English for code, comments, commits, README and `docs/`. Turkish for `learn/` and chat.
- Status: fixed.

## D-009: Learning notes
- Decision: learning material lives in `learn/`, which is git-ignored.
- Why: keeps the public repo clean and portfolio quality.
- Status: fixed.

## D-010: Project name
- Decision: `trsent-bench` for now.
- Status: may change before publishing.

## D-011: Python version and dependency pinning
- Decision: Python 3.12.10; `requirements.txt` pins only the 8 packages we install directly (pandas, scikit-learn, torch, transformers, datasets, jupyter, python-dotenv, matplotlib).
- Why: simple and readable; sub-dependencies are resolved by pip.
- Alternatives: pin the full `pip freeze` (more exact, much longer file).
- Status: fixed. Re-check in Phase 6 with a fresh environment.

## D-012: Training device
- Decision: use Apple MPS when available (checked in Phase 0: MPS works, torch 2.15.0).
- Why: it is the fastest local option on this machine.
- Alternatives: CPU (slow), Colab (ask first, see Phase 3).
- Status: to be confirmed in Phase 3 (some MPS operations can be slow or missing).

## D-013: Label meaning and label quality
- Decision: `label` 0 = negative, 1 = positive. Source: Hugging Face repo `turkish-nlp-suite/SentiTurca` (a mirror `Setitic/SentiTurca` has identical files). Split sizes match the dataset card: train 60,411 / validation 8,905 / test 8,934.
- Why: checked the card and then read random train examples of both labels. Classes are imbalanced (about 71% positive in every split).
- Known issues: labels are derived from star ratings, so some are noisy (for example a clearly positive text labelled negative). The card's byte sizes for train do not match the real text lengths (mean about 46 words in all splits). We ignore the byte sizes.
- Status: fixed.

## D-014: Cleaning and duplicates
- Decision: (1) drop empty texts (same rule for all splits, only 2 train rows hit); (2) drop train rows whose text also appears in validation or test (58 rows); (3) drop repeated texts inside train, keep the first (57 rows). Train 60,411 -> 60,294. Validation and test are untouched.
- Matching: lowercase and collapsed whitespace, used only to find duplicates. The saved text stays raw.
- Kept on purpose: 5 test rows have the same text as a validation row. Validation and test stay official. Phase 5 can report results with and without them.
- Not cleaned: URLs, `[spoiler]` tags, emojis, very short texts (they are real reviews and carry signal; BERT and LLMs read raw text).
- Why not "keep everything": a few exact repeats between train and test would leak. Why not touch test: it must stay the official split.
- Known limitation: for 17 texts inside train the label conflicts across copies (for example "vasat"). We keep the first copy, which is arbitrary.
- Log: removed rows are listed in `data/processed/cleaning_log.csv` (git-ignored).
- Status: fixed (chosen by the user, option B).

## D-015: LLM subset and few-shot pool
- Decision: LLM subset = 1000 random test ids (seed 42, no stratification, 714 positive / 286 negative). Few-shot pool = 10 train ids (5 per class), drawn from reviews with 10-60 words, in one fixed shuffled order.
- Why: the length range keeps prompts short and cheap. The 5-shot prompt needs 5 of the 10, and an odd number cannot be balanced. How to pick 5 from the pool (for example the first 5 in file order) is decided in Phase 4 and must be logged then.
- Files: `data/llm_subset_ids.json`, `data/fewshot_ids.json` (ids are positions in the official split, `example_id` column).
- Status: subset fixed, few-shot selection to be finalized in Phase 4.

## Majority baseline (validation)
- Always predicts positive: accuracy 0.7087, macro-F1 0.4148. Saved as `results/majority_validation.json`.

## D-016: TF-IDF + Logistic Regression setup and result
- Decision: one sklearn Pipeline: word TF-IDF (`min_df=2`) and character TF-IDF (`char_wb`, 3-5 grams, `min_df=2`) side by side, then Logistic Regression (`max_iter=1000`, seed 42). Shared in `src/tfidf_pipeline.py`.
- Tuning: plain loop over 12 combinations on validation only (`C` in 0.1/1/10, word n-gram max 1/2, `class_weight` none/balanced). `GridSearchCV` was not used, because its inner cross-validation would split train and blur the "tune on validation only" rule. Grid table: `results/tfidf_lr_grid_validation.csv`.
- Best on validation: `C=1`, word n-grams up to 2, `class_weight="balanced"` (macro-F1 0.8124, accuracy 0.840).
- `class_weight="balanced"` raised macro-F1 in all 6 (`C`, n-gram) pairs (+0.03 to +0.04 at `C=0.1`, only +0.002 to +0.004 at `C=10`) and lowered accuracy by 0.006 to 0.017. It helps the minority class (negative).
- Not tested: Turkish-aware lowercasing (default `lowercase=True` was used); `min_df` and the char n-gram range were fixed, not tuned.
- Final test (done once, with the user's permission): macro-F1 0.8162, accuracy 0.8435 on 8,934 examples; 0.8228 / 0.850 on the 1000-example subset. Train time 14.8 s, 0.19 ms per example (batch prediction on CPU, warm-up excluded). Files: `results/tfidf_lr_test*.json` and predictions CSVs.
- Status: fixed. Single run, no seeds to average (Logistic Regression with lbfgs is deterministic for a fixed seed and data).

## D-017: BERTurk fine-tuning setup and result (single seed)
- Decision: `dbmdz/bert-base-turkish-cased` fine-tuned with HF `Trainer` on the cleaned train split (60,294 rows), 3 epochs, learning rate 2e-5, batch size 32, weight decay 0.01, warmup 10%, seed 42. Shared code in `src/berturk_common.py`.
- `max_length=192`: about the 95th percentile of train token lengths (198); 5.3% of reviews are cut. Chosen to keep training time and memory under control on a laptop.
- `train_sampling_strategy="group_by_length"` (batches of similar length, less padding): about 2x faster on MPS. Side effect: batches are less random than plain shuffling.
- Device: Apple MPS (D-012). `accelerate` was added to `requirements.txt` because `Trainer` needs it. Training time: 91.7 min for 3 epochs.
- Checkpoint chosen by validation macro-F1: epoch 1 (0.8519; epochs 2 and 3: 0.8503, 0.8510). Validation loss rose after epoch 1 (0.315 -> 0.345) while train loss kept falling: mild overfitting, so 3 epochs were more than needed.
- Final test (done once, with the user's permission): macro-F1 0.8546, accuracy 0.8801 on 8,934 examples; macro-F1 0.8561, accuracy 0.882 on the 1000-example subset (`results/berturk_test_subset1000_seed42.json`). Prediction 18.4 ms per example (MPS, batch 64, warm-up excluded). Cost 0.
- Compared with TF-IDF + LR (D-016): macro-F1 +0.038, accuracy +0.037, but about 100x slower to predict and about 370x slower to train.
- Seeds: the plan said "3 seeds if time". Seed 43 was trained but the disk became full while saving, the run crashed after epoch 3 (validation macro-F1 0.8535) and was resumed from the epoch-2 checkpoint. The user then decided to stop because the laptop overheated. Seeds 43 and 44 are NOT part of the results. Only seed 42 counts.
- Limitation: one seed, so no standard deviation. The 0.038 gap to TF-IDF is large compared with typical seed noise for BERT fine-tuning (usually a few tenths of a point), but we did not measure it.
- Status: fixed (single seed).

## D-018: LLM setup and result (Gemini 3.6 Flash)
- Decision: `gemini-3.6-flash` through the `google-genai` SDK (`google-genai==3.3.0`, added to `requirements.txt`). Temperature 0, thinking level MINIMAL (a one-word answer does not need reasoning), `max_output_tokens=50`. Prices used (given by the user, 2026-11-14): $0.75 per 1M input tokens, $3.75 per 1M output tokens.
- Prompts: `prompts/fewshot_v1` (0-shot and 5-shot, frozen after its validation run). 5-shot uses the first 5 ids of `data/fewshot_ids.json` (3 positive, 2 negative), fixed order. This closes the open point of D-015.
- Validation (200 random examples, seed 42): 0-shot macro-F1 0.879, 5-shot 0.866, no parse errors. Difference is within noise; 5-shot uses about 3x more input tokens. Rule fixed before the test run: 0-shot is the main result (equal quality, cheaper), 5-shot is reported as secondary. Test results were not used to choose.
- Test, 1000-example subset (done once per prompt, with the user's permission): 0-shot macro-F1 0.8428, accuracy 0.866, cost $0.132, 0 parse errors. 5-shot macro-F1 0.8459, accuracy 0.869, cost $0.382, 0 parse errors. Latency about 0.8 s per example (API and network dependent, not comparable to local timing).
- Same subset, other models: BERTurk 0.8561 / 0.882, TF-IDF + LR 0.8228 / 0.850. Gemini 0-shot is close to TF-IDF, below BERTurk; its extra errors are mostly positive reviews called negative (89 of 714).
- Token comparison (100 validation reviews): BERTurk 74.5 tokens per review (1.56 per word), Gemini 96.7 (2.03 per word). Gemini needs about 30% more tokens for the same Turkish text.
- Process incident: one 5-shot test run started before the price was in `config.py`, so its first report had no cost. All answers are cached, the report was recomputed for free from the cache. The 0-shot run was interrupted once and resumed from the cache without paying twice.
- Limitations: Gemini may have seen these reviews in pretraining; one run per prompt (temperature 0 is not fully deterministic on an API); model name and price are valid for the date above.
- Status: fixed.
