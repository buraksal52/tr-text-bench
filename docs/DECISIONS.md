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
- Decision: API model, to be chosen in Phase 4 together with the user.
- Why: fast and cheap enough for 1000 examples, no local setup.
- Alternatives: a local open model (bonus in Phase 7).
- Status: open.

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
