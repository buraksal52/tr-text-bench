# CLAUDE.md

Project: **trsent-bench**, a Turkish sentiment classification benchmark.
Question: on the same data and the same metrics, how do (1) TF-IDF + Logistic Regression,
(2) fine-tuned BERTurk and (3) a prompted LLM compare in accuracy, macro-F1, speed, cost and error types?

This is a learning/warm-up project. The user is a CS student who wants to understand every step.
Simple and readable beats clever and short.

## Read before working
- `docs/PLAN.md` (read only the current phase)
- `docs/EVAL_PROTOCOL.md` (the rules; never change without asking)
- `docs/DECISIONS.md` (decisions so far)
- Phase 4 only: `prompts/`. Phase 5 only: `docs/ERROR_TAXONOMY.md`.
- Follow these documents. If reality differs from them, tell the user. Do not silently change the plan.

## Workflow and permissions
1. Start of a phase: give a plan summary (max 10 lines) and wait for approval.
2. Inside a phase: work without asking, in small commits.
3. Always ask first before:
   - calling a paid API in bulk (say how many calls and the estimated cost)
   - downloads over ~500 MB
   - any evaluation on the test split
   - deleting files
   - installing packages that are not in the plan
   - touching `docs/EVAL_PROTOCOL.md` or the data splits
4. Before a key experiment, ask the user to guess the result (for example the expected macro-F1). Chat only. Do not write guesses into files.
5. End of a phase, in this order:
   a. Write `learn/walkthrough/F<n>.md` from `learn/walkthrough/_TEMPLATE.md`.
   b. Tick the boxes in `docs/PLAN.md` and add entries to `docs/DECISIONS.md`.
   c. Commit and tag (`phase-<n>`).
   d. Explain the phase to the user in Turkish, using the walkthrough as the base.
   e. Ask 3 short check questions in chat (the user may skip them).
   f. Ask for approval to start the next phase.
6. If you meet a concept a junior must know and it is missing in `learn/CONCEPTS.md`, add it there.

## Code style (important)
Write boring, beginner-readable code. The user does not want clever or "technical" code, even for simple tasks like data cleaning.
- Plain Python with pandas, scikit-learn and standard Hugging Face APIs (`datasets`, `transformers`, `Trainer`). Use library defaults. Do not re-implement what a library already does (splits, metrics, tokenization).
- Not allowed: long one-liners, nested comprehensions, chained `.apply(lambda ...)` pipelines, decorators, metaclasses, `functools` tricks, custom classes unless a library requires one, abstractions "for later".
- Data cleaning: a few separate steps, one thing per step, plain pandas, print row counts before and after each step.
- One variable per idea, descriptive names (`train_df`, not `d`).
- Functions: aim for 30 lines or fewer. Scripts: aim for 150 lines or fewer. If a file grows beyond that, stop and ask before restructuring.
- Comments explain why, not what.
- Minimal error handling: retry on API calls, a clear message for missing files. No broad `try/except`.
- No new dependencies without asking.
- When two options work, choose the more boring one.
- Settings (seed, paths, model names) live only in `src/config.py`.
- Layout: `src/` has numbered scripts per phase (`01_prepare_data.py`, ...). Notebooks are only for EDA (phase 1) and error analysis (phase 5). Do not duplicate code between scripts and notebooks.

## Evaluation rules (details in docs/EVAL_PROTOCOL.md)
- Use the official train/validation/test splits. Tune only on validation.
- Prompts are developed on train/validation only.
- Primary metric is macro-F1.
- Cache every API response on disk. Log model name, date, tokens and cost.
- Unparseable LLM output counts as wrong and is reported.
- If a result looks too good or too bad, assume a bug first and run the sanity checks in `learn/CONCEPTS.md`.

## Repo rules
- Never commit `.env`, raw data, model weights or checkpoints.
- `learn/` is private and git-ignored. Learning notes never go into README, docs or code comments.
- English for code, comments, commits, README and `docs/`. Turkish for `learn/` and for chat with the user.
- Chat tone: casual, clear, no lecturing. Explain the why.
