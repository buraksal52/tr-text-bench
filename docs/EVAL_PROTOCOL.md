# Evaluation protocol

These rules keep the comparison fair. Changing them needs the user's approval and a `DECISIONS.md` entry.

## 1. Data
- Dataset: SentiTurca, `movies` config (binary). Verify config name, columns and label meaning in Phase 1 and record it in `DECISIONS.md`.
- Use the official train / validation / test splits. Never re-split.
- Same cleaning for all splits and all models. List removed rows in the log.
- Check exact and near duplicates inside and across splits. Decide what to do and record it.
- Anything that learns from data (TF-IDF, scalers) is fitted on train only.

## 2. Test split discipline
- Validation is for every choice: hyperparameters, checkpoints, prompts, thresholds.
- Test is used once per final model, by the final eval script only, after asking the user.
- Do not read test examples while developing. Error analysis (Phase 5) happens after all test results are fixed.

## 3. LLM subset
- 1000 examples sampled from test with the seed in `config.py`. Ids saved in `data/llm_subset_ids.json`.
- Few-shot examples come from train. Ids saved in `data/fewshot_ids.json`. Same examples, same order, every run.
- In the final table every model is also reported on this same 1000-example subset.

## 4. Metrics
- Primary: macro-F1.
- Also: accuracy, precision and recall per class, confusion matrix.
- Always show the majority-class baseline next to the models.
- Uncertainty: bootstrap 95% confidence interval for macro-F1, and McNemar test for paired comparisons on the same examples. Do not call a difference a win when the intervals overlap heavily.
- BERTurk: mean and standard deviation over 3 seeds if time allows. Otherwise one seed, and say so.

## 5. Speed and cost
- Speed: inference time per example. Same machine for classical and BERTurk, warm-up run excluded, batch size and hardware noted.
- LLM speed is API latency and depends on the network. Report it separately and say so.
- BERTurk training time is reported.
- Cost of the LLM: input tokens x price + output tokens x price. Record the price and the date.
- Local models have no API cost, but training time and hardware are reported.
- Tokenization: on a sample of sentences, compare token counts of the BERTurk tokenizer and the LLM tokenizer.

## 6. LLM rules
- Prompt versions live in `prompts/`. A version is never edited after its first validation run. Changes become a new version.
- Temperature 0. Record model name, date and any API version information.
- Output must be exactly one allowed label word. Anything else is a parse error and counts as wrong. Report the parse error count.
- Cache every response (prompt version, example id, raw output) on disk. Never pay twice for the same call.
- The LLM may have seen these reviews during pretraining. State this as a limitation.

## 7. Results format
- One JSON per run in `results/`: model, split, n_examples, seed, accuracy, macro_f1, per-class precision and recall, time_per_example_ms, cost_usd, date.
- Predictions as CSV: example_id, true_label, predicted_label, model.
- File names: `<model>_<split>[_<seed>].json`.

## 8. Leakage checklist (run before every final evaluation)
- [ ] No test example was used in training, tuning, prompts or few-shot.
- [ ] Vectorizer fitted on train only.
- [ ] No duplicates across splits (or they are handled and logged).
- [ ] Label mapping verified (which id is positive).
- [ ] Prediction distribution looks sane (not all one class).
