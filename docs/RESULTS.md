# Results in detail

The README shows the main numbers. This file has the longer reading notes. All numbers come from the files in `results/`. How each result was produced and why: `docs/DECISIONS.md`.

## Main comparison (1000 test examples)

Macro-F1 with a 95% bootstrap interval (1000 resamples). Files: `results/comparison_table.csv`, `comparison_table_full_test.csv`, `pairwise_mcnemar.csv`.

| Model | Macro-F1 | 95% interval | Accuracy | Time / example | Cost for 1000 examples |
|---|---|---|---|---|---|
| Majority class (validation set) | 0.415 | n/a | 0.709 | n/a | 0 |
| TF-IDF + LogReg | 0.823 | 0.798 to 0.848 | 0.850 | 0.19 ms (CPU) | 0 (train 15 s) |
| BERTurk (fine-tuned, seed 42) | 0.856 | 0.833 to 0.879 | 0.882 | 18 ms (Apple MPS) | 0 (train 92 min) |
| Gemini 3.6 Flash, 0-shot | 0.843 | 0.819 to 0.867 | 0.866 | ~840 ms (API latency) | $0.13 |
| Gemini 3.6 Flash, 5-shot | 0.846 | 0.821 to 0.870 | 0.869 | ~770 ms (API latency) | $0.38 |

Full test split (8,934 examples): TF-IDF + LogReg 0.816 (0.808 to 0.824), BERTurk 0.855 (0.847 to 0.862).

## How to read this

- **BERTurk vs TF-IDF:** a real difference (McNemar test p = 0.003; on the full test split the intervals do not overlap).
- **Gemini vs the others:** not significant on 1000 examples (p between 0.10 and 0.20). With six pairwise tests, only BERTurk vs TF-IDF is treated as real.
- **0-shot vs 5-shot:** the same (p = 0.71), and 5-shot costs about 3x more input tokens. 0-shot was chosen as the main LLM setting on validation, before looking at test.
- **Speed is not directly comparable.** TF-IDF and BERTurk run locally on one Mac (warm-up excluded, batch size 64 for BERTurk). Gemini speed is API latency and depends on the network. Gemini prices: $0.75 / $3.75 per 1M input / output tokens (checked on 2026-11-14).
- The majority class row is from the validation set (always predicts "positive").
- For the same Turkish text, Gemini's tokenizer needs about 30% more tokens than BERTurk's (2.03 vs 1.56 tokens per word, 100 validation reviews).

## Learning curve (validation only)

Macro-F1 on the validation split when training on the first N rows of a shuffled train split (one seed). Decision record: D-021.

| Training examples | TF-IDF + LogReg | BERTurk | BERTurk training time |
|---|---|---|---|
| 1,000 | 0.739 | 0.803 (15 epochs). 3 epochs gave 0.519: only 96 steps, the model stayed at the majority class | 31 min (15 epochs) |
| 5,000 | 0.783 | 0.826 | 10.5 min |
| 20,000 | 0.793 | 0.843 | 34.5 min |
| 60,294 (full) | 0.812 | 0.852 | 92 min |

- BERTurk is ahead at every size once it gets enough training steps (+6.4, +4.3, +5.0, +4.0 points).
- From 20K to the full train split BERTurk gains 0.9 points for about 2.7x the training time.
- TF-IDF gains a bit more from extra data (5K to 60K: +3.0 vs +2.6) but stays below BERTurk.

## Prompt sensitivity (validation only)

Four wordings of the 0-shot prompt, Gemini 3.6 Flash, the same 200 validation examples. Decision record: D-022. Prompts: `prompts/`.

| Prompt | Macro-F1 | Accuracy | Parse errors |
|---|---|---|---|
| v1 (Turkish, frozen) | 0.879 | 0.905 | 0 |
| v2 (other Turkish wording) | 0.872 | 0.900 | 0 |
| v3 (English instruction) | 0.866 | 0.895 | 0 |
| v4 ("one word, no explanation") | 0.879 | 0.905 | 0 |

The rule was fixed in advance: a variant replaces v1 only if it is better by more than 0.02. None was, so v1 stays. The spread is 0.013, about 2 to 3 examples out of 200, so on this task the wording matters little.

## Error analysis

100 errors were sampled (35 where all three models are wrong, 40 where exactly one is wrong, 25 where two are wrong) from TF-IDF, BERTurk and Gemini 0-shot, and sorted into the categories in `docs/ERROR_TAXONOMY.md`. The category was assigned from the text and the true label, before looking at which model was wrong.

| Category | Count (primary) |
|---|---|
| Mixed sentiment (praise and criticism) | 29 |
| Label noise (text clearly disagrees with label) | 19 |
| Not about quality (plot summary, quote, question) | 17 |
| Needs outside context (other films, other reviews) | 12 |
| Short / low information | 5 |
| Long with sentiment shift | 5 |
| Informal writing | 4 |
| Negation | 4 |
| Other | 4 |
| Irony | 1 |

- Label noise counted as primary or secondary category: 28 of 100, and in 24 of those all three models are wrong. A good part of the shared errors is the label's fault, so no model can reach 100%.
- TF-IDF has more errors than the other two on mixed sentiment and on reviews that need outside context. Gemini has more errors on texts that are not opinions and on very short texts. The counts are small, treat this as a hint only.
- The sample is stratified, so the percentages describe this sample, not all errors.
- Files: `results/error_analysis.csv`, `notebooks/02_error_analysis.ipynb`.
