# Limitations

Read these before quoting a number from this project.

- **One dataset, one domain:** movie reviews only. Results may not transfer to other Turkish text.
- **One seed for BERTurk.** A 3-seed run was planned, but one run crashed when the disk was full and the rest was stopped. No standard deviation over seeds is reported. The one other seed that reached the end of training (validation macro-F1 0.8535 vs 0.8519 for seed 42) suggests the seed effect is small, but it was not evaluated on test. That checkpoint was deleted afterwards.
- **Subset of 1000 test examples** for the model comparison, so the intervals are wide (about ±0.025). Small differences cannot be ranked.
- **One LLM, one run.** Gemini output through an API is not fully deterministic even at temperature 0. The prompt was chosen on 200 validation examples only. The prompt sensitivity check used 3 extra variants, 0-shot only.
- **The learning curve is one seed and validation only.** At 1K and 5K examples the seed effect is probably large. The best BERTurk epoch was picked on validation (slightly optimistic), and the 1K point uses 15 epochs instead of 3.
- **The LLM may have seen these reviews during pretraining.** We cannot check this.
- **Weak labels:** star ratings, not human sentiment judgments. In the error sample, label noise appears in about 28% of the examples.
- **Error analysis:** one labeler (an AI assistant), no agreement score, stratified sample of 100, and long reviews were read only up to about 350 characters.
- **Train overlap removal:** removing train rows that also appear in validation or test used test texts for that one decision (`docs/DECISIONS.md`, D-014).
- **Speed numbers** come from one Mac and one network connection.
- **Not done:** calibration of LLM confidence (Gemini 3.6 Flash does not return token probabilities), a small local open model, a 10-class version.
