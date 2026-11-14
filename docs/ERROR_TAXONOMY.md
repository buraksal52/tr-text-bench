# Error taxonomy

Used in Phase 5 to sort model mistakes into categories. Define the categories first, then label. The point is to see *why* models fail, not only how often.

## Categories
Pick one primary category per example. A secondary one is optional.

| Code | Category | Meaning | Example idea |
|---|---|---|---|
| NEG | Negation | Meaning flips because of a negation | "Kötü değil" |
| IRO | Irony / sarcasm | Positive words, negative meaning (or the opposite) | "Harika bir film, tam bir uyku ilacı." |
| MIX | Mixed sentiment | Praise and criticism in one review; the label depends on the balance | "Oyunculuk güzel ama senaryo çok zayıf." |
| NOI | Label noise | The text clearly disagrees with the label. Decide from the text alone | Very negative text, positive label |
| SHT | Short / low information | About 5 words or fewer and no clear sentiment cue | "Bir ara bakılır." |
| OFF | Not about quality | Plot summary, question, talk about actors or the cinema, not an opinion | "Filmi dün akşam izledik, konusu şöyle..." |
| REF | Needs outside context | Depends on other films, a previous part or a known debate | "Birincisinin yanına bile yaklaşamıyor." |
| INF | Informal writing | Slang, typos, missing diacritics, emojis, abbreviations | "cok guzeldi yaw" |
| LNG | Long with shift | Long review where the sentiment changes along the text | |
| OTH | Other | Describe in the notes column | |

## Procedure
1. Take errors from the final test predictions of each model.
2. Sample with a fixed seed: about 100 examples in total, covering (a) all three models wrong, (b) exactly one model wrong, and (c) the rest.
3. Read the text and assign the category before looking at which model was wrong.
4. Save to `results/error_analysis.csv` with columns: example_id, text, true_label, category, secondary_category, notes, and one correct/wrong column per model.
5. Report counts and percentages per category and per model.
6. Do not change the categories after you start. If a new category is really needed, add it, then re-check all examples.
7. Label noise is judged conservatively. Only mark NOI when the text is unambiguous.
