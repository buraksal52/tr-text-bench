# fewshot_v1

Status: frozen (first validation run: 2026-11-14, gemini-3.6-flash). Changes go into `fewshot_v2.md`.

## Rules
- Developed with train and validation examples only. Never with test examples.
- Output must be exactly one word: `olumlu` or `olumsuz`.
- Few-shot examples come from `data/fewshot_ids.json` (train split, fixed seed, balanced between classes) and are always in the same order.
- Which dataset label id means positive is confirmed in Phase 1. The mapping lives in `src/config.py`, not in this file.

## 0-shot prompt

System message:

```
Sen bir duygu analizi sınıflandırıcısısın. Sana bir Türkçe film yorumu verilecek.
Yorumun genel duygusunu belirle.
Sadece şu iki kelimeden birini yaz: olumlu veya olumsuz.
Başka hiçbir şey yazma.
```

User message:

```
Yorum: {text}
Duygu:
```

## 5-shot prompt

Same system message. The five examples come first, as alternating turns:

```
user:      Yorum: {example_text_1}
           Duygu:
assistant: {olumlu | olumsuz}
... (5 examples in total)
user:      Yorum: {text}
           Duygu:
```

## Output parsing
- Lowercase the answer, strip spaces and trailing punctuation.
- Accept only an exact `olumlu` or `olumsuz`.
- Anything else is a parse error. It counts as wrong and is counted in the report.

## Notes
- The review text goes into its own field. If a review contains instructions, the model should still treat it as text to classify.
- Record for every run: model name, date, temperature (0), number of examples, parse error count, token counts, cost.
