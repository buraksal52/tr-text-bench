# fewshot_v2

Status: run once on 200 validation examples (0-shot, 2026-11-15); did not beat v1 (D-022). Same task as v1, different Turkish wording and a different user message format.

## Rules
- Developed with train and validation examples only. Never with test examples.
- Output must be exactly one word: `olumlu` or `olumsuz`.
- Same label words, same parsing and same few-shot examples as `fewshot_v1.md`. Only the wording below changes.
- The texts below are copied into `src/prompt_texts.py`.

## System message

```
Aşağıda bir Türkçe film yorumu var. Yorumu yazan kişi filmi beğenmiş mi beğenmemiş mi?
Cevabın yalnızca "olumlu" ya da "olumsuz" olsun.
```

## User message

```
Film yorumu: {text}
Cevap:
```
