# fewshot_v4

Status: run once on 200 validation examples (0-shot, 2026-11-15); did not beat v1 (D-022). Same task as v1, with a strong 'one word, no explanation' emphasis.

## Rules
- Developed with train and validation examples only. Never with test examples.
- Output must be exactly one word: `olumlu` or `olumsuz`.
- Same label words, same parsing and same few-shot examples as `fewshot_v1.md`. Only the wording below changes.
- The texts below are copied into `src/prompt_texts.py`.

## System message

```
Görevin Türkçe film yorumlarını sınıflandırmak.
Cevabın TEK KELİME olmalı: olumlu veya olumsuz.
Açıklama yapma, gerekçe verme, noktalama işareti ekleme, başka kelime yazma.
Yorum karışık duygular içerse bile baskın duyguya göre tek kelime seç.
```

## User message

```
Yorum: {text}
Tek kelimelik cevap:
```
