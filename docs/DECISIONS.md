# Decisions

Short log. Format: what, why, alternatives, status.
The walkthrough files in `learn/` explain the reasoning in more detail and only link here.

## D-001: Dataset
- Decision: SentiTurca, `movies` config (binary sentiment).
- Why: official splits, binary task, part of a published Turkish benchmark, film reviews give good error analysis material.
- Alternatives: BüyükSinema 10-class (kept as a bonus), Beyazperde-only sets, other Turkish sentiment sets.
- Status: label meaning and config name still to be verified in Phase 1.

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
