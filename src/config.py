from pathlib import Path

# All settings of the project live here and nowhere else.

SEED = 42

# Paths are built from the project root, so scripts work from any folder.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
PROMPTS_DIR = PROJECT_ROOT / "prompts"

LLM_SUBSET_IDS_FILE = DATA_DIR / "llm_subset_ids.json"
FEWSHOT_IDS_FILE = DATA_DIR / "fewshot_ids.json"

# Dataset (verified in Phase 1: config "movies" exists, 0 = negative, 1 = positive)
DATASET_NAME = "turkish-nlp-suite/SentiTurca"
DATASET_CONFIG = "movies"
NEGATIVE_LABEL_ID = 0
POSITIVE_LABEL_ID = 1

# Models
BERTURK_MODEL_NAME = "dbmdz/bert-base-turkish-cased"

# LLM subset size (see docs/EVAL_PROTOCOL.md)
LLM_SUBSET_SIZE = 1000

# Few-shot pool: examples per class and a word range, so that very short
# or very long reviews do not blow up the prompt
FEWSHOT_PER_CLASS = 5
FEWSHOT_MIN_WORDS = 10
FEWSHOT_MAX_WORDS = 60

# BERTurk fine-tuning (max_length: the 95th percentile of train is ~198 tokens, see 03_check_token_lengths.py)
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
BERT_MAX_LENGTH = 192
BERT_LEARNING_RATE = 2e-5
BERT_EPOCHS = 3
BERT_BATCH_SIZE = 32

# LLM (Phase 4). Chosen together with the user, see D-004 / D-018.
LLM_MODEL_NAME = "gemini-3.6-flash"
LLM_TEMPERATURE = 0
# Prompt version is part of every cache entry, so a new prompt never reuses old answers.
PROMPT_VERSION = "fewshot_v1"
# 5-shot uses the first 5 ids of data/fewshot_ids.json (file order is fixed): 3 positive, 2 negative.
FEWSHOT_USED = 5
LLM_CACHE_DIR = RESULTS_DIR / "llm_cache"
# Prices in USD per 1M tokens. Copy them from the official pricing page and write the date.
# None means "not filled in yet": the script then prints tokens but no cost.
LLM_PRICE_INPUT_PER_M = 0.75
LLM_PRICE_OUTPUT_PER_M = 3.75
LLM_PRICE_DATE = "2026-11-14"
