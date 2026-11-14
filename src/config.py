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

# Dataset (config name and label meaning are verified in Phase 1)
DATASET_NAME = "SentiTurca"
DATASET_CONFIG = "movies"

# Models
BERTURK_MODEL_NAME = "dbmdz/bert-base-turkish-cased"

# LLM subset size (see docs/EVAL_PROTOCOL.md)
LLM_SUBSET_SIZE = 1000
