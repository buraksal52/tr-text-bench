import os

import pandas as pd
from dotenv import load_dotenv
from google import genai
from transformers import AutoTokenizer

import config

# Mini analysis: how many tokens does the same review need for BERTurk and for Gemini?
# Uses 100 validation reviews and the (free) count_tokens endpoint. No test data.
SAMPLE_SIZE = 100

load_dotenv(config.PROJECT_ROOT / ".env")
client = genai.Client(api_key=os.getenv("LLM_API_KEY"))
bert_tokenizer = AutoTokenizer.from_pretrained(config.BERTURK_MODEL_NAME)
sample_df = pd.read_csv(config.PROCESSED_DIR / "validation.csv").sample(n=SAMPLE_SIZE, random_state=config.SEED)

bert_counts = []
gemini_counts = []
for review_text in sample_df["text"]:
    # No special tokens ([CLS], [SEP]) so both sides count only the review itself.
    bert_counts.append(len(bert_tokenizer(review_text, add_special_tokens=False)["input_ids"]))
    answer = client.models.count_tokens(model=config.LLM_MODEL_NAME, contents=review_text)
    gemini_counts.append(answer.total_tokens)

word_counts = sample_df["text"].str.split().str.len()
print("reviews:", SAMPLE_SIZE)
print("mean words per review:", round(word_counts.mean(), 1))
print("mean BERTurk tokens:", round(sum(bert_counts) / SAMPLE_SIZE, 1))
print("mean Gemini tokens:", round(sum(gemini_counts) / SAMPLE_SIZE, 1))
print("tokens per word, BERTurk:", round(sum(bert_counts) / word_counts.sum(), 2))
print("tokens per word, Gemini:", round(sum(gemini_counts) / word_counts.sum(), 2))
