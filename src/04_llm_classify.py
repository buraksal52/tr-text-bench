import json
import os
import sys
import time
from datetime import date

import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

import config
from prompt_texts import PROMPTS

# Usage: python src/04_llm_classify.py <validation|test> <zeroshot|fiveshot> <number of examples> [prompt version]
# The prompt version is optional (default: config.PROMPT_VERSION, e.g. fewshot_v2). Texts live in src/prompt_texts.py.
# validation: random sample of that size (seed from config). test: the first N ids of the fixed LLM subset.
# Test is the final evaluation: only run it after the user said yes.
LABEL_WORDS = {config.POSITIVE_LABEL_ID: "olumlu", config.NEGATIVE_LABEL_ID: "olumsuz"}
WORD_TO_LABEL = {"olumlu": config.POSITIVE_LABEL_ID, "olumsuz": config.NEGATIVE_LABEL_ID}


def make_user_text(review_text, prompt_version):
    return PROMPTS[prompt_version]["user"].format(text=review_text)


def build_contents(review_text, mode, fewshot_df, prompt_version):
    contents = []
    if mode == "fiveshot":
        for _, example in fewshot_df.iterrows():
            contents.append(types.Content(role="user", parts=[types.Part(text=make_user_text(example["text"], prompt_version))]))
            contents.append(types.Content(role="model", parts=[types.Part(text=LABEL_WORDS[example["label"]])]))
    contents.append(types.Content(role="user", parts=[types.Part(text=make_user_text(review_text, prompt_version))]))
    return contents


def ask_model(client, contents, prompt_version):
    model_config = types.GenerateContentConfig(
        system_instruction=PROMPTS[prompt_version]["system"],
        temperature=config.LLM_TEMPERATURE,
        max_output_tokens=50,
        # Thinking is not needed for a one-word answer. It would only cost more tokens.
        thinking_config=types.ThinkingConfig(thinking_level="MINIMAL"),
    )
    for attempt in range(1, 5):
        start_time = time.time()
        try:
            response = client.models.generate_content(model=config.LLM_MODEL_NAME, contents=contents, config=model_config)
        except errors.APIError as error:
            print("API error, attempt", attempt, ":", error.code)
            time.sleep(5 * attempt)
            continue
        usage = response.usage_metadata
        return {
            "raw_output": response.text or "",
            "input_tokens": usage.prompt_token_count or 0,
            "output_tokens": (usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0),
            "seconds": time.time() - start_time,
        }
    return {"raw_output": "API_FAILED", "input_tokens": 0, "output_tokens": 0, "seconds": 0.0}


def parse_answer(raw_output):
    cleaned = raw_output.lower().strip().rstrip(".!,;: ")
    return WORD_TO_LABEL.get(cleaned)  # None means parse error


def load_cache(cache_file):
    cache = {}
    if cache_file.exists():
        for line in cache_file.read_text().splitlines():
            row = json.loads(line)
            cache[row["example_id"]] = row
    return cache


def main():
    split_name, mode, n_examples = sys.argv[1], sys.argv[2], int(sys.argv[3])
    prompt_version = sys.argv[4] if len(sys.argv) > 4 else config.PROMPT_VERSION
    load_dotenv(config.PROJECT_ROOT / ".env")
    client = genai.Client(api_key=os.getenv("LLM_API_KEY"))

    if split_name == "test":
        subset_ids = json.loads(config.LLM_SUBSET_IDS_FILE.read_text())[:n_examples]
        split_df = pd.read_csv(config.PROCESSED_DIR / "test.csv").set_index("example_id").loc[subset_ids].reset_index()
    else:
        split_df = pd.read_csv(config.PROCESSED_DIR / "validation.csv").sample(n=n_examples, random_state=config.SEED)

    train_df = pd.read_csv(config.PROCESSED_DIR / "train.csv").set_index("example_id")
    fewshot_ids = json.loads(config.FEWSHOT_IDS_FILE.read_text())[: config.FEWSHOT_USED]
    fewshot_df = train_df.loc[fewshot_ids].reset_index()

    config.LLM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = config.LLM_CACHE_DIR / f"{config.LLM_MODEL_NAME}_{prompt_version}_{mode}_{split_name}.jsonl"
    cache = load_cache(cache_file)
    print("answers already in the cache:", len(cache))

    for _, review in split_df.iterrows():
        example_id = int(review["example_id"])
        if example_id in cache:
            continue
        answer = ask_model(client, build_contents(review["text"], mode, fewshot_df, prompt_version), prompt_version)
        answer["example_id"] = example_id
        cache[example_id] = answer
        with open(cache_file, "a") as file:
            file.write(json.dumps(answer, ensure_ascii=False) + "\n")

    report(split_df, cache, split_name, mode, prompt_version)


def report(split_df, cache, split_name, mode, prompt_version):
    true_labels = split_df["label"].tolist()
    predictions = []
    n_parse_errors = 0
    total_input_tokens = 0
    total_output_tokens = 0
    total_seconds = 0.0
    for _, review in split_df.iterrows():
        answer = cache[int(review["example_id"])]
        total_input_tokens += answer["input_tokens"]
        total_output_tokens += answer["output_tokens"]
        total_seconds += answer["seconds"]
        predicted_label = parse_answer(answer["raw_output"])
        if predicted_label is None:
            # Unparseable output counts as wrong: we store the opposite of the true label.
            n_parse_errors += 1
            predicted_label = 1 - review["label"]
        predictions.append(predicted_label)

    n_examples = len(true_labels)
    accuracy = accuracy_score(true_labels, predictions)
    macro_f1 = f1_score(true_labels, predictions, average="macro")
    print(f"--- {split_name}, {mode}, {n_examples} examples")
    print("accuracy:", round(accuracy, 4), "| macro-F1:", round(macro_f1, 4), "| parse errors:", n_parse_errors)
    print(confusion_matrix(true_labels, predictions, labels=[0, 1]))
    print("input tokens:", total_input_tokens, "| output tokens:", total_output_tokens)

    cost = None
    if config.LLM_PRICE_INPUT_PER_M is not None:
        cost = (total_input_tokens * config.LLM_PRICE_INPUT_PER_M + total_output_tokens * config.LLM_PRICE_OUTPUT_PER_M) / 1_000_000
        print("cost USD:", round(cost, 4))
    print("seconds per example (API latency, depends on the network):", round(total_seconds / n_examples, 2))

    result = {
        "model": config.LLM_MODEL_NAME, "prompt_version": prompt_version, "mode": mode, "split": split_name,
        "n_examples": n_examples, "accuracy": accuracy, "macro_f1": macro_f1, "parse_errors": n_parse_errors,
        "input_tokens": total_input_tokens, "output_tokens": total_output_tokens, "cost_usd": cost,
        "price_input_per_m": config.LLM_PRICE_INPUT_PER_M, "price_output_per_m": config.LLM_PRICE_OUTPUT_PER_M,
        "price_date": config.LLM_PRICE_DATE, "temperature": config.LLM_TEMPERATURE,
        "seconds_per_example": total_seconds / n_examples, "date": date.today().isoformat(),
    }
    name = f"llm_{config.LLM_MODEL_NAME}_{mode}_{split_name}_n{n_examples}"
    if prompt_version != config.PROMPT_VERSION:
        name = name + "_" + prompt_version  # keeps the v1 result files untouched
    (config.RESULTS_DIR / f"{name}.json").write_text(json.dumps(result, indent=2))
    predictions_df = pd.DataFrame({"example_id": split_df["example_id"].tolist(), "true_label": true_labels, "predicted_label": predictions})
    predictions_df.to_csv(config.RESULTS_DIR / f"{name}_predictions.csv", index=False)


main()
