from __future__ import annotations

import json
import os
from pathlib import Path
from collections import Counter

import torch
from peft import PeftModel
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoModelForCausalLM, AutoTokenizer


BASE_MODEL = os.getenv(
    "ROUTER_BASE_MODEL",
    "Qwen/Qwen2.5-0.5B-Instruct",
)

ADAPTER_DIR = os.getenv(
    "ROUTER_SFT_OUTPUT_DIR",
    "models/router-qwen2.5-0.5b-lora",
)

TEST_META_PATH = Path("data/router/sft/test_with_meta.jsonl")

INTENTS = ["order", "consultant", "faq", "ignore"]


SYSTEM_PROMPT = """
You are an intent classification model for a coffee shop assistant.

Classify the user message into EXACTLY one intent:

- order
- consultant
- faq
- ignore

Rules:
- order:
    direct ordering or buying menu items
- consultant:
    asking recommendation, suggestion, preference
- faq:
    asking store information, wifi, payment, opening hour, delivery
- ignore:
    greeting, nonsense, unrelated, filler

Return ONLY the intent label.
""".strip()


def load_jsonl(path: Path) -> list[dict]:
    assert path.exists(), f"File not found: {path}"

    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    return rows


def parse_intent(text: str) -> str:
    text = text.strip().lower()

    for intent in INTENTS:
        if intent in text:
            return intent

    return "parse_error"


def build_prompt(tokenizer, text: str) -> str:
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": text,
        },
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def predict(model, tokenizer, text: str) -> str:
    prompt = build_prompt(tokenizer, text)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=8,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
    decoded = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True,
    )

    return parse_intent(decoded)


def main():
    rows = load_jsonl(TEST_META_PATH)

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(
        base_model,
        ADAPTER_DIR,
    )

    model.eval()

    y_true = []
    y_pred = []

    difficulty_total = Counter()
    difficulty_correct = Counter()

    language_total = Counter()
    language_correct = Counter()

    errors = []

    for sample in rows:
        text = sample["text"]
        true_intent = sample["intent"]

        pred_intent = predict(model, tokenizer, text)

        y_true.append(true_intent)
        y_pred.append(pred_intent)

        difficulty = sample.get("difficulty", "unknown")
        language = sample.get("language", "unknown")

        difficulty_total[difficulty] += 1
        language_total[language] += 1

        if pred_intent == true_intent:
            difficulty_correct[difficulty] += 1
            language_correct[language] += 1
        else:
            errors.append(
                {
                    "text": text,
                    "true": true_intent,
                    "pred": pred_intent,
                    "difficulty": difficulty,
                    "language": language,
                }
            )

    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = correct / len(y_true)

    print("=" * 80)
    print("ROUTER SFT EVALUATION")
    print("=" * 80)
    print(f"Base model: {BASE_MODEL}")
    print(f"Adapter: {ADAPTER_DIR}")
    print(f"Test samples: {len(rows)}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

    print("\nCLASSIFICATION REPORT")
    print("-" * 80)
    print(
        classification_report(
            y_true,
            y_pred,
            labels=INTENTS,
            zero_division=0,
        )
    )

    print("\nCONFUSION MATRIX")
    print("-" * 80)
    print("labels:", INTENTS)
    print(
        confusion_matrix(
            y_true,
            y_pred,
            labels=INTENTS,
        )
    )

    print("\nDIFFICULTY ACCURACY")
    print("-" * 80)
    for difficulty, total in difficulty_total.items():
        correct = difficulty_correct[difficulty]
        print(f"{difficulty}: {correct}/{total} = {correct / total:.2%}")

    print("\nLANGUAGE ACCURACY")
    print("-" * 80)
    for language, total in language_total.items():
        correct = language_correct[language]
        print(f"{language}: {correct}/{total} = {correct / total:.2%}")

    print("\nERROR SAMPLES")
    print("-" * 80)
    for idx, err in enumerate(errors[:30], start=1):
        print(
            f"{idx}. [{err['language']} | {err['difficulty']}] "
            f"true={err['true']} pred={err['pred']} | {err['text']}"
        )

    print(f"\nTotal errors: {len(errors)}")


if __name__ == "__main__":
    main()