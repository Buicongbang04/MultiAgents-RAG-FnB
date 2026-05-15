from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = os.getenv("INTENT_EXTRACTOR_BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
ADAPTER_DIR = os.getenv(
    "INTENT_EXTRACTOR_ADAPTER_DIR",
    "models/intent-extractor-qwen2.5-0.5b-lora",
)

TEST_META = Path("data/intent_extraction/sft/test_with_meta.jsonl")

VALID_INTENTS = {"order", "consultant", "faq", "ignore"}
VALID_LANGUAGES = {"vi", "en"}

FIELDS = ["subject", "action", "context", "cache_key", "intent", "language"]

# Các field thật sự ảnh hưởng Intelligent Cache.
CACHE_RELEVANT_FIELDS = ["action", "context", "cache_key", "intent", "language"]

SYSTEM_PROMPT = (
    "You are an intent extraction model for an F&B multi-agent assistant. "
    "Extract structured fields from the customer query. "
    "Return valid JSON only. "
    "Do not add markdown. "
    "Required fields: subject, action, context, cache_key, intent, language. "
    "intent must be one of: order, consultant, faq, ignore. "
    "language must be vi or en."
)


def load_test_rows() -> List[Dict[str, Any]]:
    rows = []
    with TEST_META.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    value = str(value).strip().lower()
    value = re.sub(r"\s+#\d+$", "", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None

    try:
        obj = json.loads(match.group(0))
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None

    return None


def is_valid_prediction(obj: Dict[str, Any]) -> bool:
    required = set(FIELDS)
    if not required.issubset(obj.keys()):
        return False

    if obj.get("intent") not in VALID_INTENTS:
        return False

    if obj.get("language") not in VALID_LANGUAGES:
        return False

    return True


def fields_match(pred: Dict[str, Any], expected: Dict[str, Any], fields: List[str]) -> bool:
    for field in fields:
        if normalize_text(pred.get(field)) != normalize_text(expected.get(field)):
            return False
    return True


def generate_prediction(model, tokenizer, text: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = outputs[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def main() -> None:
    print("=" * 80)
    print("EVALUATE INTENT EXTRACTOR SFT")
    print("=" * 80)
    print(f"BASE_MODEL: {BASE_MODEL}")
    print(f"ADAPTER_DIR: {ADAPTER_DIR}")

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        use_fast=False,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    model.eval()

    rows = load_test_rows()

    total = len(rows)
    parse_ok = 0
    valid_ok = 0

    field_correct = {field: 0 for field in FIELDS}
    exact_all_correct = 0
    cache_relevant_correct = 0

    hard_total = 0
    hard_exact_correct = 0
    hard_cache_relevant_correct = 0

    errors: List[Dict[str, Any]] = []
    cache_errors: List[Dict[str, Any]] = []
    subject_only_errors: List[Dict[str, Any]] = []

    for idx, row in enumerate(rows, start=1):
        text = row["text"]
        expected = row["expected"]
        is_hard = bool(row.get("meta", {}).get("is_hard", False))

        if is_hard:
            hard_total += 1

        raw = generate_prediction(model, tokenizer, text)
        pred = extract_json_object(raw)

        if pred is None:
            error = {
                "idx": idx,
                "text": text,
                "error": "parse_error",
                "raw": raw,
                "expected": expected,
            }
            errors.append(error)
            cache_errors.append(error)
            continue

        parse_ok += 1

        if not is_valid_prediction(pred):
            error = {
                "idx": idx,
                "text": text,
                "error": "invalid_schema",
                "raw": raw,
                "pred": pred,
                "expected": expected,
            }
            errors.append(error)
            cache_errors.append(error)
            continue

        valid_ok += 1

        all_correct = fields_match(pred, expected, FIELDS)
        cache_ok = fields_match(pred, expected, CACHE_RELEVANT_FIELDS)

        for field in FIELDS:
            if normalize_text(pred.get(field)) == normalize_text(expected.get(field)):
                field_correct[field] += 1

        if all_correct:
            exact_all_correct += 1

        if cache_ok:
            cache_relevant_correct += 1

        if is_hard:
            if all_correct:
                hard_exact_correct += 1
            if cache_ok:
                hard_cache_relevant_correct += 1

        if not all_correct:
            error = {
                "idx": idx,
                "text": text,
                "error": "field_mismatch",
                "pred": pred,
                "expected": expected,
                "raw": raw,
                "cache_relevant_ok": cache_ok,
            }

            if len(errors) < 50:
                errors.append(error)

            if cache_ok:
                if len(subject_only_errors) < 50:
                    subject_only_errors.append(error)
            else:
                if len(cache_errors) < 50:
                    cache_errors.append(error)

    metrics = {
        "total": total,
        "parse_rate": round(parse_ok / total, 4),
        "valid_json_rate": round(valid_ok / total, 4),

        # Strict metric: tất cả field đều đúng, gồm subject.
        "exact_all_fields_accuracy": round(exact_all_correct / total, 4),

        # Metric quan trọng nhất cho Intelligent Cache.
        "cache_relevant_accuracy": round(cache_relevant_correct / total, 4),
        "cache_relevant_fields": CACHE_RELEVANT_FIELDS,

        "field_accuracy": {
            field: round(count / total, 4)
            for field, count in field_correct.items()
        },

        "hard_total": hard_total,
        "hard_exact_accuracy": round(hard_exact_correct / max(1, hard_total), 4),
        "hard_cache_relevant_accuracy": round(
            hard_cache_relevant_correct / max(1, hard_total),
            4,
        ),

        "num_errors_logged": len(errors),
        "num_cache_errors_logged": len(cache_errors),
        "num_subject_only_errors_logged": len(subject_only_errors),
    }

    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    out_dir = Path("reports")
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "intent_extractor_eval_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    with (out_dir / "intent_extractor_eval_errors.json").open("w", encoding="utf-8") as f:
        json.dump(errors, f, ensure_ascii=False, indent=2)

    with (out_dir / "intent_extractor_eval_cache_errors.json").open("w", encoding="utf-8") as f:
        json.dump(cache_errors, f, ensure_ascii=False, indent=2)

    with (out_dir / "intent_extractor_eval_subject_only_errors.json").open("w", encoding="utf-8") as f:
        json.dump(subject_only_errors, f, ensure_ascii=False, indent=2)

    print()
    print("[DONE] Saved reports/intent_extractor_eval_metrics.json")
    print("[DONE] Saved reports/intent_extractor_eval_errors.json")
    print("[DONE] Saved reports/intent_extractor_eval_cache_errors.json")
    print("[DONE] Saved reports/intent_extractor_eval_subject_only_errors.json")


if __name__ == "__main__":
    main()