from __future__ import annotations

import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = os.getenv("INTENT_EXTRACTOR_BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
ADAPTER_DIR = os.getenv(
    "INTENT_EXTRACTOR_ADAPTER_DIR",
    "models/intent-extractor-qwen2.5-0.5b-lora",
)
MERGED_DIR = os.getenv(
    "INTENT_EXTRACTOR_MERGED_MODEL_DIR",
    "models/intent-extractor-qwen2.5-0.5b-merged",
)


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        use_fast=False,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    merged = model.merge_and_unload()

    merged.save_pretrained(MERGED_DIR, safe_serialization=True)
    tokenizer.save_pretrained(MERGED_DIR)

    print(f"[DONE] Saved merged extractor to: {MERGED_DIR}")


if __name__ == "__main__":
    main()