from __future__ import annotations


import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.core.config import get_settings, get_training_config

HF_CONFIG = get_settings().intent_extractor.hf
MERGE_CONFIG = get_training_config().merge
BASE_MODEL = HF_CONFIG.base_model
ADAPTER_DIR = HF_CONFIG.adapter_dir
MERGED_DIR = HF_CONFIG.merged_model_dir


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        use_fast=False,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=getattr(torch, MERGE_CONFIG.dtype),
        device_map=MERGE_CONFIG.device_map,
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    merged = model.merge_and_unload()

    merged.save_pretrained(MERGED_DIR, safe_serialization=True)
    tokenizer.save_pretrained(MERGED_DIR)

    print(f"[DONE] Saved merged extractor to: {MERGED_DIR}")


if __name__ == "__main__":
    main()