from __future__ import annotations

from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


from app.core.config import get_settings, get_training_config

HF_CONFIG = get_settings().router.hf
MERGE_CONFIG = get_training_config().merge
BASE_MODEL = HF_CONFIG.base_model
ADAPTER_DIR = HF_CONFIG.adapter_dir
OUTPUT_DIR = HF_CONFIG.merged_model_dir


def main() -> None:
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("MERGE ROUTER LORA")
    print("=" * 80)
    print(f"Base model:  {BASE_MODEL}")
    print(f"Adapter dir: {ADAPTER_DIR}")
    print(f"Output dir:  {OUTPUT_DIR}")

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=getattr(torch, MERGE_CONFIG.dtype),
        device_map=MERGE_CONFIG.device_map,
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(
        base_model,
        ADAPTER_DIR,
    )

    print("[INFO] Merging adapter into base model...")
    merged_model = model.merge_and_unload()

    print("[INFO] Saving merged model...")
    merged_model.save_pretrained(
        OUTPUT_DIR,
        safe_serialization=True,
    )

    tokenizer.save_pretrained(OUTPUT_DIR)

    print("=" * 80)
    print("MERGE DONE")
    print("=" * 80)
    print(f"Saved merged router to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()