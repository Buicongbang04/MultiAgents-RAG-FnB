from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

from app.core.config import get_settings, get_training_config
from scripts.train_router_sft import build_bnb_config

# Model/đường dẫn: configs/app.yaml (intent_extractor.hf) — siêu tham số: configs/training.yaml
CONFIG = get_training_config().intent_extractor_sft
BASE_MODEL = get_settings().intent_extractor.hf.base_model
OUTPUT_DIR = get_settings().intent_extractor.hf.adapter_dir

TRAIN_FILE = CONFIG.train_path
VAL_FILE = CONFIG.val_path

MAX_SEQ_LENGTH = CONFIG.max_length


def build_text(tokenizer, messages: List[Dict[str, str]]) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


def tokenize_example(example):
    text = build_text(tokenizer, example["messages"])

    encoded = tokenizer(
        text,
        max_length=MAX_SEQ_LENGTH,
        truncation=True,
        padding=False,
    )

    encoded["labels"] = encoded["input_ids"].copy()
    return encoded


def main() -> None:
    global tokenizer

    print("=" * 80)
    print("TRAIN INTENT EXTRACTOR SFT - TRANSFORMERS TRAINER")
    print("=" * 80)
    print(f"BASE_MODEL: {BASE_MODEL}")
    print(f"OUTPUT_DIR: {OUTPUT_DIR}")

    dataset = load_dataset(
        "json",
        data_files={
            "train": TRAIN_FILE,
            "validation": VAL_FILE,
        },
    )

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        use_fast=False,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = build_bnb_config(CONFIG.quantization) if CONFIG.use_4bit else None

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    model.config.use_cache = False
    if CONFIG.use_4bit:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(**CONFIG.lora)

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    tokenized = dataset.map(
        tokenize_example,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing intent extraction dataset",
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
        return_tensors="pt",
    )

    training_args = TrainingArguments(output_dir=OUTPUT_DIR, **CONFIG.training_args)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("=" * 80)
    print("INTENT EXTRACTOR SFT DONE")
    print("=" * 80)
    print(f"Saved adapter to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()