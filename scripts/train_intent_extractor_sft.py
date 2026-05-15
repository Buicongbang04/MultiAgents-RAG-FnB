from __future__ import annotations

import os
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

BASE_MODEL = os.getenv("INTENT_EXTRACTOR_BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")

DATA_DIR = Path("data/intent_extraction/sft")
TRAIN_FILE = str(DATA_DIR / "train.jsonl")
VAL_FILE = str(DATA_DIR / "val.jsonl")

OUTPUT_DIR = os.getenv(
    "INTENT_EXTRACTOR_ADAPTER_DIR",
    "models/intent-extractor-qwen2.5-0.5b-lora",
)

MAX_SEQ_LENGTH = 512


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

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

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

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=True,
        optim="paged_adamw_8bit",
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=True,
    )

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