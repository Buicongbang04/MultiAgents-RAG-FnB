from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


MODEL_NAME = os.getenv(
    "ROUTER_BASE_MODEL",
    "Qwen/Qwen2.5-0.5B-Instruct",
)

TRAIN_PATH = Path("data/router/sft/train.jsonl")
VAL_PATH = Path("data/router/sft/val.jsonl")

OUTPUT_DIR = os.getenv(
    "ROUTER_SFT_OUTPUT_DIR",
    "models/router-qwen2.5-0.5b-lora",
)

MAX_LENGTH = int(os.getenv("ROUTER_MAX_LENGTH", "192"))
USE_4BIT = os.getenv("ROUTER_USE_4BIT", "true").lower() == "true"


class RouterSFTDataset(Dataset):
    def __init__(self, path: Path, tokenizer, max_length: int):
        self.rows = self._load_jsonl(path)
        self.tokenizer = tokenizer
        self.max_length = max_length

    @staticmethod
    def _load_jsonl(path: Path) -> list[dict]:
        assert path.exists(), f"File not found: {path}"

        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))

        return rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        messages = row["messages"]

        system_msg = messages[0]["content"]
        user_msg = messages[1]["content"]
        assistant_msg = messages[2]["content"].strip()

        prompt_messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

        prompt_text = self.tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        full_text = prompt_text + assistant_msg + self.tokenizer.eos_token

        prompt_ids = self.tokenizer(
            prompt_text,
            add_special_tokens=False,
        )["input_ids"]

        full = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            add_special_tokens=False,
        )

        input_ids = full["input_ids"]
        attention_mask = full["attention_mask"]

        labels = input_ids.copy()
        prompt_len = min(len(prompt_ids), len(labels))

        for i in range(prompt_len):
            labels[i] = -100

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class DataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        input_ids = [f["input_ids"] for f in features]
        attention_mask = [f["attention_mask"] for f in features]
        labels = [f["labels"] for f in features]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )

        attention_mask = torch.nn.utils.rnn.pad_sequence(
            attention_mask,
            batch_first=True,
            padding_value=0,
        )

        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=-100,
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def main():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if USE_4BIT:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        quant_config = None

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
    )

    if USE_4BIT:
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

    train_dataset = RouterSFTDataset(TRAIN_PATH, tokenizer, MAX_LENGTH)
    val_dataset = RouterSFTDataset(VAL_PATH, tokenizer, MAX_LENGTH)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        warmup_ratio=0.05,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=True,
        bf16=False,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollator(tokenizer),
    )

    trainer.train()

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("=" * 80)
    print("ROUTER SFT TRAINING DONE")
    print("=" * 80)
    print(f"Saved adapter to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()