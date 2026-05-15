from __future__ import annotations

import json
import random
from pathlib import Path


SEED = 42

INPUT_PATH = Path(
    "data/router/router_dataset_800.json"
)

OUTPUT_DIR = Path(
    "data/router/sft"
)

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1


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


def load_dataset():
    with INPUT_PATH.open(
        "r",
        encoding="utf-8"
    ) as f:
        return json.load(f)


def to_chat_format(sample):

    return {
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": sample["text"],
            },
            {
                "role": "assistant",
                "content": sample["intent"],
            },
        ]
    }


def stratified_split(dataset):

    grouped = {}

    for sample in dataset:
        grouped.setdefault(
            sample["intent"],
            []
        ).append(sample)

    train = []
    val = []
    test = []

    for intent, items in grouped.items():

        random.shuffle(items)

        n = len(items)

        train_end = int(
            n * TRAIN_RATIO
        )

        val_end = train_end + int(
            n * VAL_RATIO
        )

        train.extend(
            items[:train_end]
        )

        val.extend(
            items[train_end:val_end]
        )

        test.extend(
            items[val_end:]
        )

    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    return train, val, test


def save_jsonl(path, rows):

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        for row in rows:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False
                )
                + "\n"
            )


def print_stats(name, data):

    counts = {}

    for x in data:
        intent = (
            x["messages"][-1]
            ["content"]
        )

        counts[intent] = (
            counts.get(intent, 0)
            + 1
        )

    print(
        f"{name}: "
        f"{len(data)}"
    )

    print(counts)


def main():

    random.seed(SEED)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset = load_dataset()

    train, val, test = stratified_split(
        dataset
    )

    train_chat = [
        to_chat_format(x)
        for x in train
    ]

    val_chat = [
        to_chat_format(x)
        for x in val
    ]

    test_chat = [
        to_chat_format(x)
        for x in test
    ]

    save_jsonl(
        OUTPUT_DIR / "train.jsonl",
        train_chat,
    )

    save_jsonl(
        OUTPUT_DIR / "val.jsonl",
        val_chat,
    )

    save_jsonl(
        OUTPUT_DIR / "test.jsonl",
        test_chat,
    )

    print("=" * 80)
    print(
        "SFT DATASET READY"
    )
    print("=" * 80)

    print_stats(
        "TRAIN",
        train_chat,
    )

    print_stats(
        "VAL",
        val_chat,
    )

    print_stats(
        "TEST",
        test_chat,
    )


if __name__ == "__main__":
    main()