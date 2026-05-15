from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List

RANDOM_SEED = 42

INPUT_PATH = Path("data/intent_extraction/intent_extraction_dataset.json")
OUTPUT_DIR = Path("data/intent_extraction/sft")

TRAIN_PATH = OUTPUT_DIR / "train.jsonl"
VAL_PATH = OUTPUT_DIR / "val.jsonl"
TEST_PATH = OUTPUT_DIR / "test.jsonl"
TEST_META_PATH = OUTPUT_DIR / "test_with_meta.jsonl"
STATS_PATH = OUTPUT_DIR / "split_stats.json"

INTENTS = ["order", "consultant", "faq", "ignore"]

SYSTEM_PROMPT = (
    "You are an intent extraction model for an F&B multi-agent assistant. "
    "Extract structured fields from the customer query. "
    "Return valid JSON only. "
    "Do not add markdown. "
    "Required fields: subject, action, context, cache_key, intent, language. "
    "intent must be one of: order, consultant, faq, ignore. "
    "language must be vi or en."
)


def load_dataset() -> List[Dict[str, Any]]:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {INPUT_PATH}")

    with INPUT_PATH.open("r", encoding="utf-8") as f:
        samples = json.load(f)

    if not isinstance(samples, list):
        raise ValueError("Dataset must be a list")

    return samples


def build_target_json(sample: Dict[str, Any]) -> str:
    target = {
        "subject": sample["subject"],
        "action": sample["action"],
        "context": sample["context"],
        "cache_key": sample["cache_key"],
        "intent": sample["intent"],
        "language": sample["language"],
    }
    return json.dumps(target, ensure_ascii=False, separators=(",", ":"))


def to_sft_record(sample: Dict[str, Any]) -> Dict[str, Any]:
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
                "content": build_target_json(sample),
            },
        ]
    }


def to_meta_record(sample: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "text": sample["text"],
        "expected": {
            "subject": sample["subject"],
            "action": sample["action"],
            "context": sample["context"],
            "cache_key": sample["cache_key"],
            "intent": sample["intent"],
            "language": sample["language"],
        },
        "meta": {
            "is_hard": sample["is_hard"],
            "source_template": sample["source_template"],
        },
    }


def stratified_split(
    samples: List[Dict[str, Any]],
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    random.seed(RANDOM_SEED)

    train: List[Dict[str, Any]] = []
    val: List[Dict[str, Any]] = []
    test: List[Dict[str, Any]] = []

    for intent in INTENTS:
        group = [s for s in samples if s["intent"] == intent]
        random.shuffle(group)

        n = len(group)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train.extend(group[:n_train])
        val.extend(group[n_train : n_train + n_val])
        test.extend(group[n_train + n_val :])

    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    return train, val, test


def write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def count_by_key(samples: List[Dict[str, Any]], key: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for sample in samples:
        value = str(sample[key])
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def build_stats(
    train: List[Dict[str, Any]],
    val: List[Dict[str, Any]],
    test: List[Dict[str, Any]],
) -> Dict[str, Any]:
    def one_split_stats(split_samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        hard_count = sum(1 for s in split_samples if s["is_hard"])
        return {
            "total": len(split_samples),
            "by_intent": count_by_key(split_samples, "intent"),
            "by_language": count_by_key(split_samples, "language"),
            "hard_count": hard_count,
            "hard_ratio": round(hard_count / max(1, len(split_samples)), 4),
        }

    return {
        "train": one_split_stats(train),
        "val": one_split_stats(val),
        "test": one_split_stats(test),
    }


def validate_splits(
    train: List[Dict[str, Any]],
    val: List[Dict[str, Any]],
    test: List[Dict[str, Any]],
) -> None:
    all_texts = [s["text"] for s in train + val + test]
    if len(all_texts) != len(set(all_texts)):
        raise ValueError("Duplicate text across SFT splits")

    for split_name, split_samples in {
        "train": train,
        "val": val,
        "test": test,
    }.items():
        by_intent = count_by_key(split_samples, "intent")
        missing = [intent for intent in INTENTS if intent not in by_intent]
        if missing:
            raise ValueError(f"{split_name} missing intents: {missing}")

        for sample in split_samples:
            target = json.loads(build_target_json(sample))
            required = {
                "subject",
                "action",
                "context",
                "cache_key",
                "intent",
                "language",
            }
            if set(target) != required:
                raise ValueError(f"Invalid target keys: {target}")

            if target["intent"] not in INTENTS:
                raise ValueError(f"Invalid intent: {target}")

            if target["language"] not in {"vi", "en"}:
                raise ValueError(f"Invalid language: {target}")


def main() -> None:
    samples = load_dataset()
    train, val, test = stratified_split(samples)

    validate_splits(train, val, test)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    write_jsonl(TRAIN_PATH, [to_sft_record(s) for s in train])
    write_jsonl(VAL_PATH, [to_sft_record(s) for s in val])
    write_jsonl(TEST_PATH, [to_sft_record(s) for s in test])
    write_jsonl(TEST_META_PATH, [to_meta_record(s) for s in test])

    stats = build_stats(train, val, test)
    with STATS_PATH.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("=" * 80)
    print("INTENT EXTRACTION SFT DATASET READY")
    print("=" * 80)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print()
    print(f"TRAIN: {TRAIN_PATH}")
    print(f"VAL:   {VAL_PATH}")
    print(f"TEST:  {TEST_PATH}")
    print(f"META:  {TEST_META_PATH}")


if __name__ == "__main__":
    main()