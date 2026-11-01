from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

DATASET_PATH = Path("data/intent_extraction/intent_extraction_dataset.json")

EXPECTED_TOTAL = 1000
EXPECTED_PER_INTENT = 250
VALID_INTENTS = {"order", "consultant", "faq", "ignore"}
VALID_LANGUAGES = {"vi", "en"}

REQUIRED_FIELDS = {
    "text",
    "intent",
    "language",
    "subject",
    "action",
    "context",
    "cache_key",
    "is_hard",
    "source_template",
}


def load_dataset() -> List[Dict[str, Any]]:
    assert DATASET_PATH.exists(), f"Dataset not found: {DATASET_PATH}"
    with DATASET_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def test_dataset_quality() -> None:
    samples = load_dataset()

    assert len(samples) == EXPECTED_TOTAL

    texts = [sample["text"].strip().lower() for sample in samples]
    assert len(texts) == len(set(texts)), "Duplicate text detected"

    by_intent: Dict[str, int] = {}
    hard_count = 0

    for sample in samples:
        missing = REQUIRED_FIELDS - set(sample)
        assert not missing, f"Missing fields {missing}: {sample}"

        assert sample["text"].strip()
        assert sample["intent"] in VALID_INTENTS
        assert sample["language"] in VALID_LANGUAGES

        if sample["intent"] in {"order", "consultant", "faq"}:
            assert sample["action"].strip(), f"Empty action: {sample}"
            assert sample["cache_key"].strip(), f"Empty cache_key: {sample}"

        by_intent[sample["intent"]] = by_intent.get(sample["intent"], 0) + 1
        hard_count += int(bool(sample["is_hard"]))

    for intent in VALID_INTENTS:
        assert by_intent.get(intent, 0) == EXPECTED_PER_INTENT

    hard_ratio = hard_count / len(samples)
    assert 0.10 <= hard_ratio <= 0.20, f"Unexpected hard ratio: {hard_ratio:.3f}"

    print("DATASET QUALITY PASS")
    print("by_intent:", by_intent)
    print("hard_ratio:", round(hard_ratio, 4))